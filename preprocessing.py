"""Preprocessing pipeline for DaT-SPECT NIfTI volumes.

Pipeline (see CFG for parameters):
  1. Load the NIfTI volume and reorient to RAS (canonical) if needed.
  2. Resample to isotropic voxel spacing.
  3. Estimate an uptake centroid from thresholded intensities.
  4. Extract a centroid-centered cubic crop.
  5. Clip intensities to the per-volume low/high percentiles.
  6. Scale the clipped volume to [0, 1].

Public entry point: ``preprocess_volume(path) -> np.ndarray`` of shape
``(CFG["crop_size"],) * 3``, dtype float32, values in [0, 1].

``preprocess_volume_verbose`` returns the same array plus a metadata dict
(centroid location, resampled shape, crop-boundary touch flags, intensity
stats) used for cache manifests and QC reporting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output

CFG: dict[str, float | int] = {
    "target_spacing_mm": 2.9,
    "crop_size": 64,
    "centroid_threshold_pct": 0.4,  # keep the top 40% of nonzero intensity mass
    "intensity_p_low": 1,
    "intensity_p_high": 99,
}

# Bump this when CFG's *semantics* change in a way that isn't captured by the
# CFG values themselves (e.g. a different centroid algorithm). Cached arrays
# are keyed on hash(CFG, PREPROCESSING_VERSION), so any change here or to CFG
# invalidates old cache entries instead of silently reusing them.
PREPROCESSING_VERSION = "v1"


def config_hash(cfg: dict[str, Any] | None = None) -> str:
    """Short, stable hash identifying a preprocessing configuration."""
    cfg = CFG if cfg is None else cfg
    payload = json.dumps({"version": PREPROCESSING_VERSION, **cfg}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


CONFIG_HASH = config_hash()


@dataclass
class PreprocessMeta:
    """Metadata describing how a single volume was preprocessed."""

    uid: str
    source_path: str
    original_shape: tuple[int, ...]
    original_zooms: tuple[float, ...]
    resampled_shape: tuple[int, ...]
    centroid_voxel_resampled: tuple[float, float, float]
    crop_origin: tuple[int, int, int]
    crop_size: int
    intensity_low: float
    intensity_high: float
    nonzero_fraction: float
    vmin: float
    vmax: float
    vmean: float
    vstd: float
    all_finite: bool
    touches_boundary: bool
    boundary_axes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["original_shape"] = list(self.original_shape)
        d["original_zooms"] = [float(z) for z in self.original_zooms]
        d["resampled_shape"] = list(self.resampled_shape)
        d["centroid_voxel_resampled"] = [float(c) for c in self.centroid_voxel_resampled]
        d["crop_origin"] = list(self.crop_origin)
        return d


def _estimate_centroid_voxel(data: np.ndarray, threshold_pct: float) -> np.ndarray:
    """Unweighted voxel-space centroid of the top ``threshold_pct`` fraction of
    nonzero intensity mass. Falls back to the array's geometric center when the
    volume has no signal at all.

    This is deliberately a cheap intensity-threshold heuristic rather than an
    atlas registration: for DaT-SPECT, tracer uptake is concentrated in the
    striatum, so thresholding out the low-intensity background and centroiding
    the remainder reliably lands on/near the striatal region without any
    external atlas (validated visually during EDA).
    """
    nz = data[data > 0]
    if nz.size == 0:
        return (np.array(data.shape, dtype=np.float64) - 1) / 2.0
    threshold = np.percentile(nz, (1.0 - threshold_pct) * 100.0)
    mask = data > threshold
    if not mask.any():
        return (np.array(data.shape, dtype=np.float64) - 1) / 2.0
    coords = np.array(np.nonzero(mask), dtype=np.float64)
    return coords.mean(axis=1)


def _centroid_crop(
    data: np.ndarray, centroid_voxel: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a ``size**3`` cube centered on ``centroid_voxel``, zero-padding
    where the crop extends past the volume bounds. Returns the crop and the
    (possibly negative / out-of-bounds) crop origin in source-voxel coords.
    """
    half = size // 2
    start = np.round(centroid_voxel).astype(int) - half
    end = start + size

    out = np.zeros((size, size, size), dtype=data.dtype)
    src_shape = np.array(data.shape)

    src_start = np.clip(start, 0, src_shape)
    src_end = np.clip(end, 0, src_shape)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    if np.all(src_end > src_start):
        out[
            dst_start[0]:dst_end[0],
            dst_start[1]:dst_end[1],
            dst_start[2]:dst_end[2],
        ] = data[
            src_start[0]:src_end[0],
            src_start[1]:src_end[1],
            src_start[2]:src_end[2],
        ]
    return out, start


def _normalize_intensity(
    vol: np.ndarray, p_low: float, p_high: float
) -> tuple[np.ndarray, float, float]:
    """Clip to [p_low, p_high] percentiles of this volume and rescale to [0, 1]."""
    lo, hi = np.percentile(vol, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip(vol, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled.astype(np.float32), float(lo), float(hi)


def _boundary_touch(mask: np.ndarray) -> list[str]:
    """Which faces of the crop cube have signal touching the outer voxel layer."""
    axes = []
    if mask[0, :, :].any() or mask[-1, :, :].any():
        axes.append("x")
    if mask[:, 0, :].any() or mask[:, -1, :].any():
        axes.append("y")
    if mask[:, :, 0].any() or mask[:, :, -1].any():
        axes.append("z")
    return axes


def crop_quality_warnings(volume: np.ndarray) -> list[str]:
    """Recompute QC warnings from a finished (crop_size,)*3 normalized volume
    alone -- works identically whether the volume was just produced by
    ``preprocess_volume`` or reloaded from an on-disk cache, so cache hits and
    misses get exactly the same QC treatment.
    """
    warnings: list[str] = []
    all_finite = bool(np.all(np.isfinite(volume)))
    if not all_finite:
        warnings.append("non_finite_values")
    nonzero_fraction = float((volume > 0).mean()) if all_finite else 0.0
    if nonzero_fraction < 0.01:
        warnings.append("nearly_empty_crop")
    high_mask = volume > np.percentile(volume, 90) if all_finite and volume.max() > 0 else (volume > 1)
    boundary_axes = _boundary_touch(high_mask)
    if boundary_axes:
        warnings.append(f"high_intensity_touches_boundary:{','.join(boundary_axes)}")
    return warnings


def preprocess_volume_verbose(
    path: str | Path, uid: str | None = None, cfg: dict[str, Any] | None = None
) -> tuple[np.ndarray, PreprocessMeta]:
    """Run the full preprocessing pipeline and return ``(array, metadata)``."""
    cfg = CFG if cfg is None else cfg
    path = Path(path)
    uid = uid if uid is not None else path.name.split(".")[0]

    img = nib.load(str(path))
    original_shape = img.shape
    original_zooms = tuple(float(z) for z in img.header.get_zooms()[:3])

    img = nib.as_closest_canonical(img)  # ensure RAS orientation (no-op if already RAS)
    resampled = resample_to_output(
        img, voxel_sizes=float(cfg["target_spacing_mm"]), order=1
    )
    data = np.asarray(resampled.get_fdata(dtype=np.float32))
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)  # SPECT counts shouldn't be negative; guard against
    # small negative reconstruction-filter artifacts seen in the raw data (see EDA).

    resampled_shape = data.shape
    centroid_voxel = _estimate_centroid_voxel(data, float(cfg["centroid_threshold_pct"]))

    crop_size = int(cfg["crop_size"])
    crop, crop_origin = _centroid_crop(data, centroid_voxel, crop_size)

    normalized, lo, hi = _normalize_intensity(
        crop, float(cfg["intensity_p_low"]), float(cfg["intensity_p_high"])
    )

    all_finite = bool(np.all(np.isfinite(normalized)))
    nonzero_fraction = float((normalized > 0).mean())
    warnings = crop_quality_warnings(normalized)
    boundary_axes = []
    for w in warnings:
        if w.startswith("high_intensity_touches_boundary:"):
            boundary_axes = w.split(":", 1)[1].split(",")

    meta = PreprocessMeta(
        uid=uid,
        source_path=str(path),
        original_shape=tuple(int(s) for s in original_shape),
        original_zooms=original_zooms,
        resampled_shape=tuple(int(s) for s in resampled_shape),
        centroid_voxel_resampled=tuple(float(c) for c in centroid_voxel),
        crop_origin=tuple(int(c) for c in crop_origin),
        crop_size=crop_size,
        intensity_low=lo,
        intensity_high=hi,
        nonzero_fraction=nonzero_fraction,
        vmin=float(normalized.min()),
        vmax=float(normalized.max()),
        vmean=float(normalized.mean()),
        vstd=float(normalized.std()),
        all_finite=all_finite,
        touches_boundary=bool(boundary_axes),
        boundary_axes=boundary_axes,
        warnings=warnings,
    )
    return normalized, meta


def preprocess_volume(path: str | Path, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Load, resample, centroid-crop, and intensity-normalize one NIfTI volume.

    Returns a ``(crop_size, crop_size, crop_size)`` float32 array in [0, 1].
    """
    array, _ = preprocess_volume_verbose(path, cfg=cfg)
    return array


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is None:
        print("Usage: python preprocessing.py <path-to-nifti>")
        raise SystemExit(1)
    arr, meta = preprocess_volume_verbose(target)
    print(f"config_hash = {CONFIG_HASH}")
    print(f"output shape = {arr.shape}, dtype = {arr.dtype}")
    print(json.dumps(meta.to_dict(), indent=2))
