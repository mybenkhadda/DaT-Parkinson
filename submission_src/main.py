"""DaT-Parkinson code-execution submission entry point.

Runs the trained 2-path ensemble (2.5D CNN + 3D CNN, weighted-averaged and
calibrated) against whatever competition data directory is mounted, and
writes submission.csv beside it.

NOTE on the removed third path: the pipeline originally had a third base
model ("Path A": handcrafted uptake features + an XGBoost GBM). Its joblib
artifact pickled an ``xgboost.sklearn.XGBClassifier``, which requires the
``xgboost`` package to unpickle -- a hard runtime dependency the official
competition environment is not guaranteed to provide (and empirically does
not: this is what previously produced ``ModuleNotFoundError`` right after
"Loading inference assets."). Path A was removed rather than made to work,
because the trained, cross-validated ensemble already assigned it an exact
weight of 0.0 (see final_ensemble.joblib's history / project notes) -- it
contributed nothing to any prediction, so dropping it changes no output
while eliminating the fragile dependency entirely. This is not a fallback or
a skipped-on-failure shortcut: it is a mathematically inert component that
was safe to remove unconditionally.

Layout is NOT assumed. The evaluator has been observed to extract the
archive either directly into the working directory (main.py alongside
data/) or into a src/ subdirectory (data/ one level up from main.py) --
resolve_paths() below probes for both instead of hardcoding either one.

No network access, no internet-downloaded weights: every model file is loaded
from the artifacts/ directory bundled in this same archive, resolved relative
to this file's own location (never relative to the current working directory).
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import traceback
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

import config
import models
import preprocessing as pp

# Every third-party package actually imported anywhere in this package (this
# file, config.py, models.py, preprocessing.py). Checked proactively before
# any model loading is attempted, so a missing package produces one clean,
# generic diagnostic instead of an incidental failure deep inside joblib/
# torch deserialization. Deliberately does NOT include xgboost/scipy/
# sklearn -- none of the remaining code paths need them (see module
# docstring on the Path A removal).
REQUIRED_MODULES = ("numpy", "pandas", "torch", "torchvision", "joblib", "nibabel")


# ---------------------------------------------------------------------------
# Runtime-stage tracking (generic, static stage names only -- never a uid,
# filename, path, prediction, or count; see run_entrypoint()'s exception
# handling, which reports the stage alongside the exception type)
# ---------------------------------------------------------------------------
_RUNTIME_STAGE = "startup"

# The only strings _RUNTIME_STAGE may ever hold. Kept as a closed set so a
# future edit can't accidentally pass something dynamic (a filename, a uid)
# into set_runtime_stage() and have it leak into an exception message.
_PERMITTED_STAGES = frozenset({
    "startup",
    "input validation",
    "asset loading",
    "neural-model loading",
    "ensemble loading",
    "calibration loading",
    "asset validation",
    "inference",
    "output validation",
    "output writing",
})


def set_runtime_stage(stage: str) -> None:
    global _RUNTIME_STAGE
    assert stage in _PERMITTED_STAGES, f"not a permitted runtime stage: {stage!r}"
    _RUNTIME_STAGE = stage


def debug_enabled() -> bool:
    """Whether verbose local debugging (tracebacks) is opted into. Backed by
    config.SUBMISSION_DEBUG (the single source of truth for the
    SUBMISSION_DEBUG=1 environment variable) so this and config agree."""
    return config.SUBMISSION_DEBUG


# ---------------------------------------------------------------------------
# Static preflight (dependencies + bundled assets only -- never touches test
# scans or anything derived from the hidden test set)
# ---------------------------------------------------------------------------


def verify_runtime_dependencies() -> None:
    """Proactively check every required third-party package is importable.
    Raises the same ModuleNotFoundError shape Python itself would raise on a
    failed import, so it is handled identically by run_entrypoint()."""
    for module_name in REQUIRED_MODULES:
        if find_spec(module_name) is None:
            raise ModuleNotFoundError(
                f"Required module unavailable: {module_name}", name=module_name
            )


def require_artifact(path: Path) -> None:
    """Fail fast, with a clean generic message, if a bundled artifact file
    the ZIP is supposed to contain is missing -- rather than surfacing
    whatever raw (path-containing) error the underlying loader would raise."""
    if not path.is_file():
        raise FileNotFoundError("Required inference artifact is unavailable.")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    nifti_dir: Path
    template_path: Path
    output_path: Path
    artifacts_dir: Path


def _candidate_work_dirs(main_file: Path, cwd: Path) -> list[Path]:
    """Ordered, de-duplicated list of directories that might be the
    competition working directory (the one containing data/), given where
    this file actually lives and where the process was launched from.

    Checked in this order:
      1. cwd -- the documented contract is that the process runs with
         /code_execution/ as its working directory, regardless of where
         main.py sits inside it.
      2. this file's own directory -- covers main.py extracted directly
         into the working directory, alongside data/.
      3. this file's parent's parent -- covers main.py extracted into a
         src/ subdirectory, with data/ one level up.
    """
    main_file = main_file.resolve()
    file_parent = main_file.parent
    raw_candidates = [cwd.resolve(), file_parent, file_parent.parent]

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in raw_candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _resolve_work_dir(main_file: Path, cwd: Path) -> Optional[Path]:
    """First candidate directory that actually contains data/submission_format.csv,
    or None if none of them do."""
    for candidate in _candidate_work_dirs(main_file, cwd):
        if (candidate / "data" / "submission_format.csv").is_file():
            return candidate
    return None


def resolve_paths() -> Paths:
    """Resolve the competition working directory without assuming a fixed
    main.py location. Bundled assets (artifacts/) are always resolved
    relative to this file's own directory, never relative to the current
    working directory or to the (possibly different) data working directory.

    This function ONLY locates paths -- it does not validate that the
    imaging directory contains anything, nor that bundled artifacts are
    present. That's validate_runtime_inputs()'s job (called immediately
    after this, before any model loading), so all data-availability
    failures surface at the same, early point in main() regardless of which
    specific input is missing.

    Raises FileNotFoundError with a deliberately generic message if the
    competition working directory itself cannot be located -- the searched
    paths are never included in the exception text (see
    config.SUBMISSION_DEBUG for local debugging)."""
    main_file = Path(__file__).resolve()
    artifacts_dir = main_file.parent / "artifacts"

    work_dir = _resolve_work_dir(main_file, Path.cwd())
    if work_dir is None:
        raise FileNotFoundError("Required competition input files are unavailable.")

    data_dir = work_dir / "data"
    return Paths(
        data_dir=data_dir,
        nifti_dir=data_dir / "niftis",
        template_path=data_dir / "submission_format.csv",
        output_path=work_dir / "submission.csv",
        artifacts_dir=artifacts_dir,
    )


def validate_runtime_inputs(paths: Paths) -> None:
    """Cheap, filesystem-only checks that the competition actually mounted
    its inputs, run BEFORE any model artifact is loaded. This is what
    catches a missing/empty data mount immediately, rather than several
    minutes and 50+ MB of model loading later.

    Deliberately does not parse submission_format.csv's contents or count
    NIfTI files -- that's load_submission_template()'s / discover_exam_files()'s
    job (also called before model loading in main(), just after this).
    Never prints the searched paths, a directory listing, filenames, uids,
    or any count -- only a generic message per failure mode."""
    if not paths.template_path.is_file():
        raise FileNotFoundError("Required competition input files are unavailable.")

    if not paths.nifti_dir.is_dir():
        raise FileNotFoundError("Required imaging input directory is unavailable.")

    has_supported_file = any(
        path.is_file() and (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz"))
        for path in paths.nifti_dir.rglob("*")
    )
    if not has_supported_file:
        raise FileNotFoundError("Required imaging inputs are unavailable.")


def load_submission_template(paths: Paths) -> pd.DataFrame:
    """Load submission_format.csv, the authoritative source of expected uids
    and row order."""
    df = pd.read_csv(paths.template_path)
    if "uid" not in df.columns:
        raise ValueError("submission_format.csv is missing the required 'uid' column")
    df["uid"] = df["uid"].astype(str)  # normalize dtype so later string comparisons are consistent
    if df["uid"].duplicated().any():
        raise ValueError("submission_format.csv contains duplicate uid values")
    if df.empty:
        raise ValueError("submission_format.csv contains zero rows")
    return df


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_exam_files(nifti_dir: Path, uids: list[str]) -> dict[str, Path]:
    """Deterministically map each uid to exactly one NIfTI file under
    ``nifti_dir`` (searched recursively).

    Matching is done on the *exact* filename stem after stripping a known
    NIfTI suffix (".nii.gz" or ".nii") -- never substring/``in`` matching --
    so a uid can never accidentally match another uid's file (e.g. "abc"
    matching "abc123.nii.gz").

    Raises ValueError if any requested uid has zero or more-than-one
    candidate file. Never prints or includes in exception text any filename,
    uid, or count derived from the (potentially hidden) test set -- detailed
    counts are only included when config.SUBMISSION_DEBUG is enabled.
    """
    candidates: dict[str, list[Path]] = {}
    for path in sorted(nifti_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        stem = None
        for suffix in config.NIFTI_SUFFIXES:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                break
        if stem is None:
            continue  # not a recognized NIfTI file; ignore silently (e.g. .DS_Store)
        candidates.setdefault(stem, []).append(path)

    resolved: dict[str, Path] = {}
    missing = 0
    ambiguous = 0
    for uid in uids:
        matches = sorted(candidates.get(uid, []))
        if len(matches) == 0:
            missing += 1
        elif len(matches) > 1:
            ambiguous += 1
        else:
            resolved[uid] = matches[0]

    if missing or ambiguous:
        if config.SUBMISSION_DEBUG:
            raise ValueError(
                f"NIfTI file discovery failed: {missing} uid(s) with no matching file, "
                f"{ambiguous} uid(s) with multiple matching files "
                f"(expected exactly one '<uid>.nii' or '<uid>.nii.gz' per uid)."
            )
        raise ValueError("Required imaging files could not be matched to every examination.")
    return resolved


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------


def load_nifti_volume(nifti_path: Path):
    """Run the exact training-time preprocessing chain on one NIfTI file.
    Returns (volume, meta). Raises on any read/processing failure -- the
    caller decides whether a fallback is appropriate."""
    if not nifti_path.is_file():
        raise FileNotFoundError("Resolved NIfTI path no longer exists")
    return pp.preprocess_volume_verbose(nifti_path)


def preprocess_volume(nifti_path: Path) -> np.ndarray:
    vol, _meta = load_nifti_volume(nifti_path)
    return vol


# ---------------------------------------------------------------------------
# Model loading (each artifact loaded exactly once, before the exam loop)
# ---------------------------------------------------------------------------


@dataclass
class LoadedModels:
    path_b: Optional[torch.nn.Module]
    path_b_n_slices: Optional[int]
    path_c: Optional[torch.nn.Module]
    ensemble: dict
    calibrator: dict


def build_model(kind: str, n_slices: Optional[int] = None, backbone: Optional[str] = None):
    """Construct an unfit model of the given path ('path_b' or 'path_c')."""
    if kind == "path_b":
        return models.build_25d_model(n_slices, backbone=backbone or config.CNN25D_BACKBONE, pretrained=False)
    if kind == "path_c":
        return models.Compact3DCNN()
    raise ValueError(f"Unknown model kind: {kind!r}")


def load_model_weights() -> LoadedModels:
    """Load every enabled base model, the ensemble combiner, and the
    calibrator exactly once. Verifies the bundled preprocessing config
    matches what these weights were trained against.

    No fallback of any kind on failure here: if any required artifact is
    missing, corrupt, or needs an unavailable package, this raises and the
    whole run aborts -- model-loading failures must never be papered over
    with substitute predictions."""
    if pp.CONFIG_HASH != config.PREPROCESSING_VERSION_HASH:
        raise RuntimeError(
            "Bundled preprocessing.py does not match the preprocessing config "
            "recorded at training time (config hash mismatch) -- refusing to "
            "run inference with mismatched preprocessing."
        )

    set_runtime_stage("neural-model loading")
    print("Loading neural models.", flush=True)
    path_b = None
    path_b_n_slices = None
    if config.BASE_MODELS_ENABLED.get("path_b"):
        require_artifact(config.PATH_B_WEIGHTS)
        ckpt = torch.load(config.PATH_B_WEIGHTS, map_location=config.DEVICE, weights_only=True)
        path_b_n_slices = ckpt["n_slices"]
        path_b = build_model("path_b", n_slices=path_b_n_slices, backbone=ckpt.get("backbone"))
        path_b.load_state_dict(ckpt["state_dict"])
        if path_b_n_slices != config.N_SLICES_25D:
            print("WARNING: checkpoint n_slices differs from manifest n_slices_25d; using checkpoint value.",
                  flush=True)
        path_b.to(config.DEVICE).eval()

    path_c = None
    if config.BASE_MODELS_ENABLED.get("path_c"):
        require_artifact(config.PATH_C_WEIGHTS)
        ckpt = torch.load(config.PATH_C_WEIGHTS, map_location=config.DEVICE, weights_only=True)
        path_c = build_model("path_c")
        path_c.load_state_dict(ckpt["state_dict"])
        path_c.to(config.DEVICE).eval()

    set_runtime_stage("ensemble loading")
    print("Loading ensemble configuration.", flush=True)
    require_artifact(config.ENSEMBLE_WEIGHTS)
    ensemble = joblib.load(config.ENSEMBLE_WEIGHTS)

    set_runtime_stage("calibration loading")
    print("Loading calibration configuration.", flush=True)
    require_artifact(config.CALIBRATOR_WEIGHTS)
    calibrator = joblib.load(config.CALIBRATOR_WEIGHTS)

    return LoadedModels(
        path_b=path_b, path_b_n_slices=path_b_n_slices,
        path_c=path_c, ensemble=ensemble, calibrator=calibrator,
    )


def validate_loaded_assets(loaded: LoadedModels) -> None:
    """Cross-checks between the loaded artifacts that must hold for
    predict_exam()'s ensemble combination to be meaningful. Run once, right
    after loading, so a packaged-asset inconsistency is caught in its own
    "asset validation" stage -- distinguishable from a deserialization
    failure (raised earlier, during "neural-model loading" / "ensemble
    loading" / "calibration loading") and from a later inference failure.

    Never renormalizes weights or silently drops a model to make things
    "work" -- any mismatch here means the packaged archive itself is wrong,
    which must abort the run, not be papered over."""
    available_cols = []
    if config.BASE_MODELS_ENABLED.get("path_b") and loaded.path_b is not None:
        available_cols.append("oof_pred_b")
    if config.BASE_MODELS_ENABLED.get("path_c") and loaded.path_c is not None:
        available_cols.append("oof_pred_c")

    # No stale reference to the removed Path A (handcrafted features + GBM)
    # anywhere in the manifest or the loaded ensemble.
    if config.BASE_MODELS_ENABLED.get("path_a"):
        raise ValueError("Invalid packaged ensemble configuration.")
    if "path_a" in config.MODEL_MANIFEST.get("artifacts", {}):
        raise ValueError("Invalid packaged ensemble configuration.")
    if "oof_pred_a" in config.BASE_MODEL_COLS:
        raise ValueError("Invalid packaged ensemble configuration.")

    ensemble_cols = loaded.ensemble.get("base_model_cols")
    if ensemble_cols is None or list(ensemble_cols) != list(config.BASE_MODEL_COLS):
        raise ValueError("Invalid packaged ensemble configuration.")
    if set(ensemble_cols) != set(available_cols):
        raise ValueError("Invalid packaged ensemble configuration.")

    if loaded.ensemble.get("type") == "weighted_average":
        weights = loaded.ensemble.get("weights")
        if weights is None or len(weights) != len(ensemble_cols):
            raise ValueError("Invalid packaged ensemble configuration.")
        if not np.isfinite(np.asarray(weights, dtype=np.float64)).all():
            raise ValueError("Invalid packaged ensemble configuration.")

    method = loaded.calibrator.get("method")
    if method not in ("none", "platt", "isotonic"):
        raise ValueError("Invalid packaged calibration configuration.")
    if method in ("platt", "isotonic") and loaded.calibrator.get("calibrator") is None:
        raise ValueError("Invalid packaged calibration configuration.")


# ---------------------------------------------------------------------------
# Per-exam inference
# ---------------------------------------------------------------------------


def _run_path_b(loaded: LoadedModels, vol: np.ndarray) -> float:
    z = models.peak_axial_slice(vol)
    half = loaded.path_b_n_slices // 2
    zs = np.clip(np.arange(z - half, z + half + 1), 0, vol.shape[2] - 1)
    stack = vol[:, :, zs].transpose(2, 0, 1).copy()
    x = torch.from_numpy(stack.astype(np.float32))[None].to(config.DEVICE)
    use_amp = config.USE_AMP and config.DEVICE == "cuda"
    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logit = loaded.path_b(x).squeeze(-1)
        else:
            logit = loaded.path_b(x).squeeze(-1)
        prob = torch.sigmoid(logit).float().cpu().item()
    return float(prob)


def _run_path_c(loaded: LoadedModels, vol: np.ndarray) -> float:
    x = torch.from_numpy(vol[None, None].astype(np.float32)).to(config.DEVICE)
    use_amp = config.USE_AMP and config.DEVICE == "cuda"
    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logit = loaded.path_c(x).squeeze(-1)
        else:
            logit = loaded.path_c(x).squeeze(-1)
        prob = torch.sigmoid(logit).float().cpu().item()
    return float(prob)


def predict_exam(loaded: LoadedModels, nifti_path: Path) -> float:
    """Run the full pipeline (preprocessing -> base models -> ensemble ->
    calibration) for exactly one examination. Fully self-contained: does not
    read or depend on any other examination's data."""
    vol, _meta = load_nifti_volume(nifti_path)

    base_preds: dict[str, float] = {}
    if config.BASE_MODELS_ENABLED.get("path_b") and loaded.path_b is not None:
        base_preds["oof_pred_b"] = _run_path_b(loaded, vol)
    if config.BASE_MODELS_ENABLED.get("path_c") and loaded.path_c is not None:
        base_preds["oof_pred_c"] = _run_path_c(loaded, vol)

    if config.FINAL_MODEL_COL in base_preds:
        raw_pred = base_preds[config.FINAL_MODEL_COL]
    else:
        prob_vec = np.array([[base_preds[c] for c in config.BASE_MODEL_COLS]])
        raw_pred = models.combine_ensemble(loaded.ensemble["type"], prob_vec, loaded.ensemble)

    method = loaded.calibrator["method"]
    if method == "none":
        final_prob = raw_pred
    elif method == "platt":
        final_prob = float(loaded.calibrator["calibrator"].predict_proba([[raw_pred]])[:, 1][0])
    elif method == "isotonic":
        final_prob = float(loaded.calibrator["calibrator"].predict([raw_pred])[0])
    else:
        raise ValueError(f"Unknown calibration method: {method!r}")

    if not np.isfinite(final_prob):
        raise ValueError("Model produced a non-finite probability")

    return float(np.clip(final_prob, config.PROB_CLIP_LOW, config.PROB_CLIP_HIGH))


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def validate_predictions(predictions: pd.DataFrame, template: pd.DataFrame) -> None:
    """Strict pre-write validation. Raises AssertionError with a specific,
    but deliberately generic (no echoed values/counts), message on the first
    violation found."""
    expected_cols = config.OUTPUT_COLUMNS
    assert list(predictions.columns) == expected_cols, "submission has invalid columns"
    assert len(predictions) == len(template), "submission has an invalid number of rows"
    assert not predictions["uid"].duplicated().any(), "submission contains duplicate identifiers"

    template_uids = template["uid"].tolist()
    pred_uids = predictions["uid"].tolist()
    assert set(pred_uids) == set(template_uids), "submission identifiers do not match the required set"
    assert pred_uids == template_uids, "submission row order is invalid"

    probs = predictions["is_pathologic"]
    assert pd.api.types.is_numeric_dtype(probs), "submission contains non-numeric probabilities"
    assert np.isfinite(probs.to_numpy()).all(), "submission contains invalid (NaN/infinite) probabilities"
    assert probs.between(0.0, 1.0).all(), "submission contains out-of-range probabilities"


def write_submission(predictions: pd.DataFrame, output_path: Path) -> None:
    """Atomic write: write to a temp file in the same directory, then
    os.replace into place, then read back and re-validate as a final guard
    against a truncated/corrupt write."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".submission_", suffix=".csv.tmp", dir=str(output_path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        predictions.to_csv(tmp_path, index=False)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    reread = pd.read_csv(output_path)
    assert list(reread.columns) == config.OUTPUT_COLUMNS, "post-write validation: invalid columns"
    assert len(reread) == len(predictions), "post-write validation: invalid row count"
    assert reread["uid"].tolist() == predictions["uid"].tolist(), "post-write validation: invalid row order"
    assert np.isfinite(reread["is_pathologic"].to_numpy()).all(), "post-write validation: invalid probabilities"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Cap on how many per-exam fallback warnings are ever printed, so log volume
# stays bounded (well under the 300-line budget) no matter how large the
# hidden test set is. Failure counting/abort logic below is unaffected --
# only the printing is capped.
_MAX_FALLBACK_WARNINGS_LOGGED = 5


def main() -> None:
    set_runtime_stage("startup")
    _seed_everything(config.SEED)
    verify_runtime_dependencies()

    # --- input validation: entirely before any model artifact is loaded ---
    # A missing or empty data mount must be caught here, cheaply, rather
    # than after loading ~50 MB of model weights. This is the exact ordering
    # fix for the incident where the reported log showed all four
    # "Loading ... configuration." lines before the FATAL ValueError: that
    # ValueError came from load_submission_template()/discover_exam_files(),
    # which used to run AFTER load_model_weights(). See config.py's history
    # note for the verified root-cause writeup.
    set_runtime_stage("input validation")
    paths = resolve_paths()
    validate_runtime_inputs(paths)
    template = load_submission_template(paths)
    uids = template["uid"].astype(str).tolist()
    n = len(uids)
    exam_files = discover_exam_files(paths.nifti_dir, uids)

    # --- asset loading: only reached once every input has been validated ---
    set_runtime_stage("asset loading")
    print("Loading inference assets.", flush=True)
    loaded = load_model_weights()  # sets its own finer-grained stages internally

    set_runtime_stage("asset validation")
    validate_loaded_assets(loaded)

    # --- inference ---
    set_runtime_stage("inference")
    results: dict[str, float] = {}
    n_failed = 0
    max_allowed_failures = max(config.MAX_FAILURE_COUNT, int(np.ceil(config.MAX_FAILURE_FRACTION * n)))

    print("Starting inference.", flush=True)
    for uid in uids:
        try:
            prob = predict_exam(loaded, exam_files[uid])
        except Exception:
            n_failed += 1
            # Deliberately no per-exam identifier, path, or traceback content
            # that could leak hidden-test details; bounded, generic warning.
            if n_failed <= _MAX_FALLBACK_WARNINGS_LOGGED:
                print("WARNING: fallback applied for an examination.", flush=True)
            if n_failed > max_allowed_failures:
                raise RuntimeError(
                    "Aborting: too many examinations failed inference."
                ) from None
            prob = config.TRAIN_BASE_RATE
        results[uid] = prob

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictions = pd.DataFrame({
        "uid": uids,
        "is_pathologic": [results[u] for u in uids],
    })

    # --- output validation + writing ---
    set_runtime_stage("output validation")
    validate_predictions(predictions, template)

    set_runtime_stage("output writing")
    print("Writing output.", flush=True)
    write_submission(predictions, paths.output_path)

    print("Inference completed.", flush=True)


def run_entrypoint() -> None:
    """Production-safe entry point. Each exception category gets its own
    branch so the message can name what actually matters without leaking
    detail: a missing package names itself, a missing input says only that
    inputs are unavailable, a validation failure says only that validation
    failed -- all three also report the generic _RUNTIME_STAGE ("input
    validation", "asset loading", "inference", etc.) so a failure can be
    localized without a traceback. Full tracebacks are available for local
    debugging via SUBMISSION_DEBUG=1 (debug_enabled()), never in competition
    execution."""
    try:
        main()
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "unknown"
        print(
            f"FATAL: required Python module unavailable during {_RUNTIME_STAGE}: {missing_module}.",
            flush=True,
        )
        if debug_enabled():
            traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
    except FileNotFoundError:
        print(f"FATAL: required runtime inputs unavailable during {_RUNTIME_STAGE}.", flush=True)
        if debug_enabled():
            traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
    except ValueError:
        print(f"FATAL: validation failed during {_RUNTIME_STAGE}.", flush=True)
        if debug_enabled():
            traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001 -- top-level: must always exit non-zero with a safe, generic message
        print(f"FATAL: submission failed during {_RUNTIME_STAGE} ({type(exc).__name__}).", flush=True)
        if debug_enabled():
            traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    run_entrypoint()
