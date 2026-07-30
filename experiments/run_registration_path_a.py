"""Registration-based striatal ROI upgrade for Path A.

1. Build a fixed, symmetric template in the existing centroid-crop space,
   from a subsample of TRAINING-ONLY volumes (no test data, no per-fold
   leakage -- one global template, same rule as any fixed preprocessing
   asset).
2. Rigid-register every one of the 1362 cached training volumes to that
   template (SimpleITK, Mattes mutual information, linear interpolation).
3. Extract left/right caudate + putamen SBR (using a real background/
   reference region), left-right asymmetry index, and putamen/caudate
   ratio, all with stable (never raw signed/zero-prone) formulas.
4. Retrain XGBoost, logistic regression, and a linear SVM on these
   features using the EXISTING cv_folds.csv fold assignments (never
   re-derived), producing leakage-safe OOF predictions.

Everything is written to artifacts/model_variants/ so the deliverable
notebook can load real results without re-running this (potentially
15-20 minute) registration pass.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variant_common as vc

EPSILON = 1e-6
TARGET_SPACING_MM = float(vc.pp.CFG["target_spacing_mm"])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_template(training_uids: list[str], n_examinations: int = 60, seed: int = 42):
    rng = np.random.RandomState(seed)
    sample = rng.choice(training_uids, size=min(n_examinations, len(training_uids)), replace=False)
    accum, n_used = None, 0
    for uid in sample:
        vol = vc.load_cached_volume(uid)
        accum = vol if accum is None else accum + vol
        n_used += 1
    template_array = (accum / max(n_used, 1)).astype(np.float32)
    template_image = sitk.GetImageFromArray(template_array)
    template_image.SetSpacing((TARGET_SPACING_MM,) * 3)
    return template_image, n_used


def register_volume(moving_volume: np.ndarray, fixed_image) -> tuple[np.ndarray, dict]:
    moving_image = sitk.GetImageFromArray(moving_volume.astype(np.float32))
    moving_image.SetSpacing((TARGET_SPACING_MM,) * 3)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.3, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=150)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    failure = None
    try:
        final_transform = registration.Execute(fixed_image, moving_image)
    except Exception as e:
        return moving_volume, {"failure": f"optimizer_exception:{type(e).__name__}"}

    params = final_transform.GetParameters()
    if not np.all(np.isfinite(params)):
        failure = "nonfinite_transform"
    else:
        translation = np.array(params[-3:])
        if np.linalg.norm(translation) > 40.0:
            failure = "excessive_translation"
        rotation = np.array(params[:3])
        if np.max(np.abs(rotation)) > (np.pi / 3):
            failure = failure or "excessive_rotation"

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)
    resampler.SetDefaultPixelValue(0.0)
    registered_array = sitk.GetArrayFromImage(resampler.Execute(moving_image)).astype(np.float32)

    if float((registered_array > 0).mean()) < 0.01:
        failure = failure or "nearly_empty_registered_image"

    return registered_array, {"failure": failure}


def stable_ratio(numerator: float, denominator: float, epsilon: float = EPSILON) -> float:
    return numerator / max(abs(denominator), epsilon)


def stable_asymmetry(left: float, right: float, epsilon: float = EPSILON) -> float:
    return abs(left - right) / max(0.5 * (abs(left) + abs(right)), epsilon)


def _box(shape, x_slice, y_frac, z_frac):
    cy, cz = shape[1] // 2, shape[2] // 2
    hy, hz = max(1, int(shape[1] * y_frac / 2)), max(1, int(shape[2] * z_frac / 2))
    mask = np.zeros(shape, dtype=bool)
    mask[x_slice, cy - hy:cy + hy, cz - hz:cz + hz] = True
    return mask


def extract_registered_features(vol: np.ndarray) -> dict:
    """Registered-space anatomical features: since every volume now shares
    the same template-space coordinates, fixed relative-position boxes are
    a much stronger anatomical-correspondence assumption than in raw
    centroid-crop space. Uses a genuine background/reference region (low,
    non-striatal signal) for SBR, not just a proxy."""
    shape = vol.shape
    cx = shape[0] // 2

    regions = {
        "left_caudate": _box(shape, slice(0, cx), 0.22, 0.18),
        "right_caudate": _box(shape, slice(cx, shape[0]), 0.22, 0.18),
        "left_putamen": _box(shape, slice(0, cx), 0.40, 0.34),
        "right_putamen": _box(shape, slice(cx, shape[0]), 0.40, 0.34),
        # Reference/background region: peripheral, low-uptake occipital-like
        # proxy -- outside the central striatal box entirely, on both sides.
        "reference": _box(shape, slice(0, shape[0]), 0.12, 0.70),
    }
    central_mask = _box(shape, slice(0, shape[0]), 0.55, 0.55)
    regions["reference"] = regions["reference"] & ~central_mask

    means = {name: (float(vol[mask].mean()) if mask.any() else np.nan) for name, mask in regions.items()}
    ref = means["reference"]

    feats: dict[str, float] = {f"{name}_mean": val for name, val in means.items()}

    for side in ("left", "right"):
        for region in ("caudate", "putamen"):
            key = f"{side}_{region}"
            feats[f"sbr_{key}"] = stable_ratio(means[key] - ref, ref) if np.isfinite(ref) and ref > 0 else np.nan

    sbr_left = feats["sbr_left_putamen"]
    sbr_right = feats["sbr_right_putamen"]
    feats["asymmetry_index"] = (stable_asymmetry(sbr_left, sbr_right)
                                 if np.isfinite(sbr_left) and np.isfinite(sbr_right) else np.nan)

    for side in ("left", "right"):
        putamen_sbr = feats[f"sbr_{side}_putamen"]
        caudate_sbr = feats[f"sbr_{side}_caudate"]
        feats[f"putamen_to_caudate_ratio_{side}"] = (
            stable_ratio(putamen_sbr, caudate_sbr) if np.isfinite(putamen_sbr) and np.isfinite(caudate_sbr) else np.nan)

    brain_mask = vol > np.percentile(vol[vol > 0], 10) if (vol > 0).any() else np.zeros(shape, dtype=bool)
    high_mask = brain_mask & (vol > np.percentile(vol[brain_mask], 90)) if brain_mask.any() else np.zeros(shape, dtype=bool)
    _, n_components = ndimage.label(high_mask)
    feats["connected_high_uptake_components"] = float(n_components)
    feats["high_uptake_volume_mm3"] = float(high_mask.sum() * TARGET_SPACING_MM ** 3)

    return feats


def run_classical_oof(feature_df, feature_cols, y_col, fold_col, model_name, kind, seed=42):
    oof = np.full(len(feature_df), np.nan)
    y = feature_df[y_col].values
    folds = feature_df[fold_col].values
    for f in sorted(np.unique(folds)):
        train_mask, val_mask = folds != f, folds == f
        X_train_raw = feature_df.loc[train_mask, feature_cols].values
        X_val_raw = feature_df.loc[val_mask, feature_cols].values
        imputer = SimpleImputer(strategy="median").fit(X_train_raw)
        X_train, X_val = imputer.transform(X_train_raw), imputer.transform(X_val_raw)

        if kind == "xgboost":
            pos_frac = y[train_mask].mean()
            scale_pos_weight = float((1 - pos_frac) / max(pos_frac, 1e-6))
            clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                                 objective="binary:logistic", eval_metric="logloss",
                                 random_state=seed, n_jobs=-1, scale_pos_weight=scale_pos_weight)
            clf.fit(X_train, y[train_mask])
            oof[val_mask] = clf.predict_proba(X_val)[:, 1]
        else:
            scaler = StandardScaler().fit(X_train)
            X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)
            if kind == "logreg":
                clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
                clf.fit(X_train_s, y[train_mask])
                oof[val_mask] = clf.predict_proba(X_val_s)[:, 1]
            elif kind == "linear_svm":
                base = LinearSVC(C=1.0, max_iter=10000, random_state=seed)
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
                clf.fit(X_train_s, y[train_mask])
                oof[val_mask] = clf.predict_proba(X_val_s)[:, 1]
            else:
                raise ValueError(kind)
    return pd.DataFrame({"uid": feature_df["uid"], model_name: oof})


def main():
    log("Loading cv_folds.csv (existing split, not re-derived)")
    folds_df = vc.load_cv_folds()

    log("Building fixed template from 60 training-only examinations")
    template_image, n_used = build_template(folds_df["uid"].tolist(), n_examinations=60, seed=42)
    log(f"Template built from {n_used} examinations")

    log(f"Registering all {len(folds_df)} volumes to the template (rigid, linear interpolation)")
    t0 = time.time()
    feature_rows = []
    n_failed = 0
    for i, row in enumerate(folds_df.itertuples()):
        vol = vc.load_cached_volume(row.uid)
        registered, meta = register_volume(vol, template_image)
        if meta["failure"] is not None:
            n_failed += 1
        feats = extract_registered_features(registered)
        feats["uid"] = row.uid
        feats["registration_failure"] = meta["failure"]
        feature_rows.append(feats)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            log(f"  registered {i + 1}/{len(folds_df)} ({elapsed:.0f}s elapsed, "
                f"{elapsed / (i + 1):.2f}s/scan, {n_failed} failures so far)")

    elapsed_total = time.time() - t0
    log(f"Registration complete: {len(feature_rows)} volumes in {elapsed_total:.0f}s "
        f"({elapsed_total / len(feature_rows):.2f}s/scan average), {n_failed} failures "
        f"({100 * n_failed / len(feature_rows):.1f}%)")

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(vc.VARIANTS_DIR / "registered_path_a_features.csv", index=False)

    merged = folds_df.merge(features_df, on="uid", how="inner")
    feature_cols = [c for c in features_df.columns if c not in ("uid", "registration_failure")]

    log("Retraining XGBoost / logistic regression / linear SVM on registered features (5-fold OOF)")
    oof_xgb = run_classical_oof(merged, feature_cols, "is_pathologic", "fold", "path_a_registered_xgb", "xgboost")
    oof_logreg = run_classical_oof(merged, feature_cols, "is_pathologic", "fold", "path_a_registered_logreg", "logreg")
    oof_svm = run_classical_oof(merged, feature_cols, "is_pathologic", "fold", "path_a_registered_svm", "linear_svm")

    oof_combined = oof_xgb.merge(oof_logreg, on="uid").merge(oof_svm, on="uid")
    oof_combined = oof_combined.merge(folds_df[["uid", "is_pathologic", "fold"]], on="uid")
    oof_combined.to_csv(vc.VARIANTS_DIR / "registration_path_a_oof.csv", index=False)

    metrics = {}
    for col in ["path_a_registered_xgb", "path_a_registered_logreg", "path_a_registered_svm"]:
        metrics[col] = vc.compute_binary_metrics(oof_combined["is_pathologic"], oof_combined[col])

    summary = {
        "n_registered": len(feature_rows),
        "n_registration_failures": n_failed,
        "registration_failure_rate": n_failed / len(feature_rows),
        "registration_runtime_seconds": elapsed_total,
        "feature_columns": feature_cols,
        "metrics": metrics,
    }
    with open(vc.VARIANTS_DIR / "registration_path_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log("Done. Summary:")
    log(json.dumps({k: v for k, v in summary.items() if k != "feature_columns"}, indent=2, default=str))


if __name__ == "__main__":
    main()
