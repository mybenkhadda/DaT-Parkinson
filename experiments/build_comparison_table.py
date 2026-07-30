"""Rebuilds the full model/ensemble comparison table, folding in the three
new variants (registration-based Path A, MedicalNet, Model Genesis) and
applying the SAME cross-fitted calibration search (none/platt/isotonic) to
EVERY row -- individual paths and every ensemble variant, old and new alike
-- so it's transparent whether calibration helped or was correctly rejected,
rather than silently identical raw/calibrated numbers looking skipped.

Gracefully skips any new-variant OOF file that doesn't exist yet, so this
can be run (and re-run) while background training is still in progress.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variant_common as vc

ARTIFACTS_DIR = vc.ARTIFACTS_DIR
VARIANTS_DIR = vc.VARIANTS_DIR


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    n_classes = len(set(y_true.tolist()))
    return {
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "auroc": roc_auc_score(y_true, y_prob) if n_classes > 1 else np.nan,
        "brier": brier_score_loss(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def cross_fitted_calibrate(y_true, y_prob, folds, method: str) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    folds = np.asarray(folds)
    out = np.full(len(y_prob), np.nan)
    for fold in np.unique(folds):
        train_mask, val_mask = folds != fold, folds == fold
        yt, yp = y_true[train_mask], y_prob[train_mask]
        if method == "platt":
            calibrator = LogisticRegression()
            calibrator.fit(yp.reshape(-1, 1), yt)
            out[val_mask] = calibrator.predict_proba(y_prob[val_mask].reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(yp, yt)
            out[val_mask] = calibrator.predict(y_prob[val_mask])
        else:
            raise ValueError(method)
    assert not np.isnan(out).any()
    return np.clip(out, 1e-7, 1 - 1e-7)


def calibrate_and_score(name: str, y_true, y_prob, folds) -> dict:
    """Tries none/platt/isotonic, cross-fitted, and picks the lowest OOF log
    loss -- the SAME selection rule already used for individual paths in the
    original pipeline, now applied uniformly to every row including
    ensembles, so a rejection of calibration is a verified result, not a
    silent skip."""
    candidates = {"none": np.clip(y_prob, 1e-7, 1 - 1e-7)}
    for method in ["platt", "isotonic"]:
        candidates[method] = cross_fitted_calibrate(y_true, y_prob, folds, method)
    scored = [(m, log_loss(y_true, p, labels=[0, 1]), brier_score_loss(y_true, p)) for m, p in candidates.items()]
    best_method, best_ll, best_brier = sorted(scored, key=lambda t: (t[1], t[2]))[0]
    raw_metrics = compute_binary_metrics(y_true, candidates["none"])
    cal_metrics = compute_binary_metrics(y_true, candidates[best_method])
    return {
        "model": name,
        "selected_calibration": best_method,
        "calibration_helped": best_method != "none",
        "raw_log_loss": raw_metrics["log_loss"], "raw_auroc": raw_metrics["auroc"], "raw_brier": raw_metrics["brier"],
        "calibrated_log_loss": cal_metrics["log_loss"], "calibrated_brier": cal_metrics["brier"],
        "sensitivity": cal_metrics["sensitivity"], "specificity": cal_metrics["specificity"],
        "_calibrated_probs": candidates[best_method],
    }


def fit_nonneg_weights(probs, y):
    n = probs.shape[1]

    def obj(w):
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
        return log_loss(y, np.clip(probs @ w, 1e-7, 1 - 1e-7), labels=[0, 1])

    res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n,
                    constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, options={"maxiter": 300})
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else np.ones(n) / n


def cross_fitted_ensemble(y_true, probs, folds, fit_fn, predict_fn):
    out = np.full(len(y_true), np.nan)
    for fold in np.unique(folds):
        train_mask, val_mask = folds != fold, folds == fold
        params = fit_fn(probs[train_mask], y_true[train_mask])
        out[val_mask] = predict_fn(params, probs[val_mask])
    assert not np.isnan(out).any()
    return np.clip(out, 1e-7, 1 - 1e-7), None


def fit_logreg_stack(probs, y):
    X = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / (1 - np.clip(probs, 1e-7, 1 - 1e-7)))
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, y)
    return model


def predict_logreg_stack(model, probs):
    X = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / (1 - np.clip(probs, 1e-7, 1 - 1e-7)))
    return model.predict_proba(X)[:, 1]


def predict_weighted(w, probs):
    return np.clip(probs @ w, 1e-7, 1 - 1e-7)


def load_base_table() -> pd.DataFrame:
    """Original pipeline's OOF table: uid, is_pathologic, fold, and raw
    (uncalibrated) predictions for Path A/B/C -- the calibrated_* columns
    already baked into oof_predictions.csv are IGNORED here; every row in
    the final table below is (re-)calibrated fresh, uniformly, in this
    script, so old and new variants are judged by the identical procedure."""
    oof = pd.read_csv(ARTIFACTS_DIR / "oof_predictions.csv")
    base = oof[["uid", "is_pathologic", "fold"]].copy()
    base["Path A (handcrafted + GBM)"] = oof["pred__oof_pred_a"]
    base["Path B (2.5D CNN)"] = oof["pred__oof_pred_b"]
    base["Path C (3D CNN)"] = oof["pred__oof_pred_c"]
    return base


def try_load_variant(path: Path, col: str, new_name: str, table: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        print(f"  [skip] {new_name}: {path.name} not found yet")
        return table, False
    variant_df = pd.read_csv(path)[["uid", col]].rename(columns={col: new_name})
    table = table.merge(variant_df, on="uid", how="left")
    n_missing = table[new_name].isna().sum()
    print(f"  [loaded] {new_name}: {len(variant_df)} predictions, {n_missing} missing after merge")
    return table, True


def main():
    print("Loading base OOF table (Path A/B/C, existing cv_folds.csv split)")
    table = load_base_table()

    individual_model_cols = ["Path A (handcrafted + GBM)", "Path B (2.5D CNN)", "Path C (3D CNN)"]

    print("Attempting to load the three new variants:")
    table, has_reg = try_load_variant(
        VARIANTS_DIR / "registration_path_a_oof.csv", "path_a_registered_xgb",
        "Path A-registered (XGBoost)", table)
    if has_reg:
        individual_model_cols.append("Path A-registered (XGBoost)")
        reg_oof = pd.read_csv(VARIANTS_DIR / "registration_path_a_oof.csv")
        for raw_col, label in [("path_a_registered_logreg", "Path A-registered (logistic regression)"),
                                ("path_a_registered_svm", "Path A-registered (linear SVM)")]:
            table = table.merge(reg_oof[["uid", raw_col]].rename(columns={raw_col: label}), on="uid", how="left")
            individual_model_cols.append(label)

    table, has_medicalnet = try_load_variant(
        VARIANTS_DIR / "medicalnet_oof.csv", "medicalnet_pretrained", "Path C-MedicalNet (pretrained 3D ResNet-18)", table)
    if has_medicalnet:
        individual_model_cols.append("Path C-MedicalNet (pretrained 3D ResNet-18)")

    table, has_genesis = try_load_variant(
        VARIANTS_DIR / "model_genesis_oof.csv", "model_genesis_pretrained", "Path C-ModelGenesis (self-supervised pretrained)", table)
    if has_genesis:
        individual_model_cols.append("Path C-ModelGenesis (self-supervised pretrained)")

    # Only keep rows where every active column is present (should be all 1362
    # unless a variant genuinely failed on some uid).
    complete_cols = ["is_pathologic", "fold"] + individual_model_cols
    n_before = len(table)
    table = table.dropna(subset=complete_cols).reset_index(drop=True)
    if len(table) != n_before:
        print(f"WARNING: dropped {n_before - len(table)} rows with missing predictions in at least one active column")

    y_true = table["is_pathologic"].values
    folds = table["fold"].values

    print(f"\nCalibrating {len(individual_model_cols)} individual model(s), cross-fitted none/platt/isotonic:")
    rows = []
    calibrated_cols = {}
    for name in individual_model_cols:
        result = calibrate_and_score(name, y_true, table[name].values, folds)
        calibrated_cols[name] = result.pop("_calibrated_probs")
        result["kind"] = "individual"
        rows.append(result)
        print(f"  {name}: raw={result['raw_log_loss']:.4f} -> calibrated={result['calibrated_log_loss']:.4f} "
              f"(method={result['selected_calibration']})")

    print("\nBuilding ensembles over the FULL expanded model set:")
    prob_matrix = table[individual_model_cols].values.astype(np.float64)
    ensembles = {}
    ensembles["Equal-weight average (expanded)"] = np.clip(prob_matrix.mean(axis=1), 1e-7, 1 - 1e-7)
    logits = np.log(np.clip(prob_matrix, 1e-7, 1 - 1e-7) / (1 - np.clip(prob_matrix, 1e-7, 1 - 1e-7)))
    ensembles["Logit-space average (expanded)"] = np.clip(1 / (1 + np.exp(-logits.mean(axis=1))), 1e-7, 1 - 1e-7)
    w_preds, _ = cross_fitted_ensemble(y_true, prob_matrix, folds, fit_nonneg_weights, predict_weighted)
    ensembles["Weighted average, optimized (expanded)"] = w_preds
    stack_preds, _ = cross_fitted_ensemble(y_true, prob_matrix, folds, fit_logreg_stack, predict_logreg_stack)
    ensembles["Logistic stacker (expanded)"] = stack_preds

    # Also keep the ORIGINAL (Path A/B/C-only) best ensembles for direct
    # apples-to-apples comparison against the expanded-model-set ensembles.
    orig_matrix = table[["Path A (handcrafted + GBM)", "Path B (2.5D CNN)", "Path C (3D CNN)"]].values.astype(np.float64)
    w_preds_orig, _ = cross_fitted_ensemble(y_true, orig_matrix, folds, fit_nonneg_weights, predict_weighted)
    ensembles["Weighted average, optimized (original A+B+C)"] = w_preds_orig
    stack_preds_orig, _ = cross_fitted_ensemble(y_true, orig_matrix, folds, fit_logreg_stack, predict_logreg_stack)
    ensembles["Logistic stacker (original A+B+C)"] = stack_preds_orig

    print("\nCalibrating every ensemble variant, cross-fitted none/platt/isotonic:")
    for name, preds in ensembles.items():
        result = calibrate_and_score(name, y_true, preds, folds)
        calibrated_cols[name] = result.pop("_calibrated_probs")
        result["kind"] = "ensemble"
        rows.append(result)
        helped = "YES" if result["calibration_helped"] else "no (raw already optimal)"
        print(f"  {name}: raw={result['raw_log_loss']:.4f} -> calibrated={result['calibrated_log_loss']:.4f} "
              f"(method={result['selected_calibration']}, calibration helped: {helped})")

    comparison_df = pd.DataFrame(rows).sort_values("calibrated_log_loss").reset_index(drop=True)
    comparison_df.to_csv(VARIANTS_DIR / "full_comparison_table.csv", index=False)

    oof_export = table[["uid", "is_pathologic", "fold"]].copy()
    for name, col_name in [(n, n) for n in list(individual_model_cols) + list(ensembles.keys())]:
        oof_export[f"raw__{col_name}"] = table[col_name] if col_name in table.columns else ensembles.get(col_name)
        oof_export[f"calibrated__{col_name}"] = calibrated_cols[col_name]
    oof_export.to_csv(VARIANTS_DIR / "full_oof_with_calibration.csv", index=False)

    status = {
        "has_registration_variant": has_reg, "has_medicalnet_variant": has_medicalnet,
        "has_model_genesis_variant": has_genesis, "n_rows": len(table),
        "n_individual_models": len(individual_model_cols), "n_ensembles": len(ensembles),
    }
    with open(VARIANTS_DIR / "comparison_build_status.json", "w") as f:
        json.dump(status, f, indent=2)

    print("\n" + "=" * 100)
    print("FULL COMPARISON TABLE (sorted by calibrated log loss)")
    print("=" * 100)
    print(comparison_df[["model", "kind", "raw_log_loss", "calibrated_log_loss", "selected_calibration",
                          "raw_auroc", "calibrated_brier", "sensitivity", "specificity"]].to_string(index=False))
    print(f"\nStatus: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    main()
