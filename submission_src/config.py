"""Central configuration for the DaT-Parkinson code-execution submission.

Every dataset- or environment-specific assumption lives here so the rest of
the codebase never hardcodes a path, filename, or magic number. If the real
runtime layout, file-naming convention, or artifact set changes, this is the
only file that should need edits.

Verified facts (as of this writing):
  * niftis/, model_summary.json, pipeline_config.json all come from the same
    training run captured in dat_scan_full_pipeline.ipynb and artifacts/ at
    the repo root.
  * The DrivenData smoke-test archive (smoke_test_data_2gePzfM.tar.gz) confirms
    ONE NIfTI file per uid, named exactly ``<uid>.nii.gz`` -- 20 uids, 20
    files, no ambiguity. This is the same convention used for training
    (niftis/<uid>.nii.gz against train_labels_JNDlMjr.csv).

Assumptions NOT independently verified against the official runtime repo
(flagged again in the final report -- verify before relying on them):
  * Exact package versions available in the competition container (this repo's
    requirements.txt reflects the LOCAL dev environment, not necessarily the
    official runtime image).
  * Whether hidden test scans are guaranteed single-file-per-uid like the
    smoke test, or whether some edge case in the full hidden test set differs.
  * Whether /code_execution/data/niftis/ is flat or nested -- discover_exam_files
    below handles both via a recursive search, but this hasn't been exercised
    against the real runtime's directory layout.
  * Whether the evaluator always extracts main.py into a src/ subdirectory, or
    sometimes directly into the working directory -- main.resolve_paths()
    probes for both (see its docstring) rather than assuming either one.

Path A removal (2026-07-29): the original 3-base-model ensemble had a
handcrafted-feature + XGBoost path ("Path A"). Its joblib artifact pickled an
xgboost.sklearn.XGBClassifier, which requires the xgboost package to
unpickle -- and xgboost is not confirmed to be part of the official
runtime's uv.lock. This produced ModuleNotFoundError immediately after
"Loading inference assets." Path A's weight in the trained, optimized
ensemble was exactly 0.0 (mathematically inert -- verified by inspecting the
original final_ensemble.joblib before it was rewritten), so it was dropped
entirely rather than patched: final_ensemble.joblib and model_manifest.json
were regenerated without it, features.py and final_feature_model.joblib were
removed from the package, and xgboost/scipy/sklearn are no longer runtime
dependencies of this submission at all.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
# NOTE: the competition working directory (where data/ lives and where
# submission.csv must be written) is NOT a fixed constant here -- it depends
# on whether the evaluator extracted this archive directly into the working
# directory or into a src/ subdirectory, and that has been observed to vary.
# See main.resolve_paths() / main._resolve_work_dir(), which probe for both
# layouts at runtime instead of assuming one.
#
# Bundled assets (artifacts/) are NOT ambiguous: they always ship alongside
# this file, so they're resolved relative to SRC_DIR regardless of which
# working-directory layout is in play.
SRC_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SRC_DIR / "artifacts"
MANIFEST_PATH = ARTIFACTS_DIR / "model_manifest.json"

# ---------------------------------------------------------------------------
# NIfTI file discovery
# ---------------------------------------------------------------------------
# Confirmed convention (training data + official smoke-test archive): exactly
# one file per uid, named "<uid>.nii.gz". ".nii" is also accepted since
# nibabel supports it and it costs nothing to be permissive on read.
NIFTI_SUFFIXES = (".nii.gz", ".nii")

# ---------------------------------------------------------------------------
# Model manifest (sanitized copy of the training run's pipeline_config.json)
# ---------------------------------------------------------------------------
with open(MANIFEST_PATH, encoding="utf-8") as _f:
    MODEL_MANIFEST: dict = json.load(_f)

# Artifact filenames are stored as basenames in the manifest and resolved
# against ARTIFACTS_DIR here -- never trust absolute paths recorded at
# training time, since the training machine's filesystem layout is not the
# competition container's. Path A (see docstring above) is intentionally
# absent: no PATH_A_WEIGHTS constant, no feature_columns.
PATH_B_WEIGHTS = ARTIFACTS_DIR / MODEL_MANIFEST["artifacts"]["path_b"]
PATH_C_WEIGHTS = ARTIFACTS_DIR / MODEL_MANIFEST["artifacts"]["path_c"]
ENSEMBLE_WEIGHTS = ARTIFACTS_DIR / MODEL_MANIFEST["final_ensemble_path"]
CALIBRATOR_WEIGHTS = ARTIFACTS_DIR / MODEL_MANIFEST["final_calibrator_path"]

BASE_MODELS_ENABLED = MODEL_MANIFEST["base_models_enabled"]
BASE_MODEL_COLS = MODEL_MANIFEST["base_model_cols"]
FINAL_MODEL_COL = MODEL_MANIFEST["final_model_col"]
N_SLICES_25D = MODEL_MANIFEST["n_slices_25d"]
CNN25D_BACKBONE = MODEL_MANIFEST["cnn25d_backbone"]
PREPROCESSING_CONFIG = MODEL_MANIFEST["preprocessing_config"]
PREPROCESSING_VERSION_HASH = MODEL_MANIFEST["preprocessing_version_hash"]

# ---------------------------------------------------------------------------
# Runtime / device
# ---------------------------------------------------------------------------
SEED = int(MODEL_MANIFEST.get("seed", 42))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True  # only ever applied when DEVICE == "cuda"; see models.py

# Local-development-only verbosity switch. Must default to OFF for
# competition execution: production logs/exceptions never include paths,
# uids, filenames, counts, or other per-examination detail. Set the
# environment variable SUBMISSION_DEBUG=1 to opt into detailed exception
# text and full tracebacks when debugging locally.
SUBMISSION_DEBUG = os.environ.get("SUBMISSION_DEBUG") == "1"

# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = ["uid", "is_pathologic"]
PROB_CLIP_LOW = 1e-6
PROB_CLIP_HIGH = 1 - 1e-6

# ---------------------------------------------------------------------------
# Failure-handling policy
# ---------------------------------------------------------------------------
# A single unreadable/corrupt scan should not zero out an otherwise-valid
# submission (code-execution competitions score log loss on ALL rows; a
# missing row or a crash forfeits the whole run). We therefore allow a
# narrow, explicit, logged fallback to the empirical training-set base rate
# (a scientifically defensible prior in the total absence of other
# information) for individual failures -- never silently, always counted.
#
# If failures exceed this threshold, something systemic is wrong (bad
# artifact, broken preprocessing, wrong package version) rather than one bad
# file, and we abort the whole run instead of masking it with fallbacks.
TRAIN_BASE_RATE = 0.5484581497797357  # train_labels_JNDlMjr.csv: mean(is_pathologic), n=1362
MAX_FAILURE_COUNT = 3
MAX_FAILURE_FRACTION = 0.05
