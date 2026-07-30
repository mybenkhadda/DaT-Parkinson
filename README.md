# DaT-Parkinson

DaT-SPECT pathology (normal vs. pathologic) classification pipeline for the
DrivenData DaT Parkinson's Challenge.

## Repo layout

- `dat_scan_full_pipeline.ipynb` -- the main pipeline: preprocessing, CV
  splits, Path A (handcrafted features + GBM), Path B (2.5D CNN), Path C
  (3D CNN), calibration, ensembling, plus a research extension (orientation/
  registration audits, classical baselines, calibration/ensemble comparisons).
- `model_variants_comparison.ipynb` -- three additional model variants
  (MedicalNet-pretrained 3D CNN, Model Genesis self-supervised pretraining,
  registration-based Path A features) evaluated against the main pipeline's
  baselines on the same CV split, plus a global calibration fix.
- `preprocessing.py` -- shared NIfTI preprocessing (canonical orientation,
  resampling, centroid crop, intensity normalization).
- `experiments/` -- driver scripts for the three new model variants
  (`run_medicalnet_cv.py`, `run_model_genesis_cv.py`,
  `run_registration_path_a.py`) and shared training code
  (`variant_common.py`).
- `submission_src/` -- the offline DrivenData code-execution submission
  package (build with `scripts/build_submission.py`).
- `tests/` -- pytest suite for `submission_src/`.
- `artifacts/` -- small, trained-model artifacts (final calibrator/ensemble
  weights, OOF predictions, config, plots) checked into the repo. Large
  per-fold checkpoints, the downloaded MedicalNet weights, and scratch
  caches are **not** tracked (see `.gitignore`) -- regenerate them by
  re-running the relevant notebook/script.

## What's NOT in this repo (and why)

- `niftis/`, `preproc_cache/`, `data-demo/`, `train_labels_JNDlMjr.csv` --
  the competition's actual scan data / labels. Not redistributed here;
  source your own copy (see "Running on Colab" below).
- `artifacts/pretrained_weights/medicalnet_resnet18_23dataset.pth` -- a
  132MB third-party file (Tencent/MedicalNet, MIT license). Not committed;
  `experiments/variant_common.py`'s `ensure_medicalnet_weights()` downloads
  and checksum-verifies it automatically the first time it's needed (called
  from both `experiments/run_medicalnet_cv.py` and
  `model_variants_comparison.ipynb`), so no manual step is required on a
  fresh clone -- Colab included. To pre-fetch it manually instead (e.g. to
  cache it on Drive across sessions):

  ```bash
  mkdir -p artifacts/pretrained_weights
  curl -sL "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/main/resnet_18_23dataset.pth" \
    -o artifacts/pretrained_weights/medicalnet_resnet18_23dataset.pth
  # sha256: 61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4
  ```
- Per-fold model checkpoints (`artifacts/path_b_fold*.pt`,
  `artifacts/path_c_fold*.pt`, `artifacts/model_variants/*.pt`) -- each
  under GitHub's 100MB limit individually, but large in aggregate and fully
  reproducible by re-running training. The **final**, full-data-trained
  models (`artifacts/final_25d_model.pt`, `artifacts/final_3d_model.pt`) are
  tracked, since those are what `submission_src/` actually needs.

## Running on Colab

1. Clone the repo:

   ```bash
   !git clone https://github.com/mybenkhadda/DaT-Parkinson.git
   %cd DaT-Parkinson
   !pip install -r requirements.txt
   ```

2. Get the data. This repo does not ship `niftis/`. Upload your own copy of
   the competition's NIfTI files + labels to Google Drive, then mount it:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")

   import os
   os.symlink("/content/drive/MyDrive/<your-niftis-folder>", "niftis")
   os.symlink("/content/drive/MyDrive/<your-labels-csv>", "train_labels_JNDlMjr.csv")
   ```

   Preprocessing will populate `preproc_cache/` on first run (cached
   thereafter -- put `preproc_cache/` on Drive too if you want it to persist
   across sessions).

3. The MedicalNet variant's pretrained weights are fetched automatically
   (checksum-verified) the first time `model_variants_comparison.ipynb` /
   `experiments/run_medicalnet_cv.py` needs them -- no manual step required.

4. Open and run `dat_scan_full_pipeline.ipynb` top to bottom (GPU runtime
   recommended: `Runtime > Change runtime type > GPU`). Set `CFG["fast_dev_run"] = True`
   for a quick smoke test before committing to the full run.
