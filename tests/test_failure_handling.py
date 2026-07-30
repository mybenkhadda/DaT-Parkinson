"""Tests for the failure-handling policy documented in config.py: malformed
scans and invalid model weights must raise clearly; isolated per-exam
failures fall back to the training base rate (loudly, counted); failures
beyond the allowed threshold abort the whole run rather than masking a
systemic problem."""
from pathlib import Path

import pandas as pd
import pytest

import config
import main as m


def test_load_nifti_volume_raises_on_malformed_file(tmp_path):
    garbage = tmp_path / "not_a_real_scan.nii.gz"
    garbage.write_bytes(b"this is not valid gzip/nifti content")

    with pytest.raises(Exception):
        m.load_nifti_volume(garbage)


def test_load_nifti_volume_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.nii.gz"
    with pytest.raises(FileNotFoundError):
        m.load_nifti_volume(missing)


def test_load_model_weights_raises_on_invalid_ensemble_artifact(tmp_path, monkeypatch):
    bad_path = tmp_path / "corrupt_ensemble.joblib"
    bad_path.write_bytes(b"not a real joblib pickle")
    monkeypatch.setattr(config, "ENSEMBLE_WEIGHTS", bad_path)

    with pytest.raises(Exception):
        m.load_model_weights()


def test_main_falls_back_on_isolated_failure(tmp_path, monkeypatch):
    uids = ["good1", "good2", "bad", "good3"]
    data_dir = tmp_path / "data"
    nifti_dir = data_dir / "niftis"
    nifti_dir.mkdir(parents=True)
    for uid in uids:
        (nifti_dir / f"{uid}.nii.gz").write_bytes(b"placeholder")
    template_path = data_dir / "submission_format.csv"
    pd.DataFrame({"uid": uids, "is_pathologic": [0.5] * len(uids)}).to_csv(template_path, index=False)
    output_path = tmp_path / "submission.csv"
    (tmp_path / "artifacts").mkdir()

    monkeypatch.setattr(m, "__file__", str(tmp_path / "main.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m, "load_model_weights", lambda: object())
    monkeypatch.setattr(m, "validate_loaded_assets", lambda _loaded: None)

    def fake_predict_exam(_loaded, nifti_path: Path) -> float:
        if nifti_path.name.startswith("bad"):
            raise RuntimeError("simulated corrupt scan")
        return 0.42

    monkeypatch.setattr(m, "predict_exam", fake_predict_exam)

    m.main()

    result = pd.read_csv(output_path)
    by_uid = dict(zip(result["uid"], result["is_pathologic"]))
    assert by_uid["good1"] == pytest.approx(0.42)
    assert by_uid["bad"] == pytest.approx(config.TRAIN_BASE_RATE)


def test_main_aborts_when_failures_exceed_threshold(tmp_path, monkeypatch):
    uids = [f"uid{i}" for i in range(10)]
    data_dir = tmp_path / "data"
    nifti_dir = data_dir / "niftis"
    nifti_dir.mkdir(parents=True)
    for uid in uids:
        (nifti_dir / f"{uid}.nii.gz").write_bytes(b"placeholder")
    template_path = data_dir / "submission_format.csv"
    pd.DataFrame({"uid": uids, "is_pathologic": [0.5] * len(uids)}).to_csv(template_path, index=False)
    output_path = tmp_path / "submission.csv"
    (tmp_path / "artifacts").mkdir()

    monkeypatch.setattr(m, "__file__", str(tmp_path / "main.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m, "load_model_weights", lambda: object())
    monkeypatch.setattr(m, "validate_loaded_assets", lambda _loaded: None)

    def always_fails(_loaded, _nifti_path):
        raise RuntimeError("simulated systemic failure")

    monkeypatch.setattr(m, "predict_exam", always_fails)

    with pytest.raises(RuntimeError, match="Aborting"):
        m.main()

    assert not output_path.exists()
