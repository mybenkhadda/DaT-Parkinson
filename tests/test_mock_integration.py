"""Mock end-to-end integration test (no real model weights or images):
confirms submission.csv is written beside the competition's data/ directory
-- specifically the src/-subdirectory layout with main.py one level below
work_dir -- even when the process is launched from an unrelated current
working directory."""
import pandas as pd

import main as m

UIDS = ["examA", "examB", "examC"]


def test_submission_written_beside_data_not_cwd_or_root(tmp_path, monkeypatch):
    work_dir = tmp_path / "code_execution"
    src_dir = work_dir / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "artifacts").mkdir()

    data_dir = work_dir / "data"
    nifti_dir = data_dir / "niftis"
    nifti_dir.mkdir(parents=True)
    for uid in UIDS:
        (nifti_dir / f"{uid}.nii.gz").write_bytes(b"placeholder")
    pd.DataFrame({"uid": UIDS, "is_pathologic": [0.5] * len(UIDS)}).to_csv(
        data_dir / "submission_format.csv", index=False
    )

    decoy_cwd = tmp_path / "some_other_directory"
    decoy_cwd.mkdir()

    monkeypatch.setattr(m, "__file__", str(src_dir / "main.py"))
    monkeypatch.setattr(m, "load_model_weights", lambda: object())
    monkeypatch.setattr(m, "validate_loaded_assets", lambda _loaded: None)
    monkeypatch.setattr(m, "predict_exam", lambda _loaded, _path: 0.7)
    monkeypatch.chdir(decoy_cwd)  # process launched from somewhere else entirely

    m.main()

    expected_output = work_dir / "submission.csv"
    assert expected_output.is_file()
    assert not (decoy_cwd / "submission.csv").exists()
    assert not (tmp_path / "submission.csv").exists()

    submission = pd.read_csv(expected_output)
    assert submission["uid"].tolist() == UIDS
    assert (submission["is_pathologic"] == 0.7).all()
