"""Verifies main()'s stdout contains only the four safe stage-level messages
-- never uids, filenames, paths, dataset counts, or predictions -- using a
fully mocked model (no real weights/images needed)."""
import pandas as pd
import pytest

import main as m

HIDDEN_UIDS = ["hidden_uid_alpha", "hidden_uid_beta", "hidden_uid_gamma"]
FAKE_PROB = 0.123456
ALLOWED_LINES = {
    "Loading inference assets.",
    "Starting inference.",
    "Writing output.",
    "Inference completed.",
}


@pytest.fixture
def mock_run(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    nifti_dir = data_dir / "niftis"
    nifti_dir.mkdir(parents=True)
    for uid in HIDDEN_UIDS:
        (nifti_dir / f"{uid}.nii.gz").write_bytes(b"placeholder")
    template_path = data_dir / "submission_format.csv"
    pd.DataFrame({"uid": HIDDEN_UIDS, "is_pathologic": [0.5] * len(HIDDEN_UIDS)}).to_csv(
        template_path, index=False
    )
    output_path = tmp_path / "submission.csv"

    monkeypatch.setattr(m, "load_model_weights", lambda: object())
    monkeypatch.setattr(m, "validate_loaded_assets", lambda _loaded: None)
    monkeypatch.setattr(m, "predict_exam", lambda _loaded, _path: FAKE_PROB)

    # resolve_paths() no longer trusts config.* directly -- point __file__ at
    # a location whose parent *is* tmp_path so the work-dir search finds it.
    monkeypatch.setattr(m, "__file__", str(tmp_path / "main.py"))
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)

    return tmp_path, output_path


def test_stdout_contains_only_safe_stage_messages(mock_run, capsys):
    tmp_path, output_path = mock_run

    m.main()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]

    assert lines, "expected at least the stage-level log lines"
    for line in lines:
        assert line in ALLOWED_LINES, f"unexpected / unsafe log line: {line!r}"

    combined = captured.out + captured.err
    for uid in HIDDEN_UIDS:
        assert uid not in combined
    assert str(FAKE_PROB) not in combined
    assert "niftis" not in combined.lower()
    assert str(tmp_path) not in combined
    assert "Found" not in combined
    assert "Processed" not in combined
    assert str(len(HIDDEN_UIDS)) not in combined
