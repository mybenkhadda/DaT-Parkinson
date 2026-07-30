"""End-to-end integration smoke test: runs the real, packaged main.py against
the official DrivenData smoke-test archive (20 examinations, extracted to
data-demo/ at the repo root), exactly as the competition runtime would.

This is the closest thing to `make test-submission` available without the
official runtime repo/Docker image (see the assumptions list in the project
report). It uses the actual bundled artifacts, not a mock model.
"""
import shutil
import time
from pathlib import Path

import pandas as pd
import pytest

import main as m

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DEMO = REPO_ROOT / "data-demo"

pytestmark = pytest.mark.skipif(
    not (DATA_DEMO / "niftis").is_dir(),
    reason="data-demo/ (smoke-test archive extraction) not present",
)


def test_full_pipeline_on_smoke_test_archive(tmp_path, monkeypatch):
    # resolve_paths() looks for <work_dir>/data/submission_format.csv;
    # data-demo/ mirrors that contract but under a different directory name,
    # so mirror it into a real "data" dir under a throwaway work_dir. Model
    # weights are NOT copied -- load_model_weights() reads them via
    # config.ARTIFACTS_DIR, which stays pointed at the real submission_src/artifacts
    # regardless of this test's work_dir, so this still exercises the actual
    # trained model.
    work_dir = tmp_path
    shutil.copytree(DATA_DEMO, work_dir / "data")
    (work_dir / "artifacts").mkdir()  # dummy: satisfies resolve_paths()'s existence check only
    output_path = work_dir / "submission.csv"

    monkeypatch.setattr(m, "__file__", str(work_dir / "main.py"))
    monkeypatch.chdir(work_dir)

    start = time.monotonic()
    m.main()
    elapsed = time.monotonic() - start

    assert output_path.is_file()
    submission = pd.read_csv(output_path)
    template = pd.read_csv(DATA_DEMO / "submission_format.csv")

    assert list(submission.columns) == ["uid", "is_pathologic"]
    assert submission["uid"].tolist() == template["uid"].astype(str).tolist()
    assert submission["is_pathologic"].between(0.0, 1.0).all()
    assert submission["is_pathologic"].notna().all()

    # Smoke-test time budget from the spec: 6 minutes for ~20 examinations.
    assert elapsed < 360, f"smoke test took {elapsed:.1f}s, expected < 360s"

    # Not a scored assertion (20 held-out-ish samples is too small to gate CI
    # on), just a sanity print so a regression is visible locally.
    test_labels = pd.read_csv(DATA_DEMO / "test_labels.csv")
    merged = submission.merge(test_labels, on="uid", suffixes=("_pred", "_true"))
    accuracy = (
        (merged["is_pathologic_pred"] >= 0.5).astype(int) == merged["is_pathologic_true"].astype(int)
    ).mean()
    print(f"\n[integration] smoke-test accuracy: {accuracy:.2f} in {elapsed:.1f}s")
