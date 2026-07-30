"""End-to-end scenario tests matching the incident ticket's "Required Tests"
1 and 5, run as real subprocesses against the actual submission_src/main.py
(not monkeypatched) so the printed output and exit behavior are exactly what
the competition runtime would see."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SUBMISSION_SRC = Path(__file__).resolve().parents[1] / "submission_src"
PYTHON = sys.executable


def _run(cwd: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os
    env = os.environ.copy()
    env.pop("SUBMISSION_DEBUG", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [PYTHON, str(cwd / "main.py")],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def _copy_submission_src(dest: Path) -> None:
    import shutil
    shutil.copytree(SUBMISSION_SRC, dest, dirs_exist_ok=True)


def test_empty_full_runtime_data_fails_before_asset_loading(tmp_path):
    """Required Test #1: runtime/{main.py,artifacts/,data/} with data/ empty.
    Must fail during "input validation" WITHOUT ever printing
    "Loading inference assets." -- proving input validation runs first."""
    runtime = tmp_path / "runtime"
    _copy_submission_src(runtime)
    (runtime / "data").mkdir(exist_ok=True)  # empty

    result = _run(runtime)

    assert result.returncode == 1
    assert "Loading inference assets." not in result.stdout
    assert result.stdout.strip() == "FATAL: required runtime inputs unavailable during input validation."
    assert "Traceback" not in result.stderr


def test_smoke_style_input_proceeds_past_input_validation(tmp_path):
    """Required Test #5: a valid small mock template + mock NIfTI files must
    let startup proceed all the way to asset loading (it will then fail on
    the FAKE nifti content during inference -- that's expected and fine;
    the point is it gets past input validation)."""
    runtime = tmp_path / "runtime"
    _copy_submission_src(runtime)
    data_dir = runtime / "data"
    nifti_dir = data_dir / "niftis"
    nifti_dir.mkdir(parents=True)
    (nifti_dir / "mockuid.nii.gz").write_bytes(b"not a real nifti, just a placeholder")
    pd.DataFrame({"uid": ["mockuid"], "is_pathologic": [0.5]}).to_csv(
        data_dir / "submission_format.csv", index=False
    )

    result = _run(runtime)

    assert "Loading inference assets." in result.stdout
    assert "Loading neural models." in result.stdout


def test_missing_template_generic_message(tmp_path):
    """Required Test #2."""
    runtime = tmp_path / "runtime"
    _copy_submission_src(runtime)
    (runtime / "data" / "niftis").mkdir(parents=True)
    (runtime / "data" / "niftis" / "a.nii.gz").write_bytes(b"x")

    result = _run(runtime)

    assert result.returncode == 1
    assert result.stdout.strip() == "FATAL: required runtime inputs unavailable during input validation."
    assert "Loading inference assets." not in result.stdout


def test_missing_nifti_dir_generic_message(tmp_path):
    """Required Test #3."""
    runtime = tmp_path / "runtime"
    _copy_submission_src(runtime)
    data_dir = runtime / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"uid": ["a"], "is_pathologic": [0.5]}).to_csv(
        data_dir / "submission_format.csv", index=False
    )

    result = _run(runtime)

    assert result.returncode == 1
    assert result.stdout.strip() == "FATAL: required runtime inputs unavailable during input validation."
    assert "Loading inference assets." not in result.stdout


def test_empty_nifti_dir_generic_message(tmp_path):
    """Required Test #4: submission_format.csv + niftis/ both present, but
    niftis/ contains no .nii/.nii.gz files."""
    runtime = tmp_path / "runtime"
    _copy_submission_src(runtime)
    data_dir = runtime / "data"
    (data_dir / "niftis").mkdir(parents=True)
    pd.DataFrame({"uid": ["a"], "is_pathologic": [0.5]}).to_csv(
        data_dir / "submission_format.csv", index=False
    )

    result = _run(runtime)

    assert result.returncode == 1
    assert result.stdout.strip() == "FATAL: required runtime inputs unavailable during input validation."
    assert "Loading inference assets." not in result.stdout


def test_debug_mode_shows_traceback_only_locally(tmp_path):
    """Required Test #9."""
    runtime = tmp_path / "runtime"
    _copy_submission_src(runtime)
    (runtime / "data").mkdir(exist_ok=True)

    result_default = _run(runtime)
    result_debug = _run(runtime, env_extra={"SUBMISSION_DEBUG": "1"})

    assert "Traceback" not in result_default.stderr
    assert "Traceback" not in result_default.stdout
    assert "Traceback" in result_debug.stderr
