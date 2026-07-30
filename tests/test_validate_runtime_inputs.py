"""Tests for validate_runtime_inputs() -- the cheap, filesystem-only checks
that must catch a missing/empty competition data mount BEFORE any model
artifact is loaded (Required Tests #2, #3, #4 in the incident ticket: missing
template, missing niftis dir, empty niftis dir)."""
import pandas as pd
import pytest

import main as m


def _paths(tmp_path, template=True, nifti_dir=True, nifti_file=True):
    data_dir = tmp_path / "data"
    nifti = data_dir / "niftis"
    if nifti_dir:
        nifti.mkdir(parents=True)
    else:
        data_dir.mkdir(parents=True)
    template_path = data_dir / "submission_format.csv"
    if template:
        pd.DataFrame({"uid": ["a"], "is_pathologic": [0.5]}).to_csv(template_path, index=False)
    if nifti_dir and nifti_file:
        (nifti / "a.nii.gz").write_bytes(b"placeholder")
    return m.Paths(
        data_dir=data_dir,
        nifti_dir=nifti,
        template_path=template_path,
        output_path=tmp_path / "submission.csv",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_valid_inputs_pass(tmp_path):
    paths = _paths(tmp_path)
    m.validate_runtime_inputs(paths)  # must not raise


def test_missing_template_raises_generic(tmp_path):
    paths = _paths(tmp_path, template=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        m.validate_runtime_inputs(paths)
    assert str(excinfo.value) == "Required competition input files are unavailable."
    assert str(tmp_path) not in str(excinfo.value)


def test_missing_nifti_dir_raises_generic(tmp_path):
    paths = _paths(tmp_path, nifti_dir=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        m.validate_runtime_inputs(paths)
    assert str(excinfo.value) == "Required imaging input directory is unavailable."
    assert str(tmp_path) not in str(excinfo.value)


def test_empty_nifti_dir_raises_generic(tmp_path):
    """niftis/ exists but contains zero .nii/.nii.gz files -- this is the
    condition that, before the fix, slipped past resolve_paths() (which only
    checked nifti_dir.is_dir()) and wasn't caught until discover_exam_files()
    ran AFTER model loading, producing the reported "4 stages then
    ValueError" log signature."""
    paths = _paths(tmp_path, nifti_file=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        m.validate_runtime_inputs(paths)
    assert str(excinfo.value) == "Required imaging inputs are unavailable."
    assert str(tmp_path) not in str(excinfo.value)


def test_nifti_dir_with_only_unrelated_files_raises_generic(tmp_path):
    paths = _paths(tmp_path, nifti_file=False)
    (paths.nifti_dir / "readme.txt").write_text("not a scan")
    with pytest.raises(FileNotFoundError) as excinfo:
        m.validate_runtime_inputs(paths)
    assert str(excinfo.value) == "Required imaging inputs are unavailable."


def test_nifti_dir_with_nii_uncompressed_file_passes(tmp_path):
    paths = _paths(tmp_path, nifti_file=False)
    (paths.nifti_dir / "a.nii").write_bytes(b"placeholder")
    m.validate_runtime_inputs(paths)  # must not raise


def test_error_messages_do_not_leak_paths_or_counts(tmp_path):
    paths = _paths(tmp_path, template=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        m.validate_runtime_inputs(paths)
    message = str(excinfo.value)
    assert str(paths.template_path) not in message
    assert str(paths.data_dir) not in message
