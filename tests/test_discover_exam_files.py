"""Unit tests for main.discover_exam_files -- deterministic uid -> file
resolution, including the failure modes called out in the submission spec:
missing scans, duplicate/ambiguous matches, and substring-collision safety.
"""
import pytest

import config
import main as m


def _touch(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_single_file_per_uid_resolves(tmp_path):
    _touch(tmp_path / "abc123.nii.gz")
    _touch(tmp_path / "def456.nii.gz")

    resolved = m.discover_exam_files(tmp_path, ["abc123", "def456"])

    assert set(resolved.keys()) == {"abc123", "def456"}
    assert resolved["abc123"].name == "abc123.nii.gz"
    assert resolved["def456"].name == "def456.nii.gz"


def test_nii_suffix_also_supported(tmp_path):
    _touch(tmp_path / "abc123.nii")

    resolved = m.discover_exam_files(tmp_path, ["abc123"])

    assert resolved["abc123"].name == "abc123.nii"


def test_nested_directories_are_searched(tmp_path):
    _touch(tmp_path / "sub" / "dir" / "abc123.nii.gz")

    resolved = m.discover_exam_files(tmp_path, ["abc123"])

    assert resolved["abc123"].name == "abc123.nii.gz"


def test_missing_scan_raises_generic_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SUBMISSION_DEBUG", False)
    _touch(tmp_path / "abc123.nii.gz")

    with pytest.raises(ValueError) as excinfo:
        m.discover_exam_files(tmp_path, ["abc123", "does_not_exist"])
    assert str(excinfo.value) == "Required imaging files could not be matched to every examination."


def test_duplicate_scan_raises_generic_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SUBMISSION_DEBUG", False)
    _touch(tmp_path / "abc123.nii.gz")
    _touch(tmp_path / "abc123.nii")  # same uid, two recognized extensions

    with pytest.raises(ValueError) as excinfo:
        m.discover_exam_files(tmp_path, ["abc123"])
    assert str(excinfo.value) == "Required imaging files could not be matched to every examination."


def test_missing_scan_raises_detailed_message_in_debug_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SUBMISSION_DEBUG", True)
    _touch(tmp_path / "abc123.nii.gz")

    with pytest.raises(ValueError, match="no matching file"):
        m.discover_exam_files(tmp_path, ["abc123", "does_not_exist"])


def test_duplicate_scan_raises_detailed_message_in_debug_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SUBMISSION_DEBUG", True)
    _touch(tmp_path / "abc123.nii.gz")
    _touch(tmp_path / "abc123.nii")  # same uid, two recognized extensions

    with pytest.raises(ValueError, match="multiple matching files"):
        m.discover_exam_files(tmp_path, ["abc123"])


def test_no_substring_collision(tmp_path):
    """A uid must never match another uid's file via 'in' / substring logic."""
    _touch(tmp_path / "abc.nii.gz")
    _touch(tmp_path / "abcdef.nii.gz")

    resolved = m.discover_exam_files(tmp_path, ["abc", "abcdef"])

    assert resolved["abc"].name == "abc.nii.gz"
    assert resolved["abcdef"].name == "abcdef.nii.gz"


def test_unrecognized_extension_ignored(tmp_path):
    _touch(tmp_path / "abc123.nii.gz")
    _touch(tmp_path / "notes.txt")

    resolved = m.discover_exam_files(tmp_path, ["abc123"])

    assert set(resolved.keys()) == {"abc123"}


def test_error_messages_do_not_leak_uids(tmp_path):
    """Logging restrictions: error text must not echo the specific uid or
    filename involved, only counts."""
    _touch(tmp_path / "abc123.nii.gz")
    secret_uid = "totally_hidden_test_uid"

    with pytest.raises(ValueError) as excinfo:
        m.discover_exam_files(tmp_path, ["abc123", secret_uid])

    assert secret_uid not in str(excinfo.value)
