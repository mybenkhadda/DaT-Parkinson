"""Unit tests for load_submission_template and write_submission (atomic write
+ read-back validation). resolve_paths() itself is covered by
test_path_resolution.py."""
import pandas as pd
import pytest

import main as m


def test_load_submission_template_missing_uid_column_raises(tmp_path):
    path = tmp_path / "submission_format.csv"
    path.write_text("id,is_pathologic\nabc,0.5\n")

    class FakePaths:
        template_path = path

    with pytest.raises(ValueError, match="uid"):
        m.load_submission_template(FakePaths())


def test_load_submission_template_duplicate_uid_raises(tmp_path):
    path = tmp_path / "submission_format.csv"
    path.write_text("uid,is_pathologic\nabc,0.5\nabc,0.5\n")

    class FakePaths:
        template_path = path

    with pytest.raises(ValueError, match="duplicate"):
        m.load_submission_template(FakePaths())


def test_write_submission_round_trips(tmp_path):
    output_path = tmp_path / "submission.csv"
    df = pd.DataFrame({"uid": ["a", "b", "c"], "is_pathologic": [0.1, 0.5, 0.9]})

    m.write_submission(df, output_path)

    assert output_path.is_file()
    reread = pd.read_csv(output_path)
    assert list(reread.columns) == ["uid", "is_pathologic"]
    assert reread["uid"].tolist() == ["a", "b", "c"]

    # no leftover temp files
    leftovers = list(tmp_path.glob(".submission_*"))
    assert leftovers == []
