"""Tests for resolve_paths() -- must work whether the evaluator extracts the
archive with main.py at the working-directory root or inside a src/
subdirectory, must resolve bundled assets relative to main.py's own location
regardless, and must fail with a generic (non-leaking) message when required
inputs are absent."""
import pytest

import main as m


def _make_data_dir(root, with_niftis=True):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "submission_format.csv").write_text("uid,is_pathologic\na,0.5\n")
    if with_niftis:
        (data / "niftis").mkdir(exist_ok=True)
    return data


def test_layout1_main_at_root(tmp_path, monkeypatch):
    """main.py directly in the working directory, alongside data/ and artifacts/."""
    root = tmp_path
    (root / "artifacts").mkdir()
    data = _make_data_dir(root)

    monkeypatch.setattr(m, "__file__", str(root / "main.py"))
    monkeypatch.chdir(root)

    paths = m.resolve_paths()

    assert paths.data_dir == data
    assert paths.nifti_dir == data / "niftis"
    assert paths.template_path == data / "submission_format.csv"
    assert paths.output_path == root / "submission.csv"
    assert paths.artifacts_dir == root / "artifacts"


def test_layout2_main_in_src_subdir(tmp_path, monkeypatch):
    """main.py inside src/, data/ one level up -- the documented
    /code_execution/{data,src}/ layout, with cwd == /code_execution/."""
    root = tmp_path
    src = root / "src"
    (src / "artifacts").mkdir(parents=True)
    data = _make_data_dir(root)

    monkeypatch.setattr(m, "__file__", str(src / "main.py"))
    monkeypatch.chdir(root)  # documented working directory is /code_execution/

    paths = m.resolve_paths()

    assert paths.data_dir == data
    assert paths.output_path == root / "submission.csv"
    assert paths.artifacts_dir == src / "artifacts"


def test_layout3_missing_template_raises_generic(tmp_path, monkeypatch):
    root = tmp_path
    (root / "artifacts").mkdir()
    (root / "data").mkdir()  # present but empty: no submission_format.csv

    monkeypatch.setattr(m, "__file__", str(root / "main.py"))
    monkeypatch.chdir(root)

    with pytest.raises(FileNotFoundError) as excinfo:
        m.resolve_paths()

    assert str(excinfo.value) == "Required competition input files are unavailable."
    assert str(root) not in str(excinfo.value)


def test_layout4_missing_nifti_dir_raises_generic(tmp_path, monkeypatch):
    """resolve_paths() itself only locates paths; the niftis/ check now lives
    in validate_runtime_inputs(), called right after it in main() -- see
    test_validate_runtime_inputs.py for that function's own tests. This
    still exercises resolve_paths() succeeding despite the missing dir."""
    root = tmp_path
    (root / "artifacts").mkdir()
    _make_data_dir(root, with_niftis=False)

    monkeypatch.setattr(m, "__file__", str(root / "main.py"))
    monkeypatch.chdir(root)

    paths = m.resolve_paths()  # succeeds: resolve_paths() no longer validates niftis/
    assert paths.nifti_dir == root / "data" / "niftis"

    with pytest.raises(FileNotFoundError) as excinfo:
        m.validate_runtime_inputs(paths)
    assert str(excinfo.value) == "Required imaging input directory is unavailable."
    assert str(root) not in str(excinfo.value)


def test_layout5_misleading_cwd_still_resolves_via_file_location(tmp_path, monkeypatch):
    """cwd points somewhere with no data/ at all -- resolution must fall back
    to locations derived from main.py's own file path rather than failing."""
    root = tmp_path / "real_root"
    src = root / "src"
    (src / "artifacts").mkdir(parents=True)
    data = _make_data_dir(root)

    decoy_cwd = tmp_path / "unrelated_cwd"
    decoy_cwd.mkdir()

    monkeypatch.setattr(m, "__file__", str(src / "main.py"))
    monkeypatch.chdir(decoy_cwd)

    paths = m.resolve_paths()

    assert paths.data_dir == data
    assert paths.artifacts_dir == src / "artifacts"


def test_resolve_paths_succeeds_even_if_artifacts_dir_is_missing(tmp_path, monkeypatch):
    """resolve_paths() only locates paths; it deliberately does not validate
    that artifacts/ exists or is complete -- that's require_artifact()'s job,
    checked file-by-file at the point each artifact is actually loaded (see
    test_preflight_and_entrypoint.py), which gives a more precise failure
    than a single whole-directory check would."""
    root = tmp_path
    _make_data_dir(root)
    # artifacts/ deliberately not created

    monkeypatch.setattr(m, "__file__", str(root / "main.py"))
    monkeypatch.chdir(root)

    paths = m.resolve_paths()  # must not raise
    assert paths.artifacts_dir == root / "artifacts"
    assert not paths.artifacts_dir.exists()
