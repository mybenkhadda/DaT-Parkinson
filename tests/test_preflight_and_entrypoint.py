"""Tests for the Phase 5 static preflight (verify_runtime_dependencies,
require_artifact) and the ModuleNotFoundError-aware run_entrypoint(), added
after tracking down a real ModuleNotFoundError: final_feature_model.joblib
pickled an xgboost.sklearn.XGBClassifier, and xgboost is not a confirmed
runtime dependency. Path A was removed (its trained ensemble weight was
exactly 0.0); these tests guard against the failure mode recurring for any
future artifact."""
import pytest

import config
import main as m


def test_verify_runtime_dependencies_passes_for_real_modules():
    m.verify_runtime_dependencies()  # must not raise: numpy/pandas/torch/etc. are installed


def test_verify_runtime_dependencies_raises_module_not_found_with_name(monkeypatch):
    monkeypatch.setattr(m, "REQUIRED_MODULES", ("numpy", "this_module_does_not_exist_anywhere"))

    with pytest.raises(ModuleNotFoundError) as excinfo:
        m.verify_runtime_dependencies()

    assert excinfo.value.name == "this_module_does_not_exist_anywhere"


def test_require_artifact_passes_for_existing_file(tmp_path):
    path = tmp_path / "present.joblib"
    path.write_bytes(b"x")
    m.require_artifact(path)  # must not raise


def test_require_artifact_raises_generic_for_missing_file(tmp_path):
    path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError) as excinfo:
        m.require_artifact(path)

    assert str(excinfo.value) == "Required inference artifact is unavailable."
    assert str(path) not in str(excinfo.value)


def test_load_model_weights_raises_module_not_found_when_dependency_missing_before_any_artifact_load(
    tmp_path, monkeypatch
):
    """Reproduces the originally-reported failure shape end-to-end: a
    required package unavailable should surface as ModuleNotFoundError with
    the module name preserved, not some other exception type."""
    monkeypatch.setattr(m, "REQUIRED_MODULES", ("this_module_does_not_exist_anywhere",))

    with pytest.raises(ModuleNotFoundError) as excinfo:
        m.main()

    assert excinfo.value.name == "this_module_does_not_exist_anywhere"


def test_run_entrypoint_reports_missing_module_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(config, "SUBMISSION_DEBUG", False)
    monkeypatch.setattr(m, "main", lambda: (_ for _ in ()).throw(
        ModuleNotFoundError("No module named 'xgboost'", name="xgboost")
    ))

    with pytest.raises(SystemExit) as excinfo:
        m.run_entrypoint()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == "FATAL: required Python module unavailable during startup: xgboost."
    assert "Traceback" not in captured.err
    assert "xgboost" not in captured.err  # no traceback leakage either


def test_run_entrypoint_shows_traceback_only_in_debug_mode(monkeypatch, capsys):
    monkeypatch.setattr(config, "SUBMISSION_DEBUG", True)
    monkeypatch.setattr(m, "main", lambda: (_ for _ in ()).throw(
        ModuleNotFoundError("No module named 'xgboost'", name="xgboost")
    ))

    with pytest.raises(SystemExit):
        m.run_entrypoint()

    captured = capsys.readouterr()
    assert "Traceback" in captured.err


def test_run_entrypoint_generic_exception_gets_type_name_only(monkeypatch, capsys):
    monkeypatch.setattr(config, "SUBMISSION_DEBUG", False)
    monkeypatch.setattr(m, "main", lambda: (_ for _ in ()).throw(RuntimeError("some internal detail")))

    with pytest.raises(SystemExit) as excinfo:
        m.run_entrypoint()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == "FATAL: submission failed during startup (RuntimeError)."
    assert "some internal detail" not in (captured.out + captured.err)


def test_loaded_models_has_no_path_a_field():
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(m.LoadedModels)}
    assert "path_a" not in field_names
    assert field_names == {"path_b", "path_b_n_slices", "path_c", "ensemble", "calibrator"}


def test_config_has_no_path_a_weights_constant():
    assert not hasattr(config, "PATH_A_WEIGHTS")
    assert not hasattr(config, "FEATURE_COLUMNS")


def test_manifest_disables_path_a_and_ensemble_has_two_models():
    assert config.BASE_MODELS_ENABLED.get("path_a") in (False, None)
    assert config.BASE_MODEL_COLS == ["oof_pred_b", "oof_pred_c"]

    import joblib
    ensemble = joblib.load(config.ENSEMBLE_WEIGHTS)
    assert ensemble["base_model_cols"] == ["oof_pred_b", "oof_pred_c"]
    assert len(ensemble["weights"]) == 2
    assert abs(sum(ensemble["weights"]) - 1.0) < 1e-9
