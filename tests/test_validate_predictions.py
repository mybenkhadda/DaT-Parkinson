"""Unit tests for main.validate_predictions -- every check listed in the
submission spec's Output Validation section, exercised both for the passing
case and each individual failure mode.
"""
import numpy as np
import pandas as pd
import pytest

import main as m


def _template(uids):
    return pd.DataFrame({"uid": uids, "is_pathologic": [0.5] * len(uids)})


def _preds(uids, probs):
    return pd.DataFrame({"uid": uids, "is_pathologic": probs})


def test_valid_predictions_pass():
    uids = ["a", "b", "c"]
    template = _template(uids)
    preds = _preds(uids, [0.1, 0.5, 0.9])
    m.validate_predictions(preds, template)  # must not raise


def test_wrong_row_count_fails():
    template = _template(["a", "b", "c"])
    preds = _preds(["a", "b"], [0.1, 0.5])
    with pytest.raises(AssertionError, match="number of rows"):
        m.validate_predictions(preds, template)


def test_duplicate_uid_fails():
    template = _template(["a", "b", "c"])
    preds = _preds(["a", "a", "c"], [0.1, 0.5, 0.9])
    with pytest.raises(AssertionError, match="duplicate identifiers"):
        m.validate_predictions(preds, template)


def test_missing_uid_fails():
    template = _template(["a", "b", "c"])
    preds = _preds(["a", "b", "d"], [0.1, 0.5, 0.9])
    with pytest.raises(AssertionError, match="do not match the required set"):
        m.validate_predictions(preds, template)


def test_extra_uid_fails():
    template = _template(["a", "b"])
    preds = _preds(["a", "b", "c"], [0.1, 0.5, 0.9])
    with pytest.raises(AssertionError, match="number of rows|do not match the required set"):
        m.validate_predictions(preds, template)


def test_wrong_order_fails():
    template = _template(["a", "b", "c"])
    preds = _preds(["c", "b", "a"], [0.9, 0.5, 0.1])
    with pytest.raises(AssertionError, match="order"):
        m.validate_predictions(preds, template)


def test_nan_prediction_fails():
    template = _template(["a", "b", "c"])
    preds = _preds(["a", "b", "c"], [0.1, np.nan, 0.9])
    with pytest.raises(AssertionError, match="NaN|infinite"):
        m.validate_predictions(preds, template)


def test_infinite_prediction_fails():
    template = _template(["a", "b", "c"])
    preds = _preds(["a", "b", "c"], [0.1, float("inf"), 0.9])
    with pytest.raises(AssertionError, match="NaN|infinite"):
        m.validate_predictions(preds, template)


def test_out_of_range_probability_fails():
    template = _template(["a", "b", "c"])
    preds = _preds(["a", "b", "c"], [0.1, 1.5, 0.9])
    with pytest.raises(AssertionError, match="out-of-range"):
        m.validate_predictions(preds, template)


def test_negative_probability_fails():
    template = _template(["a", "b", "c"])
    preds = _preds(["a", "b", "c"], [-0.1, 0.5, 0.9])
    with pytest.raises(AssertionError, match="out-of-range"):
        m.validate_predictions(preds, template)


def test_wrong_columns_fails():
    template = _template(["a", "b", "c"])
    preds = pd.DataFrame({"uid": ["a", "b", "c"], "probability": [0.1, 0.5, 0.9]})
    with pytest.raises(AssertionError, match="columns"):
        m.validate_predictions(preds, template)


def test_non_numeric_probability_fails():
    template = _template(["a", "b", "c"])
    preds = _preds(["a", "b", "c"], ["low", "mid", "high"])
    with pytest.raises(AssertionError, match="numeric"):
        m.validate_predictions(preds, template)
