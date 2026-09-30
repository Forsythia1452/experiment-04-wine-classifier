import numpy as np
import pandas as pd
import pytest

from src.features import SchemaError, assess_distribution, predict_rows, validate_frame


def median_frame(meta):
    return pd.DataFrame([{name: meta["training_stats"][name]["median"] for name in meta["feature_names"]}])


def test_pipeline_accepts_raw_schema(model, meta):
    frame = median_frame(meta)
    assert model.predict(frame).shape == (1,)
    assert model.predict_proba(frame).shape == (1, 3)


def test_missing_feature_is_rejected(meta):
    frame = median_frame(meta).drop(columns=[meta["feature_names"][0]])
    with pytest.raises(SchemaError, match="缺少特征"):
        validate_frame(frame, meta)


def test_unknown_feature_is_rejected(meta):
    frame = median_frame(meta); frame["unknown"] = 1
    with pytest.raises(SchemaError, match="未知特征"):
        validate_frame(frame, meta)


def test_non_numeric_value_is_rejected(meta):
    frame = median_frame(meta).astype(object); frame.iloc[0, 0] = "bad"
    with pytest.raises(SchemaError, match="空值或非数字"):
        validate_frame(frame, meta)


def test_infinite_value_is_rejected(meta):
    frame = median_frame(meta); frame.iloc[0, 0] = np.inf
    with pytest.raises(SchemaError, match="有限数值"):
        validate_frame(frame, meta)


def test_column_order_is_normalized(meta):
    frame = median_frame(meta)[list(reversed(meta["feature_names"]))]
    clean = validate_frame(frame, meta)
    assert clean.columns.tolist() == meta["feature_names"]


def test_loaded_model_prediction_matches_known_fixture(model, meta):
    frame = median_frame(meta)
    first = model.predict(frame).tolist()
    second = model.predict(validate_frame(frame, meta)).tolist()
    assert first == second


def test_normal_input_has_no_drift_warning(meta):
    frame = median_frame(meta)
    result = assess_distribution(frame.iloc[0], meta)
    assert result.level == "normal" and result.count == 0


def test_one_out_of_range_feature_warns(meta):
    frame = median_frame(meta)
    name = meta["feature_names"][0]
    frame.loc[0, name] = meta["training_stats"][name]["max"] + .01
    result = assess_distribution(frame.iloc[0], meta)
    assert result.level == "medium"
    assert name in result.details[0]


def test_multiple_out_of_range_features_are_high_risk(meta):
    frame = median_frame(meta)
    for name in meta["feature_names"][:2]:
        frame.loc[0, name] = meta["training_stats"][name]["max"] + .01
    assert assess_distribution(frame.iloc[0], meta).level == "high"


def test_prediction_result_contains_drift_evidence(model, meta):
    result = predict_rows(model, median_frame(meta), meta)
    assert set(result.columns) == {"prediction", "confidence", "distribution_level", "distribution_warnings"}
    assert 0 <= result.iloc[0]["confidence"] <= 1

