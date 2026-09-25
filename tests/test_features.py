"""Tests for column selection, leakage exclusion and pipeline construction."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.features import (
    build_pipeline,
    drop_excluded_columns,
    get_feature_names,
    input_columns,
    prettify_feature_name,
    split_feature_columns,
)


def test_leakage_columns_are_dropped(synthetic_frame: pd.DataFrame) -> None:
    """Every identifier/leakage column disappears from the feature frame."""
    reduced = drop_excluded_columns(synthetic_frame)
    for column in config.LEAKAGE_COLUMNS:
        assert column in synthetic_frame.columns
        assert column not in reduced.columns


def test_keep_leakage_flag_preserves_columns(synthetic_frame: pd.DataFrame) -> None:
    """The demo flag keeps the leaky columns so the inflated score can be shown."""
    kept = drop_excluded_columns(synthetic_frame, keep_leakage=True)
    for column in config.LEAKAGE_COLUMNS:
        assert column in kept.columns


def test_feature_columns_constant_excludes_target_and_leakage() -> None:
    """``FEATURE_COLUMNS`` is the raw schema minus target, id and leakage columns."""
    assert config.TARGET_COLUMN not in config.FEATURE_COLUMNS
    assert not set(config.FEATURE_COLUMNS) & set(config.LEAKAGE_COLUMNS)
    assert len(config.FEATURE_COLUMNS) == len(config.NUMERIC_FIELDS) + len(
        config.CATEGORICAL_FIELDS
    )


def test_split_feature_columns_uses_overrides(synthetic_frame: pd.DataFrame) -> None:
    """Numeric and categorical groups match the override lists in config."""
    reduced = drop_excluded_columns(synthetic_frame).drop(columns=[config.TARGET_COLUMN])
    numeric, categorical = split_feature_columns(reduced)

    assert set(numeric) == set(config.NUMERIC_OVERRIDES)
    assert set(categorical) == set(config.CATEGORICAL_OVERRIDES)
    assert not set(numeric) & set(categorical)


def test_fitted_feature_names_contain_no_leakage(fitted_pipeline) -> None:
    """The decisive check: no transformed feature traces back to a leaky column."""
    names = get_feature_names(fitted_pipeline)
    assert names, "the fitted preprocessor should report feature names"

    for column in config.LEAKAGE_COLUMNS:
        assert not any(column in name for name in names), f"{column} leaked into the model"

    assert config.TARGET_COLUMN not in input_columns(fitted_pipeline)


def test_pipeline_has_expected_structure(synthetic_frame: pd.DataFrame) -> None:
    """The pipeline is preprocessor + estimator, with both imputers wired up."""
    features = synthetic_frame.drop(columns=[config.TARGET_COLUMN])
    pipeline, numeric, categorical = build_pipeline(features, "rf", seed=0)

    assert list(pipeline.named_steps) == ["preprocessor", "estimator"]
    transformers = dict(
        (name, transformer)
        for name, transformer, _columns in pipeline.named_steps["preprocessor"].transformers
    )
    assert transformers["numeric"].named_steps["imputer"].strategy == "median"
    assert transformers["categorical"].named_steps["imputer"].strategy == "most_frequent"
    assert transformers["categorical"].named_steps["encoder"].handle_unknown == "ignore"
    assert numeric and categorical


def test_build_pipeline_rejects_unknown_model(synthetic_frame: pd.DataFrame) -> None:
    """An unknown estimator name fails loudly rather than defaulting."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline(synthetic_frame, "not-a-model")


def test_prettify_feature_name() -> None:
    """One-hot names are rendered as 'Column: value' for the UI."""
    assert prettify_feature_name("categorical__Smoking_Status_Current") == "Smoking Status: Current"
    assert prettify_feature_name("numeric__HbA1c") == "HbA1c"
