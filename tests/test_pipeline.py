"""Tests for data preparation, training and prediction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.data import class_balance, clean_dataset, split_dataset, split_features_target
from src.predict import (
    ModelNotAvailableError,
    global_feature_importance,
    load_model,
    predict_records,
)
from src.train import train


def test_split_is_deterministic(synthetic_frame: pd.DataFrame) -> None:
    """The same seed yields byte-identical splits; a different seed does not."""
    first_train, first_test = split_dataset(synthetic_frame, test_size=0.25, seed=42)
    second_train, second_test = split_dataset(synthetic_frame, test_size=0.25, seed=42)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)

    other_train, _other_test = split_dataset(synthetic_frame, test_size=0.25, seed=7)
    assert not other_train.equals(first_train)


def test_split_is_stratified_and_disjoint(synthetic_frame: pd.DataFrame) -> None:
    """Both sides keep every class, and no row appears twice."""
    train_df, test_df = split_dataset(synthetic_frame, test_size=0.25, seed=42)

    assert len(train_df) + len(test_df) == len(synthetic_frame)
    assert set(train_df[config.TARGET_COLUMN]) == set(config.CLASS_ORDER)
    assert set(test_df[config.TARGET_COLUMN]) == set(config.CLASS_ORDER)
    assert not set(train_df["Patient_ID"]) & set(test_df["Patient_ID"])


def test_invalid_test_size_is_rejected(synthetic_frame: pd.DataFrame) -> None:
    """A test fraction outside (0, 1) is a hard error."""
    with pytest.raises(ValueError, match="test_size"):
        split_dataset(synthetic_frame, test_size=1.5)


def test_clean_dataset_drops_unknown_labels(synthetic_frame: pd.DataFrame) -> None:
    """Rows whose target is missing or unrecognised are removed."""
    dirty = synthetic_frame.copy()
    dirty.loc[0, config.TARGET_COLUMN] = "Unknown"
    dirty.loc[1, config.TARGET_COLUMN] = None

    cleaned = clean_dataset(dirty)
    assert len(cleaned) == len(dirty) - 2
    assert set(cleaned[config.TARGET_COLUMN]).issubset(set(config.CLASS_ORDER))


def test_class_balance_orders_by_severity(synthetic_frame: pd.DataFrame) -> None:
    """The balance table follows CLASS_ORDER and sums to 100%."""
    balance = class_balance(synthetic_frame[config.TARGET_COLUMN])
    assert list(balance.index) == list(config.CLASS_ORDER)
    assert balance["count"].sum() == len(synthetic_frame)
    assert round(float(balance["percent"].sum())) == 100


def test_train_writes_model_and_metadata(synthetic_frame: pd.DataFrame, tmp_path: Path) -> None:
    """A short training run persists a usable model plus honest metadata."""
    train_csv = tmp_path / "train.csv"
    synthetic_frame.to_csv(train_csv, index=False)
    model_path = tmp_path / "model.joblib"

    metadata = train(
        train_csv=train_csv,
        model_choice="logreg",
        model_path=model_path,
        keep_leakage=False,
        cv_folds=2,
        scoring=config.SCORING,
        n_jobs=1,
        seed=0,
    )

    assert model_path.is_file()
    written = json.loads(model_path.with_name("metadata.json").read_text(encoding="utf-8"))
    assert written["model_name"] == "logreg"
    assert written["keep_leakage"] is False
    assert written["cv_folds"] == 2
    assert set(written["class_order"]) == set(config.CLASS_ORDER)
    assert not set(written["input_columns"]) & set(config.LEAKAGE_COLUMNS)
    assert metadata["cv_score"] == written["cv_score"]


def test_predict_on_sample_record(fitted_pipeline, sample_patient: dict[str, object]) -> None:
    """The committed sample JSON scores end to end and sums to one."""
    results = predict_records(fitted_pipeline, [sample_patient])

    assert len(results) == 1
    result = results[0]
    assert result["prediction"] in config.CLASS_ORDER
    assert set(result["probabilities"]) == set(config.CLASS_ORDER)
    assert round(sum(result["probabilities"].values()), 6) == 1.0
    assert list(result["probabilities"]) == list(config.CLASS_ORDER)


def test_predict_batch_from_sample_csv(fitted_pipeline) -> None:
    """The committed sample CSV scores as a batch, one prediction per row."""
    frame = pd.read_csv(Path(__file__).resolve().parents[1] / "samples" / "sample_patients.csv")
    results = predict_records(fitted_pipeline, frame.to_dict(orient="records"))

    assert len(results) == len(frame)
    assert all(item["prediction"] in config.CLASS_ORDER for item in results)


def test_predict_tolerates_missing_fields(fitted_pipeline, sample_patient) -> None:
    """A partially filled record is imputed rather than rejected."""
    partial = {key: value for key, value in sample_patient.items() if key != "HbA1c"}
    results = predict_records(fitted_pipeline, [partial])
    assert results[0]["prediction"] in config.CLASS_ORDER


def test_load_model_missing_file_is_explicit(tmp_path: Path) -> None:
    """A missing artifact produces the 'train the model first' message."""
    with pytest.raises(ModelNotAvailableError, match="python -m src.train"):
        load_model(tmp_path / "nope.joblib")


def test_global_feature_importance_falls_back_to_coefficients(
    fitted_pipeline, tmp_path: Path
) -> None:
    """Without a permutation report, coefficients are used instead."""
    drivers = global_feature_importance(
        fitted_pipeline, metadata=None, top_k=5, importance_path=tmp_path / "absent.json"
    )
    assert len(drivers) == 5
    assert all(driver["score"] >= 0 for driver in drivers)
    assert all(
        not any(column in driver["feature"] for column in config.LEAKAGE_COLUMNS)
        for driver in drivers
    )


def test_split_features_target_separates_label(synthetic_frame: pd.DataFrame) -> None:
    """Splitting features from the target leaves the label out of the frame."""
    features, labels = split_features_target(synthetic_frame)
    assert config.TARGET_COLUMN not in features.columns
    assert len(labels) == len(synthetic_frame)
