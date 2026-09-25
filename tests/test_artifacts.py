"""Checks on the artifacts committed to the repository.

These guard the files a fresh clone relies on: the raw dataset, the processed
train/test split, the trained model and the evaluation report. Each test is
skipped when its file is absent (for example in a trimmed fork).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.forms import FIELDS
from src import config
from src.features import input_columns
from src.predict import load_model, predict_records


def _require(path):
    if not path.is_file():
        pytest.skip(f"{path} is not present")
    return path


def test_processed_split_is_stratified_and_disjoint() -> None:
    train = pd.read_csv(_require(config.TRAIN_CSV_PATH))
    test = pd.read_csv(_require(config.TEST_CSV_PATH))

    assert len(train) + len(test) == 50_000
    assert set(train["Patient_ID"]).isdisjoint(test["Patient_ID"])
    train_share = train[config.TARGET_COLUMN].value_counts(normalize=True)
    test_share = test[config.TARGET_COLUMN].value_counts(normalize=True)
    for label in config.CLASS_ORDER:
        assert abs(train_share[label] - test_share[label]) < 0.002


def test_raw_dataset_has_the_expected_schema() -> None:
    raw = pd.read_csv(_require(config.RAW_CSV_PATH), nrows=5)
    assert tuple(raw.columns) == config.EXPECTED_COLUMNS


def test_committed_model_matches_the_form() -> None:
    pipeline, metadata = load_model(_require(config.MODEL_PATH))
    columns = input_columns(pipeline)

    assert sorted(columns) == sorted(field.name for field in FIELDS)
    assert sorted(columns) == sorted(config.FEATURE_COLUMNS)
    assert not set(columns) & set(config.LEAKAGE_COLUMNS)
    assert metadata["keep_leakage"] is False
    assert config.MODEL_PATH.stat().st_size < 25 * 1024 * 1024


def test_committed_model_scores_the_sample_patient() -> None:
    pipeline, _metadata = load_model(_require(config.MODEL_PATH))
    record = json.loads((config.SAMPLES_DIR / "sample_patient.json").read_text(encoding="utf-8"))
    result = predict_records(pipeline, [record])[0]
    assert result["prediction"] in config.CLASS_ORDER
    assert round(sum(result["probabilities"].values()), 6) == 1.0


def test_metrics_report_matches_the_model() -> None:
    metrics = json.loads(_require(config.METRICS_PATH).read_text(encoding="utf-8"))
    metadata = json.loads(_require(config.METADATA_PATH).read_text(encoding="utf-8"))

    assert metrics["n_test_rows"] == 10_000
    assert metrics["model"]["model_name"] == metadata["model_name"]
    assert metrics["model"]["trained_at_utc"] == metadata["trained_at_utc"]
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    for figure in metrics["figures"]:
        assert (config.PROJECT_ROOT / figure).is_file()
