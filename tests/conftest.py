"""Shared pytest fixtures.

Everything here runs on a small synthetic frame built from the field
specifications in :mod:`src.config`, so the suite never needs the 50k-row
dataset and finishes in a couple of seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.features import build_pipeline  # noqa: E402

ROWS_PER_CLASS = 40


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    """A balanced, fully populated frame with every raw column present.

    Numeric values are drawn around each class's own offset so that the label is
    genuinely learnable from the non-leaky features; the leakage columns are
    filled in so tests can prove they are dropped.
    """
    rng = np.random.default_rng(20240501)
    records: list[dict[str, object]] = []

    for class_index, label in enumerate(config.CLASS_ORDER):
        shift = (class_index - 1) * 0.9
        for _ in range(ROWS_PER_CLASS):
            record: dict[str, object] = {}
            for spec in config.NUMERIC_FIELDS:
                span = (spec.maximum - spec.minimum) / 10.0
                centre = spec.default + shift * span
                value = float(np.clip(rng.normal(centre, span / 3), spec.minimum, spec.maximum))
                record[spec.name] = round(value, 2)
            for spec in config.CATEGORICAL_FIELDS:
                weights = np.linspace(1.0, 1.0 + class_index, len(spec.choices))
                weights = weights / weights.sum()
                record[spec.name] = str(rng.choice(list(spec.choices), p=weights))

            record["Patient_ID"] = len(records) + 1
            record["Diabetes_Risk_Score"] = [20, 50, 85][class_index] + int(rng.integers(-3, 4))
            record["AI_Health_Recommendation"] = f"Plan {label}"
            record["Doctor_Consultation_Needed"] = "No" if label == "Low" else "Yes"
            record[config.TARGET_COLUMN] = label
            records.append(record)

    frame = pd.DataFrame(records)
    return (
        frame[list(config.EXPECTED_COLUMNS)].sample(frac=1.0, random_state=7).reset_index(drop=True)
    )


@pytest.fixture(scope="session")
def fitted_pipeline(synthetic_frame: pd.DataFrame):
    """A tiny logistic-regression pipeline fitted on the synthetic frame."""
    features = synthetic_frame.drop(columns=[config.TARGET_COLUMN])
    labels = synthetic_frame[config.TARGET_COLUMN]
    pipeline, _numeric, _categorical = build_pipeline(features, "logreg", seed=0)
    pipeline.fit(features, labels)
    return pipeline


@pytest.fixture(scope="session")
def model_file(tmp_path_factory: pytest.TempPathFactory, fitted_pipeline) -> Path:
    """Persist the fixture model plus a metadata sidecar, like ``src.train`` does."""
    directory = tmp_path_factory.mktemp("models")
    path = directory / "model.joblib"
    joblib.dump(fitted_pipeline, path)

    from src.features import get_feature_names

    metadata = {
        "model_name": "logreg",
        "cv_scoring": config.SCORING,
        "cv_score": 0.0,
        "trained_at_utc": "2024-05-01T00:00:00+00:00",
        "sklearn_version": "test-fixture",
        "class_order": list(config.CLASS_ORDER),
        "keep_leakage": False,
        "leakage_columns": list(config.LEAKAGE_COLUMNS),
        "transformed_feature_names": get_feature_names(fitted_pipeline),
    }
    path.with_name("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def app_with_model(model_file: Path):
    """Flask app configured to use the fixture model."""
    from app import create_app

    return create_app({"TESTING": True, "MODEL_PATH": str(model_file)})


@pytest.fixture
def client_with_model(app_with_model):
    """Test client for an app that has a usable model."""
    return app_with_model.test_client()


@pytest.fixture
def client_without_model(tmp_path: Path):
    """Test client for an app whose model file does not exist."""
    from app import create_app

    missing = tmp_path / "models" / "model.joblib"
    return create_app({"TESTING": True, "MODEL_PATH": str(missing)}).test_client()


@pytest.fixture(scope="session")
def sample_patient() -> dict[str, object]:
    """The committed single-record sample, used by the prediction tests."""
    path = PROJECT_ROOT / "samples" / "sample_patient.json"
    return json.loads(path.read_text(encoding="utf-8"))
