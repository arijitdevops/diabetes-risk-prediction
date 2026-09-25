"""Tests for the Flask UI and JSON API, including the model-missing paths."""

from __future__ import annotations

from app.forms import default_form_values, validate_payload
from src import config


def test_health_reports_ready_when_model_exists(client_with_model) -> None:
    """``/api/health`` is 200 and advertises the loaded model."""
    response = client_with_model.get("/api/health")
    assert response.status_code == 200

    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_available"] is True
    assert body["class_order"] == list(config.CLASS_ORDER)
    assert body["model"]["keep_leakage"] is False


def test_health_reports_degraded_without_model(client_without_model) -> None:
    """Without an artifact the service degrades instead of raising."""
    response = client_without_model.get("/api/health")
    assert response.status_code == 503

    body = response.get_json()
    assert body["status"] == "degraded"
    assert body["model_available"] is False
    assert "python -m src.train" in body["message"]


def test_index_renders_form(client_with_model) -> None:
    """The form page lists every section and field."""
    response = client_with_model.get("/")
    assert response.status_code == 200

    html = response.get_data(as_text=True)
    for section in config.SECTIONS:
        assert section in html
    for column in config.FEATURE_COLUMNS:
        assert f'name="{column}"' in html
    assert "not a medical device" in html


def test_index_warns_when_model_missing(client_without_model) -> None:
    """The banner tells the reader exactly which command to run."""
    response = client_without_model.get("/")
    assert response.status_code == 200

    html = response.get_data(as_text=True)
    assert "No trained model found" in html
    assert "python -m src.train" in html


def test_form_prediction_renders_probability_bars(client_with_model) -> None:
    """A valid submission produces a result page with one bar per class."""
    response = client_with_model.post("/predict", data=default_form_values())
    assert response.status_code == 200

    html = response.get_data(as_text=True)
    assert "Predicted risk band" in html
    for label in config.CLASS_ORDER:
        assert label in html
    assert "progress-bar" in html


def test_form_rejects_out_of_range_values(client_with_model) -> None:
    """An impossible HbA1c is reported inline and the form comes back as 400."""
    payload = default_form_values()
    payload["HbA1c"] = 99

    response = client_with_model.post("/predict", data=payload)
    assert response.status_code == 400
    assert "HbA1c must be between" in response.get_data(as_text=True)


def test_form_prediction_without_model_returns_503(client_without_model) -> None:
    """Submitting with no artifact yields the guidance page, not a traceback."""
    response = client_without_model.post("/predict", data=default_form_values())
    assert response.status_code == 503

    html = response.get_data(as_text=True)
    assert "No trained model is available" in html
    assert "python -m src.train" in html


def test_api_predict_returns_probabilities(client_with_model, sample_patient) -> None:
    """``POST /api/predict`` returns the band, per-class probabilities and model info."""
    response = client_with_model.post("/api/predict", json=sample_patient)
    assert response.status_code == 200

    body = response.get_json()
    assert body["status"] == "ok"
    assert body["prediction"] in config.CLASS_ORDER
    assert set(body["probabilities"]) == set(config.CLASS_ORDER)
    assert round(sum(body["probabilities"].values()), 6) == 1.0
    assert "not a medical device" in body["disclaimer"]


def test_api_predict_validates_input(client_with_model, sample_patient) -> None:
    """Bad values come back as a 400 with a per-field explanation."""
    payload = dict(sample_patient)
    payload["Gender"] = "Martian"
    payload["Age"] = "old"

    response = client_with_model.post("/api/predict", json=payload)
    assert response.status_code == 400

    body = response.get_json()
    assert body["status"] == "error"
    assert set(body["fields"]) == {"Gender", "Age"}


def test_api_predict_rejects_non_object_body(client_with_model) -> None:
    """A JSON array is not a patient record."""
    response = client_with_model.post("/api/predict", json=[1, 2, 3])
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


def test_api_predict_without_model_returns_503(client_without_model, sample_patient) -> None:
    """The API mirrors the UI: 503 and a clear instruction, never a stack trace."""
    response = client_without_model.post("/api/predict", json=sample_patient)
    assert response.status_code == 503

    body = response.get_json()
    assert "python -m src.train" in body["error"]


def test_unknown_route_returns_json_for_api_paths(client_with_model) -> None:
    """404s under /api/ stay JSON so clients never have to parse HTML."""
    response = client_with_model.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["status"] == "error"


def test_cross_field_validation_catches_impossible_blood_pressure() -> None:
    """Diastolic pressure above systolic is rejected by the shared validator."""
    payload = default_form_values()
    payload["Blood_Pressure_Systolic"] = 110
    payload["Blood_Pressure_Diastolic"] = 120

    _record, errors = validate_payload(payload)
    assert "Blood_Pressure_Diastolic" in errors
