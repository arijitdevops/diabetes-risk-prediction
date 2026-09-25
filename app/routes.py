"""HTTP routes: the patient form, the result page and the JSON API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, render_template, request

from app.forms import default_form_values, echo_values, validate_payload
from src import config
from src.predict import (
    ModelNotAvailableError,
    global_feature_importance,
    load_model,
    predict_records,
)
from src.utils import get_logger

logger = get_logger(__name__)

bp = Blueprint("main", __name__)

#: Bootstrap contextual colour per risk band, used for badges and bars.
BAND_STYLES: dict[str, str] = {"Low": "success", "Moderate": "warning", "High": "danger"}

BAND_SUMMARY: dict[str, str] = {
    "Low": "The model places this profile in its lowest risk band.",
    "Moderate": "The model places this profile in its middle risk band.",
    "High": "The model places this profile in its highest risk band.",
}


def _model_path() -> Path:
    """Resolve the model path from app config, falling back to the package default."""
    return Path(current_app.config.get("MODEL_PATH", config.MODEL_PATH))


def get_model() -> tuple[Any, dict[str, Any]]:
    """Load the fitted pipeline once per process and cache it on the app.

    The cache key includes the file's modification time, so retraining while the
    development server is running is picked up on the next request.
    """
    path = _model_path()
    try:
        stamp = path.stat().st_mtime_ns
    except OSError as exc:
        raise ModelNotAvailableError(
            f"Model not available. Train one first: python -m src.train (expected at '{path}')"
        ) from exc

    cache = current_app.extensions.setdefault("diabetes_model", {})
    if cache.get("key") == (str(path), stamp):
        return cache["pipeline"], cache["metadata"]

    pipeline, metadata = load_model(path)
    cache.update({"key": (str(path), stamp), "pipeline": pipeline, "metadata": metadata})
    return pipeline, metadata


def _model_status() -> tuple[bool, str | None]:
    """Return ``(available, message)`` without raising, for banners and health."""
    try:
        get_model()
    except ModelNotAvailableError as exc:
        return False, str(exc)
    return True, None


@bp.get("/")
def index() -> str:
    """Render the patient form, pre-filled with plausible defaults."""
    available, message = _model_status()
    return render_template(
        "index.html",
        sections=config.fields_by_section(),
        values=default_form_values(),
        errors={},
        model_available=available,
        model_message=message,
    )


@bp.post("/predict")
def predict() -> tuple[str, int] | str:
    """Validate the submitted form and render the prediction, or re-render errors."""
    payload = request.form.to_dict()
    record, errors = validate_payload(payload)

    if errors:
        logger.info("Rejected form submission with %d validation error(s)", len(errors))
        return (
            render_template(
                "index.html",
                sections=config.fields_by_section(),
                values=echo_values(payload),
                errors=errors,
                model_available=_model_status()[0],
                model_message=_model_status()[1],
            ),
            400,
        )

    try:
        pipeline, metadata = get_model()
        result = predict_records(pipeline, [record])[0]
        drivers = global_feature_importance(pipeline, metadata, top_k=8)
    except ModelNotAvailableError as exc:
        return render_template("errors/model_missing.html", message=str(exc)), 503
    except (ValueError, RuntimeError) as exc:
        logger.exception("Prediction failed")
        return render_template("errors/500.html", message=str(exc)), 500

    band = result["prediction"]
    probabilities = result.get("probabilities", {})
    return render_template(
        "result.html",
        record=record,
        prediction=band,
        band_style=BAND_STYLES.get(band, "secondary"),
        band_summary=BAND_SUMMARY.get(band, ""),
        probabilities=probabilities,
        band_styles=BAND_STYLES,
        confidence=result.get("confidence"),
        drivers=drivers,
        metadata=metadata,
        sections=config.fields_by_section(),
    )


@bp.post("/api/predict")
def api_predict():  # type: ignore[no-untyped-def]
    """Score a single patient record supplied as JSON.

    Returns 400 on validation failure, 503 when no model has been trained yet.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return (
            jsonify({"status": "error", "error": "Request body must be a JSON object."}),
            400,
        )

    record, errors = validate_payload(payload)
    if errors:
        return jsonify({"status": "error", "error": "Validation failed.", "fields": errors}), 400

    try:
        pipeline, metadata = get_model()
        result = predict_records(pipeline, [record])[0]
    except ModelNotAvailableError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 503
    except (ValueError, RuntimeError) as exc:
        logger.exception("API prediction failed")
        return jsonify({"status": "error", "error": str(exc)}), 500

    return jsonify(
        {
            "status": "ok",
            "prediction": result["prediction"],
            "probabilities": result.get("probabilities", {}),
            "confidence": result.get("confidence"),
            "model": {
                "name": metadata.get("model_name"),
                "trained_at_utc": metadata.get("trained_at_utc"),
                "sklearn_version": metadata.get("sklearn_version"),
                "keep_leakage": metadata.get("keep_leakage", False),
            },
            "disclaimer": current_app.config["DISCLAIMER"],
        }
    )


@bp.get("/api/health")
def api_health():  # type: ignore[no-untyped-def]
    """Report whether the service can serve predictions."""
    available, message = _model_status()
    metadata: dict[str, Any] = {}
    if available:
        _pipeline, metadata = get_model()

    body = {
        "status": "ok" if available else "degraded",
        "model_available": available,
        "model_path": str(_model_path()),
        "class_order": list(config.CLASS_ORDER),
        "model": {
            "name": metadata.get("model_name"),
            "trained_at_utc": metadata.get("trained_at_utc"),
            "cv_score": metadata.get("cv_score"),
            "keep_leakage": metadata.get("keep_leakage", False),
        },
    }
    if message:
        body["message"] = message
    return jsonify(body), 200 if available else 503
