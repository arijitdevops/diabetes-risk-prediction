"""Flask application factory for the diabetes risk demo UI."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from flask import Flask, jsonify, render_template, request

from src import config
from src.utils import configure_logging, get_logger

logger = get_logger(__name__)

DISCLAIMER = (
    "This tool is a machine learning demonstration built on a synthetic public dataset. "
    "It is not a medical device, it has not been clinically validated, and it must not be "
    "used to diagnose, treat or make decisions about any real person. Talk to a qualified "
    "clinician about your health."
)


def create_app(overrides: Mapping[str, Any] | None = None) -> Flask:
    """Build and configure the Flask application.

    Parameters
    ----------
    overrides:
        Optional configuration overrides, primarily for tests (for example
        ``{"TESTING": True, "MODEL_PATH": tmp_model}``).
    """
    configure_logging(os.getenv("LOG_LEVEL", config.LOG_LEVEL))

    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-only-insecure-key"),
        MODEL_PATH=str(config.MODEL_PATH),
        MAX_CONTENT_LENGTH=1 * 1024 * 1024,  # 1 MB is ample for a single record
        DISCLAIMER=DISCLAIMER,
    )
    # Keep probabilities in severity order (Low, Moderate, High) in JSON output.
    app.json.sort_keys = False  # type: ignore[attr-defined]
    if overrides:
        app.config.update(dict(overrides))

    if app.config["SECRET_KEY"] == "dev-only-insecure-key" and not app.config.get("TESTING"):
        logger.warning("SECRET_KEY is unset; using an insecure development key.")

    from app.routes import bp as main_blueprint

    app.register_blueprint(main_blueprint)
    _register_error_handlers(app)

    @app.context_processor
    def inject_globals() -> dict[str, Any]:
        """Expose the disclaimer and class order to every template."""
        return {
            "disclaimer": app.config["DISCLAIMER"],
            "class_order": config.CLASS_ORDER,
        }

    logger.info("Flask app created (model path: %s)", app.config["MODEL_PATH"])
    return app


def _wants_json() -> bool:
    """True when the caller is hitting the JSON API rather than the HTML UI."""
    return request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json"


def _register_error_handlers(app: Flask) -> None:
    """Attach HTML/JSON aware handlers for the errors this app can raise."""

    @app.errorhandler(400)
    def bad_request(error: Exception):  # type: ignore[no-untyped-def]
        message = getattr(error, "description", "Bad request.")
        if _wants_json():
            return jsonify({"status": "error", "error": message}), 400
        return render_template("errors/400.html", message=message), 400

    @app.errorhandler(404)
    def not_found(error: Exception):  # type: ignore[no-untyped-def]
        if _wants_json():
            return jsonify({"status": "error", "error": "Not found."}), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(413)
    def payload_too_large(error: Exception):  # type: ignore[no-untyped-def]
        message = "Request body is too large."
        if _wants_json():
            return jsonify({"status": "error", "error": message}), 413
        return render_template("errors/400.html", message=message), 413

    @app.errorhandler(500)
    def server_error(error: Exception):  # type: ignore[no-untyped-def]
        logger.exception("Unhandled application error: %s", error)
        if _wants_json():
            return jsonify({"status": "error", "error": "Internal server error."}), 500
        return render_template("errors/500.html"), 500
