"""Development entry point for the Flask application.

    python run.py                 # http://127.0.0.1:5000
    FLASK_PORT=8000 python run.py

For anything beyond local development use a WSGI server, for example:

    waitress-serve --port 8000 "app:create_app()"
"""

from __future__ import annotations

import os

from app import create_app
from src.utils import env_int, get_logger

logger = get_logger(__name__)

app = create_app()


def main() -> None:
    """Start the built-in development server."""
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = env_int("FLASK_PORT", 5000)
    debug = os.getenv("FLASK_ENV", "development").lower() == "development"

    logger.info("Starting development server on http://%s:%d (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
