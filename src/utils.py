"""Shared utilities: logging setup, JSON helpers and filesystem helpers."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once, using ``LOG_LEVEL`` when no level is given.

    Repeated calls are cheap no-ops, so every CLI entry point may call this
    without worrying about duplicated handlers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        if level is not None:
            logging.getLogger().setLevel(_coerce_level(level))
        return

    logging.basicConfig(
        level=_coerce_level(level if level is not None else os.getenv("LOG_LEVEL", "INFO")),
        format=_LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _CONFIGURED = True


def _coerce_level(level: str | int) -> int:
    """Translate ``"debug"``/``"INFO"``/``20`` into a stdlib logging level."""
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).strip().upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, making sure logging has been configured."""
    configure_logging()
    return logging.getLogger(name)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and parents) if needed and return it as a :class:`Path`."""
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        raise RuntimeError(f"Could not create directory '{directory}': {exc}") from exc
    return directory


def write_json(path: str | os.PathLike[str], payload: Any, *, indent: int = 2) -> Path:
    """Serialise ``payload`` to ``path`` as UTF-8 JSON, creating parent folders."""
    target = Path(path)
    ensure_dir(target.parent)
    try:
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, default=_json_default)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not write JSON to '{target}': {exc}") from exc
    return target


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read UTF-8 JSON from ``path`` with a readable error on failure."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: '{source}'") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON from '{source}': {exc}") from exc


def _json_default(value: Any) -> Any:
    """Fallback encoder for NumPy scalars, arrays, paths and datetimes."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item") and hasattr(value, "dtype"):  # NumPy scalar
        return value.item()
    if hasattr(value, "tolist"):  # NumPy array / pandas Index
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Environment variable %s=%r is not an integer; using %d", name, raw, default
        )
        return default


def env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Environment variable %s=%r is not a float; using %s", name, raw, default
        )
        return default


def format_duration(seconds: float) -> str:
    """Render a duration in a compact human readable form (``"1m 12.3s"``)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"
