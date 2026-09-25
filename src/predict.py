"""Inference helpers and CLI.

Used by both the command line (``python -m src.predict --input ...``) and the
Flask app, so a prediction is produced exactly the same way in either case.

Examples
--------
    python -m src.predict --input samples/sample_patient.json
    cat samples/sample_patient.json | python -m src.predict --stdin
    python -m src.predict --input samples/sample_patients.csv --output preds.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.features import prettify_feature_name
from src.utils import configure_logging, get_logger, read_json, write_json

logger = get_logger(__name__)

TRAIN_HINT = "Model not available. Train one first: python -m src.train"


class ModelNotAvailableError(RuntimeError):
    """Raised when the fitted pipeline cannot be loaded from disk."""


def load_model(model_path: Path | str | None = None) -> tuple[Pipeline, dict[str, Any]]:
    """Load the fitted pipeline and its metadata sidecar.

    Metadata is optional: a model trained by an older run still works, it just
    reports fewer details.

    Raises
    ------
    ModelNotAvailableError
        If the joblib file is missing or cannot be deserialised.
    """
    path = Path(model_path) if model_path is not None else config.MODEL_PATH
    if not path.is_file():
        raise ModelNotAvailableError(f"{TRAIN_HINT} (expected at '{path}')")

    try:
        pipeline = joblib.load(path)
    except (OSError, EOFError, ValueError, ModuleNotFoundError) as exc:
        raise ModelNotAvailableError(f"Could not load the model from '{path}': {exc}") from exc

    if not hasattr(pipeline, "predict"):
        raise ModelNotAvailableError(f"Artifact at '{path}' is not a fitted scikit-learn pipeline.")

    metadata: dict[str, Any] = {}
    metadata_path = path.with_name("metadata.json")
    if metadata_path.is_file():
        try:
            loaded = read_json(metadata_path)
            if isinstance(loaded, dict):
                metadata = loaded
        except RuntimeError as exc:
            logger.warning("Ignoring unreadable metadata at %s: %s", metadata_path, exc)

    logger.info("Loaded model from %s (%s)", path, metadata.get("model_name", "unknown estimator"))
    return pipeline, metadata


def records_to_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Turn a sequence of record dictionaries into a model-ready DataFrame."""
    if not records:
        raise ValueError("No records supplied.")

    frame = pd.DataFrame(list(records))
    for column in config.NUMERIC_OVERRIDES:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in config.CATEGORICAL_OVERRIDES:
        if column in frame.columns:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    return frame


def align_to_model(frame: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Add any missing input columns as NaN and drop unknown ones.

    Missing values are then filled by the imputers inside the pipeline, so a
    partially filled record still produces a prediction. Which columns were
    missing is logged, because silently imputing half a record is worth knowing.
    """
    from src.features import input_columns  # local import keeps the module import graph flat

    expected = input_columns(pipeline)
    aligned = frame.copy()

    missing = [column for column in expected if column not in aligned.columns]
    if missing:
        logger.warning(
            "Input is missing %d column(s), imputing: %s", len(missing), ", ".join(missing)
        )
        for column in missing:
            aligned[column] = np.nan

    unused = [column for column in aligned.columns if column not in expected]
    if unused:
        logger.info("Ignoring %d non-feature column(s): %s", len(unused), ", ".join(unused))

    return aligned[expected]


def predict_frame(pipeline: Pipeline, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Predict a risk band and per-class probabilities for every row."""
    aligned = align_to_model(frame, pipeline)
    try:
        predictions = pipeline.predict(aligned)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"The model could not score these inputs: {exc}") from exc

    classes = [str(label) for label in getattr(pipeline, "classes_", config.CLASS_ORDER)]
    probabilities: np.ndarray | None = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(aligned)
        except (ValueError, AttributeError) as exc:  # pragma: no cover - estimator dependent
            logger.warning("Estimator does not provide calibrated probabilities: %s", exc)

    results: list[dict[str, Any]] = []
    for position, label in enumerate(predictions):
        row: dict[str, Any] = {"prediction": str(label)}
        if probabilities is not None:
            per_class = {
                name: round(float(probabilities[position][index]), 6)
                for index, name in enumerate(classes)
            }
            ordered = [name for name in config.CLASS_ORDER if name in per_class]
            ordered += [name for name in per_class if name not in ordered]
            row["probabilities"] = {name: per_class[name] for name in ordered}
            row["confidence"] = max(per_class.values())
        results.append(row)
    return results


def predict_records(
    pipeline: Pipeline,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convenience wrapper: dictionaries in, prediction dictionaries out."""
    return predict_frame(pipeline, records_to_frame(records))


def global_feature_importance(
    pipeline: Pipeline,
    metadata: Mapping[str, Any] | None = None,
    top_k: int = 8,
    importance_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the model's most influential features, best source first.

    Preference order:

    1. ``reports/permutation_importance.json`` if ``src.evaluate`` has been run,
       because permutation importance is measured on held-out data;
    2. the estimator's own ``feature_importances_`` (tree models);
    3. the mean absolute logistic-regression coefficient per feature.

    These are model-level drivers, not a per-patient attribution - the UI labels
    them as such.
    """
    path = importance_path if importance_path is not None else config.IMPORTANCE_PATH
    if path.is_file():
        try:
            payload = read_json(path)
            entries = payload.get("importances", []) if isinstance(payload, dict) else []
            if entries:
                return [
                    {
                        "feature": prettify_feature_name(str(entry.get("feature", ""))),
                        "score": round(float(entry.get("importance_mean", 0.0)), 6),
                        "source": "permutation importance (held-out test set)",
                    }
                    for entry in entries[:top_k]
                ]
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Could not read permutation importance from %s: %s", path, exc)

    estimator = pipeline.named_steps.get("estimator") if hasattr(pipeline, "named_steps") else None
    if estimator is None:
        return []

    names = list((metadata or {}).get("transformed_feature_names") or [])
    if not names:
        from src.features import get_feature_names

        names = get_feature_names(pipeline)

    weights: np.ndarray | None = None
    source = ""
    if hasattr(estimator, "feature_importances_"):
        weights = np.asarray(estimator.feature_importances_, dtype=float)
        source = "impurity-based feature importance (training data)"
    elif hasattr(estimator, "coef_"):
        weights = np.abs(np.asarray(estimator.coef_, dtype=float)).mean(axis=0)
        source = "mean absolute logistic regression coefficient"

    if weights is None or not names or len(names) != len(weights):
        return []

    order = np.argsort(weights)[::-1][:top_k]
    return [
        {
            "feature": prettify_feature_name(names[index]),
            "score": round(float(weights[index]), 6),
            "source": source,
        }
        for index in order
    ]


def _load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """Read the CLI input into a DataFrame and report which mode was used."""
    if args.stdin:
        raw = sys.stdin.read().strip()
        if not raw:
            raise ValueError("No JSON received on stdin.")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"stdin is not valid JSON: {exc}") from exc
        return records_to_frame(_as_records(payload)), "stdin"

    path: Path = args.input
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: '{path}'")

    if path.suffix.lower() == ".json":
        return records_to_frame(_as_records(read_json(path))), str(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"Could not read CSV '{path}': {exc}") from exc
        if frame.empty:
            raise ValueError(f"CSV '{path}' contains no rows.")
        return frame, str(path)
    raise ValueError(f"Unsupported input extension '{path.suffix}'. Use .json or .csv.")


def _as_records(payload: Any) -> list[dict[str, Any]]:
    """Accept either a single JSON object or a list of objects."""
    if isinstance(payload, Mapping):
        return [dict(payload)]
    if isinstance(payload, list) and all(isinstance(item, Mapping) for item in payload):
        return [dict(item) for item in payload]
    raise ValueError("JSON input must be an object or a list of objects.")


def build_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Score one patient record or a CSV batch with the trained pipeline.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Path to a .json record or a .csv batch.")
    source.add_argument("--stdin", action="store_true", help="Read a JSON record from stdin.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.MODEL_PATH,
        help=f"Fitted pipeline to use (default: {config.MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results to this .json or .csv file instead of stdout.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (default: %(default)s, so stdout stays machine readable)",
    )
    return parser


def _write_output(results: list[dict[str, Any]], frame: pd.DataFrame, output: Path) -> None:
    """Persist predictions as JSON or CSV, matching the requested extension."""
    if output.suffix.lower() == ".json":
        write_json(output, results)
        return
    if output.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported output extension '{output.suffix}'. Use .json or .csv.")

    flat = pd.DataFrame(
        [
            {
                "prediction": item["prediction"],
                **{
                    f"probability_{name}": value
                    for name, value in item.get("probabilities", {}).items()
                },
            }
            for item in results
        ]
    )
    if "Patient_ID" in frame.columns:
        flat.insert(0, "Patient_ID", frame["Patient_ID"].to_numpy())
    try:
        flat.to_csv(output, index=False)
    except OSError as exc:
        raise RuntimeError(f"Could not write '{output}': {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        frame, origin = _load_input(args)
        pipeline, _metadata = load_model(args.model_path)
        results = predict_frame(pipeline, frame)
    except (ModelNotAvailableError, FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    logger.info("Scored %d record(s) from %s", len(results), origin)
    if args.output is not None:
        try:
            _write_output(results, frame, args.output)
        except (ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {len(results)} prediction(s) to {args.output}")
    else:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
