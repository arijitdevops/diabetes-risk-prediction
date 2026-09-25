"""CLI: fit and cross-validate the candidate models, persist the best pipeline.

Examples
--------
    python -m src.train --model all
    python -m src.train --model rf --cv 3
    python -m src.train --model hgb --keep-leakage   # demo only, see README
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config
from src.data import DataValidationError, load_processed, split_features_target
from src.features import MODEL_CHOICES, PARAM_GRIDS, build_pipeline, get_feature_names
from src.utils import (
    configure_logging,
    ensure_dir,
    format_duration,
    get_logger,
    utc_timestamp,
    write_json,
)

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Outcome of a single model's grid search."""

    model_name: str
    pipeline: Pipeline
    best_params: dict[str, object]
    cv_score: float
    cv_std: float
    fit_seconds: float
    numeric_columns: list[str]
    categorical_columns: list[str]


def build_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Train, cross-validate and persist the diabetes risk pipeline.",
    )
    parser.add_argument(
        "--model",
        choices=(*MODEL_CHOICES, "all"),
        default="all",
        help="Candidate estimator, or 'all' to grid-search each and keep the best.",
    )
    parser.add_argument(
        "--keep-leakage",
        action="store_true",
        help=(
            "Keep Diabetes_Risk_Score, AI_Health_Recommendation, Doctor_Consultation_Needed "
            "and Patient_ID as features. Demonstration only - the resulting score is inflated "
            "because those columns encode the label."
        ),
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=config.TRAIN_CSV_PATH,
        help=f"Processed training split (default: {config.TRAIN_CSV_PATH})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.MODEL_PATH,
        help=f"Where to write the fitted pipeline (default: {config.MODEL_PATH})",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=config.CV_FOLDS,
        help=f"Number of stratified CV folds (default: {config.CV_FOLDS})",
    )
    parser.add_argument(
        "--scoring",
        default=config.SCORING,
        help=f"scikit-learn scoring name used for selection (default: {config.SCORING})",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for the grid search (default: -1, all cores)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optionally subsample the training split, useful for a quick smoke run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help=f"Random seed (default: {config.RANDOM_SEED})",
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        help="Logging level (default: %(default)s)",
    )
    return parser


def run_grid_search(
    features: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    *,
    keep_leakage: bool,
    cv_folds: int,
    scoring: str,
    n_jobs: int,
    seed: int,
) -> TrainingResult:
    """Grid-search one estimator and return the refitted best pipeline."""
    pipeline, numeric_columns, categorical_columns = build_pipeline(
        features, model_name, keep_leakage=keep_leakage, seed=seed
    )
    grid = PARAM_GRIDS.get(model_name, {})
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring=scoring,
        cv=splitter,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )

    logger.info(
        "Grid-searching %s over %d parameter combination(s) with %d-fold CV",
        model_name,
        max(1, _grid_size(grid)),
        cv_folds,
    )
    started = time.perf_counter()
    search.fit(features, labels)
    elapsed = time.perf_counter() - started

    index = int(search.best_index_)
    std = float(search.cv_results_["std_test_score"][index])
    logger.info(
        "%s: best %s = %.4f (+/- %.4f) in %s | params=%s",
        model_name,
        scoring,
        search.best_score_,
        std,
        format_duration(elapsed),
        search.best_params_ or "{defaults}",
    )

    return TrainingResult(
        model_name=model_name,
        pipeline=search.best_estimator_,
        best_params={key: _jsonable(value) for key, value in search.best_params_.items()},
        cv_score=float(search.best_score_),
        cv_std=std,
        fit_seconds=elapsed,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def _grid_size(grid: dict[str, list[object]]) -> int:
    """Number of parameter combinations in a grid dictionary."""
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _jsonable(value: object) -> object:
    """Make a hyper-parameter value safe for JSON (``None``, numbers, strings)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def build_metadata(
    result: TrainingResult,
    *,
    keep_leakage: bool,
    scoring: str,
    cv_folds: int,
    n_train_rows: int,
    class_counts: dict[str, int],
    candidates: dict[str, float],
) -> dict[str, object]:
    """Assemble the metadata JSON written next to the model."""
    return {
        "model_name": result.model_name,
        "estimator_class": type(result.pipeline.named_steps["estimator"]).__name__,
        "best_params": result.best_params,
        "cv_scoring": scoring,
        "cv_folds": cv_folds,
        "cv_score": round(result.cv_score, 6),
        "cv_score_std": round(result.cv_std, 6),
        "candidate_cv_scores": {name: round(score, 6) for name, score in candidates.items()},
        "fit_seconds": round(result.fit_seconds, 2),
        "trained_at_utc": utc_timestamp(),
        "sklearn_version": sklearn.__version__,
        "class_order": list(config.CLASS_ORDER),
        "classes_": [str(label) for label in result.pipeline.named_steps["estimator"].classes_],
        "train_class_counts": class_counts,
        "n_train_rows": n_train_rows,
        "keep_leakage": keep_leakage,
        "leakage_columns": list(config.LEAKAGE_COLUMNS),
        "numeric_columns": result.numeric_columns,
        "categorical_columns": result.categorical_columns,
        "input_columns": [*result.numeric_columns, *result.categorical_columns],
        "transformed_feature_names": get_feature_names(result.pipeline),
    }


def train(
    train_csv: Path,
    model_choice: str,
    model_path: Path,
    *,
    keep_leakage: bool,
    cv_folds: int,
    scoring: str,
    n_jobs: int,
    seed: int,
    max_rows: int | None = None,
) -> dict[str, object]:
    """Fit the requested model(s), persist the winner and return its metadata."""
    frame = load_processed("train") if train_csv == config.TRAIN_CSV_PATH else _read_csv(train_csv)

    if max_rows is not None and max_rows < len(frame):
        logger.warning("Subsampling training data to %d rows (--max-rows)", max_rows)
        frame = frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    features, labels = split_features_target(frame)
    class_counts = {str(k): int(v) for k, v in labels.value_counts().items()}
    logger.info("Training on %d rows | class counts: %s", len(frame), class_counts)

    if labels.nunique() < 2:
        raise DataValidationError("Training split contains a single class; cannot fit a model.")

    names = list(MODEL_CHOICES) if model_choice == "all" else [model_choice]
    results: list[TrainingResult] = []
    for name in names:
        results.append(
            run_grid_search(
                features,
                labels,
                name,
                keep_leakage=keep_leakage,
                cv_folds=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                seed=seed,
            )
        )

    best = max(results, key=lambda item: item.cv_score)
    candidates = {item.model_name: item.cv_score for item in results}
    logger.info("Selected %s with %s=%.4f", best.model_name, scoring, best.cv_score)

    ensure_dir(model_path.parent)
    try:
        joblib.dump(best.pipeline, model_path, compress=3)
    except OSError as exc:
        raise RuntimeError(f"Could not write the model to '{model_path}': {exc}") from exc
    logger.info("Wrote fitted pipeline to %s", model_path)

    metadata = build_metadata(
        best,
        keep_leakage=keep_leakage,
        scoring=scoring,
        cv_folds=cv_folds,
        n_train_rows=len(frame),
        class_counts=class_counts,
        candidates=candidates,
    )
    metadata_path = model_path.with_name("metadata.json")
    write_json(metadata_path, metadata)
    logger.info("Wrote model metadata to %s", metadata_path)

    if keep_leakage:
        logger.warning(
            "This model was trained WITH leakage columns. Its CV score (%.4f) reflects "
            "the label being visible in the inputs, not clinical signal.",
            best.cv_score,
        )
    return metadata


def _read_csv(path: Path) -> pd.DataFrame:
    """Read an arbitrary processed CSV with a helpful error message."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Training split not found at '{path}'. Run 'python -m src.prepare_data' first."
        )
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not read '{path}': {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    if args.cv < 2:
        logger.error("--cv must be at least 2, got %d", args.cv)
        return 2

    try:
        train(
            train_csv=args.train_csv,
            model_choice=args.model,
            model_path=args.model_path,
            keep_leakage=args.keep_leakage,
            cv_folds=args.cv,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            seed=args.seed,
            max_rows=args.max_rows,
        )
    except (FileNotFoundError, DataValidationError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
