"""CLI: score the held-out split, write metrics, figures and feature importance.

Outputs
-------
reports/metrics.json                  accuracy, macro/weighted P/R/F1, per-class
                                      report and one-vs-rest ROC-AUC
reports/permutation_importance.json   top-N permutation importances
reports/figures/confusion_matrix.png
reports/figures/roc_curves.png
reports/figures/permutation_importance.png

Example
-------
    python -m src.evaluate --n-repeats 5 --top-k 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402

from src import config  # noqa: E402
from src.data import DataValidationError, split_features_target  # noqa: E402
from src.features import prettify_feature_name  # noqa: E402
from src.predict import ModelNotAvailableError, align_to_model, load_model  # noqa: E402
from src.utils import (  # noqa: E402
    configure_logging,
    ensure_dir,
    get_logger,
    utc_timestamp,
    write_json,
)

logger = get_logger(__name__)

FIGURE_DPI = 140


def build_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evaluate the trained pipeline on the held-out test split.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.MODEL_PATH,
        help=f"Fitted pipeline (default: {config.MODEL_PATH})",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=config.TEST_CSV_PATH,
        help=f"Processed test split (default: {config.TEST_CSV_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.REPORTS_DIR,
        help=f"Where metrics and figures are written (default: {config.REPORTS_DIR})",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Permutation importance repeats (default: %(default)s)",
    )
    parser.add_argument(
        "--importance-sample",
        type=int,
        default=2000,
        help="Rows sampled for permutation importance (default: %(default)s, 0 = all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many features to keep in the importance report (default: %(default)s)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Compute metrics only, skip matplotlib output.",
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


def present_classes(pipeline: Pipeline) -> list[str]:
    """Class labels in the canonical severity order, restricted to what the model knows."""
    known = [str(label) for label in getattr(pipeline, "classes_", ())]
    ordered = [label for label in config.CLASS_ORDER if label in known]
    ordered += [label for label in known if label not in ordered]
    return ordered or list(config.CLASS_ORDER)


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    labels: list[str],
) -> dict[str, Any]:
    """Build the metrics payload written to ``reports/metrics.json``."""
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    metrics: dict[str, Any] = {
        "evaluated_at_utc": utc_timestamp(),
        "n_test_rows": int(len(y_true)),
        "class_order": labels,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(macro[0]), 6),
        "recall_macro": round(float(macro[1]), 6),
        "f1_macro": round(float(macro[2]), 6),
        "precision_weighted": round(float(weighted[0]), 6),
        "recall_weighted": round(float(weighted[1]), 6),
        "f1_weighted": round(float(weighted[2]), 6),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": {
            "labels": labels,
            "rows_are_true_labels": True,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }

    if y_proba is not None:
        metrics["roc_auc"] = _roc_auc_block(y_true, y_proba, labels)
    else:
        metrics["roc_auc"] = None
        logger.warning("Estimator exposes no probabilities; ROC-AUC was skipped.")
    return metrics


def _display_path(path: Path) -> str:
    """Return ``path`` relative to the project root when possible (portable JSON)."""
    try:
        return Path(path).resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _roc_auc_block(y_true: pd.Series, y_proba: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """One-vs-rest ROC-AUC, macro/weighted plus per class."""
    block: dict[str, Any] = {"strategy": "one-vs-rest"}
    # roc_auc_score wants ``labels`` sorted and the probability columns in the
    # same order, so reorder from the severity order used everywhere else.
    order = sorted(range(len(labels)), key=lambda index: labels[index])
    sorted_labels = [labels[index] for index in order]
    sorted_proba = y_proba[:, order]
    try:
        for average in ("macro", "weighted"):
            score = roc_auc_score(
                y_true, sorted_proba, multi_class="ovr", average=average, labels=sorted_labels
            )
            block[average] = round(float(score), 6)
    except ValueError as exc:
        logger.warning("Aggregate ROC-AUC unavailable: %s", exc)
        block["macro"] = None
        block["weighted"] = None

    binarized = label_binarize(y_true, classes=labels)
    per_class: dict[str, float | None] = {}
    for index, label in enumerate(labels):
        column = binarized[:, index]
        if column.sum() == 0 or column.sum() == len(column):
            per_class[label] = None
            continue
        per_class[label] = round(float(roc_auc_score(column, y_proba[:, index])), 6)
    block["per_class"] = per_class
    return block


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], output: Path) -> Path:
    """Save an annotated, row-normalised confusion matrix."""
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)

    figure, axes = plt.subplots(figsize=(6.0, 5.2))
    image = axes.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    axes.set_xticks(range(len(labels)), labels)
    axes.set_yticks(range(len(labels)), labels)
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("True label")
    axes.set_title("Confusion matrix (row-normalised)")

    for row in range(len(labels)):
        for column in range(len(labels)):
            share = normalised[row, column]
            axes.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{share:.1%}",
                ha="center",
                va="center",
                color="white" if share > 0.5 else "black",
                fontsize=9,
            )

    figure.colorbar(image, ax=axes, shrink=0.85, label="Share of true class")
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI)
    plt.close(figure)
    return output


def plot_roc_curves(
    y_true: pd.Series,
    y_proba: np.ndarray,
    labels: list[str],
    output: Path,
) -> Path | None:
    """Save one-vs-rest ROC curves, one line per risk band."""
    binarized = label_binarize(y_true, classes=labels)
    figure, axes = plt.subplots(figsize=(6.2, 5.2))
    drawn = 0

    for index, label in enumerate(labels):
        column = binarized[:, index]
        if column.sum() == 0 or column.sum() == len(column):
            logger.warning("Skipping ROC curve for '%s': only one outcome present.", label)
            continue
        false_positive, true_positive, _ = roc_curve(column, y_proba[:, index])
        area = roc_auc_score(column, y_proba[:, index])
        axes.plot(false_positive, true_positive, linewidth=1.8, label=f"{label} (AUC={area:.3f})")
        drawn += 1

    if drawn == 0:
        plt.close(figure)
        return None

    axes.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="grey", label="Chance")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title("ROC curves (one-vs-rest)")
    axes.legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI)
    plt.close(figure)
    return output


def plot_importances(entries: list[dict[str, Any]], output: Path) -> Path | None:
    """Save a horizontal bar chart of permutation importances."""
    if not entries:
        return None

    names = [prettify_feature_name(str(entry["feature"])) for entry in entries][::-1]
    means = [float(entry["importance_mean"]) for entry in entries][::-1]
    errors = [float(entry["importance_std"]) for entry in entries][::-1]

    height = max(4.0, 0.34 * len(names) + 1.2)
    figure, axes = plt.subplots(figsize=(8.2, height))
    axes.barh(names, means, xerr=errors, color="#2a6f97", ecolor="#94a3b8", capsize=2.5)
    axes.set_xlabel(f"Mean drop in {config.SCORING} when the feature is shuffled")
    axes.set_title(f"Permutation importance (top {len(names)})")
    axes.grid(axis="x", linestyle=":", alpha=0.5)
    figure.tight_layout()
    figure.savefig(output, dpi=FIGURE_DPI)
    plt.close(figure)
    return output


def compute_permutation_importance(
    pipeline: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    n_repeats: int,
    sample_size: int,
    top_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Permutation importance over the raw input columns of the fitted pipeline."""
    frame = features
    target = labels
    if 0 < sample_size < len(frame):
        logger.info("Sampling %d rows for permutation importance", sample_size)
        frame = frame.sample(n=sample_size, random_state=seed)
        target = labels.loc[frame.index]

    try:
        result = permutation_importance(
            pipeline,
            frame,
            target,
            scoring=config.SCORING,
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=1,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("Permutation importance failed: %s", exc)
        return []

    entries = [
        {
            "feature": str(column),
            "importance_mean": round(float(result.importances_mean[index]), 6),
            "importance_std": round(float(result.importances_std[index]), 6),
        }
        for index, column in enumerate(frame.columns)
    ]
    entries.sort(key=lambda item: item["importance_mean"], reverse=True)
    return entries[:top_k]


def log_summary(metrics: dict[str, Any]) -> None:
    """Log the headline numbers so a CLI run is readable without opening the JSON."""
    logger.info("Accuracy            %.4f", metrics["accuracy"])
    logger.info("Balanced accuracy   %.4f", metrics["balanced_accuracy"])
    logger.info("F1 (macro)          %.4f", metrics["f1_macro"])
    logger.info("F1 (weighted)       %.4f", metrics["f1_weighted"])
    auc = (metrics.get("roc_auc") or {}).get("macro")
    if auc is not None:
        logger.info("ROC-AUC (ovr macro) %.4f", auc)


def evaluate(
    model_path: Path,
    test_csv: Path,
    output_dir: Path,
    *,
    n_repeats: int,
    importance_sample: int,
    top_k: int,
    make_figures: bool,
    seed: int,
) -> dict[str, Any]:
    """Run the full evaluation and return the metrics payload."""
    pipeline, metadata = load_model(model_path)

    if not test_csv.is_file():
        raise FileNotFoundError(
            f"Test split not found at '{test_csv}'. Run 'python -m src.prepare_data' first."
        )
    try:
        frame = pd.read_csv(test_csv)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"Could not read '{test_csv}': {exc}") from exc

    features, y_true = split_features_target(frame)
    aligned = align_to_model(features, pipeline)
    labels = present_classes(pipeline)

    y_pred = pipeline.predict(aligned)
    y_proba = pipeline.predict_proba(aligned) if hasattr(pipeline, "predict_proba") else None
    if y_proba is not None:
        # Reorder probability columns to the canonical class order.
        known = [str(label) for label in pipeline.classes_]
        y_proba = y_proba[:, [known.index(label) for label in labels]]

    metrics = compute_metrics(y_true, y_pred, y_proba, labels)
    metrics["model"] = {
        "path": _display_path(model_path),
        "model_name": metadata.get("model_name"),
        "trained_at_utc": metadata.get("trained_at_utc"),
        "cv_score": metadata.get("cv_score"),
        "keep_leakage": metadata.get("keep_leakage", False),
        "sklearn_version": metadata.get("sklearn_version"),
    }
    if metadata.get("keep_leakage"):
        metrics["warning"] = (
            "This model was trained with leakage columns (--keep-leakage). "
            "The metrics below are inflated and must not be reported as performance."
        )
        logger.warning("%s", metrics["warning"])

    ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    importances = compute_permutation_importance(
        pipeline,
        aligned,
        y_true,
        n_repeats=n_repeats,
        sample_size=importance_sample,
        top_k=top_k,
        seed=seed,
    )
    write_json(
        output_dir / "permutation_importance.json",
        {
            "computed_at_utc": utc_timestamp(),
            "scoring": config.SCORING,
            "n_repeats": n_repeats,
            "top_k": top_k,
            "importances": importances,
        },
    )

    written: list[str] = []
    if make_figures:
        matrix = np.asarray(metrics["confusion_matrix"]["matrix"], dtype=int)
        confusion_path = plot_confusion_matrix(matrix, labels, figures_dir / "confusion_matrix.png")
        written.append(_display_path(confusion_path))
        if y_proba is not None:
            roc_path = plot_roc_curves(y_true, y_proba, labels, figures_dir / "roc_curves.png")
            if roc_path is not None:
                written.append(_display_path(roc_path))
        importance_path = plot_importances(importances, figures_dir / "permutation_importance.png")
        if importance_path is not None:
            written.append(_display_path(importance_path))
        for path in written:
            logger.info("Wrote figure %s", path)

    metrics["figures"] = written
    write_json(output_dir / "metrics.json", metrics)
    logger.info("Wrote metrics to %s", output_dir / "metrics.json")
    log_summary(metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        evaluate(
            model_path=args.model_path,
            test_csv=args.test_csv,
            output_dir=args.output_dir,
            n_repeats=args.n_repeats,
            importance_sample=args.importance_sample,
            top_k=args.top_k,
            make_figures=not args.no_figures,
            seed=args.seed,
        )
    except (
        ModelNotAvailableError,
        FileNotFoundError,
        DataValidationError,
        ValueError,
        RuntimeError,
    ) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
