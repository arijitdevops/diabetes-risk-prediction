"""Feature engineering: column selection, preprocessing and estimator factories.

The headline behaviour of this module is :func:`drop_excluded_columns`, which
removes the identifier and target-leakage columns before anything else happens.
Everything downstream - training, evaluation, the web form - derives its column
list from the result, so a leaky column cannot sneak back in.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

MODEL_CHOICES: tuple[str, ...] = ("logreg", "rf", "hgb")

#: Deliberately small grids. They are meant to show the mechanics of model
#: selection on a laptop, not to squeeze out the last point of F1.
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logreg": {
        "estimator__C": [0.1, 1.0, 10.0],
    },
    # Depth and leaf size are bounded so a forest stays a few MB on disk; fully
    # grown trees on 40k rows would produce a model file of several hundred MB.
    "rf": {
        "estimator__max_depth": [12, 18],
        "estimator__min_samples_leaf": [10, 25],
    },
    "hgb": {
        "estimator__learning_rate": [0.05, 0.1],
        "estimator__class_weight": [None, "balanced"],
    },
}


def drop_excluded_columns(
    frame: pd.DataFrame,
    *,
    keep_leakage: bool = False,
) -> pd.DataFrame:
    """Return ``frame`` without the identifier and leakage columns.

    Parameters
    ----------
    frame:
        Feature frame (the target may or may not still be present).
    keep_leakage:
        When True the leakage columns are kept. This exists purely so the
        ``--keep-leakage`` demo flag on ``src.train`` can show how implausibly
        high the scores become; never use it for a model you intend to trust.
    """
    if keep_leakage:
        logger.warning(
            "keep_leakage=True: %s are being fed to the model. "
            "Any score produced this way is meaningless.",
            ", ".join(config.LEAKAGE_COLUMNS),
        )
        return frame.copy()

    to_drop = [column for column in config.LEAKAGE_COLUMNS if column in frame.columns]
    if to_drop:
        logger.info("Dropping identifier/leakage columns: %s", ", ".join(to_drop))
    return frame.drop(columns=to_drop)


def split_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split ``frame`` columns into (numeric, categorical).

    Groups are inferred from dtypes and then corrected with the explicit
    override lists in :mod:`src.config`, so a numeric column that arrived as
    text still lands in the numeric branch.
    """
    numeric: list[str] = []
    categorical: list[str] = []

    for column in frame.columns:
        if column == config.TARGET_COLUMN:
            continue
        if column in config.NUMERIC_OVERRIDES:
            numeric.append(column)
        elif column in config.CATEGORICAL_OVERRIDES:
            categorical.append(column)
        elif pd.api.types.is_numeric_dtype(frame[column]):
            numeric.append(column)
        else:
            categorical.append(column)

    if not numeric and not categorical:
        raise ValueError("No feature columns left after exclusions.")

    logger.info("Feature groups: %d numeric, %d categorical", len(numeric), len(categorical))
    return numeric, categorical


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build the median-impute/scale + most-frequent-impute/one-hot transformer.

    ``handle_unknown="ignore"`` means a country or work type that was never seen
    during training encodes as all-zeros instead of raising at request time,
    which is what the web form needs.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_estimator(name: str, seed: int = config.RANDOM_SEED):
    """Instantiate one of the candidate estimators by short name.

    Notes
    -----
    ``LogisticRegression`` with the ``lbfgs`` solver is multinomial by default
    for multiclass targets, so no ``multi_class`` argument is passed (that
    parameter is deprecated in recent scikit-learn releases).
    Class imbalance is handled through ``class_weight`` (always balanced for
    the linear model and the forest, searched over for gradient boosting).
    """
    if name not in MODEL_CHOICES:
        raise ValueError(f"Unknown model {name!r}. Choose from {', '.join(MODEL_CHOICES)}.")

    if name == "logreg":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=250,
        max_leaf_nodes=31,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )


def build_pipeline(
    frame: pd.DataFrame,
    model_name: str,
    *,
    keep_leakage: bool = False,
    seed: int = config.RANDOM_SEED,
) -> tuple[Pipeline, list[str], list[str]]:
    """Assemble the full preprocess + estimate pipeline for a feature frame.

    Returns the unfitted pipeline plus the numeric and categorical column lists
    it was configured with, so callers can record them in metadata.
    """
    features = drop_excluded_columns(frame, keep_leakage=keep_leakage)
    features = features.drop(columns=[config.TARGET_COLUMN], errors="ignore")
    numeric_columns, categorical_columns = split_feature_columns(features)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
            ("estimator", build_estimator(model_name, seed=seed)),
        ]
    )
    return pipeline, numeric_columns, categorical_columns


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the post-transform feature names of a fitted pipeline.

    Falls back to positional names if the transformer cannot report them.
    """
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline has no 'preprocessor' step.")
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except (AttributeError, ValueError) as exc:  # pragma: no cover - old sklearn only
        logger.warning("Could not read transformed feature names: %s", exc)
        return []


def prettify_feature_name(name: str) -> str:
    """Turn ``'categorical__Smoking_Status_Current'`` into ``'Smoking Status: Current'``."""
    cleaned = name.split("__", 1)[-1]
    for column in sorted(config.CATEGORICAL_OVERRIDES, key=len, reverse=True):
        if cleaned.startswith(f"{column}_"):
            value = cleaned[len(column) + 1 :]
            return f"{column.replace('_', ' ')}: {value}"
    return cleaned.replace("_", " ")


def input_columns(pipeline: Pipeline) -> list[str]:
    """List the raw input columns a fitted pipeline expects, in transformer order."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline has no 'preprocessor' step.")
    columns: list[str] = []
    for name, transformer, selection in preprocessor.transformers_:
        # The remainder holds the identifier/leakage columns the model drops.
        if name == "remainder" or transformer == "drop":
            continue
        if isinstance(selection, (list, tuple)):
            columns.extend(str(column) for column in selection)
    return columns
