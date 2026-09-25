"""Raw data loading, validation, cleaning and splitting.

The raw CSV is committed at ``data/raw/`` (``DATA_DIR`` can point elsewhere),
and :func:`split_dataset` produces the ``data/processed/`` train/test files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when the dataset on disk does not match the expected schema."""


def load_raw_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw dataset CSV.

    Parameters
    ----------
    path:
        Optional explicit CSV path. Defaults to ``DATA_DIR/RAW_CSV_NAME``.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing, with a pointer at ``scripts/download_data.py``.
    DataValidationError
        If the file is empty or unparsable.
    """
    csv_path = Path(path) if path is not None else config.RAW_CSV_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at '{csv_path}'. It ships in data/raw/; if you moved it, "
            "set DATA_DIR in your .env, or run 'python scripts/download_data.py' to fetch it "
            "from Kaggle."
        )

    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Raw dataset '{csv_path}' is empty.") from exc
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not read '{csv_path}': {exc}") from exc

    logger.info("Loaded %s rows x %s columns from %s", len(frame), frame.shape[1], csv_path)
    return frame


def validate_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...] = config.EXPECTED_COLUMNS,
    *,
    allow_extra: bool = True,
) -> None:
    """Check that every expected column is present.

    Extra columns are tolerated by default (a Kaggle re-upload may add one) but
    are logged so the difference is never silent.
    """
    present = set(frame.columns)
    missing = [column for column in expected if column not in present]
    if missing:
        raise DataValidationError(
            f"Dataset is missing {len(missing)} expected column(s): {', '.join(missing)}"
        )

    extra = [column for column in frame.columns if column not in set(expected)]
    if extra:
        message = f"Dataset has unexpected column(s): {', '.join(extra)}"
        if allow_extra:
            logger.warning("%s (ignored)", message)
        else:
            raise DataValidationError(message)


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative cleaning that does not depend on the train/test split.

    Steps: strip whitespace from text columns, coerce the known numeric columns
    to floats, drop duplicate patient identifiers, and drop rows whose target is
    missing or outside :data:`src.config.CLASS_ORDER`.

    Missing feature values are intentionally left in place - imputation happens
    inside the modelling pipeline so it is fitted on training folds only.
    """
    cleaned = frame.copy()

    for column in cleaned.columns:
        if cleaned[column].dtype == object or pd.api.types.is_string_dtype(cleaned[column]):
            stripped = cleaned[column].astype("string").str.strip()
            stripped = stripped.replace({"": pd.NA, "nan": pd.NA, "NA": pd.NA})
            # Hand plain object/np.nan back to scikit-learn: pandas' NA sentinel
            # is not understood by SimpleImputer.
            cleaned[column] = stripped.astype(object).where(stripped.notna(), np.nan)

    for column in config.NUMERIC_OVERRIDES:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "Patient_ID" in cleaned.columns:
        duplicates = int(cleaned["Patient_ID"].duplicated().sum())
        if duplicates:
            logger.warning("Dropping %d duplicated Patient_ID row(s)", duplicates)
            cleaned = cleaned.drop_duplicates(subset="Patient_ID", keep="first")

    target = config.TARGET_COLUMN
    if target not in cleaned.columns:
        raise DataValidationError(f"Target column '{target}' is not present in the dataset.")

    cleaned[target] = cleaned[target].astype("string").str.strip().str.title().astype(object)
    unknown_mask = ~cleaned[target].isin(config.CLASS_ORDER)
    unknown_count = int(unknown_mask.sum())
    if unknown_count:
        observed = sorted(set(cleaned.loc[unknown_mask, target].dropna().tolist()))
        logger.warning(
            "Dropping %d row(s) with a missing or unknown target label %s",
            unknown_count,
            observed or "(missing)",
        )
        cleaned = cleaned.loc[~unknown_mask]

    if cleaned.empty:
        raise DataValidationError("No usable rows remain after cleaning.")

    cleaned = cleaned.reset_index(drop=True)
    logger.info("Cleaned dataset: %s rows x %s columns", len(cleaned), cleaned.shape[1])
    return cleaned


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the target column.

    Stratification matters here: the ``Low`` band is under 1% of the shipped
    dataset, and a plain random split can leave a fold without it entirely.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}")

    target = frame[config.TARGET_COLUMN]
    counts = target.value_counts()
    rare = counts[counts < 2]
    stratify = target if rare.empty else None
    if stratify is None:
        logger.warning(
            "Classes %s have fewer than 2 rows; falling back to an unstratified split",
            list(rare.index),
        )

    train_df, test_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )
    logger.info("Split into %d train / %d test rows (seed=%d)", len(train_df), len(test_df), seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def class_balance(labels: pd.Series) -> pd.DataFrame:
    """Return a count/percentage table for the target, ordered by severity."""
    counts = labels.value_counts().reindex(config.CLASS_ORDER, fill_value=0)
    total = int(counts.sum())
    percentage = (counts / total * 100).round(2) if total else counts.astype(float)
    return pd.DataFrame({"count": counts.astype(int), "percent": percentage})


def load_processed(kind: str) -> pd.DataFrame:
    """Load ``data/processed/train.csv`` or ``data/processed/test.csv``."""
    if kind not in {"train", "test"}:
        raise ValueError(f"kind must be 'train' or 'test', got {kind!r}")

    path = config.TRAIN_CSV_PATH if kind == "train" else config.TEST_CSV_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed {kind} split not found at '{path}'. Run 'python -m src.prepare_data' first."
        )

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not read '{path}': {exc}") from exc

    logger.info("Loaded %s split: %s rows from %s", kind, len(frame), path)
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target column from the rest of the frame."""
    target = config.TARGET_COLUMN
    if target not in frame.columns:
        raise DataValidationError(f"Target column '{target}' is not present.")
    return frame.drop(columns=[target]), frame[target].astype("object")
