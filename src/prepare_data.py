"""CLI: load the raw CSV, validate and clean it, and write a stratified split.

Example
-------
    python -m src.prepare_data --test-size 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import config
from src.data import (
    DataValidationError,
    class_balance,
    clean_dataset,
    load_raw_dataset,
    split_dataset,
    validate_columns,
)
from src.utils import configure_logging, ensure_dir, get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m src.prepare_data",
        description="Clean the raw diabetes dataset and write stratified train/test splits.",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=config.RAW_CSV_PATH,
        help=f"Path to the raw CSV (default: {config.RAW_CSV_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.PROCESSED_DIR,
        help=f"Directory for train.csv / test.csv (default: {config.PROCESSED_DIR})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=config.TEST_SIZE,
        help=f"Fraction held out for testing (default: {config.TEST_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help=f"Random seed for the split (default: {config.RANDOM_SEED})",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="Fail instead of warning when the CSV contains unexpected columns.",
    )
    parser.add_argument(
        "--log-level",
        default=config.LOG_LEVEL,
        help="Logging level (default: %(default)s)",
    )
    return parser


def report_balance(name: str, labels: pd.Series) -> None:
    """Log the class distribution of a split as a small table."""
    balance = class_balance(labels)
    logger.info("Class balance for %s (%d rows):", name, len(labels))
    for label, row in balance.iterrows():
        logger.info("  %-9s %7d rows  %6.2f%%", label, int(row["count"]), float(row["percent"]))


def prepare(
    raw_csv: Path,
    output_dir: Path,
    test_size: float,
    seed: int,
    *,
    strict_schema: bool = False,
) -> tuple[Path, Path]:
    """Run the full preparation flow and return the written file paths."""
    frame = load_raw_dataset(raw_csv)
    validate_columns(frame, allow_extra=not strict_schema)
    cleaned = clean_dataset(frame)

    report_balance("full dataset", cleaned[config.TARGET_COLUMN])

    train_df, test_df = split_dataset(cleaned, test_size=test_size, seed=seed)
    report_balance("train", train_df[config.TARGET_COLUMN])
    report_balance("test", test_df[config.TARGET_COLUMN])

    ensure_dir(output_dir)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    try:
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
    except OSError as exc:
        raise RuntimeError(f"Could not write processed splits to '{output_dir}': {exc}") from exc

    logger.info("Wrote %s (%d rows)", train_path, len(train_df))
    logger.info("Wrote %s (%d rows)", test_path, len(test_df))
    return train_path, test_path


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        prepare(
            raw_csv=args.raw_csv,
            output_dir=args.output_dir,
            test_size=args.test_size,
            seed=args.seed,
            strict_schema=args.strict_schema,
        )
    except (FileNotFoundError, DataValidationError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
