"""Fetch the raw dataset into ``DATA_DIR``.

A copy of the dataset already ships in ``data/raw/``; use this script only to
re-download it from Kaggle (for example into a different ``DATA_DIR``). It tries
the Kaggle CLI and, when that is unavailable or unauthenticated, prints the
manual steps instead of failing silently.

    python scripts/download_data.py            # download + unzip into DATA_DIR
    python scripts/download_data.py --check    # only report what is on disk
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.utils import configure_logging, ensure_dir, get_logger  # noqa: E402

logger = get_logger(__name__)

MANUAL_STEPS = f"""
Manual download
---------------
1. Create a Kaggle account and an API token (Account -> Create New API Token).
   Place the downloaded kaggle.json at:
     Windows  %USERPROFILE%\\.kaggle\\kaggle.json
     macOS/Linux  ~/.kaggle/kaggle.json   (chmod 600)

2. Install the CLI and download the dataset:
     pip install kaggle
     kaggle datasets download -d {config.KAGGLE_DATASET} -p "{config.DATA_DIR}" --unzip

3. Or download the ZIP from the dataset page in a browser and extract it so that
   this file exists:
     {config.RAW_CSV_PATH}

4. If you keep the data somewhere else, point DATA_DIR at it in your .env file.
""".strip()


def build_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    parser = argparse.ArgumentParser(
        prog="python scripts/download_data.py",
        description="Download the diabetes risk dataset into DATA_DIR.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help=f"Destination directory (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report whether the dataset is already present.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if the CSV already exists.",
    )
    return parser


def dataset_present(data_dir: Path) -> Path | None:
    """Return the CSV path if the dataset is already extracted, else None."""
    expected = data_dir / config.RAW_CSV_NAME
    if expected.is_file():
        return expected
    matches = sorted(data_dir.glob("*.csv")) if data_dir.is_dir() else []
    return matches[0] if matches else None


def unzip_archives(data_dir: Path) -> None:
    """Extract any ZIP archives the Kaggle CLI left behind."""
    for archive in sorted(data_dir.glob("*.zip")):
        logger.info("Extracting %s", archive)
        try:
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(data_dir)
        except (zipfile.BadZipFile, OSError) as exc:
            logger.error("Could not extract '%s': %s", archive, exc)


def download_with_kaggle(data_dir: Path) -> bool:
    """Try the Kaggle CLI. Returns True on success."""
    executable = shutil.which("kaggle")
    if executable is None:
        logger.warning("The 'kaggle' CLI is not on PATH.")
        return False

    command = [
        executable,
        "datasets",
        "download",
        "-d",
        config.KAGGLE_DATASET,
        "-p",
        str(data_dir),
        "--unzip",
    ]
    logger.info("Running: %s", " ".join(command))
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=900
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.error("Kaggle download failed to start or timed out: %s", exc)
        return False

    if completed.returncode != 0:
        logger.error(
            "Kaggle CLI exited with code %d: %s",
            completed.returncode,
            (completed.stderr or completed.stdout or "").strip()[:500],
        )
        return False

    unzip_archives(data_dir)
    return dataset_present(data_dir) is not None


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging()

    data_dir = ensure_dir(args.data_dir)
    existing = dataset_present(data_dir)

    if existing and not args.force:
        logger.info("Dataset already present: %s", existing)
        if existing.name != config.RAW_CSV_NAME:
            logger.warning(
                "Expected '%s'. Rename it or set RAW_CSV_NAME in your .env.",
                config.RAW_CSV_NAME,
            )
        return 0

    if args.check:
        logger.error("Dataset not found in %s", data_dir)
        print(MANUAL_STEPS)
        return 1

    if download_with_kaggle(data_dir):
        logger.info("Dataset ready in %s", data_dir)
        return 0

    logger.error("Automatic download was not possible.")
    print(MANUAL_STEPS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
