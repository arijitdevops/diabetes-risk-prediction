"""Central configuration: paths, column groups and feature metadata.

Every runtime path is driven by an environment variable with a working
default, so the project can be cloned and run without a ``.env`` file: the raw
CSV ships in ``data/raw/`` and the processed splits in ``data/processed/``.

The field specifications at the bottom of this module are the single source of
truth shared by the Flask form, the JSON API validator and the sample data
generator: add a column here and it appears everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.utils import env_float, env_int

try:  # python-dotenv is a convenience, not a hard requirement
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without the extra

    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        """No-op stand-in used when python-dotenv is not installed."""
        return False


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: str) -> Path:
    """Resolve an env-configured path relative to the project root."""
    raw = os.getenv(name, "").strip() or default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

DATA_DIR: Path = _path_from_env("DATA_DIR", "data/raw")
RAW_CSV_NAME: str = os.getenv("RAW_CSV_NAME", "diabetes_risk_prediction_dataset.csv")
RAW_CSV_PATH: Path = DATA_DIR / RAW_CSV_NAME

PROCESSED_DIR: Path = _path_from_env("PROCESSED_DIR", "data/processed")
TRAIN_CSV_PATH: Path = PROCESSED_DIR / "train.csv"
TEST_CSV_PATH: Path = PROCESSED_DIR / "test.csv"

MODEL_PATH: Path = _path_from_env("MODEL_PATH", "models/model.joblib")
METADATA_PATH: Path = MODEL_PATH.with_name("metadata.json")

REPORTS_DIR: Path = _path_from_env("REPORTS_DIR", "reports")
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"
IMPORTANCE_PATH: Path = REPORTS_DIR / "permutation_importance.json"

SAMPLES_DIR: Path = PROJECT_ROOT / "samples"

# --------------------------------------------------------------------------- #
# Run settings
# --------------------------------------------------------------------------- #

RANDOM_SEED: int = env_int("RANDOM_SEED", 42)
TEST_SIZE: float = env_float("TEST_SIZE", 0.2)
CV_FOLDS: int = env_int("CV_FOLDS", 5)
SCORING: str = os.getenv("SCORING", "f1_macro")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

KAGGLE_DATASET: str = "mobeenfatimah/diabetes-risk-prediction-dataset-50k-patients"

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #

TARGET_COLUMN: str = "Diabetes_Risk"

#: Ordered from least to most severe. Used for plots, probability bars and the
#: ``labels=`` argument of every scikit-learn metric so that reports are stable
#: even when a split happens to be missing a rare class.
CLASS_ORDER: tuple[str, ...] = ("Low", "Moderate", "High")

#: Columns that must never reach the model.
#:
#: ``Diabetes_Risk_Score`` is the continuous score the ``Diabetes_Risk`` label
#: is bucketed from (Low 13-34, Moderate 35-64, High 65-100 in the shipped
#: dataset), so it reconstructs the target exactly. ``AI_Health_Recommendation``
#: and ``Doctor_Consultation_Needed`` are generated downstream of that score and
#: therefore encode the answer too. ``Patient_ID`` is a row identifier with no
#: clinical meaning; keeping it would let tree models memorise rows.
LEAKAGE_COLUMNS: tuple[str, ...] = (
    "Patient_ID",
    "Diabetes_Risk_Score",
    "AI_Health_Recommendation",
    "Doctor_Consultation_Needed",
)

#: Every column expected in the raw CSV, in file order. Used by the validator in
#: :mod:`src.data` to fail fast on a truncated or renamed download.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "Patient_ID",
    "Age",
    "Gender",
    "Country",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Waist_Circumference_cm",
    "Blood_Glucose",
    "HbA1c",
    "Fasting_Blood_Sugar",
    "Insulin_Level",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Total_Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
    "Heart_Rate",
    "Physical_Activity_Level",
    "Exercise_Hours_Per_Week",
    "Daily_Walking_Minutes",
    "Diet_Quality",
    "Sugar_Intake_Level",
    "Sleep_Hours",
    "Stress_Level",
    "Smoking_Status",
    "Alcohol_Consumption",
    "Family_History_Diabetes",
    "Hypertension",
    "Heart_Disease",
    "Fatty_Liver",
    "PCOS",
    "Medication_Adherence",
    "Work_Type",
    "Residence_Type",
    "Daily_Water_Intake_L",
    "Diabetes_Risk_Score",
    "AI_Health_Recommendation",
    "Doctor_Consultation_Needed",
    "Diabetes_Risk",
)

#: Explicit dtype overrides. Column groups are inferred from pandas dtypes at
#: fit time (see :func:`src.features.split_feature_columns`), but these lists win
#: whenever they disagree - for example a numeric-looking column that should be
#: treated as a category, or a numeric column that arrives as text because of a
#: stray unit suffix in a re-exported CSV.
NUMERIC_OVERRIDES: tuple[str, ...] = (
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Waist_Circumference_cm",
    "Blood_Glucose",
    "HbA1c",
    "Fasting_Blood_Sugar",
    "Insulin_Level",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Total_Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
    "Heart_Rate",
    "Exercise_Hours_Per_Week",
    "Daily_Walking_Minutes",
    "Sleep_Hours",
    "Daily_Water_Intake_L",
)

CATEGORICAL_OVERRIDES: tuple[str, ...] = (
    "Gender",
    "Country",
    "Physical_Activity_Level",
    "Diet_Quality",
    "Sugar_Intake_Level",
    "Stress_Level",
    "Smoking_Status",
    "Alcohol_Consumption",
    "Family_History_Diabetes",
    "Hypertension",
    "Heart_Disease",
    "Fatty_Liver",
    "PCOS",
    "Medication_Adherence",
    "Work_Type",
    "Residence_Type",
)

# --------------------------------------------------------------------------- #
# Feature metadata (shared by the web form, the API validator and the samples)
# --------------------------------------------------------------------------- #

#: Form section order, rendered top to bottom in the web UI.
SECTIONS: tuple[str, ...] = (
    "Demographics",
    "Body Metrics",
    "Blood Work",
    "Vitals",
    "Lifestyle",
    "Medical History",
)

YES_NO: tuple[str, ...] = ("No", "Yes")

COUNTRIES: tuple[str, ...] = (
    "Argentina",
    "Australia",
    "Bangladesh",
    "Brazil",
    "Canada",
    "China",
    "Egypt",
    "France",
    "Germany",
    "India",
    "Indonesia",
    "Italy",
    "Japan",
    "Malaysia",
    "Mexico",
    "Nigeria",
    "Pakistan",
    "Russia",
    "Saudi Arabia",
    "South Africa",
    "South Korea",
    "Spain",
    "Turkey",
    "United Kingdom",
    "United States",
)


@dataclass(frozen=True)
class NumericField:
    """A numeric input: label, unit, accepted range and a plausible default.

    ``minimum``/``maximum`` are validation bounds, deliberately a little wider
    than the range observed in the shipped dataset so that a realistic patient
    outside the sample is not rejected.
    """

    name: str
    label: str
    section: str
    minimum: float
    maximum: float
    step: float
    default: float
    unit: str = ""
    help_text: str = ""

    @property
    def is_integer(self) -> bool:
        """True when the field should be collected and echoed as a whole number."""
        return float(self.step).is_integer()


@dataclass(frozen=True)
class CategoricalField:
    """A single-choice input backed by a fixed vocabulary."""

    name: str
    label: str
    section: str
    choices: tuple[str, ...]
    default: str
    help_text: str = ""


NUMERIC_FIELDS: tuple[NumericField, ...] = (
    NumericField("Age", "Age", "Demographics", 18, 110, 1, 54, "years"),
    NumericField("Height_cm", "Height", "Body Metrics", 120, 220, 0.1, 170.0, "cm"),
    NumericField("Weight_kg", "Weight", "Body Metrics", 30, 250, 0.1, 87.5, "kg"),
    NumericField(
        "BMI",
        "Body mass index",
        "Body Metrics",
        10,
        80,
        0.1,
        30.9,
        "kg/m^2",
        "Leave as calculated from height and weight if you are unsure.",
    ),
    NumericField(
        "Waist_Circumference_cm", "Waist circumference", "Body Metrics", 50, 200, 0.1, 100.0, "cm"
    ),
    NumericField(
        "Blood_Glucose", "Random blood glucose", "Blood Work", 40, 400, 0.1, 160.0, "mg/dL"
    ),
    NumericField("HbA1c", "HbA1c", "Blood Work", 3.0, 18.0, 0.1, 8.5, "%"),
    NumericField(
        "Fasting_Blood_Sugar", "Fasting blood sugar", "Blood Work", 40, 400, 0.1, 142.6, "mg/dL"
    ),
    NumericField("Insulin_Level", "Fasting insulin", "Blood Work", 0.5, 100, 0.1, 23.5, "uU/mL"),
    NumericField(
        "Total_Cholesterol", "Total cholesterol", "Blood Work", 80, 450, 0.1, 220.0, "mg/dL"
    ),
    NumericField("HDL", "HDL cholesterol", "Blood Work", 15, 130, 0.1, 57.0, "mg/dL"),
    NumericField("LDL", "LDL cholesterol", "Blood Work", 20, 320, 0.1, 135.0, "mg/dL"),
    NumericField("Triglycerides", "Triglycerides", "Blood Work", 30, 600, 0.1, 229.0, "mg/dL"),
    NumericField(
        "Blood_Pressure_Systolic", "Systolic blood pressure", "Vitals", 70, 250, 1, 140, "mmHg"
    ),
    NumericField(
        "Blood_Pressure_Diastolic", "Diastolic blood pressure", "Vitals", 40, 160, 1, 90, "mmHg"
    ),
    NumericField("Heart_Rate", "Resting heart rate", "Vitals", 35, 200, 1, 85, "bpm"),
    NumericField(
        "Exercise_Hours_Per_Week", "Exercise per week", "Lifestyle", 0, 40, 0.1, 5.0, "hours"
    ),
    NumericField("Daily_Walking_Minutes", "Walking per day", "Lifestyle", 0, 600, 1, 90, "minutes"),
    NumericField("Sleep_Hours", "Sleep per night", "Lifestyle", 0, 16, 0.1, 6.5, "hours"),
    NumericField("Daily_Water_Intake_L", "Water per day", "Lifestyle", 0, 10, 0.1, 3.0, "litres"),
)

CATEGORICAL_FIELDS: tuple[CategoricalField, ...] = (
    CategoricalField("Gender", "Gender", "Demographics", ("Female", "Male", "Other"), "Female"),
    CategoricalField("Country", "Country", "Demographics", COUNTRIES, "India"),
    CategoricalField(
        "Work_Type",
        "Work type",
        "Demographics",
        ("Business", "Government", "Private", "Retired", "Student"),
        "Private",
    ),
    CategoricalField("Residence_Type", "Residence", "Demographics", ("Rural", "Urban"), "Urban"),
    CategoricalField(
        "Physical_Activity_Level",
        "Physical activity level",
        "Lifestyle",
        ("Low", "Moderate", "High"),
        "Moderate",
    ),
    CategoricalField(
        "Diet_Quality", "Diet quality", "Lifestyle", ("Poor", "Average", "Healthy"), "Average"
    ),
    CategoricalField(
        "Sugar_Intake_Level", "Sugar intake", "Lifestyle", ("Low", "Moderate", "High"), "Moderate"
    ),
    CategoricalField(
        "Stress_Level", "Stress level", "Lifestyle", ("Low", "Moderate", "High"), "Moderate"
    ),
    CategoricalField(
        "Smoking_Status", "Smoking status", "Lifestyle", ("Never", "Former", "Current"), "Never"
    ),
    CategoricalField(
        "Alcohol_Consumption",
        "Alcohol consumption",
        "Lifestyle",
        ("Never", "Occasionally", "Frequently"),
        "Never",
    ),
    CategoricalField(
        "Family_History_Diabetes", "Family history of diabetes", "Medical History", YES_NO, "No"
    ),
    CategoricalField("Hypertension", "Hypertension", "Medical History", YES_NO, "No"),
    CategoricalField("Heart_Disease", "Heart disease", "Medical History", YES_NO, "No"),
    CategoricalField("Fatty_Liver", "Fatty liver", "Medical History", YES_NO, "No"),
    CategoricalField(
        "PCOS", "PCOS", "Medical History", YES_NO, "No", "Applicable to female patients."
    ),
    CategoricalField(
        "Medication_Adherence",
        "Medication adherence",
        "Medical History",
        ("Poor", "Average", "Good"),
        "Average",
    ),
)

NUMERIC_FIELD_MAP: dict[str, NumericField] = {f.name: f for f in NUMERIC_FIELDS}
CATEGORICAL_FIELD_MAP: dict[str, CategoricalField] = {f.name: f for f in CATEGORICAL_FIELDS}

#: Model input columns, i.e. everything in the raw file except the target and
#: the identifier/leakage columns.
FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in EXPECTED_COLUMNS
    if column != TARGET_COLUMN and column not in LEAKAGE_COLUMNS
)


def fields_by_section() -> dict[str, list[NumericField | CategoricalField]]:
    """Group every input field by form section, preserving :data:`SECTIONS` order."""
    grouped: dict[str, list[NumericField | CategoricalField]] = {name: [] for name in SECTIONS}
    for item in (*NUMERIC_FIELDS, *CATEGORICAL_FIELDS):
        grouped.setdefault(item.section, []).append(item)
    for section in grouped:
        grouped[section].sort(key=lambda item: FEATURE_COLUMNS.index(item.name))
    return grouped


def describe_settings() -> dict[str, str]:
    """Return the effective configuration, handy for ``/api/health`` and logs."""
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "raw_csv_path": str(RAW_CSV_PATH),
        "processed_dir": str(PROCESSED_DIR),
        "model_path": str(MODEL_PATH),
        "metadata_path": str(METADATA_PATH),
        "reports_dir": str(REPORTS_DIR),
        "random_seed": str(RANDOM_SEED),
        "test_size": str(TEST_SIZE),
        "cv_folds": str(CV_FOLDS),
        "scoring": SCORING,
        "log_level": LOG_LEVEL,
    }
