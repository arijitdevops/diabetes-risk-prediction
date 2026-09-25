"""Server-side validation for the patient form and the JSON API.

There is no form library here on purpose: the field definitions already live in
:mod:`src.config`, and the same validator has to serve both an HTML form (where
values arrive as strings) and a JSON endpoint (where they arrive typed).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from src import config
from src.config import CategoricalField, NumericField

#: Every input the model expects, in the order the form renders them.
FIELDS: tuple[NumericField | CategoricalField, ...] = (
    *config.NUMERIC_FIELDS,
    *config.CATEGORICAL_FIELDS,
)


def default_form_values() -> dict[str, Any]:
    """Return a plausible, fully populated record used to pre-fill the form."""
    values: dict[str, Any] = {}
    for spec in config.NUMERIC_FIELDS:
        values[spec.name] = int(spec.default) if spec.is_integer else spec.default
    for spec in config.CATEGORICAL_FIELDS:
        values[spec.name] = spec.default
    return values


def validate_numeric(spec: NumericField, raw: Any) -> tuple[float | None, str | None]:
    """Coerce and range-check a single numeric value."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, f"{spec.label} is required."

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"{spec.label} must be a number."

    if math.isnan(value) or math.isinf(value):
        return None, f"{spec.label} must be a finite number."

    if not spec.minimum <= value <= spec.maximum:
        unit = f" {spec.unit}" if spec.unit else ""
        return None, (
            f"{spec.label} must be between {_trim(spec.minimum)}{unit} "
            f"and {_trim(spec.maximum)}{unit}."
        )
    return value, None


def validate_categorical(spec: CategoricalField, raw: Any) -> tuple[str | None, str | None]:
    """Check that a categorical value is one of the allowed choices."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, f"{spec.label} is required."

    value = str(raw).strip()
    match = {choice.casefold(): choice for choice in spec.choices}.get(value.casefold())
    if match is None:
        return None, f"{spec.label} must be one of: {', '.join(spec.choices)}."
    return match, None


def validate_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate a full patient record.

    Returns
    -------
    (record, errors)
        ``record`` holds the cleaned, typed values for every field that passed.
        ``errors`` maps field name to a human readable message and is empty when
        the record is usable.
    """
    record: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for spec in config.NUMERIC_FIELDS:
        value, error = validate_numeric(spec, payload.get(spec.name))
        if error:
            errors[spec.name] = error
        else:
            record[spec.name] = value

    for spec in config.CATEGORICAL_FIELDS:
        value, error = validate_categorical(spec, payload.get(spec.name))
        if error:
            errors[spec.name] = error
        else:
            record[spec.name] = value

    if not errors:
        errors.update(_cross_field_checks(record))
    return record, errors


def _cross_field_checks(record: Mapping[str, Any]) -> dict[str, str]:
    """Checks that only make sense once every individual field is valid."""
    problems: dict[str, str] = {}

    systolic = record.get("Blood_Pressure_Systolic")
    diastolic = record.get("Blood_Pressure_Diastolic")
    if systolic is not None and diastolic is not None and diastolic >= systolic:
        problems["Blood_Pressure_Diastolic"] = (
            "Diastolic blood pressure must be lower than systolic blood pressure."
        )

    hdl = record.get("HDL")
    total = record.get("Total_Cholesterol")
    if hdl is not None and total is not None and hdl > total:
        problems["HDL"] = "HDL cannot exceed total cholesterol."

    return problems


def echo_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the raw submitted values so a rejected form can be re-rendered."""
    return {spec.name: payload.get(spec.name, "") for spec in FIELDS}


def _trim(value: float) -> str:
    """Render a bound without a trailing ``.0``."""
    return str(int(value)) if float(value).is_integer() else str(value)
