"""Diabetes risk prediction package.

Modules
-------
config        Environment-driven settings, column groups and feature metadata.
data          Raw CSV loading, validation, cleaning and train/test splitting.
features      Preprocessing pipeline, estimators and hyper-parameter grids.
prepare_data  CLI: raw CSV -> data/processed/{train,test}.csv
train         CLI: fit + cross-validate -> models/model.joblib + metadata.json
evaluate      CLI: held-out metrics, figures and permutation importance.
predict       CLI and helpers for single-record / batch inference.
utils         Logging, JSON helpers and small filesystem utilities.
"""

__version__ = "0.1.0"
