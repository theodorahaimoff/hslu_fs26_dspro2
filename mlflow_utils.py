# mlflow_utils.py
# Shared MLflow configuration for the DSPRO2 project.
# Import this file in each modeling notebook instead of repeating setup code.

import mlflow
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# One experiment per model type. Use these constants in the notebooks
# so the names stay consistent.
EXPERIMENT_BASELINE       = "baseline"
EXPERIMENT_CLUSTERING     = "clustering"
EXPERIMENT_HMM            = "hmm_regime_detection"
EXPERIMENT_CLASSIFIERS    = "classification"
EXPERIMENT_DIVERSIFICATION = "diversification"
EXPERIMENT_LSTM           = "lstm_forecasting"


def get_or_create_experiment(name: str) -> str:
    """
    Returns the experiment ID for a given name.
    Creates the experiment if it does not exist yet.
    """
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id