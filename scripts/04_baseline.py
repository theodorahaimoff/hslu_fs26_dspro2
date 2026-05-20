"""
Do not edit by hand. The notebook is the source of truth.
Intended to be run via run_all.py, which sets the working directory to
notebooks/ so the notebook-relative paths resolve correctly.
"""

# Shim so notebook display() calls do not break headless execution.
try:
    from IPython.display import display
except Exception:
    def display(*args, **kwargs):
        for _a in args:
            print(_a)


# # 04 Baseline
#
# This notebook establishes a simple predictive baseline that later, more sophisticated models (classifiers in notebook `07`, LSTM in notebook `09`) should be able to beat. The point of a baseline is not to perform well but it is to define a reference level that separates "real signal" from "random guessing".
#
# ## Task
#
# Binary classification: **Will the 30-day average log return of Bitcoin be positive?**
#
# For each trading day `t`, the target is
#
# $$
# y_t = \mathbb{1}\left[ \frac{1}{30} \sum_{k=1}^{30} r_{BTC,\, t+k} > 0 \right]
# $$
#
# where $r_{BTC,t}$ is the BTC daily log return. A 30-day horizon is used rather than next-day direction because daily returns are essentially noise, while a 30-day average has a much higher signal-to-noise ratio and corresponds to a realistic investment horizon.
#
# ## Baselines evaluated
#
# Three reference models are fitted and logged as separate MLflow runs under the `baseline` experiment:
#
# | # | Model | Prediction rule |
# | --- | --- | --- |
# | 1 | **Majority class** | Always predict the more frequent class on the training set. |
# | 2 | **Persistence / momentum** | Predict "positive" if the *past* 30-day average BTC return was positive. |
# | 3 | **Logistic regression** | Linear classifier on four simple features: past 30d BTC return, past 30d BTC volatility, past 30d mean altcoin-BTC correlation, current VIX level. No tuning, no feature engineering. |
#
# All runs are tracked with **MLflow** so notebooks `07` and `08` can read the reference numbers from the `baseline` experiment and be compared on identical metrics.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow

sys.path.append(str(Path.cwd().resolve().parent))
from mlflow_utils import get_or_create_experiment, EXPERIMENT_BASELINE
from src.utils.plot_config import apply_plot_style

apply_plot_style()

get_or_create_experiment(EXPERIMENT_BASELINE)
mlflow.set_experiment(EXPERIMENT_BASELINE)

# ## Configuration
#
# This section defines the input paths, the prediction horizon, and the train/test split.

PROJECT_ROOT = Path.cwd().resolve().parent

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_MODEL_OUTPUTS_DIR = PROJECT_ROOT / "data" / "model_outputs"

# Input paths (outputs of notebook 02)
LOG_RETURNS_ALIGNED_PATH = DATA_PROCESSED_DIR / "crypto_wide_log_returns_aligned.csv"
MACRO_ALIGNED_PATH = DATA_PROCESSED_DIR / "macro_wide_close_aligned_filled.csv"
FEATURES_LONG_ALIGNED_PATH = DATA_PROCESSED_DIR / "crypto_features_long_aligned.csv"

# Output paths
BASELINE_PREDICTIONS_OUTPUT_PATH = DATA_MODEL_OUTPUTS_DIR / "baseline_predictions.csv"

HORIZON = 30
LOOKBACK = 30
TEST_FRACTION = 0.2
RANDOM_STATE = 42

# ## Load data
#
# The aligned log-return dataset and the aligned macro dataset are loaded. The `features_long` table is used only to compute the mean altcoin-BTC correlation per day.

log_returns_df = pd.read_csv(LOG_RETURNS_ALIGNED_PATH, parse_dates=["Date"], index_col="Date")
macro_df = pd.read_csv(MACRO_ALIGNED_PATH, parse_dates=["Date"], index_col="Date")
features_long_df = pd.read_csv(FEATURES_LONG_ALIGNED_PATH, parse_dates=["Date"])

print("Log returns shape:", log_returns_df.shape)
print("Macro shape:     ", macro_df.shape)
print("Features shape:  ", features_long_df.shape)

# ## Build the target
#
# The target is the sign of the BTC average log return over the next `HORIZON` trading days. To avoid look-ahead leakage, the average is strictly forward-looking: the label at date $t$ depends only on returns from $t+1$ to $t+30$.

def build_forward_target(btc_returns: pd.Series, horizon: int) -> pd.Series:
    """
    Binary target: 1 if the average BTC log return over the next `horizon`
    trading days is strictly positive, 0 otherwise.
    """
    forward_avg = btc_returns.shift(-1).rolling(window=horizon).mean().shift(-(horizon - 1))
    return (forward_avg > 0).astype("Int64")

btc_returns = log_returns_df["BTC"]
target = build_forward_target(btc_returns, HORIZON).rename("y")

# The last ~29 rows have no valid 30-day forward window and are silently
# labeled 0 by the function. We need to restore them to NaN so they are dropped
# before training and evaluation rather than treated as genuine labels.
forward_avg_check = btc_returns.shift(-1).rolling(window=HORIZON).mean().shift(-(HORIZON - 1))
target = target.mask(forward_avg_check.isna())

print("Target class balance:")
print(target.value_counts(dropna=False))

# ## Build the feature matrix
#
# Four simple features are used. All are known at date $t$ (no look-ahead):
#
# | Feature | Definition |
# | --- | --- |
# | `past_btc_return_30d` | Average BTC log return over the past 30 days |
# | `past_btc_vol_30d` | Standard deviation of BTC log returns over the past 30 days |
# | `past_mean_btc_corr_30d` | Mean 30-day BTC correlation across altcoins, computed in notebook 02 |
# | `vix_level` | Current VIX closing level (already forward-filled in notebook 02) |

def build_feature_matrix(
    btc_returns: pd.Series,
    features_long: pd.DataFrame,
    macro: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """
    Build the baseline feature matrix indexed by Date. Each column is known at time t.
    """
    past_btc_return = btc_returns.rolling(window=lookback).mean().rename("past_btc_return_30d")
    past_btc_vol = btc_returns.rolling(window=lookback).std().rename("past_btc_vol_30d")

    altcoin_df = features_long[features_long["Ticker"] != "BTC"]
    mean_btc_corr = (
        altcoin_df.groupby("Date")["btc_corr_30d"].mean().rename("past_mean_btc_corr_30d")
    )

    vix_level = macro["VIX"].rename("vix_level")

    X = pd.concat([past_btc_return, past_btc_vol, mean_btc_corr, vix_level], axis=1)
    return X

X_df = build_feature_matrix(btc_returns, features_long_df, macro_df, LOOKBACK)

dataset_df = pd.concat([X_df, target], axis=1).dropna()
dataset_df["y"] = dataset_df["y"].astype(int)

print("Full dataset shape after dropping NaNs:", dataset_df.shape)
dataset_df.head()

# ## Chronological train / test split
#
# Because this is a time-series task, a random split would leak future information into training. The split is strictly chronological: the first 80% of dates are used for training and the final 20% for testing.

split_idx = int(len(dataset_df) * (1 - TEST_FRACTION))

train_df = dataset_df.iloc[:split_idx]
test_df = dataset_df.iloc[split_idx:]

feature_columns = [c for c in dataset_df.columns if c != "y"]

X_train = train_df[feature_columns]
y_train = train_df["y"]
X_test = test_df[feature_columns]
y_test = test_df["y"]

TRAIN_PERIOD = f"{train_df.index.min().date()}_to_{train_df.index.max().date()}"
TEST_PERIOD = f"{test_df.index.min().date()}_to_{test_df.index.max().date()}"

print(f"Train: {X_train.shape[0]} rows | period {TRAIN_PERIOD}")
print(f"Test:  {X_test.shape[0]} rows | period {TEST_PERIOD}")
print(f"\nTrain class balance:\n{y_train.value_counts().sort_index()}")
print(f"\nTest class balance:\n{y_test.value_counts().sort_index()}")

# ## MLflow logging helper
#
# Each of the three baselines is logged with the same set of parameters and metrics so they can be compared side-by-side in the MLflow UI.

def evaluate_and_log(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    extra_params: dict | None = None,
) -> dict:
    """
    Compute classification metrics and log a single MLflow run.
    Returns the metrics dict for further display.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("dataset", "aligned_6_assets_from_2020")
        mlflow.log_param("task", "btc_30d_avg_return_sign")
        mlflow.log_param("horizon", HORIZON)
        mlflow.log_param("lookback", LOOKBACK)
        mlflow.log_param("train_period", TRAIN_PERIOD)
        mlflow.log_param("test_period", TEST_PERIOD)
        mlflow.log_param("features_used", ",".join(feature_columns))

        for k, v in (extra_params or {}).items():
            mlflow.log_param(k, v)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    return metrics

# ## Baseline 1 — Majority class
#
# The simplest possible classifier: always predict the more common class on the training set. Any real model should beat this.

majority_class = int(y_train.mode().iloc[0])
y_pred_majority = np.full(shape=len(y_test), fill_value=majority_class, dtype=int)

metrics_majority = evaluate_and_log(
    model_name="majority_class",
    y_true=y_test,
    y_pred=y_pred_majority,
    extra_params={"majority_class": majority_class},
)

metrics_majority

# ## Baseline 2 — Persistence (momentum)
#
# A null model in finance: predict that the sign of the *next* 30-day average return equals the sign of the *past* 30-day average return. This tests whether simple trend-following is enough.

y_pred_persistence = (X_test["past_btc_return_30d"] > 0).astype(int).to_numpy()

metrics_persistence = evaluate_and_log(
    model_name="persistence_momentum",
    y_true=y_test,
    y_pred=y_pred_persistence,
)

metrics_persistence

# ## Baseline 3 — Logistic regression
#
# A non-trivial model: linear classifier on the four standardized baseline features. No hyperparameter tuning, no regularization search, no feature engineering. The goal is to set a low but meaningful bar.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
logreg.fit(X_train_scaled, y_train)

y_pred_logreg = logreg.predict(X_test_scaled)

metrics_logreg = evaluate_and_log(
    model_name="logistic_regression",
    y_true=y_test,
    y_pred=y_pred_logreg,
    extra_params={"random_state": RANDOM_STATE, "max_iter": 1000},
)

metrics_logreg

# ## Compare baselines
#
# All three baselines are summarized in a single table for quick comparison.

results_df = pd.DataFrame(
    [metrics_majority, metrics_persistence, metrics_logreg],
    index=["majority_class", "persistence_momentum", "logistic_regression"],
)

results_df.round(4)

# ## Save predictions
#
# The predictions of all three baselines on the test set are saved to `data/model_outputs/` so later notebooks can re-use them for comparison or stacking.

predictions_df = pd.DataFrame(
    {
        "y_true": y_test.to_numpy(),
        "majority_class": y_pred_majority,
        "persistence_momentum": y_pred_persistence,
        "logistic_regression": y_pred_logreg,
    },
    index=y_test.index,
)

DATA_MODEL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
predictions_df.to_csv(BASELINE_PREDICTIONS_OUTPUT_PATH, index=True)
print(f"Saved baseline predictions to: {BASELINE_PREDICTIONS_OUTPUT_PATH}")

# ## Summary
#
# This notebook established three simple baselines for predicting the direction of BTC's 30-day average return.
#
# **What was done:**
# - Defined a binary target based on a forward 30-day average return (smoother than daily direction, still classification)
# - Built a minimal four-feature matrix from already-engineered columns in notebook `02`
# - Split chronologically (80/20) to avoid look-ahead leakage
# - Fit three baselines (majority class, persistence, logistic regression) and logged each to MLflow under the `baseline` experiment
# - Saved test-set predictions to `data/model_outputs/baseline_predictions.csv`
#
# **Limitations:**
# - Only four features; no tuning, no cross-validation
# - 30-day horizon means fewer effectively independent test observations than the row count suggests (overlapping windows)
# - Class balance is not enforced; the majority-class baseline captures any natural imbalance
#
# **Next steps:**
# - Notebook 07 — proper classifiers (random forest, gradient boosting) on a richer feature set should beat the logistic baseline
# - Notebook 09 — LSTM sequence models on the raw returns
# - Cross-notebook comparison: any downstream model should log the same metrics (`accuracy`, `f1_macro`) so MLflow lets us compare runs directly against these baselines
