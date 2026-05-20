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


# # 05 Clustering
#
# This notebook applies unsupervised clustering to identify distinct market regimes in the crypto market. The idea builds on the EDA findings: BTC correlations, volatility, and return dispersion all fluctuate over time in ways that suggest the market alternates between different structural states (calm vs. crisis periods).
#
# The approach:
# 1. Aggregate the per-asset features from notebook `02` into daily cross-sectional summaries that describe the overall market state on each day.
# 2. Use K-Means clustering to partition trading days into regimes.
# 3. Evaluate cluster quality with the elbow method and silhouette scores.
# 4. Characterize the discovered regimes and check whether they align with known market events.
#
# All runs (per value of $k$, plus the final fit) are tracked with **MLflow** under the `clustering` experiment so configurations can be compared in the MLflow UI.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

import mlflow

sys.path.append(str(Path.cwd().resolve().parent))
from mlflow_utils import get_or_create_experiment, EXPERIMENT_CLUSTERING
from src.utils.plot_config import COIN_COLORS, apply_plot_style

apply_plot_style()

get_or_create_experiment(EXPERIMENT_CLUSTERING)
mlflow.set_experiment(EXPERIMENT_CLUSTERING)

# ## Configuration

PROJECT_ROOT = Path.cwd().resolve().parent

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Input paths (outputs of notebook 02)
FEATURES_LONG_ALIGNED_PATH = DATA_PROCESSED_DIR / "crypto_features_long_aligned.csv"
LOG_RETURNS_ALIGNED_PATH = DATA_PROCESSED_DIR / "crypto_wide_log_returns_aligned.csv"
MACRO_RETURNS_ALIGNED_PATH = DATA_PROCESSED_DIR / "macro_wide_log_returns_aligned.csv"

# Output paths
REGIME_LABELS_OUTPUT_PATH = DATA_PROCESSED_DIR / "regime_labels_baseline.csv"

RANDOM_STATE = 42
K_RANGE = range(2, 7)

# ## Load data

features_long_df = pd.read_csv(FEATURES_LONG_ALIGNED_PATH, parse_dates=["Date"])
log_returns_df = pd.read_csv(LOG_RETURNS_ALIGNED_PATH, parse_dates=["Date"], index_col="Date")
macro_returns_df = pd.read_csv(MACRO_RETURNS_ALIGNED_PATH, parse_dates=["Date"], index_col="Date")

print("Features long shape:", features_long_df.shape)
print("Log returns shape:", log_returns_df.shape)
print("Macro returns shape:", macro_returns_df.shape)

# ## Build daily market-state features
#
# The feature table from notebook `02` contains one row per asset per day. For regime clustering, the goal is to describe the overall market state on each day — not individual asset behavior. To do this, the per-asset features are aggregated into daily cross-sectional summaries.
#
# The following daily features are computed:
#
# | Feature | Description |
# | --- | --- |
# | `mean_log_return` | Average log return across all assets |
# | `return_dispersion` | Standard deviation of log returns across assets (measures how differently assets behave on a given day) |
# | `mean_volatility` | Average 30-day rolling volatility |
# | `mean_btc_corr` | Average 30-day BTC correlation across altcoins (BTC excluded) |
# | `mean_idio_vol` | Average idiosyncratic volatility across altcoins (BTC excluded) |
# | `dxy_log_return` | DXY daily log return |
# | `vix_log_return` | VIX daily log return |
# | `gold_log_return` | Gold daily log return |
# | `sp500_log_return` | S&P 500 daily log return |

from src.features.daily import build_daily_market_features

daily_features_df = build_daily_market_features(features_long_df)

print("Daily features shape:", daily_features_df.shape)
print("\nMissing values:")
print(daily_features_df.isna().sum())

daily_features_clean_df = daily_features_df.dropna().copy()

print("Shape before dropping NaNs:", daily_features_df.shape)
print("Shape after dropping NaNs:", daily_features_clean_df.shape)
print(f"Dropped {len(daily_features_df) - len(daily_features_clean_df)} rows")

daily_features_clean_df.head()

# ## Standardize features
#
# K-Means uses Euclidean distance, which means features with larger scales would dominate the clustering. All features are standardized to zero mean and unit variance before fitting.

feature_columns = daily_features_clean_df.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(daily_features_clean_df[feature_columns])

print("Scaled feature matrix shape:", X_scaled.shape)
print("\nFeature means after scaling (should be ~0):")
print(pd.Series(X_scaled.mean(axis=0), index=feature_columns).round(6))

# ## Selecting the number of clusters
#
# Two standard methods are used to choose a reasonable value of $k$:
#
# - **Elbow method:** Plot the within-cluster sum of squares against $k$. The "elbow" is the point where adding more clusters gives diminishing returns.
# - **Silhouette score:** Measures how similar each point is to its own cluster compared to the nearest other cluster. Ranges from $-1$ (wrong cluster) to $+1$ (well matched). Higher average scores indicate better-separated clusters.

inertias = []
silhouette_scores = []

FEATURES_USED = ",".join(feature_columns)

for k in K_RANGE:
    with mlflow.start_run(run_name=f"kmeans_sweep_k={k}"):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)

        mlflow.log_param("algorithm", "kmeans")
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_init", 10)
        mlflow.log_param("dataset", "aligned_6_assets_from_2020")
        mlflow.log_param("n_features", X_scaled.shape[1])
        mlflow.log_param("features_used", FEATURES_USED)

        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("inertia", kmeans.inertia_)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(sil)

# ## Fit baseline model
#
# Based on the elbow and silhouette plots above, a value of $k$ is chosen for the baseline. For regime detection in financial markets, 3–4 clusters typically correspond to interpretable states (e.g., calm, trending, crisis).

# K=2 chosen to match the HMM regime detection in notebook 06 for a fair side-by-side comparison
# (same K isolates the model-type effect rather than mixing it with a granularity difference).
BASELINE_K = 2

with mlflow.start_run(run_name=f"kmeans_final_k={BASELINE_K}") as final_run:
    kmeans_baseline = KMeans(n_clusters=BASELINE_K, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans_baseline.fit_predict(X_scaled)

    daily_features_clean_df["regime"] = cluster_labels

    sil = silhouette_score(X_scaled, cluster_labels)

    mlflow.log_param("algorithm", "kmeans")
    mlflow.log_param("n_clusters", BASELINE_K)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("n_init", 10)
    mlflow.log_param("dataset", "aligned_6_assets_from_2020")
    mlflow.log_param("features_used", FEATURES_USED)
    mlflow.set_tag("stage", "final")

    mlflow.log_metric("silhouette_score", sil)
    mlflow.log_metric("inertia", kmeans_baseline.inertia_)
    for regime_id, size in pd.Series(cluster_labels).value_counts().sort_index().items():
        mlflow.log_metric(f"cluster_{regime_id}_size", int(size))

print(f"Final K-Means with k={BASELINE_K}")
print(f"Silhouette score: {sil:.4f}")
print(f"Inertia: {kmeans_baseline.inertia_:.2f}")
print(f"\nCluster sizes:")
print(pd.Series(cluster_labels).value_counts().sort_index())
print(f"\nMLflow run_id: {final_run.info.run_id}")

# ## Visualize clusters in PCA space
#
# The feature space has 9 dimensions, so PCA is used to project the data down to 2 dimensions for visualization. This is only for plotting so the clustering itself uses the full feature space.

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Explained variance by PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Total explained variance (2 components): {pca.explained_variance_ratio_.sum():.2%}")

# ## Regime timeline
#
# Plotting the regime labels over time shows when the market transitions between states. If the clustering is meaningful, the regimes should correspond to recognizable market periods.

# ## Regime characterization
#
# To understand what each cluster represents, the average feature values per regime are compared. This reveals the dominant characteristics of each market state.

regime_summary = daily_features_clean_df.groupby("regime")[feature_columns].mean()
regime_summary

# ### Feature distributions by regime

# ## Silhouette analysis
#
# A silhouette plot shows how well each individual data point fits its assigned cluster. Points with negative silhouette values are closer to a neighboring cluster than to their own.

# ## Regime transition frequency
#
# Counting how often the market switches between regimes gives a sense of stability. Frequent switching may indicate that the clustering is too noisy, while rare transitions suggest more persistent regimes.

regime_series = daily_features_clean_df["regime"]
transitions = (regime_series != regime_series.shift(1)).sum() - 1  # subtract first row

total_days = len(regime_series)
print(f"Total trading days: {total_days}")
print(f"Total regime transitions: {transitions}")
print(f"Average regime duration: {total_days / (transitions + 1):.1f} days")

# Transition matrix (row = from, column = to)
transition_pairs = pd.DataFrame({
    "from": regime_series.values[:-1],
    "to": regime_series.values[1:],
})
transition_matrix = pd.crosstab(transition_pairs["from"], transition_pairs["to"], normalize="index")

print("\nTransition probability matrix (row → column):")
transition_matrix

# ## PCA component loadings
#
# Understanding which features contribute most to each principal component helps interpret the structure that PCA reveals.

# ## Save outputs
#
# The regime labels are saved alongside the daily market features so they can be used in downstream analysis and modeling.

output_df = daily_features_clean_df[["regime"]].copy()
output_df.to_csv(REGIME_LABELS_OUTPUT_PATH, index=True)

print(f"Saved regime labels to: {REGIME_LABELS_OUTPUT_PATH}")
print(f"Output shape: {output_df.shape}")

# ## Summary
#
# This notebook clustered trading days into market regimes using K-Means on daily cross-sectional features.
#
# **What was done:**
# - Aggregated per-asset features into 9 daily market-state features (return statistics, volatility, BTC dependence, macro factors)
# - Evaluated cluster counts from $k=2$ to $k=6$ using the elbow method and silhouette scores with each $k$ logging as a separate MLflow run
# - Fit a final K-Means model with $k=2$ (matched to the HMM in notebook `06` for a fair comparison), characterized the resulting regimes, and logged the final run to MLflow (`clustering` experiment, `stage=final` tag)
# - Analyzed cluster quality, PCA structure, and regime transition behavior
#
# **Limitations:**
# - Feature selection was fixed and not tuned
