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


# # 02 Feature Engineering
#
# This notebook loads the cleaned cryptocurrency datasets created in notebook `01a_data_collection_and_cleaning` and the macro dataset created in notebook `01b_macro_data_collection`. It processes the macro data and computes time-series features for later analysis and modeling.
#
# The feature set is designed to capture both general market behavior and dependence on Bitcoin.
#
# In particular, the notebook creates:
#
# | Feature | Meaning                                                                                                            |
# | --- |--------------------------------------------------------------------------------------------------------------------|
# | **daily log returns** | How much did the price change from yesterday to today?                                                             |
# | **rolling volatility** | How much does this coin fluctuate?                                                                                 |
# | **rolling correlation with Bitcoin** | Does this coin move together with Bitcoin?                                                                         |
# | **idiosyncratic volatility (BTC-only)** based on residuals from a regression on Bitcoin returns | How much of this coin's movement is not explained by Bitcoin?                                                      |
# | **multi-factor idiosyncratic volatility (Leave-One-Out)** based on residuals from a rolling regression on BTC, macro factors, and the rest-of-sector index | How much of this coin's movement is not explained by BTC, broad markets, the US dollar, volatility risk, gold, or the rest of the altcoin sector? |
# | **volume-based features** | How reliable is the coin? <br> `high volume`: strong market activity <br> `low volume`: illiquid and less reliable |
# | **macro log returns** (DXY, VIX, Gold, S&P 500) | How do broader market conditions change over time, independent of crypto prices? |

from pathlib import Path

import numpy as np
import pandas as pd

# ## Configuration
#
# This section defines the input and output paths used in the notebook.

PROJECT_ROOT = Path.cwd().resolve().parent

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Full historical wide datasets
FULL_CLOSE_INPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_close_full.csv"
FULL_VOLUME_INPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_volume_full.csv"
FULL_MARKETCAP_INPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_marketcap_full.csv"

# Aligned modeling datasets
ALIGNED_CLOSE_INPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_close_aligned.csv"
ALIGNED_VOLUME_INPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_volume_aligned.csv"

# Aligned macro dataset
MACRO_ALIGNED_INPUT_PATH = DATA_PROCESSED_DIR / "macro_wide_close_aligned.csv"
MACRO_ALIGNED_FILLED_OUTPUT_PATH = DATA_PROCESSED_DIR / "macro_wide_close_aligned_filled.csv"

# Feature datasets in long format (one row per asset and date)
FEATURES_LONG_FULL_OUTPUT_PATH = DATA_PROCESSED_DIR / "crypto_features_long_full.csv"
FEATURES_LONG_ALIGNED_OUTPUT_PATH = DATA_PROCESSED_DIR / "crypto_features_long_aligned.csv"

# Wide-format log returns used for modeling
FEATURES_WIDE_RETURNS_OUTPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_log_returns_aligned.csv"
MACRO_WIDE_RETURNS_OUTPUT_PATH = DATA_PROCESSED_DIR / "macro_wide_log_returns_aligned.csv"

# Summary of missing values and data quality in the feature datasets
FEATURES_SUMMARY_OUTPUT_PATH = DATA_PROCESSED_DIR / "feature_quality_summary.csv"

VOLATILITY_WINDOW = 30
CORRELATION_WINDOW = 30
LONG_CORRELATION_WINDOW = 90
IDIO_WINDOW = 30
MF_IDIO_WINDOW = 126

# ## Load and preprocess macro datasets
#
# The aligned macro dataset is loaded and prepared for feature engineering.
#
# Because macro factors follow market trading calendars, missing values occur on weekends and holidays when no observations are available. To ensure compatibility with the crypto dataset, missing values are forward-filled.
#
# This assumes that macro conditions remain constant between trading days and allows the macro data to be aligned with the continuous crypto time series.

macro_aligned_df = pd.read_csv(MACRO_ALIGNED_INPUT_PATH, parse_dates=["Date"], index_col="Date")

macro_aligned_df = macro_aligned_df.ffill()

print("Missing values after forward fill:")
print(macro_aligned_df.isna().sum())

# ### Compute macro log returns
#
# To integrate macroeconomic market factors into the feature set, daily log returns are computed for the aligned macro series. These returns can later be used as additional explanatory variables alongside the crypto-specific features.

macro_log_returns_df = np.log(macro_aligned_df / macro_aligned_df.shift(1))
macro_log_returns_df.head()

# ## Load cleaned crypto datasets
#
# The full datasets preserve the full historical range, while the aligned datasets only contain dates for which all selected assets are available.

close_full_df = pd.read_csv(FULL_CLOSE_INPUT_PATH, parse_dates=["Date"], index_col="Date")
volume_full_df = pd.read_csv(FULL_VOLUME_INPUT_PATH, parse_dates=["Date"], index_col="Date")

close_aligned_df = pd.read_csv(ALIGNED_CLOSE_INPUT_PATH, parse_dates=["Date"], index_col="Date")
volume_aligned_df = pd.read_csv(ALIGNED_VOLUME_INPUT_PATH, parse_dates=["Date"], index_col="Date")

# Market cap data is only available as a full history file. It is reindexed
# to the aligned date range for use in the multi-factor IVol sector index.
marketcap_full_df = pd.read_csv(FULL_MARKETCAP_INPUT_PATH, parse_dates=["Date"], index_col="Date")
marketcap_aligned_df = marketcap_full_df.reindex(close_aligned_df.index)

print("Full close shape:", close_full_df.shape)
print("Full volume shape:", volume_full_df.shape)
print("Aligned close shape:", close_aligned_df.shape)
print("Aligned volume shape:", volume_aligned_df.shape)

# ## Compute log returns
#
# Log returns are used instead of raw prices because they are more suitable for time-series analysis and make price changes comparable across assets with different price levels.

def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a wide-format price DataFrame.
    """
    return np.log(price_df / price_df.shift(1))

log_returns_full_df = compute_log_returns(close_full_df)
log_returns_aligned_df = compute_log_returns(close_aligned_df)

log_returns_aligned_df.head()

# ## Compute rolling volatility
#
# Rolling volatility is estimated as the standard deviation of log returns over a fixed time window.

def compute_rolling_volatility(return_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling volatility from log returns.
    """
    return return_df.rolling(window=window).std()

rolling_volatility_full_df = compute_rolling_volatility(log_returns_full_df, VOLATILITY_WINDOW)
rolling_volatility_aligned_df = compute_rolling_volatility(log_returns_aligned_df, VOLATILITY_WINDOW)

# ## Compute rolling correlation with Bitcoin
#
# Rolling correlation with Bitcoin is one of the main dependence features in this project. It measures how strongly each asset moves together with Bitcoin over time.

def compute_rolling_correlation_with_btc(return_df: pd.DataFrame, window: int, btc_col: str = "BTC") -> pd.DataFrame:
    """
    Compute rolling correlation of each asset with Bitcoin.
    """
    btc_series = return_df[btc_col]
    corr_df = pd.DataFrame(index=return_df.index)

    for col in return_df.columns:
        corr_df[col] = return_df[col].rolling(window=window).corr(btc_series)

    return corr_df

rolling_corr_30_full_df = compute_rolling_correlation_with_btc(log_returns_full_df, CORRELATION_WINDOW)
rolling_corr_30_aligned_df = compute_rolling_correlation_with_btc(log_returns_aligned_df, CORRELATION_WINDOW)

rolling_corr_90_full_df = compute_rolling_correlation_with_btc(log_returns_full_df, LONG_CORRELATION_WINDOW)
rolling_corr_90_aligned_df = compute_rolling_correlation_with_btc(log_returns_aligned_df, LONG_CORRELATION_WINDOW)

# ## Compute idiosyncratic volatility
#
# Idiosyncratic volatility is used here as an estimate of asset-specific risk that is not explained by Bitcoin.

def compute_idiosyncratic_volatility(
    return_df: pd.DataFrame,
    window: int,
    btc_col: str = "BTC",
    ) -> pd.DataFrame:
    """
    Compute rolling idiosyncratic volatility for each asset relative to Bitcoin.

    For each rolling window, the asset return is regressed on Bitcoin return using
    ordinary least squares. The standard deviation of the residuals is then used
    as the idiosyncratic volatility.
    """
    btc_returns = return_df[btc_col]
    output_df = pd.DataFrame(index=return_df.index, columns=return_df.columns, dtype=float)

    for asset in return_df.columns:
        if asset == btc_col:
            output_df[asset] = 0.0
            continue

        asset_series = return_df[asset]
        values = []

        for i in range(len(return_df)):
            if i < window - 1:
                values.append(np.nan)
                continue

            y = asset_series.iloc[i - window + 1 : i + 1]
            x = btc_returns.iloc[i - window + 1 : i + 1]

            window_df = pd.DataFrame({"y": y, "x": x}).dropna()

            if len(window_df) < window:
                values.append(np.nan)
                continue

            x_vals = window_df["x"].to_numpy()
            y_vals = window_df["y"].to_numpy()

            x_mean = x_vals.mean()
            y_mean = y_vals.mean()

            denom = ((x_vals - x_mean) ** 2).sum()
            if denom == 0:
                values.append(np.nan)
                continue

            beta = ((x_vals - x_mean) * (y_vals - y_mean)).sum() / denom
            alpha = y_mean - beta * x_mean

            residuals = y_vals - (alpha + beta * x_vals)
            values.append(np.std(residuals, ddof=1))

        output_df[asset] = values

    return output_df

idiosyncratic_vol_full_df = compute_idiosyncratic_volatility(log_returns_full_df, IDIO_WINDOW)
idiosyncratic_vol_aligned_df = compute_idiosyncratic_volatility(log_returns_aligned_df, IDIO_WINDOW)

# ### Mathematical explanation
#
# For each asset and each rolling window of length $W$, a simple linear regression is estimated:
#
# $$
# r_{i,t} = \alpha_i + \beta_i \,  \times r_{BTC,t} + \varepsilon_{i,t}
# $$
#
# where:
#
# - $r_{i,t}$: log return of asset $i$ at time $t$
# - $\alpha_i$: baseline return independent of Bitcoin
# - $\beta_i$: sensitivity of the asset to Bitcoin
# - $r_{BTC,t}$: log return of Bitcoin at time $t$
# - $\varepsilon_{i,t}$: residual, the unexplained component
#
# #### Estimation within a rolling window
#
# For each window, the regression parameters are estimated using ordinary least squares:
#
# $$
# \beta_i =
# \frac{\sum_{t=1}^{W}(x_t - \bar{x})(y_t - \bar{y})}
# {\sum_{t=1}^{W}(x_t - \bar{x})^2}
# $$
#
# $$
# \alpha_i = \bar{y} - \beta_i \bar{x}
# $$
#
# where:
#
# - $x_t = r_{BTC,t}$
# - $y_t = r_{i,t}$
# - $\bar{x}$ and $\bar{y}$ are the mean values within the rolling window
#
# #### Residuals
#
# The residuals represent the part of the asset return that is not explained by Bitcoin:
#
# $$
# \varepsilon_{i,t} = y_t - (\alpha_i + \beta_i x_t)
# $$
#
# #### Idiosyncratic volatility
#
# The idiosyncratic volatility is defined as the standard deviation of the residuals within the rolling window:
#
# $$
# \sigma^{idio}_i =
# \sqrt{\frac{1}{W-1}\sum_{t=1}^{W}(\varepsilon_{i,t} - \bar{\varepsilon})^2}
# $$
#
# #### Interpretation
#
# | Idiosyncratic Volatility | Meaning |
# | --- | --- |
# | High | the asset shows stronger asset-specific behavior and is less explained by Bitcoin |
# | Low | the asset is more strongly driven by Bitcoin movements |
#
# In this project, idiosyncratic volatility is used as an indicator of how independent an altcoin behaves relative to Bitcoin.

# ## Multi-factor idiosyncratic volatility (Leave-One-Out)
#
# The simple BTC-only residual volatility above treats Bitcoin as the sole systematic driver of altcoin returns. In practice, altcoin returns are also influenced by:
#
# - **broad risk sentiment** — approximated by the S&P 500
# - **safe-haven demand / dollar strength** — approximated by Gold and the US Dollar Index (DXY)
# - **market fear / volatility regime** — approximated by the VIX
# - **crypto-sector co-movement that is not Bitcoin-specific** — the weighted average return of the *other* altcoins
#
# To isolate the component of each altcoin's return that is genuinely asset-specific, a **multi-factor rolling OLS regression** is estimated against all of these drivers simultaneously. The idiosyncratic volatility is then defined as the rolling standard deviation of the residuals.
#
# ### Leave-One-Out sector index (market-cap weighted)
#
# A naive "crypto sector" index that includes the target altcoin would create circular logic (the target explains itself). To avoid this, the sector factor is constructed per target as a **market-cap-weighted** average of the log returns of *all other altcoins*, excluding the target and Bitcoin:
#
# $$
# R_{sector,t}^{(-i)} = \sum_{j \neq i,\, j \neq BTC} w_{j,t}^{(-i)} \cdot r_{j,t}, \qquad
# w_{j,t}^{(-i)} = \frac{MC_{j,t}}{\sum_{k \neq i,\, k \neq BTC} MC_{k,t}}
# $$
#
# where $MC_{j,t}$ is the market capitalization of asset $j$ at time $t$. The weights are **renormalized per day** over the peer set (all altcoins except the target $i$ and BTC), and they **change over time** as the relative sizes of the altcoins evolve.
#
# **Why market-cap weighting instead of equal weighting?**
#
# Equal weighting would treat a ~$500B coin like ETH and a ~$10B coin like TRX as equally influential drivers of the "rest of the crypto sector", which misrepresents how the market actually moves: large-cap coins dominate cross-sectional co-movement. Market-cap weighting gives each peer a voice proportional to its economic footprint, matching standard practice in equity index construction (e.g., S&P 500) and producing a sector factor that better reflects what a diversified altcoin investor would actually experience. It also makes the residuals, and therefore the resulting idiosyncratic volatility, more meaningful: movements that remain unexplained are truly specific to the target coin, not artefacts of overweighting small coins in the sector index.
#
# ### Regression model
#
# For each altcoin $i$ and each rolling window of length $W$ (default 126 trading days, roughly half a year), we estimate by OLS:
#
# $$
# r_{i,t} = \alpha_i + \beta_1 r_{BTC,t} + \beta_2 r_{SP500,t} + \beta_3 r_{DXY,t} + \beta_4 r_{VIX,t} + \beta_5 r_{Gold,t} + \beta_6 R_{sector,t}^{(-i)} + \varepsilon_{i,t}
# $$
#
# The multi-factor idiosyncratic volatility is the standard deviation of the residuals inside the window:
#
# $$
# \sigma^{idio, MF}_{i,t} = \sqrt{\frac{1}{W-1} \sum_{k=0}^{W-1} \left(\varepsilon_{i,t-k} - \bar{\varepsilon}\right)^2}
# $$
#
# ### Interpretation
#
# | Multi-factor IVol | Meaning |
# | --- | --- |
# | High | the asset has strong idiosyncratic movement that is not explained by BTC, macro factors, or the rest of the altcoin sector |
# | Low | the asset's returns are well explained by the systematic factors — it behaves like a leveraged combination of BTC, macro risk, and the sector |
#
# This feature is a more defensible measure of "asset-specific risk" than a BTC-only residual and is used downstream as the primary input to regime detection (HMM) and forecasting.

def compute_multifactor_idiosyncratic_volatility(
    return_df: pd.DataFrame,
    macro_return_df: pd.DataFrame,
    marketcap_df: pd.DataFrame,
    window: int,
    btc_col: str = "BTC",
    macro_cols: tuple = ("SP500", "DXY", "VIX", "Gold"),
) -> pd.DataFrame:
    """
    Compute rolling multi-factor idiosyncratic volatility for each altcoin, using
    a market-cap-weighted Leave-One-Out sector index.

    For each asset i, the daily log return is regressed (OLS, rolling window of `window`
    observations) on:
        - BTC return
        - macro factor returns (SP500, DXY, VIX, Gold by default)
        - a Leave-One-Out sector index: the MARKET-CAP-WEIGHTED mean return of all
          other altcoins in the basket, excluding asset i and BTC. Weights are
          recomputed each day from the contemporaneous market caps of the peer set,
          so a larger coin such as ETH naturally dominates the sector return while a
          smaller coin such as TRX contributes proportionally less.
    The rolling standard deviation of the residuals is returned as the idiosyncratic
    volatility. BTC itself has no meaningful idiosyncratic decomposition in this setup,
    so the BTC column is filled with zeros.
    """
    merged = return_df.join(macro_return_df[list(macro_cols)], how="left")
    altcoins = [c for c in return_df.columns if c != btc_col]
    output_df = pd.DataFrame(index=return_df.index, columns=return_df.columns, dtype=float)
    output_df[btc_col] = 0.0

    for asset in altcoins:
        peers = [c for c in altcoins if c != asset]

        # Market-cap weights per day, restricted to the peer set, renormalized
        # so weights sum to 1. Days with missing market caps for any peer yield
        # NaN weights, which later drop out via dropna().
        peer_mc = marketcap_df[peers]
        peer_weights = peer_mc.div(peer_mc.sum(axis=1), axis=0)
        sector_idx = (return_df[peers] * peer_weights).sum(axis=1, min_count=1)

        factors = pd.concat(
            [merged[btc_col], merged[list(macro_cols)], sector_idx.rename("SECTOR")],
            axis=1,
        )
        y_full = merged[asset]
        df_full = pd.concat([y_full.rename("y"), factors], axis=1).dropna()

        resid_series = pd.Series(index=return_df.index, dtype=float)

        for end_pos in range(window - 1, len(df_full)):
            win = df_full.iloc[end_pos - window + 1 : end_pos + 1]
            y = win["y"].to_numpy()
            X = win.drop(columns="y").to_numpy()
            X = np.column_stack([np.ones(len(X)), X])

            # OLS via lstsq (robust to occasional collinearity)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            resid_series.loc[win.index[-1]] = np.std(resid, ddof=1)

        output_df[asset] = resid_series

    return output_df

mf_idio_vol_aligned_df = compute_multifactor_idiosyncratic_volatility(
    return_df=log_returns_aligned_df,
    macro_return_df=macro_log_returns_df,
    marketcap_df=marketcap_aligned_df,
    window=MF_IDIO_WINDOW,
)

mf_idio_vol_aligned_df.tail()

mf_idio_vol_aligned_df.describe()

# ## Volume-based features
#
# In addition to price-based features, simple volume features are included in the form of log-transformed volume and daily log volume change. Including volume-based features allows the model to account for variations in market activity that are not reflected in price changes alone. \
# For example, a price increase accompanied by high trading volume may indicate strong market participation, whereas the same price increase with low volume may be less reliable and driven by fewer market participants.

def compute_log_volume(volume_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log-transformed trading volume.
    """
    return np.log1p(volume_df)


def compute_log_volume_change(volume_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily change in log-transformed trading volume.
    """
    log_volume_df = np.log1p(volume_df)
    return log_volume_df.diff()

log_volume_full_df = compute_log_volume(volume_full_df)
log_volume_aligned_df = compute_log_volume(volume_aligned_df)

log_volume_change_full_df = compute_log_volume_change(volume_full_df)
log_volume_change_aligned_df = compute_log_volume_change(volume_aligned_df)

# ## Crypto & Macro dataset safety check
# Confirm that the macro and crypto indexes match before aligning them.

print("Crypto aligned index equals macro aligned index:", close_aligned_df.index.equals(macro_log_returns_df.index))

# ## Convert wide features to long format
#
# A long-format feature table is created so that each row corresponds to one asset on one date. This format is convenient for clustering, visualization, and later export.

def wide_to_long_feature(wide_df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Convert a wide-format feature DataFrame to long format.
    """
    long_df = (
        wide_df
        .reset_index()
        .melt(id_vars="Date", var_name="Ticker", value_name=feature_name)
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    return long_df

def build_feature_table(
    close_df: pd.DataFrame,
    return_df: pd.DataFrame,
    volatility_df: pd.DataFrame,
    corr_30_df: pd.DataFrame,
    corr_90_df: pd.DataFrame,
    idio_df: pd.DataFrame,
    log_volume_df: pd.DataFrame,
    log_volume_change_df: pd.DataFrame,
    macro_return_df: pd.DataFrame | None = None,
    mf_idio_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
    """
    Merge all feature DataFrames into one long-format feature table.
    """
    feature_tables = [
        wide_to_long_feature(close_df, "close"),
        wide_to_long_feature(return_df, "log_return"),
        wide_to_long_feature(volatility_df, f"volatility_{VOLATILITY_WINDOW}d"),
        wide_to_long_feature(corr_30_df, f"btc_corr_{CORRELATION_WINDOW}d"),
        wide_to_long_feature(corr_90_df, f"btc_corr_{LONG_CORRELATION_WINDOW}d"),
        wide_to_long_feature(idio_df, f"idio_vol_{IDIO_WINDOW}d"),
        wide_to_long_feature(log_volume_df, "log_volume"),
        wide_to_long_feature(log_volume_change_df, "log_volume_change"),
    ]

    if mf_idio_df is not None:
        feature_tables.append(
            wide_to_long_feature(mf_idio_df, f"idio_vol_mf_{MF_IDIO_WINDOW}d")
        )

    merged_df = feature_tables[0]
    for df in feature_tables[1:]:
        merged_df = merged_df.merge(df, on=["Date", "Ticker"], how="left")

    if macro_return_df is not None:
        macro_feature_df = macro_return_df.reset_index().copy()
        macro_feature_df.columns = ["Date"] + [f"{col.lower()}_log_return" for col in macro_return_df.columns]
        merged_df = merged_df.merge(macro_feature_df, on="Date", how="left")

    return merged_df

features_long_full_df = build_feature_table(
    close_df=close_full_df,
    return_df=log_returns_full_df,
    volatility_df=rolling_volatility_full_df,
    corr_30_df=rolling_corr_30_full_df,
    corr_90_df=rolling_corr_90_full_df,
    idio_df=idiosyncratic_vol_full_df,
    log_volume_df=log_volume_full_df,
    log_volume_change_df=log_volume_change_full_df,
    macro_return_df=None,
    mf_idio_df=None,
)

features_long_aligned_df = build_feature_table(
    close_df=close_aligned_df,
    return_df=log_returns_aligned_df,
    volatility_df=rolling_volatility_aligned_df,
    corr_30_df=rolling_corr_30_aligned_df,
    corr_90_df=rolling_corr_90_aligned_df,
    idio_df=idiosyncratic_vol_aligned_df,
    log_volume_df=log_volume_aligned_df,
    log_volume_change_df=log_volume_change_aligned_df,
    macro_return_df=macro_log_returns_df,
    mf_idio_df=mf_idio_vol_aligned_df,
)

features_long_aligned_df.head()

features_long_aligned_df.describe()

# ## Inspect missing values in the feature set
#
# Some missing values are expected at the beginning of the time series because rolling statistics require a minimum number of observations.

feature_quality_summary = pd.DataFrame({
    "missing_full": features_long_full_df.isna().sum(),
    "missing_aligned": features_long_aligned_df.isna().sum(),
})

feature_quality_summary

feature_quality_summary.to_csv(FEATURES_SUMMARY_OUTPUT_PATH, index=True)
print(f"Saved feature quality summary to: {FEATURES_SUMMARY_OUTPUT_PATH}")

# ## Modeling subset
#
# For some downstream methods, it may be useful to remove rows with incomplete features after the rolling windows have been applied.

features_long_aligned_complete_df = features_long_aligned_df.dropna().copy()

print("Aligned feature table shape before dropping NaNs:", features_long_aligned_df.shape)
print("Aligned feature table shape after dropping NaNs:", features_long_aligned_complete_df.shape)

# ## Save outputs
#
# The main outputs of this notebook are:
#
# - forward-filled aligned macro dataset
# - full long-format feature dataset
# - aligned long-format feature dataset
# - aligned wide-format log returns
# - feature quality summary

macro_aligned_df.to_csv(MACRO_ALIGNED_FILLED_OUTPUT_PATH, index=True)
macro_log_returns_df.to_csv(MACRO_WIDE_RETURNS_OUTPUT_PATH, index=True)
features_long_full_df.to_csv(FEATURES_LONG_FULL_OUTPUT_PATH, index=False)
features_long_aligned_df.to_csv(FEATURES_LONG_ALIGNED_OUTPUT_PATH, index=False)
log_returns_aligned_df.to_csv(FEATURES_WIDE_RETURNS_OUTPUT_PATH, index=True)

print(f"Saved forward-filled macro dataset to: {MACRO_ALIGNED_FILLED_OUTPUT_PATH}")
print(f"Saved aligned macro log returns to: {MACRO_WIDE_RETURNS_OUTPUT_PATH}")
print(f"Saved full long-format features to: {FEATURES_LONG_FULL_OUTPUT_PATH}")
print(f"Saved aligned long-format features to: {FEATURES_LONG_ALIGNED_OUTPUT_PATH}")
print(f"Saved aligned wide-format log returns to: {FEATURES_WIDE_RETURNS_OUTPUT_PATH}")
