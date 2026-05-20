# src/app/backend.py
#
# Data loading and preprocessing functions for the CryptoLens Streamlit app.
# All raw pd.read_csv calls happen here. Each function returns
# a clean DataFrame ready for plotting. Functions that are pure data reads are
# cached with st.cache_data so repeated sidebar interactions do not re-read
# files from disk.
#


import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MODEL_OUTPUTS = PROJECT_ROOT / "data" / "model_outputs"


# ===========================================================================
# Tab 1: Price History
# ===========================================================================

@st.cache_data
def load_close_prices() -> pd.DataFrame:
    """
    Load wide-format daily close prices produced by notebook 01a.
    Returns a DataFrame indexed by Date with one column per coin.
    Uses the full dataset (from 2017) so Tab 1 shows the complete history
    even though modeling notebooks use the aligned dataset from April 2020.
    """
    df = pd.read_csv(
        DATA_PROCESSED / "crypto_wide_close_full.csv",
        parse_dates=["Date"],
        index_col="Date",
    )
    return df


def filter_close(
    close_df: pd.DataFrame,
    coins: list,
    date_range,
) -> pd.DataFrame:
    """
    Filter close prices to the selected coins and date range.
    date_range is the two-element list returned by st.date_input.
    Drops rows where all selected coins are NaN (e.g. before a coin launched).
    """
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1])
    cols = [c for c in coins if c in close_df.columns]
    return close_df.loc[start:end, cols].dropna(how="all")


def normalize_to_100(close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each column so that its first non-NaN value equals 100.
    Used for the CHF 100 hypothetical investment growth chart in Tab 1.
    Each coin is anchored independently so coins that launched later still
    start at 100 on their own first trading day.
    """
    normalized = close_df.copy().astype(float)
    for col in normalized.columns:
        first_idx = normalized[col].first_valid_index()
        if first_idx is not None:
            base = normalized.loc[first_idx, col]
            if base != 0:
                normalized[col] = normalized[col] / base * 100
    return normalized


# ===========================================================================
# Tab 2: Risk Overview
# ===========================================================================

@st.cache_data
def load_features_long() -> pd.DataFrame:
    """
    Load the per-asset long-format feature table produced by notebook 02.
    Returns one row per (Date, Ticker) combination. Key columns used by the
    app are: log_return, volatility_30d, and btc_corr_30d.
    """
    df = pd.read_csv(
        DATA_PROCESSED / "crypto_features_long_aligned.csv",
        parse_dates=["Date"],
    )
    return df


def compute_risk_summary(features_df: pd.DataFrame, coins: list) -> pd.DataFrame:
    """
    Compute average 30-day rolling volatility per coin and assign a risk bucket.
    Thresholds are set relative to the distribution in the aligned dataset
    (mean ~3.97%, std ~2.34% from notebook 02 describe output):
      Low:    volatility_30d < 0.030
      Medium: 0.030 to 0.050
      High:   > 0.050
    Returns a DataFrame with columns: Coin, Avg Volatility, Risk Level.
    """
    subset = features_df[features_df["Ticker"].isin(coins)].copy()
    avg_vol = (
        subset.groupby("Ticker")["volatility_30d"]
        .mean()
        .reindex(coins)
    )

    def to_bucket(v):
        if pd.isna(v):
            return "Unknown"
        if v < 0.030:
            return "Low"
        if v <= 0.050:
            return "Medium"
        return "High"

    risk_level = avg_vol.apply(to_bucket)

    result = pd.DataFrame({
        "Coin": avg_vol.index,
        "Avg Volatility (30d)": (avg_vol.values * 100).round(2),
        "Risk Level": risk_level.values,
    })
    return result.reset_index(drop=True)


def get_rolling_volatility(
    features_df: pd.DataFrame,
    coins: list,
    date_range,
) -> pd.DataFrame:
    """
    Pivot 30-day rolling volatility to wide format for the selected coins and
    date range. Returns a DataFrame indexed by Date with one column per coin.
    """
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1])
    subset = features_df[
        features_df["Ticker"].isin(coins)
        & features_df["Date"].between(start, end)
    ][["Date", "Ticker", "volatility_30d"]].dropna(subset=["volatility_30d"])
    return subset.pivot(index="Date", columns="Ticker", values="volatility_30d")


def compute_max_drawdown(close_df: pd.DataFrame, coins: list) -> pd.DataFrame:
    """
    Compute the maximum drawdown per coin over the full price history.
    Max drawdown is the largest peak-to-trough percentage decline from any
    point to any subsequent point. Returns a DataFrame indexed by Coin with
    column Max Drawdown (%).
    """
    records = []
    for coin in coins:
        if coin not in close_df.columns:
            continue
        series = close_df[coin].dropna()
        rolling_max = series.cummax()
        drawdown = (series - rolling_max) / rolling_max
        records.append({
            "Coin": coin,
            "Max Drawdown (%)": round(drawdown.min() * 100, 1),
        })
    return pd.DataFrame(records).set_index("Coin")


def get_log_returns_wide(
    features_df: pd.DataFrame,
    coins: list,
    date_range,
) -> pd.DataFrame:
    """
    Pivot daily log returns to wide format for the selected coins and date
    range. Used for the box plot of typical daily price swings in Tab 2.
    """
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1])
    subset = features_df[
        features_df["Ticker"].isin(coins)
        & features_df["Date"].between(start, end)
    ][["Date", "Ticker", "log_return"]].dropna(subset=["log_return"])
    return subset.pivot(index="Date", columns="Ticker", values="log_return")


# ===========================================================================
# Tab 3: Coin Independence
# ===========================================================================

def get_coin_independence_table(features_df: pd.DataFrame, coins: list) -> pd.DataFrame:
    """
    Compute average 30-day BTC correlation per coin and assign an independence
    label. BTC is excluded because its correlation with itself is always 1.
    Thresholds:
      Moves with Bitcoin:   avg btc_corr_30d >= 0.80
      Somewhat Independent: avg btc_corr_30d >= 0.50
      Acts Independently:   avg btc_corr_30d < 0.50
    Returns a DataFrame with columns: Coin, Avg BTC Correlation, Independence.
    """
    altcoins = [c for c in coins if c != "BTC"]
    subset = features_df[features_df["Ticker"].isin(altcoins)]
    avg_corr = (
        subset.groupby("Ticker")["btc_corr_30d"]
        .mean()
        .reindex(altcoins)
    )

    def to_label(v):
        if pd.isna(v):
            return "Unknown"
        if v >= 0.80:
            return "Moves with Bitcoin"
        if v >= 0.50:
            return "Somewhat Independent"
        return "Acts Independently"

    independence = avg_corr.apply(to_label)
    return pd.DataFrame({
        "Coin": avg_corr.index,
        "Avg BTC Correlation": avg_corr.round(3).values,
        "Independence": independence.values,
    }).reset_index(drop=True)


def get_rolling_btc_correlation(
    features_df: pd.DataFrame,
    coins: list,
    date_range,
) -> pd.DataFrame:
    """
    Pivot 30-day rolling BTC correlation to wide format as a percentage.
    BTC is excluded from the output (correlation with itself is always 100%).
    Returns a DataFrame indexed by Date with one column per altcoin.
    """
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1])
    altcoins = [c for c in coins if c != "BTC"]
    subset = features_df[
        features_df["Ticker"].isin(altcoins)
        & features_df["Date"].between(start, end)
    ][["Date", "Ticker", "btc_corr_30d"]].dropna(subset=["btc_corr_30d"])
    wide = subset.pivot(index="Date", columns="Ticker", values="btc_corr_30d")
    return wide * 100


# ===========================================================================
# Tab 4: Market Conditions
# ===========================================================================

@st.cache_data
def load_hmm_labels() -> pd.DataFrame:
    """
    Load HMM regime labels from notebook 06 (hmm_regime_labels.csv).
    Returns a DataFrame indexed by Date with columns:
    hmm_regime (0 or 1), hmm_prob_state_0, hmm_prob_state_1.
    K=2 was the BIC-selected final configuration in notebook 06.
    """
    df = pd.read_csv(
        DATA_MODEL_OUTPUTS / "hmm_regime_labels.csv",
        parse_dates=["Date"],
        index_col="Date",
    )
    return df


def get_regime_label_map(hmm_df: pd.DataFrame, features_df: pd.DataFrame) -> dict:
    """
    Assign plain-language names to HMM regimes by comparing their average
    volatility. The regime with higher average volatility is labeled
    'High Volatility' and the other 'Low Volatility / Trending'.
    This avoids hardcoding which numeric state corresponds to which market
    condition, since HMM state IDs are arbitrary and can flip between runs.
    """
    daily_vol = (
        features_df.groupby("Date")["volatility_30d"]
        .mean()
        .rename("vol")
    )
    merged = hmm_df[["hmm_regime"]].join(daily_vol, how="inner")
    avg_vol_per_regime = merged.groupby("hmm_regime")["vol"].mean()
    high_vol_state = int(avg_vol_per_regime.idxmax())
    low_vol_state = int(avg_vol_per_regime.idxmin())
    return {
        high_vol_state: "High Volatility",
        low_vol_state: "Low Volatility / Trending",
    }


def get_current_regime_info(hmm_df: pd.DataFrame, label_map: dict) -> dict:
    """
    Return the current market phase label and how many consecutive days the
    market has been in that phase (streak from the most recent date backwards).
    """
    latest_state = int(hmm_df["hmm_regime"].iloc[-1])
    label = label_map.get(latest_state, f"Regime {latest_state}")

    regimes = hmm_df["hmm_regime"].values
    streak = 1
    for i in range(len(regimes) - 2, -1, -1):
        if int(regimes[i]) == latest_state:
            streak += 1
        else:
            break

    return {"label": label, "duration_days": streak}


def get_regime_summary_table(
    hmm_df: pd.DataFrame,
    features_df: pd.DataFrame,
    label_map: dict,
) -> pd.DataFrame:
    """
    Compute average daily return and average volatility per HMM regime over
    the full history. Returns a DataFrame with human-readable regime names
    as the index and percentage columns for display with st.dataframe.
    """
    daily_stats = features_df.groupby("Date").agg(
        mean_log_return=("log_return", "mean"),
        mean_volatility=("volatility_30d", "mean"),
    )
    merged = hmm_df[["hmm_regime"]].join(daily_stats, how="inner")
    summary = merged.groupby("hmm_regime").agg(
        avg_daily_return=("mean_log_return", "mean"),
        avg_volatility=("mean_volatility", "mean"),
        trading_days=("mean_log_return", "count"),
    )
    summary.index = summary.index.map(
        lambda x: label_map.get(int(x), f"Regime {x}")
    )
    summary["avg_daily_return"] = (summary["avg_daily_return"] * 100).round(3)
    summary["avg_volatility"] = (summary["avg_volatility"] * 100).round(3)
    summary.columns = ["Avg Daily Return (%)", "Avg Volatility (%)", "Trading Days"]
    return summary


# ===========================================================================
# Tab 5: Price Forecast
# ===========================================================================

@st.cache_data
def load_lstm_predictions() -> pd.DataFrame:
    """
    Load LSTM test-set predictions from notebook 09 (lstm_predictions.csv).
    Columns: pred (0/1 direction), proba (probability BTC goes up),
    regime (HMM state at the time of prediction).
    Index: Date, covering the test period Dec 2024 to Feb 2026, BTC only.

    Important: this model predicts the sign of BTC's 30-day average log
    return, not a price level. The forecast tab shows this direction signal
    rather than a price line.
    """
    df = pd.read_csv(
        DATA_MODEL_OUTPUTS / "lstm_predictions.csv",
        parse_dates=["Date"],
        index_col="Date",
    )
    return df


@st.cache_data
def load_baseline_predictions() -> pd.DataFrame:
    """
    Load baseline model predictions from notebook 04 (baseline_predictions.csv).
    Columns: y_true, majority_class, persistence_momentum, logistic_regression.
    Index: Date (same test period as the LSTM).
    """
    df = pd.read_csv(
        DATA_MODEL_OUTPUTS / "baseline_predictions.csv",
        parse_dates=["Date"],
        index_col="Date",
    )
    return df


def get_btc_with_signal(
    close_df: pd.DataFrame,
    lstm_df: pd.DataFrame,
) -> tuple:
    """
    Return BTC close prices and the aligned LSTM direction signal for the
    test period. Used to plot the historical price with overlaid up/down
    markers showing what the model predicted at each date.

    Returns:
        btc_full: full BTC close price Series (for the historical line)
        btc_test: BTC close price Series restricted to the test period
        signal: lstm_df['pred'] aligned to the test period dates
        proba: lstm_df['proba'] aligned to the test period dates
    """
    btc_full = close_df["BTC"].dropna()
    test_dates = lstm_df.index
    btc_test = btc_full.reindex(test_dates).dropna()
    signal = lstm_df["pred"].reindex(btc_test.index)
    proba = lstm_df["proba"].reindex(btc_test.index)
    return btc_full, btc_test, signal, proba


def get_forecast_accuracy_table(
    lstm_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare LSTM direction accuracy against the three baselines on the shared
    test period. Returns a DataFrame with one row per model and columns:
    Model, Accuracy (%), F1 Macro. F1 macro is used because the test set has
    slightly unequal class balance (235 down vs 188 up days).
    """
    y_true = baseline_df["y_true"]

    baseline_cols = {
        "Majority Class": "majority_class",
        "Persistence / Momentum": "persistence_momentum",
        "Logistic Regression": "logistic_regression",
    }

    rows = []
    for display_name, col in baseline_cols.items():
        preds = baseline_df[col].reindex(y_true.index)
        rows.append({
            "Model": display_name,
            "Accuracy (%)": round(accuracy_score(y_true, preds) * 100, 1),
            "F1 Macro": round(f1_score(y_true, preds, average="macro"), 3),
        })

    # align LSTM predictions to the baseline test dates
    lstm_preds = lstm_df["pred"].reindex(y_true.index).dropna().astype(int)
    y_true_aligned = y_true.reindex(lstm_preds.index)
    rows.append({
        "Model": "LSTM (L=90, regime-aware)",
        "Accuracy (%)": round(accuracy_score(y_true_aligned, lstm_preds) * 100, 1),
        "F1 Macro": round(f1_score(y_true_aligned, lstm_preds, average="macro"), 3),
    })

    return pd.DataFrame(rows).set_index("Model")


def get_forecast_accuracy_by_regime(
    lstm_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Break the LSTM's direction accuracy down by HMM market regime. The model's
    predictive edge is not spread evenly across market conditions — it
    concentrates in one regime — and this split is what exposes that.

    Returns a DataFrame indexed by regime (int) with columns:
    n, accuracy (%), f1_macro.

    The calm regime has a small test sample (~46 days), so its metrics carry
    wide confidence intervals; callers should surface that caveat (see 09).
    """
    y_true = baseline_df["y_true"]
    aligned = pd.DataFrame({
        "y_true": y_true,
        "pred":   lstm_df["pred"].reindex(y_true.index),
        "regime": lstm_df["regime"].reindex(y_true.index),
    }).dropna()
    aligned["pred"]   = aligned["pred"].astype(int)
    aligned["regime"] = aligned["regime"].astype(int)

    rows = []
    for regime, sub in aligned.groupby("regime"):
        rows.append({
            "regime":   int(regime),
            "n":        len(sub),
            "accuracy": round(accuracy_score(sub["y_true"], sub["pred"]) * 100, 1),
            "f1_macro": round(f1_score(sub["y_true"], sub["pred"], average="macro"), 3),
        })
    return pd.DataFrame(rows).set_index("regime").sort_index()