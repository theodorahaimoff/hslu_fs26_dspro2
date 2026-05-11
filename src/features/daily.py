"""Shared daily-feature aggregation used by 05, 06, 07, and 09."""

import pandas as pd


def build_daily_market_features(features_long: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-asset features into daily market-state summaries.

    Returns a DataFrame indexed by Date with one row per trading day and
    nine columns: cross-sectional return statistics, average volatility,
    average BTC correlation and idiosyncratic volatility (altcoins only,
    BTC excluded), and the four macro log returns.
    """
    altcoin_df = features_long[features_long["Ticker"] != "BTC"]

    return_stats = features_long.groupby("Date")["log_return"].agg(
        mean_log_return="mean",
        return_dispersion="std",
    )
    mean_vol = features_long.groupby("Date")["volatility_30d"].mean().rename("mean_volatility")
    mean_btc_corr = altcoin_df.groupby("Date")["btc_corr_30d"].mean().rename("mean_btc_corr")
    mean_idio = altcoin_df.groupby("Date")["idio_vol_30d"].mean().rename("mean_idio_vol")

    macro_cols = ["dxy_log_return", "gold_log_return", "sp500_log_return", "vix_log_return"]
    macro_daily = features_long.groupby("Date")[macro_cols].first()

    return pd.concat([return_stats, mean_vol, mean_btc_corr, mean_idio, macro_daily], axis=1)
