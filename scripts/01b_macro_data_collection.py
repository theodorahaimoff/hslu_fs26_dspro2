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


# # 01 Macro Data Collection
#
# This notebook downloads a small set of additional market factors that may help interpret cryptocurrency market behavior in later project stages.
#
# | Macro Factor | Meaning          |
# | --- |------------------|
# | **DXY** | US dollar index  |
# | **VIX** | Volatility index |
# | **Gold** | Gold futures (safe haven) |
# | **SP500** | S&P 500 index (risk appetite) |
#
# These series are downloaded, cleaned, aligned to the crypto dataset date range, and saved for later use in regime interpretation and possible model extensions.

from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

# ## Configuration
#
# This section defines the project paths, selected macro tickers, and output locations.

PROJECT_ROOT = Path.cwd().resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MACRO_TICKERS: Dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "Gold": "GC=F",
    "SP500": "^GSPC",
}

START_DATE = "2017-01-01"
END_DATE = None
INTERVAL = "1d"

# Reference dataset for alignment
CRYPTO_ALIGNED_CLOSE_INPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_close_aligned.csv"

# Raw macro dataset
MACRO_RAW_OUTPUT_PATH = DATA_RAW_DIR / "macro_long_raw.csv"

# Clean macro datasets
MACRO_WIDE_FULL_OUTPUT_PATH = DATA_PROCESSED_DIR / "macro_wide_close_full.csv"
MACRO_WIDE_ALIGNED_OUTPUT_PATH = DATA_PROCESSED_DIR / "macro_wide_close_aligned.csv"

# Macro datasets
MACRO_QUALITY_SUMMARY_OUTPUT_PATH = DATA_PROCESSED_DIR / "macro_data_quality_summary.csv"

# ## Download raw macro data
#
# The following function downloads daily market data for a single macro factor and returns it in a consistent tabular structure.

def download_single_series(
    ticker: str,
    symbol: str,
    start_date: str,
    end_date: str | None,
    interval: str,
    ) -> pd.DataFrame:
    """
    Download daily market data for a single ticker from yfinance.

    Parameters
    ----------
    ticker : str
        Internal ticker name used in the project, e.g. DXY.
    symbol : str
        Yahoo Finance symbol.
    start_date : str
        Download start date in YYYY-MM-DD format.
    end_date : str | None
        Download end date in YYYY-MM-DD format. If None, the latest available date is used.
    interval : str
        Data interval, e.g. '1d'.

    Returns
    -------
    pd.DataFrame
        Raw market data with one row per date.
    """
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({symbol}).")

    df = df.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    expected_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_columns = [col for col in expected_columns if col in df.columns]

    df = df[available_columns].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = ticker
    df["Symbol"] = symbol

    return df

raw_frames = []

for ticker, symbol in MACRO_TICKERS.items():
    df_ticker = download_single_series(
        ticker=ticker,
        symbol=symbol,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
    )
    raw_frames.append(df_ticker)

macro_raw_df = pd.concat(raw_frames, ignore_index=True)
macro_raw_df = macro_raw_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

macro_raw_df.head()

# ## Save raw combined dataset
#
# The raw combined dataset is stored before further transformation so the original download remains available for inspection and reproducibility.

macro_raw_df.to_csv(MACRO_RAW_OUTPUT_PATH, index=False)
print(f"Saved raw macro dataset to: {MACRO_RAW_OUTPUT_PATH}")

# ## Basic inspection

print(macro_raw_df.info())

macro_raw_df.isna().sum()

macro_raw_df.duplicated(subset=["Ticker", "Date"]).sum()

# ## Data quality summary by factor
#
# This summary shows the available history and missing values for each selected macro factor.

summary_dict = {
    "symbol": ("Symbol", "first"),
    "first_date": ("Date", "min"),
    "last_date": ("Date", "max"),
    "n_rows": ("Date", "count"),
    "missing_open": ("Open", lambda s: s.isna().sum()),
    "missing_high": ("High", lambda s: s.isna().sum()),
    "missing_low": ("Low", lambda s: s.isna().sum()),
    "missing_close": ("Close", lambda s: s.isna().sum()),
    "missing_volume": ("Volume", lambda s: s.isna().sum()),
}

if "Adj Close" in macro_raw_df.columns:
    summary_dict["missing_adj_close"] = ("Adj Close", lambda s: s.isna().sum())

macro_quality_summary = (
    macro_raw_df
    .groupby("Ticker")
    .agg(**summary_dict)
    .reset_index()
)

macro_quality_summary

macro_quality_summary.to_csv(MACRO_QUALITY_SUMMARY_OUTPUT_PATH, index=False)
print(f"Saved macro quality summary to: {MACRO_QUALITY_SUMMARY_OUTPUT_PATH}")

# ## Create wide-format close dataset
#
# For later use, the macro series are stored in wide format with one column per factor.

macro_wide_full_df = (
    macro_raw_df
    .pivot(index="Date", columns="Ticker", values="Close")
    .sort_index()
)

macro_wide_full_df = macro_wide_full_df.dropna(how="all")

print("Macro full shape:", macro_wide_full_df.shape)
macro_wide_full_df.head()

# ## Align macro data to the crypto modeling period
#
# The aligned macro dataset is restricted to the date range of the aligned crypto dataset so that both can be merged later without additional date filtering.

crypto_aligned_close_df = pd.read_csv(
    CRYPTO_ALIGNED_CLOSE_INPUT_PATH,
    parse_dates=["Date"],
    index_col="Date",
)

print("Crypto aligned start date:", crypto_aligned_close_df.index.min())
print("Crypto aligned end date:", crypto_aligned_close_df.index.max())

macro_wide_aligned_df = macro_wide_full_df.loc[
    crypto_aligned_close_df.index.min():crypto_aligned_close_df.index.max()
].copy()

print("Macro aligned shape before reindex:", macro_wide_aligned_df.shape)

# ## Reindex to the crypto aligned dates
#
# The macro series may not naturally have exactly the same date index as the crypto dataset. To ensure compatibility, the macro data is reindexed to the aligned crypto dates.

macro_wide_aligned_df = macro_wide_aligned_df.reindex(crypto_aligned_close_df.index)

print("Macro aligned shape after reindex:", macro_wide_aligned_df.shape)
macro_wide_aligned_df.head()

# ## Missing values after alignment
#
# Some missing values may appear because macro series and crypto series do not always share exactly the same calendar. This is expected and should be documented before later use.

print("Missing values in macro full dataset:")
print(macro_wide_full_df.isna().sum())

print()
print("Missing values in macro aligned dataset:")
print(macro_wide_aligned_df.isna().sum())

# <div class="alert alert-block alert-info">
# These missing values mainly occur on weekends and market holidays, since macro factors follow trading calendars while cryptocurrencies trade continuously.
# </div>

# ## Save cleaned outputs
#
# Two versions are stored:
#
# - full macro dataset with the complete historical range
# - aligned macro dataset restricted to the crypto modeling period

macro_wide_full_df.to_csv(MACRO_WIDE_FULL_OUTPUT_PATH, index=True)
macro_wide_aligned_df.to_csv(MACRO_WIDE_ALIGNED_OUTPUT_PATH, index=True)

print(f"Saved macro full dataset to: {MACRO_WIDE_FULL_OUTPUT_PATH}")
print(f"Saved macro aligned dataset to: {MACRO_WIDE_ALIGNED_OUTPUT_PATH}")
