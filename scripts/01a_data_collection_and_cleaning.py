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


# # 01 Data Collection and Cleaning
#
# This notebook downloads historical daily cryptocurrency market data, performs initial cleaning, checks basic data quality, and stores both a full historical dataset and an aligned modeling dataset.
#
# Two dataset versions are created:
#
# - Full dataset: preserves the full history from 2017 onward, even if some assets were not yet available during the early period
# - Aligned dataset: only retains dates for which all selected assets have data
#
# This distinction is important because some assets, such as Solana, have a shorter trading history than Bitcoin and the other selected altcoins. The full dataset is useful for exploratory analysis, while the aligned dataset is used for methods that require complete feature vectors, such as clustering, HMMs, and LSTM models.

from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

# ## Configuration
#
# This section defines the project paths, selected assets, and download settings.
#
# ### Output file naming convention
# Processed datasets follow the pattern: `dataset_format_metric_scope`
#
# | Chunk | Meaning                                                                 |
# | --- |-------------------------------------------------------------------------|
# | **dataset** | data source used in this project (e.g. crypto, macro)                   |
# | **format** |                                                                         |
# | `long` | one row per asset and date                                              |
# | `wide` | one column per asset                                                    |
# | **metric** |                                                                         |
# | `close` | closing price                                                           |
# | `volume` | traded volume                                                           |
# | **scope** |                                                                         |
# | `raw` | direct download from the data source                                    |
# | `clean` | cleaned long-format dataset                                             |
# | `full` | full historical dataset (may contain missing values for younger assets) |
# | `aligned` | dataset restricted to dates where all assets are available              |
#
# Additional files store data quality summaries and asset availability diagnostics.

# Project paths
PROJECT_ROOT = Path.cwd().resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Asset selection
TICKERS: Dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "BNB": "BNB-USD",
    "TRX": "TRX-USD",
}

START_DATE = "2017-01-01"
END_DATE = None
INTERVAL = "1d"

# Raw data
RAW_OUTPUT_PATH = DATA_RAW_DIR / "crypto_long_raw.csv"

# Clean datasets
CLEAN_LONG_OUTPUT_PATH = DATA_PROCESSED_DIR / "crypto_long_clean.csv"

# Full historical wide datasets
FULL_CLOSE_OUTPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_close_full.csv"
FULL_VOLUME_OUTPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_volume_full.csv"

# Aligned modeling datasets
ALIGNED_CLOSE_OUTPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_close_aligned.csv"
ALIGNED_VOLUME_OUTPUT_PATH = DATA_PROCESSED_DIR / "crypto_wide_volume_aligned.csv"

# Diagnostics
QUALITY_SUMMARY_OUTPUT_PATH = DATA_PROCESSED_DIR / "data_quality_summary.csv"
AVAILABILITY_MATRIX_OUTPUT_PATH = DATA_PROCESSED_DIR / "asset_availability_matrix.csv"

# ## Download raw market data
#
# The following function downloads daily **OHLCV** data for each selected asset and stores the result in a consistent tabular format.
#
# | Indicator | Meaning                                            |
# | --- |----------------------------------------------------|
# | **Open** | The first traded price at the beginning of the day |
# | **High** | The highest traded price for the day               |
# | **Low** | The lowest traded price for the day                |
# | **Close** | The final traded price at the end of the day       |
# | **Volume** | The number of coins traded in the day              |

def download_single_ticker(ticker: str, symbol: str, start_date: str, end_date: str | None, interval: str) -> pd.DataFrame:
    """
    Download daily market data for a single ticker from yfinance and return a cleaned raw DataFrame.

    Parameters
    ----------
    ticker : str
        Internal short name, e.g. BTC.
    symbol : str
        Yahoo Finance ticker symbol, e.g. BTC-USD.
    start_date : str
        Download start date in YYYY-MM-DD format.
    end_date : str | None
        Download end date in YYYY-MM-DD format. If None, yfinance uses the latest available date.
    interval : str
        Data interval, e.g. '1d'.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per date and standard OHLCV columns.
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

    # Flatten possible multi-index columns returned by yfinance
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    expected_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_columns = [col for col in expected_columns if col in df.columns]
    df = df[available_columns].copy()

    df["Ticker"] = ticker
    df["Symbol"] = symbol
    df["Date"] = pd.to_datetime(df["Date"])

    return df

raw_frames = []

for ticker, symbol in TICKERS.items():
    df_ticker = download_single_ticker(
        ticker=ticker,
        symbol=symbol,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
    )
    raw_frames.append(df_ticker)

raw_df = pd.concat(raw_frames, ignore_index=True)
raw_df = raw_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

raw_df.head()

# ## Save raw combined dataset
#
# The raw combined dataset is stored before further cleaning so that the original download can be reproduced and inspected later.

raw_df.to_csv(RAW_OUTPUT_PATH, index=False)
print(f"Saved raw dataset to: {RAW_OUTPUT_PATH}")

# ## Basic raw data inspection
#
# This section checks the structure and completeness of the downloaded dataset.

print(raw_df.info())
print()
print(raw_df.describe(include="all"))

raw_df.isna().sum()

raw_df.duplicated(subset=["Ticker", "Date"]).sum()

# ## Data quality summary by asset
#
# The following summary provides a first overview of the available history and missing values for each asset.

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

if "Adj Close" in raw_df.columns:
    summary_dict["missing_adj_close"] = ("Adj Close", lambda s: s.isna().sum())

quality_summary = (
    raw_df
    .groupby("Ticker")
    .agg(**summary_dict)
    .reset_index()
)

quality_summary

quality_summary.to_csv(QUALITY_SUMMARY_OUTPUT_PATH, index=False)
print(f"Saved quality summary to: {QUALITY_SUMMARY_OUTPUT_PATH}")

# ## Keep relevant columns
#
# The most relevant columns for our project are date, close price, and volume. The full OHLCV data remains available in the raw export if needed later.

clean_long_df = raw_df[["Date", "Ticker", "Symbol", "Close", "Volume"]].copy()
clean_long_df = clean_long_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

clean_long_df.head()

# ## Create wide-format datasets
#
# Two wide-format versions are created, a full and an aligned dataset.

close_wide_full_df = clean_long_df.pivot(index="Date", columns="Ticker", values="Close").sort_index()
volume_wide_full_df = clean_long_df.pivot(index="Date", columns="Ticker", values="Volume").sort_index()

print("Full close shape:", close_wide_full_df.shape)
print("Full volume shape:", volume_wide_full_df.shape)

# ### Remove rows without any available data
#
# Some of the earliest dates returned by the download may contain no observations for any of the selected assets. These rows do not contain useful information and are removed.
#
# Rows are only dropped when **all assets are missing**, so the full dataset still preserves the early history of assets that started trading earlier than others.

close_wide_full_df = close_wide_full_df.dropna(how="all")
volume_wide_full_df = volume_wide_full_df.dropna(how="all")

# ## Asset availability over time
#
# The following matrix indicates whether price data is available for each asset on each date. This helps identify the start date of younger assets such as Solana.

availability_matrix = close_wide_full_df.notna().astype(int)
availability_matrix.head()

availability_matrix.to_csv(AVAILABILITY_MATRIX_OUTPUT_PATH, index=True)
print(f"Saved availability matrix to: {AVAILABILITY_MATRIX_OUTPUT_PATH}")

asset_start_dates = close_wide_full_df.apply(lambda col: col.first_valid_index()).to_frame(name="first_valid_date")
asset_end_dates = close_wide_full_df.apply(lambda col: col.last_valid_index()).to_frame(name="last_valid_date")

availability_summary = asset_start_dates.join(asset_end_dates)
availability_summary

# ## Create aligned modeling dataset
#
# The aligned dataset only keeps dates where all selected assets have valid observations. This is the version used later for modeling steps that require complete feature vectors.

close_wide_aligned_df = close_wide_full_df.dropna(how="any").copy()
volume_wide_aligned_df = volume_wide_full_df.loc[close_wide_aligned_df.index].copy()

print("Aligned close shape:", close_wide_aligned_df.shape)
print("Aligned volume shape:", volume_wide_aligned_df.shape)

print("Aligned start date:", close_wide_aligned_df.index.min())
print("Aligned end date:", close_wide_aligned_df.index.max())

# ## Final checks
#
# These checks confirm that the cleaned aligned datasets no longer contain missing values.

print("Missing values in full close dataset:")
print(close_wide_full_df.isna().sum())

print()
print("Missing values in aligned close dataset:")
print(close_wide_aligned_df.isna().sum())

print("Missing values in full volume dataset:")
print(volume_wide_full_df.isna().sum())

print()
print("Missing values in aligned volume dataset:")
print(volume_wide_aligned_df.isna().sum())

# ## Save cleaned outputs
#
# The cleaned outputs are saved in both long and wide format for later notebooks.

clean_long_df.to_csv(CLEAN_LONG_OUTPUT_PATH, index=False)

close_wide_full_df.to_csv(FULL_CLOSE_OUTPUT_PATH, index=True)
volume_wide_full_df.to_csv(FULL_VOLUME_OUTPUT_PATH, index=True)

close_wide_aligned_df.to_csv(ALIGNED_CLOSE_OUTPUT_PATH, index=True)
volume_wide_aligned_df.to_csv(ALIGNED_VOLUME_OUTPUT_PATH, index=True)

print(f"Saved clean long dataset to: {CLEAN_LONG_OUTPUT_PATH}")
print(f"Saved full close dataset to: {FULL_CLOSE_OUTPUT_PATH}")
print(f"Saved full volume dataset to: {FULL_VOLUME_OUTPUT_PATH}")
print(f"Saved aligned close dataset to: {ALIGNED_CLOSE_OUTPUT_PATH}")
print(f"Saved aligned volume dataset to: {ALIGNED_VOLUME_OUTPUT_PATH}")
