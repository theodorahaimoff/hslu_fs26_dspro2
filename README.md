# Investigating Market Dependence Between Bitcoin and Major Altcoins
In this project, we analyze the relationship between Bitcoin and major alternative cryptocurrencies using historical market data. The analysis focuses on five major cryptocurrencies other than Bitcoin: Ethereum, Solana, XRP, BNB, and Tron. Historical data dating back to 2017 will be utilized to examine how these assets behave over time and how strongly their price fluctuations are related to Bitcoin.

The goal of the project is to identify how strongly these cryptocurrencies depend on Bitcoin and to thereby determine their diversification potential. In general, a high correlation between an altcoin (alternative coin) and Bitcoin indicates that both assets tend to move in the same direction and including both coins in a portfolio may not significantly reduce overall risk. In contrast, cryptocurrencies with lower correlation to Bitcoin may behave more independently and therefore offer greater diversification potential.

## Data
Historical cryptocurrency market data is obtained using the  [yfinance](https://pypi.org/project/yfinance/) API.

Assets analyzed:
- BTC-USD (Bitcoin)
- ETH-USD (Ethereum)
- SOL-USD (Solana)
- XRP-USD (XRP)
- BNB-USD (Binance Coin)
- TRX-USD (Tron)

Data frequency: daily  \
Start date: 2017-01-01 \
Currency: USD

## Modeling Strategy

The project has two parallel tracks that share the feature table from notebook 02:

### Track A — Supervised BTC direction prediction (04 → 07 → 09)
Same target across all three notebooks: **sign of BTC's average log return over the next 30 trading days** (binary classification). Train/test split is chronological (first 80% train, last 20% test).

- `04`: baselines (majority class, trailing 30-day sign, logistic regression). Defines the beat-this line.
- `07`: Random Forest and Gradient Boosting classifiers with regime context from Track B as additional features. Tests how much the regime information actually helps. Best run uses threshold calibration to reduce over-prediction.
- `09`: LSTM on the same target. Tests whether a sequence model beats the static classifiers from `07`. Hyperparameters selected via `09b`.
- `09b`: hyperparameter sweep over patience, recurrent dropout, and learning rate. Run once before `09`.

### Track B — Unsupervised regime detection (05, 06)
Goal: label each trading day with a market state from cross-sectional features (volatility, BTC-correlation, dispersion across the 5 altcoins).

- `05` — K-Means. Treats each day as independent.
- `06` — HMM. Same labeling task, but explicitly models state persistence and transition probabilities.

These are alternative approaches to the same labeling problem, not a chain. The outputs of Track B feed Track A as additional features.

### How the tracks connect
The headline question is *whether altcoin diversification potential is regime-dependent*. Track B identifies the regimes while Track A uses them as context to predict BTC direction and reports per-regime accuracy. Notebook `08` answers the headline question directly: a regime-conditional portfolio backtest that measures the diversification benefit (Sharpe ratio, diversification ratio) separately in each Track B regime. A regime-aware model that beats baseline only in low-altcoin correlation regimes is itself a finding about diversification.

## Results

This project has two modeling tracks on the same six-asset universe (Bitcoin plus ETH, SOL, XRP, BNB, TRX): Track A predicts the 30-day direction of Bitcoin, and Track B detects market regimes. The project's headline question, whether altcoin diversification is regime-dependent, is answered by a regime-conditional portfolio backtest built on the Track B regimes.

**Headline finding.** The diversification benefit of holding altcoins alongside Bitcoin is not constant. It is strongly regime-dependent, an equal-weight crypto portfolio earns a Sharpe ratio of 1.21 in high-volatility regimes versus 0.50 in calm regimes, while a BTC-only portfolio shows the opposite pattern (0.26 vs 0.69). Diversification pays precisely when markets are turbulent.

This is measured by a regime-conditional portfolio backtest (notebook 08): three daily-rebalanced portfolios over the full sample (May 2020 to March 2026), split by the Track B regimes.

Full sample:

| Portfolio | Ann. return | Ann. volatility | Sharpe | Diversification ratio |
|---|---|---|---|---|
| BTC only | 23.3% | 48.3% | 0.48 | 1.00 |
| Equal weight (1/6 each) | 45.5% | 57.3% | 0.80 | 1.28 |
| BTC heavy (60% BTC / 40% altcoins) | 35.3% | 50.2% | 0.70 | 1.20 |

Sharpe ratio by regime:

| Portfolio | High-volatility regime | Calm regime |
|---|---|---|
| BTC only | 0.26 | 0.69 |
| Equal weight | 1.21 | 0.50 |
| BTC heavy | 0.86 | 0.63 |

The Sharpe ranking reverses between regimes. Equal-weight diversification beats BTC-only by a wide margin in the high-volatility regime and loses to it in the calm regime. The diversification ratio stays above 1 in both regimes (1.31 high-vol, 1.21 calm), so altcoins always reduce portfolio volatility, but the risk-adjusted payoff is concentrated in turbulent periods.

### Track A: BTC direction prediction (notebooks 04, 07, 09)

Binary task: predict the sign of Bitcoin's average log return over the next 30 trading days. Chronological 80/20 split, test period December 2024 to February 2026 (423 days). Baselines first (04), then a regime-aware Random Forest (07), then an LSTM with a lookback sweep (09, 09b).

| Model | Accuracy | F1 macro |
|---|---|---|
| Majority class (baseline) | 44.4% | 0.31 |
| Persistence, trailing-30d sign (baseline) | 48.7% | 0.48 |
| Logistic regression (baseline) | 42.1% | 0.34 |
| Random Forest, regime-aware (notebook 07) | 52.5% | 0.52 |
| LSTM, 90-day lookback (notebook 09) | 61.0% | 0.61 |

The LSTM is the strongest model, beating the hardest baseline (persistence) by 0.13 F1 and the Random Forest by 0.09 F1. Splitting its test predictions by regime shows the edge is concentrated in the high-volatility regime:

| Regime | Test days | Accuracy | F1 macro |
|---|---|---|---|
| 0, High volatility | 46 | 78.3% | 0.75 |
| 1, Calm | 377 | 58.9% | 0.59 |

The high-volatility figure rests on only 46 test days, so it is a strong lead rather than a confirmed result.

### Track B: Market regime detection (notebooks 05, 06)

A Hidden Markov Model labels every trading day as one of two market regimes. K = 2 was chosen as the principled split: BIC keeps decreasing as K grows, so it does not single out a value on its own, but K = 2 maps cleanly onto the calm-versus-volatile market view, converges reliably across random restarts, and was independently confirmed by notebook 07 over K = 3 and K = 4 for predictive value. K-Means (notebook 05) is a time-agnostic cross-check on the same labeling task, and its silhouette score is highest at K = 2.

| Regime | Label | Days | Share | Annualized BTC volatility |
|---|---|---|---|---|
| 0 | High volatility | 674 | 31% | ~65% |
| 1 | Calm / trending | 1470 | 69% | ~38% |

### Human-in-the-loop

CryptoLens is a decision-support tool, not an automated trader. The model outputs (regime label, direction signal, per-regime accuracy) are surfaced to a human in the Streamlit app alongside explicit caveats, and the human makes the allocation decision.

## Repo Layout
```bash
HSLU_FS26_DSPRO2/
├── README.md
├── requirements.txt
├── .python-version             # pins Python 3.12 for Streamlit Community Cloud
├── .gitignore
├── run_all.py                  # runs the full modeling pipeline
├── run_app.py                  # launches the CryptoLens Streamlit dashboard
├── mlflow_utils.py             # shared MLflow experiment configuration
├── .streamlit/
│   └── config.toml             # contains Streamlit global configuration
├── data/
│   ├── raw/                    # downloaded raw market data
│   ├── processed/              # cleaned and feature-engineered datasets
│   └── model_outputs/          # predictions, regime labels, cluster assignments
├── notebooks/                  # exploration, experiments, prototyping
├── scripts/                    # standalone .py exports of the notebooks, run by run_all.py
├── src/
│   ├── features/
│   │   └── daily.py            # shared daily feature aggregation used by notebooks 05, 06, 07, 09                
│   ├── utils/
│   │   └── plot_config.py      # shared matplotlib style and coin color palette                  
│   └── app/
│       ├── app.py              # Streamlit entry point, layout and plotting only
│       ├── backend.py          # data loading and preprocessing functions for the app
│       └── logo.svg            # CryptoLens logo
└── mlruns/                     # MLflow tracking directory (auto-generated, see .gitignore)
```

## Local Setup

### Windows Terminal
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Mac / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Reproduce the results

With the virtual environment active, run the full modeling pipeline:

```bash
python run_all.py
```

This runs the exported notebook scripts in `scripts/` one step at a time, from
feature engineering through the LSTM. It does not re-download market data. The terminal shows a
one-line progress entry per step. (~10min on CPU)

To re-download the raw market data from yfinance before running the pipeline:

```bash
python run_all.py --refresh_data
```

Once the pipeline has run, launch the dashboard with
`python run_app.py` (see CryptoLens section below)

## CryptoLens Streamlit App
 
CryptoLens is the interactive front end for this project. It reads the pre-computed
CSV files produced by the notebooks and presents the results through a browser-based
dashboard aimed at users without a finance or data science background.
 
All model training happens in the notebooks. The app does not re-train or re-run any
model at runtime. It only loads data from `data/processed/` and `data/model_outputs/`
and visualizes it.
 
### Architecture
 
The app is split into two files:
 
- `src/app/backend.py`: all `pd.read_csv` calls and data preprocessing logic.
  Every function that reads from disk is decorated with `@st.cache_data` so repeated
  sidebar interactions do not re-read files. Helper functions (filtering, derived
  metrics) accept DataFrames as arguments and are not cached.
- `src/app/app.py`: layout, sidebar controls, and plotting only. It imports from
  `backend.py` and never calls `pd.read_csv` directly.
This separation keeps the plotting code readable and makes it straightforward to
update a data source or preprocessing step without touching the layout.
 
### Running the app
 
Make sure the virtual environment is active and the modeling pipeline has run
 (`python run_all.py`) so the model output files exist. Then, from the
project root:
 
```bash
python run_app.py
```
 
`run_app.py` checks that the required data files are present and then starts
Streamlit. The app opens in your default browser at `http://localhost:8501`.
 
### Sidebar controls
 
| Control | Description |
|---|---|
| Cryptocurrencies | Multiselect for the five altcoins. Bitcoin is always included as the reference coin. |
| Time period | Date range picker. Affects all charts that show data over time. The minimum start date is April 2020, when all six coins were available simultaneously. |
 
### Tabs
 
**Price History**
 
Shows raw closing prices in USD for the selected coins over the chosen date range,
and a normalized chart anchored to CHF 100 at the start of the period. The
normalized view makes it easier to compare growth rates across coins that trade at
very different price levels.
 
Data source: `data/processed/crypto_wide_close_full.csv`
 
**Risk Overview**
 
Summarizes how unpredictable each coin's price has been using three views: a risk
bucket table (Low / Medium / High) based on average 30-day rolling volatility, a
rolling volatility line chart over the selected period, a maximum drawdown bar chart
showing the worst peak-to-trough drop in the full history, and a box plot of daily
log returns showing the typical range of daily price swings.
 
Data sources: `data/processed/crypto_wide_close_full.csv`,
`data/processed/crypto_features_long_aligned.csv`
 
**Coin Independence**
 
Addresses the central research question of the project: do the selected altcoins
actually move independently of Bitcoin? The tab shows an independence ratings table
derived from average 30-day BTC correlation over the full history, and a rolling
correlation chart over the selected period so users can see how the relationship has
changed over time.
 
Data source: `data/processed/crypto_features_long_aligned.csv`
 
**Market Conditions**
 
Uses the Hidden Markov Model output from notebook 06 to label each trading day as
either Low Volatility / Trending or High Volatility. The tab shows the current
market phase and how many consecutive days it has lasted, a Bitcoin price chart with
colored background bands indicating which phase was active at each point in time,
and a summary table of average daily return and average volatility per phase.
 
Data source: `data/model_outputs/hmm_regime_labels.csv`,
`data/processed/crypto_features_long_aligned.csv`
 
**Price Forecast**
 
Presents the output of the LSTM model from notebook 09. The model predicts the sign
of Bitcoin's 30-day average log return (up or down), not a price level. The tab
shows the Bitcoin price chart with colored markers indicating the model's predicted
direction at each date in the test period (December 2024 to February 2026), a
prediction accuracy chart showing which days were called correctly, and an accuracy
comparison table of the LSTM against the three baselines from notebook 04.
 
Because the LSTM was trained on Bitcoin data only, forecasts for other coins are not
available. Selecting an altcoin in the sidebar shows a note and defaults to the
Bitcoin signal.
 
Data sources: `data/model_outputs/lstm_predictions.csv`,
`data/model_outputs/baseline_predictions.csv`
 
### Files the app depends on
 
All files below are produced by running the notebooks in order. If any file is
missing, run the corresponding notebook first.
 
| File | Produced by |
|---|---|
| `data/processed/crypto_wide_close_full.csv` | Notebook 01a |
| `data/processed/crypto_features_long_aligned.csv` | Notebook 02 |
| `data/model_outputs/hmm_regime_labels.csv` | Notebook 06 |
| `data/model_outputs/lstm_predictions.csv` | Notebook 09 |
| `data/model_outputs/baseline_predictions.csv` | Notebook 04 |

## MLflow Experiment Tracking

This project uses MLflow to track model experiments across notebooks 04 to 09b.
Every time you train a model or try a different configuration, the run should be
logged with MLflow so results can be compared later.

### Setup

To view the MLflow UI after running any experiment, open a terminal in the
project root and run:

```bash
mlflow ui
```

Then open http://127.0.0.1:5000 in your browser.

### How to log a run in a modeling notebook

At the top of each modeling notebook, import the shared config:

```python
import mlflow
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().resolve().parent))
from mlflow_utils import get_or_create_experiment, EXPERIMENT_X
```

Then wrap your model code in a run block:

```python
experiment_id = get_or_create_experiment(EXPERIMENT_X)
mlflow.set_experiment(EXPERIMENT_X)

with mlflow.start_run(run_name="your_run_name_here"):
    mlflow.log_param("model_type", "your_model")
    mlflow.log_param("dataset", "aligned_6_assets_from_2020")

    # your model code goes here

    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_artifact("../data/processed/your_output.csv")
```

Each different configuration you try (for example a different number of clusters
or a different window size) should be logged as a separate run with a descriptive
`run_name` so it can be identified in the UI.

Replace `EXPERIMENT_X` with the constant that matches your notebook:

| Notebook | Experiment constant | Example params | Example metrics |
|---|---|---|---|
| 04 Baseline | `EXPERIMENT_BASELINE` | model_type, dataset, train/test period | accuracy, f1_macro |
| 05 Clustering | `EXPERIMENT_CLUSTERING` | algorithm, n_clusters, features_used | silhouette_score, inertia |
| 06 HMM | `EXPERIMENT_HMM` | n_states, covariance_type, n_iter | log_likelihood, AIC, BIC |
| 07 Classification | `EXPERIMENT_CLASSIFIERS` | model_type, feature_block, k_states | accuracy, f1_macro, precision, recall |
| 08 Diversification | `EXPERIMENT_DIVERSIFICATION` | portfolio, regime_source, date range | sharpe, ann_return, ann_vol, diversification_ratio |
| 09 LSTM | `EXPERIMENT_LSTM` | lookback, lstm_units, dropout, learning_rate | accuracy, f1_macro, precision, recall |
| 09b LSTM Sweep | `lstm_sweep` | config_label, patience, recurrent_dropout, l2_reg | accuracy, f1_macro |

## Plot Style Guide

All charts in this project use a shared visual style defined in `src/utils/plot_config.py`.
This ensures that every notebook and the Streamlit app produce consistent-looking graphs
that match the CryptoLens color scheme.

### Setup

At the top of every notebook that produces plots, add the following after your imports:

```python
import sys
import matplotlib as mpl
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
from src.utils.plot_config import COIN_COLORS, apply_plot_style

apply_plot_style()

# Re-apply background settings after seaborn initializes,
# as seaborn can override these during its own setup.
mpl.rcParams["figure.facecolor"] = "#F9F7FC"
mpl.rcParams["axes.facecolor"]   = "#FFFFFF"
mpl.rcParams["font.family"]      = "sans-serif"
```

### Coin colors

Each coin has a fixed color defined in `COIN_COLORS`. Always use these when
coloring lines, bars, or markers by coin so charts are consistent across notebooks.

```python
# Line chart
for ticker in selected_tickers:
    ax.plot(df.index, df[ticker], color=COIN_COLORS[ticker], label=ticker)

# Bar chart
ax.bar(tickers, values, color=[COIN_COLORS[t] for t in tickers])

# Seaborn chart (pass as palette)
sns.boxplot(data=df, x="Ticker", y="Value", palette=COIN_COLORS, ax=ax)
```

The current coin colors are:

| Coin | Hex |
|------|-----|
| BTC  | `#5B6EF5` |
| ETH  | `#F49D37` |
| SOL  | `#C2E812` |
| XRP  | `#549F93` |
| BNB  | `#E2C2FF` |
| TRX  | `#F76F8E` |

If you need to update a color, change it in `src/utils/plot_config.py` only.
Every notebook and the app will pick up the change automatically on the next run.

### Macro factor colors

Macro factors (DXY, VIX, Gold, SP500) are not in the shared config since they
are only used in the EDA notebook. They are defined locally in that notebook as:

```python
COLORS = {
    **COIN_COLORS,
    "DXY":   "#2E86C1",
    "VIX":   "#E74C3C",
    "Gold":  "#FFD700",
    "SP500": "#1B5E20",
}
```

### Saving figures

When using matplotlib inside Streamlit, always close the figure after rendering
to prevent memory from accumulating across reruns:

```python
fig, ax = plt.subplots()
# ... your plot code ...
st.pyplot(fig)
plt.close(fig)
```

In notebooks, use `plt.show()` as normal.

## Notes for Collaborators

- If the error `ModuleNotFound` pops up, there's a dependency issue. Either there's a mismatch of package versions or a package isn't supported by the Streamlit Python version.
- Use Conventional Commit messages when committing changes so the history remains structured and easy to read. \
  Format: `<type>: short description` \
  Common types used in this repository: 
  ```
  feat: add new functionality
  fix: bug fix
  refactor: code restructuring without behavior change
  docs: documentation changes
  chore: maintenance tasks
  ``` 
  Example: 
  ```
  feat: add data preprocessing pipeline
  fix: correct calculation
  docs: update README setup instructions
  ```
