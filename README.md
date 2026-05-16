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
- `08`: portfolio backtest broken down by HMM regime. Three portfolios (BTC-only, equal-weight, BTC-heavy) compared on Sharpe ratio and diversification ratio across calm and volatile market periods.
- `09`: LSTM on the same target. Tests whether a sequence model beats the static classifiers from `07`. Hyperparameters selected via `09b`.
- `09b`: hyperparameter sweep over patience, recurrent dropout, and learning rate. Run once before `09`.

### Track B — Unsupervised regime detection (05, 06)
Goal: label each trading day with a market state from cross-sectional features (volatility, BTC-correlation, dispersion across the 5 altcoins).

- `05` — K-Means. Treats each day as independent.
- `06` — HMM. Same labeling task, but explicitly models state persistence and transition probabilities.

These are alternative approaches to the same labeling problem, not a chain. The outputs of Track B feed Track A as additional features.

### How the tracks connect
The headline question is *whether altcoin diversification potential is regime-dependent*. Track B identifies the regimes while Track A uses them as context to predict BTC direction and reports per-regime accuracy. A regime-aware model that beats baseline only in low-altcoin-correlation regimes is itself a finding about diversification, not a failure.

## Repo Layout
```bash
HSLU_FS26_DSPRO2/
├── README.md
├── requirements.txt
├── .python-version             # pins Python 3.12 for Streamlit Community Cloud
├── .gitignore
├── mlflow_utils.py             # shared MLflow experiment configuration
├── .streamlit/
│   └── config.toml             # contains Streamlit global configuration
├── data/
│   ├── raw/                    # downloaded raw market data
│   ├── processed/              # cleaned and feature-engineered datasets
│   └── model_outputs/          # predictions, regime labels, cluster assignments
├── notebooks/                  # exploration, experiments, prototyping
├── src/
│   ├── models/                 # clustering, HMM, LSTM implementations
│   ├── utils/                  # general helper functions used across the project
│   └── app/                    # Streamlit app code
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

- If you make any changes to the notebooks, you can export them into a script
    ```bash
      jupyter nbconvert --to script notebooks/NAME.ipynb --output "app_backend" --output-dir=src --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["noexport"]'
    ```
    > 👉 **Note** \
    Any cells that shouldn't be exported into the backend should be tagged as `noexport`. Make sure the ones you do export are actually needed for the app backend.
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
