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

## Repo Layout
```bash
HSLU_HS25_DSPRO2/
├── README.md
├── requirements.txt
├── .python-version             # pins Python 3.12 for Streamlit Community Cloud
├── .gitignore
├── mlflow_utils.py             # shared MLflow experiment configuration
├── .streamlit/
│   └── config.toml             # contains Streamlit global configuration
├── data/
│   ├── raw/                    # downloaded raw market data
│   └── processed/              # cleaned and feature-engineered datasets
├── notebooks/                  # exploration, experiments, prototyping
├── src/
│   ├── logs/                   # application or experiment logs
│   ├── data/                   # data loading and preprocessing code
│   ├── features/               # feature engineering code
│   ├── models/                 # clustering, HMM, LSTM implementations
│   ├── utils/                  # general helper functions used across the project
│   └── app/                    # Streamlit app code
├── store/                      # local artifacts, cached outputs and intermediate files
└── mlruns/                     # MLflow tracking directory (auto-generated, see .gitignore)
```

## Local Setup
Fixed Python version: 3.12

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Initialisation

## MLflow Experiment Tracking

This project uses MLflow to track model experiments across notebooks 04 to 08.
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

sys.path.append(str(Path.cwd().parent))
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
| 07 Classification | `EXPERIMENT_CLASSIFIERS` | model_type, hyperparameters, features_used | accuracy, f1_macro, precision, recall |
| 08 LSTM | `EXPERIMENT_LSTM` | window_size, lstm_units, epochs, learning_rate | train_mae, val_mae, test_mae, test_rmse |

## Notes for Collaborators

- If you make any changes to the notebooks, you can export them into a script
    ```bash
      jupyter nbconvert --to script notebooks/NAME.ipynb --output "app_backend" --output-dir=src --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["noexport"]'
    ```
    > 👉 **Note** \
    Any cells that shouldn't be exported into the backend should be tagged as `noexport`. Make sure the ones you do export are actually needed for the app backend.
- If the error `ModuleNotFound` pops up, there's a dependency issue. Either there's a mismatch of package versions or a package isn't supported by the Streamlit Python version (3.13.9).
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
