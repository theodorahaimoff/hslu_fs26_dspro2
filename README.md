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
├── .gitignore
├── .streamlit/
│   └── config.toml         # contains Streamlit global configuration
├── data/
│   ├── raw/                # downloaded raw market data
│   └── processed/          # cleaned and feature-engineered datasets
├── notebooks/              # exploration, experiments, prototyping
├── src/  
│   ├── logs/               # application or experiment logs
│   ├── data/               # data loading and preprocessing code
│   ├── features/           # feature engineering code
│   ├── models/             # clustering, HMM, LSTM implementations
│   ├── utils/              # general helper functions used across the project
│   └── app/                # Streamlit app code
└── store/                  # local artifacts, cached outputs, MLflow or intermediate files
```

## Local Setup
Recommended Python version: 3.12

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Initialisation

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
