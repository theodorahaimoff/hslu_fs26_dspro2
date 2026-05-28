# run_app.py
#
# Starts the CryptoLens Streamlit dashboard in a local browser window.
#
# Usage
# -----
#   python run_app.py
#
# Before running this script, the full modeling pipeline must have been
# executed at least once so that the required data and model output files
# exist. If any file is missing the script prints a clear error message
# and exits with a non-zero code rather than letting Streamlit fail with
# a confusing import error mid-startup.
#
# If you have not run the pipeline yet:
#   python run_all.py
#
# Required files
# --------------
# The dashboard needs five files produced by the pipeline:
#
#   data/processed/crypto_wide_close_full.csv
#       Daily closing prices for all six coins, wide format (one column per coin).
#       Produced by notebook 01a (data collection and cleaning).
#
#   data/processed/crypto_features_long_aligned.csv
#       Engineered features (rolling volatility, correlation, log returns, etc.)
#       in long format (one row per coin per date). Produced by notebook 02.
#
#   data/model_outputs/hmm_regime_labels.csv
#       HMM market regime label assigned to each trading day (0 or 1).
#       Produced by notebook 06.
#
#   data/model_outputs/lstm_predictions.csv
#       LSTM direction predictions and probabilities for the test period.
#       Produced by notebook 09.
#
#   data/model_outputs/baseline_predictions.csv
#       Baseline model predictions and ground-truth labels used for comparison.
#       Produced by notebook 04.

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP = REPO_ROOT / "src" / "app" / "app.py"

# Files the dashboard cannot start without. Each entry maps to a specific
# pipeline step; see the header comment above for details.
REQUIRED = [
    REPO_ROOT / "data" / "processed" / "crypto_wide_close_full.csv",
    REPO_ROOT / "data" / "processed" / "crypto_features_long_aligned.csv",
    REPO_ROOT / "data" / "model_outputs" / "hmm_regime_labels.csv",
    REPO_ROOT / "data" / "model_outputs" / "lstm_predictions.csv",
    REPO_ROOT / "data" / "model_outputs" / "baseline_predictions.csv",
]


def main() -> None:
    # Check all required files exist before handing off to Streamlit.
    # Failing early with a descriptive message is much easier to debug
    # than a FileNotFoundError buried inside a backend import.
    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        print("Cannot start the app. These required files are missing:")
        for p in missing:
            print(f"  {p.relative_to(REPO_ROOT)}")
        print()
        print("Run the pipeline first:  python run_all.py")
        sys.exit(1)

    # Run from REPO_ROOT so that relative imports inside src/ resolve correctly.
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP)],
        cwd=str(REPO_ROOT),
    )


if __name__ == "__main__":
    main()