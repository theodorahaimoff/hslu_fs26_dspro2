# run_all.py
#
# Runs the full CryptoLens modeling pipeline from start to finish.
#
# Usage
# -----
#   python run_all.py                   reproduce all results from committed data
#   python run_all.py --refresh_data    also re-download raw data first (01a, 01b)
#
# What this script does
# ---------------------
# Each notebook in the project has been exported as a plain Python script
# and saved in the scripts/ folder. This script executes them in the correct
# dependency order, printing a one-line status update per step.
#
# If a step fails, the pipeline stops immediately and prints the last 20 lines
# of that step's log file so you can diagnose the error without searching
# through individual log files.
#
# Output locations
# ----------------
#   data/processed/      intermediate datasets (features, aligned prices)
#   data/model_outputs/  final model artifacts (HMM labels, LSTM predictions)
#   mlruns/              MLflow experiment tracking (view with: mlflow ui)
#   logs/                one .log file per step with full stdout and stderr
#
# Step overview
# -------------
# Data steps (only run with --refresh_data):
#   01a  data_collection_and_cleaning   download and clean raw OHLCV data
#   01b  macro_data_collection          download macro indicators (VIX, DXY, etc.)
#
# Model steps (always run):
#   02   feature_engineering            compute rolling vol, correlation, log returns
#   04   baseline                       fit and evaluate simple baseline classifiers
#   05   clustering                     k-Means and GMM coin grouping
#   06   hmm                            Hidden Markov Model regime detection
#   07   classification                 supervised diversification classifiers
#   08   diversification                diversification class assignment and analysis
#   09b  lstm_sweep                     hyperparameter sweep to select best LSTM config
#   09   lstm                           train final LSTM and generate test predictions
#
# Dependencies between steps
# --------------------------
#   01a, 01b --> 02 --> 04, 05, 06, 07, 08
#                   --> 09b --> 09
#
# Note: steps 04-08 are independent of each other and can be run in any
# order after 02. Steps 09b and 09 must run in that order.

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent
SCRIPTS_DIR  = REPO_ROOT / "scripts"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
LOGS_DIR     = REPO_ROOT / "logs"

# Steps that re-download raw data from yfinance. Only included in the run
# order when --refresh_data is passed, because the committed CSV files are
# already clean and re-downloading takes time and requires internet access.
DATA_STEPS = [
    "01a_data_collection_and_cleaning",
    "01b_macro_data_collection",
]

# Core modeling steps. Always run. Order matters: feature engineering (02)
# must complete before any downstream step can start.
MODEL_STEPS = [
    "02_feature_engineering",
    "04_baseline",
    "05_clustering",
    "06_hmm",
    "07_classification",
    "08_diversification",
    "09b_lstm_sweep",
    "09_lstm",
]


def run_step(index: int, total: int, name: str, next_name: str, env: dict) -> tuple:
    """
    Execute one pipeline script and capture its output to a log file.

    Parameters
    ----------
    index     : 1-based position of this step in the full run order.
    total     : total number of steps being run.
    name      : script filename without the .py extension.
    next_name : name of the following step (used for the progress hint), or
                an empty string if this is the last step.
    env       : environment variables to pass to the subprocess.

    Returns
    -------
    (ok, elapsed) where ok is True if the script exited with code 0 and
    elapsed is the wall-clock time in seconds.
    """
    script = SCRIPTS_DIR / f"{name}.py"
    label  = f"[{index}/{total}] {name}"

    if not script.exists():
        print(f"{label}: script not found at {script}")
        return False, 0.0

    # Print a progress line with the name of the next step so the user knows
    # what is coming while waiting for a long-running step to finish.
    hint = f"   next: {next_name}" if next_name else ""
    print(f"{label:.<46} running{hint}", end="", flush=True)

    log_path = LOGS_DIR / f"{name}.log"
    start = time.time()

    # Run the script from NOTEBOOKS_DIR so that relative paths inside each
    # notebook script (e.g. Path.cwd().parent for the project root) resolve
    # consistently regardless of where run_all.py is called from.
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(NOTEBOOKS_DIR),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    elapsed = time.time() - start
    ok      = proc.returncode == 0
    status  = "done" if ok else "FAILED"

    # Overwrite the "running" line with the final status on the same line.
    print(f"\r{label:.<46} {status} ({elapsed:.0f}s){' ' * 30}", flush=True)
    return ok, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CryptoLens modeling pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refresh_data",
        action="store_true",
        help=(
            "Also run steps 01a and 01b to re-download raw price data from "
            "yfinance before running the rest of the pipeline. Requires an "
            "internet connection. Omit this flag to reproduce results from "
            "the data files already committed to the repository."
        ),
    )
    args = parser.parse_args()

    steps = (DATA_STEPS if args.refresh_data else []) + MODEL_STEPS
    total = len(steps)
    LOGS_DIR.mkdir(exist_ok=True)

    # MPLBACKEND=Agg prevents matplotlib from trying to open a GUI window
    # when running headlessly (e.g. on a server or in CI). PYTHONIOENCODING
    # ensures log files are written as UTF-8 on all platforms.
    env = dict(os.environ)
    env["MPLBACKEND"]       = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"

    mode = "full pipeline with data refresh" if args.refresh_data else "pipeline from committed data"
    print(f"Running {mode}: {total} steps")
    print()

    results = []
    pipeline_start = time.time()

    for i, name in enumerate(steps, 1):
        next_name = steps[i] if i < total else ""
        ok, elapsed = run_step(i, total, name, next_name, env)
        results.append((name, ok, elapsed))

        if not ok:
            # Print the tail of the log so the user can see the error without
            # having to open a separate file.
            log_path = LOGS_DIR / f"{name}.log"
            print()
            print(f"Step '{name}' failed. Last lines of {log_path.relative_to(REPO_ROOT)}:")
            if log_path.exists():
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
                for line in tail:
                    print(f"  {line}")
            print()
            print("Pipeline stopped.")
            sys.exit(1)

    total_elapsed = time.time() - pipeline_start
    print()
    print("Pipeline complete.")

    # Print a per-step summary table so the user can see which steps were
    # slow and might benefit from caching or optimisation.
    for name, ok, elapsed in results:
        print(f"  {'OK' if ok else 'FAIL':<5}{name:<34}{elapsed:6.0f}s")
    print(f"  {'':5}{'total':<34}{total_elapsed:6.0f}s")
    print()
    print("Model outputs: data/model_outputs/   MLflow runs: mlruns/ (view with: mlflow ui)")
    print("To launch the dashboard:  python run_app.py")


if __name__ == "__main__":
    main()