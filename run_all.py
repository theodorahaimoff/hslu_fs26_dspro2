"""Run the full modeling pipeline end to end.

Executes the exported scripts in scripts/ in dependency order.

  python run_all.py                 reproduce all results from committed data
  python run_all.py --refresh_data  also re-download raw data first (01a, 01b)

The terminal shows one line per step.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
LOGS_DIR = REPO_ROOT / "logs"

# redownload step
DATA_STEPS = [
    "01a_data_collection_and_cleaning",
    "01b_macro_data_collection",
]

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
    """Run one script, capturing its output to a log file. Returns (ok, seconds)."""
    script = SCRIPTS_DIR / f"{name}.py"
    label = f"[{index}/{total}] {name}"
    if not script.exists():
        print(f"{label}: script not found at {script}")
        return False, 0.0

    hint = f"   next: {next_name}" if next_name else ""
    print(f"{label:.<46} running{hint}", end="", flush=True)

    log_path = LOGS_DIR / f"{name}.log"
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(NOTEBOOKS_DIR),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    elapsed = time.time() - start
    ok = proc.returncode == 0
    status = "done" if ok else "FAILED"
    print(f"\r{label:.<46} {status} ({elapsed:.0f}s){' ' * 30}", flush=True)
    return ok, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the modeling pipeline.")
    parser.add_argument(
        "--refresh_data",
        action="store_true",
        help="Also run 01a/01b to re-download raw data from yfinance.",
    )
    args = parser.parse_args()

    steps = (DATA_STEPS if args.refresh_data else []) + MODEL_STEPS
    total = len(steps)
    LOGS_DIR.mkdir(exist_ok=True)


    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
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
    for name, ok, elapsed in results:
        print(f"  {'OK' if ok else 'FAIL':<5}{name:<34}{elapsed:6.0f}s")
    print(f"  {'':5}{'total':<34}{total_elapsed:6.0f}s")
    print()
    print("Model outputs: data/model_outputs/   MLflow runs: mlruns/ (view with: mlflow ui)")
    print("To launch the dashboard:  python run_app.py")


if __name__ == "__main__":
    main()
