"""Launch the CryptoLens Streamlit dashboard locally.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
APP = REPO_ROOT / "src" / "app" / "app.py"

REQUIRED = [
    REPO_ROOT / "data" / "processed" / "crypto_wide_close_full.csv",
    REPO_ROOT / "data" / "processed" / "crypto_features_long_aligned.csv",
    REPO_ROOT / "data" / "model_outputs" / "hmm_regime_labels.csv",
    REPO_ROOT / "data" / "model_outputs" / "lstm_predictions.csv",
    REPO_ROOT / "data" / "model_outputs" / "baseline_predictions.csv",
]


def main() -> None:
    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        print("Cannot start the app. These required files are missing:")
        for p in missing:
            print(f"  {p.relative_to(REPO_ROOT)}")
        print()
        print("Run the pipeline first:  python run_all.py")
        sys.exit(1)

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP)],
        cwd=str(REPO_ROOT),
    )


if __name__ == "__main__":
    main()
