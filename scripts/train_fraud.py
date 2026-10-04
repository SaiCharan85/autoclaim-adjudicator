"""Benchmark the fraud models, train and save the production artifacts, and write the report.

Usage: uv run python scripts/train_fraud.py [--skip-benchmark] [--models catboost lightgbm]
"""

import sys

from autoclaim.ml.train import main

if __name__ == "__main__":
    sys.exit(main())
