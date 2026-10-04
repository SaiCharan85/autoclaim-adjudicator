"""Compare fraud-model variants on validation (never the locked test).

Usage: uv run python scripts/fraud_experiments.py [--levers repair_cost weighting ensemble grid]
"""

import sys

from autoclaim.ml.experiment import main

if __name__ == "__main__":
    sys.exit(main())
