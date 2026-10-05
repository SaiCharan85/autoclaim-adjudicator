"""Two-stage fraud triage (first notice + after the independent appraisal).

Usage:
  uv run python scripts/fraud_triage.py           # validation: fit the stage-2 threshold
  uv run python scripts/fraud_triage.py --final   # locked test with the frozen threshold (logged)
"""

import sys

from autoclaim.lines.auto.fraud_triage import main

if __name__ == "__main__":
    sys.exit(main())
