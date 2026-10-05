"""ML vs LLM-only vs hybrid fraud scoring on the same claims.

Usage:
  uv run python scripts/fraud_llm_benchmark.py --n 50 --dry-run   # estimate, spends nothing
  uv run python scripts/fraud_llm_benchmark.py --n 50             # pilot
  uv run python scripts/fraud_llm_benchmark.py --n 200            # validation (pilot reused)
  uv run python scripts/fraud_llm_benchmark.py --n 200 --final    # locked test, once, logged
"""

import sys

from autoclaim.lines.auto.fraud_benchmark import main

if __name__ == "__main__":
    sys.exit(main())
