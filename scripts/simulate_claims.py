"""Build the grounded-hybrid claims dataset (real CRSS crashes + simulated policy/fraud/traps).

Usage: uv run python scripts/simulate_claims.py [--n 50000]
"""

import sys

from autoclaim.lines.auto.simulator.build import main

if __name__ == "__main__":
    sys.exit(main())
