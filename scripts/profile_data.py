"""Validate and profile the fraud table. Usage: uv run python scripts/profile_data.py"""

import sys

from autoclaim.datasets.profile import main

if __name__ == "__main__":
    sys.exit(main())
