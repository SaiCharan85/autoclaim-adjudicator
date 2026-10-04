"""Download Kaggle datasets into data/raw/.

Usage: uv run python scripts/download_data.py [--force]
"""

import sys

from autoclaim.datasets.download import main

if __name__ == "__main__":
    sys.exit(main())
