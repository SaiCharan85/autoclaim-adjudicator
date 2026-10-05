"""Train on one era, test on later eras (2002-2023, before the locked test) -> docs/drift_study.md.

Usage: uv run python scripts/drift_study.py
"""

import sys

from autoclaim.config import load_carrier_config
from autoclaim.datasets.profile import to_markdown
from autoclaim.lines.auto.datasets import get_spec
from autoclaim.lines.auto.drift import drift_matrix
from autoclaim.ml import models
from autoclaim.paths import REPO_ROOT


def main() -> int:
    models.MAX_TREES = 800  # comparisons across eras only; production models are not capped
    cfg = load_carrier_config().fraud_model
    raw = get_spec("sim_decades", cfg).load_raw()
    res = drift_matrix(raw, cfg).round(3)
    path = REPO_ROOT / "docs" / "drift_study.md"
    path.write_text(
        "# Drift study: train on one era, test on later eras\n\n"
        "Multi-decade world (real GES 2002-2015 / CRSS 2016-2024 crashes, BLS-priced, simulated "
        "policy and fraud). Data before the locked test window only. Metrics vs TRUE fraud.\n\n"
        + to_markdown(res.set_index("stage"), "stage")
        + "\n",
        encoding="utf-8",
    )
    print(res.to_string())
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
