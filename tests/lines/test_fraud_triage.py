import numpy as np
import pandas as pd

from autoclaim.lines.auto.fraud_triage import render, summarize
from autoclaim.ml.triage import TwoStagePolicy


def _pool(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true = rng.random(n) < 0.1
    types = np.where(true, rng.choice(["inflated_damage", "owner_give_up"], n), "")
    return pd.DataFrame({
        "s1": true * 0.3 + rng.random(n) * 0.7,
        "s2": true * 0.6 + rng.random(n) * 0.4,
        "confirmed": true & (rng.random(n) < 0.75),
        "true": true,
        "fraud_type": types,
        "amount": rng.uniform(500, 5000, n),
    })  # fmt: skip


def test_summarize_tables_and_render() -> None:
    pool = _pool()
    tables = summarize(pool, TwoStagePolicy(0.05, 0.6))
    assert set(tables["policy"].index) == {"true fraud", "confirmed fraud"}
    stage2 = tables["models"].loc["stage 2 (after appraisal)", "roc_auc_true"]
    assert stage2 > tables["models"].loc["stage 1 (first notice)", "roc_auc_true"]
    assert set(tables["by_type"].index) == {"inflated_damage", "owner_give_up"}
    assert tables["policy"].loc["true fraud", "recall"] > 0.9  # s2 separates perfectly at 0.6
    md = render("validation", "v1, v2", TwoStagePolicy(0.05, 0.6), tables)
    assert md.startswith("# Two-stage fraud triage") and "By fraud type" in md
