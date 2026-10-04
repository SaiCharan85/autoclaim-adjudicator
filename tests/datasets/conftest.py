import numpy as np
import pandas as pd
import pytest

from autoclaim.datasets.vehicle_fraud import SCHEMA

N_ROWS = 200


def make_valid_frame(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Schema-valid synthetic stand-in for fraud_oracle.csv (no real data in tests)."""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}
    for spec in SCHEMA.columns:
        if spec.kind == "binary":
            data[spec.name] = (rng.random(n) < 0.1).astype(int)
        elif spec.kind == "integer":
            lo = spec.min_value if spec.min_value is not None else 0
            hi = min(spec.max_value if spec.max_value is not None else lo + 10, lo + 50)
            data[spec.name] = rng.integers(lo, hi + 1, n)
        elif spec.allowed is not None:
            data[spec.name] = rng.choice(sorted(spec.allowed), n)
        else:
            data[spec.name] = rng.choice(["a", "b", "c"], n)
    df = pd.DataFrame(data)
    df["PolicyNumber"] = np.arange(1, n + 1)
    return df


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    return make_valid_frame()
