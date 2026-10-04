from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoclaim.config import FraudModelConfig
from autoclaim.datasets.vehicle_fraud import SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[1]
N_ROWS = 200


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


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


def make_fraud_frame(n: int = 900, seed: int = 1) -> pd.DataFrame:
    """Valid frame with a learnable fraud signal and all three years present."""
    rng = np.random.default_rng(seed)
    df = make_valid_frame(n, seed)
    df["Year"] = rng.choice([1994, 1995, 1996], n)
    at_fault = df["Fault"].eq("Policy Holder")
    rural = df["AccidentArea"].eq("Rural")
    p = 0.03 + 0.35 * (at_fault & rural) + 0.10 * at_fault
    df["FraudFound_P"] = (rng.random(n) < p).astype(int)
    return df


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    return make_valid_frame()


@pytest.fixture
def fraud_frame() -> pd.DataFrame:
    return make_fraud_frame()


@pytest.fixture
def small_fraud_cfg() -> FraudModelConfig:
    return FraudModelConfig(
        train_years=[1994, 1995],
        test_years=[1996],
        review_budget=0.1,
        cv_folds=2,
        sensitive_features=["Sex", "MaritalStatus"],
    )


@pytest.fixture(scope="session")
def trained_artifacts():
    """Tiny real CatBoost + Isolation Forest, trained once per test session."""
    from autoclaim.ml import fraud_model, models
    from autoclaim.ml.features import clean

    cfg = FraudModelConfig(
        train_years=[1994, 1995],
        test_years=[1996],
        review_budget=0.1,
        cv_folds=2,
        sensitive_features=["Sex", "MaritalStatus"],
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(models, "MAX_TREES", 80)
        return fraud_model.train_final(clean(make_fraud_frame()), cfg, seed=0)
