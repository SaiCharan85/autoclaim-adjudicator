from pathlib import Path

import pandas as pd
import pytest

from autoclaim.config import DatasetSplit, FraudModelConfig
from fakes import make_fraud_frame, make_valid_frame

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    return make_valid_frame()


@pytest.fixture
def fraud_frame() -> pd.DataFrame:
    return make_fraud_frame()


@pytest.fixture(scope="session")
def fraud_cfg() -> FraudModelConfig:
    """Small, fast config over the 1990s-style schema (1994-95 train/val, 1996 test)."""
    return FraudModelConfig(
        production_dataset="legacy_1990s",
        review_budget=0.1,
        seeds=[0, 1],
        siu_review_cost_usd=300,
        datasets={
            "legacy_1990s": DatasetSplit(
                train_years=[1994, 1995],
                test_years=[1996],
                val_fraction=0.2,
                sensitive_features=["Sex", "MaritalStatus"],
            )
        },
    )


@pytest.fixture(scope="session")
def legacy_spec(fraud_cfg: FraudModelConfig):
    from autoclaim.lines.auto.datasets import get_spec

    return get_spec("legacy_1990s", fraud_cfg)


@pytest.fixture(scope="session")
def fraud_std(legacy_spec) -> pd.DataFrame:
    """Standard modeling frame (features + y + t) built from the synthetic 1990s-style rows."""
    return legacy_spec.prepare(make_fraud_frame())


@pytest.fixture(scope="session")
def trained_artifacts(legacy_spec, fraud_cfg: FraudModelConfig):
    """Tiny real CatBoost + Isolation Forest, trained once per test session."""
    from autoclaim.ml import fraud_model, models

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(models, "MAX_TREES", 80)
        frame = legacy_spec.prepare(make_fraud_frame())
        return fraud_model.train_final(frame, legacy_spec, fraud_cfg, seed=0, final=True)
