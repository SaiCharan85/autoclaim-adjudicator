import numpy as np
import pandas as pd
import pytest

from autoclaim.ml.frame import Y
from autoclaim.ml.stages import ExpectedAmountStage, apply_stages, fit_stages, stage_outputs


def _claims(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Amount driven by damage level and vehicle value; fraud rows inflated 3x."""
    rng = np.random.default_rng(seed)
    extent = rng.choice(["minor", "functional", "disabling"], n)
    value = rng.uniform(5_000, 40_000, n)
    base = {"minor": 0.05, "functional": 0.15, "disabling": 0.5}
    true_cost = value * pd.Series(extent).map(base).to_numpy() * np.exp(rng.normal(0, 0.2, n))
    fraud = (rng.random(n) < 0.1).astype(int)
    amount = true_cost * np.where(fraud == 1, 3.0, 1.0)
    return pd.DataFrame(
        {"damage_extent": extent, "vehicle_acv": value, "claimed_amount": amount, Y: fraud}
    )


def _stage(seed: int = 0) -> ExpectedAmountStage:
    return ExpectedAmountStage(["damage_extent", "vehicle_acv"], "claimed_amount", seed=seed)


def test_outputs_and_residual_math() -> None:
    df = _claims()
    out = _stage().fit_oof(df)
    assert set(ExpectedAmountStage.outputs) <= set(out.columns)
    np.testing.assert_allclose(
        out["amount_residual"], np.log1p(df["claimed_amount"]) - out["expected_log_amount"]
    )


def test_inflated_claims_get_large_residuals() -> None:
    out = _stage().fit_oof(_claims())
    fraud, legit = out[out[Y] == 1], out[out[Y] == 0]
    assert fraud["amount_residual"].median() > np.log(2)  # ~log(3): flagged as "too expensive"
    assert abs(legit["amount_residual"].median()) < 0.15


def test_never_learns_from_confirmed_fraud() -> None:
    df = _claims()
    stage = _stage()
    stage.fit_oof(df)
    huge = df.assign(
        claimed_amount=np.where(df[Y] == 1, df["claimed_amount"] * 1000, df["claimed_amount"])
    )
    stage_huge = _stage()
    stage_huge.fit_oof(huge)
    probe = df[df[Y] == 0].head(50)
    np.testing.assert_allclose(  # absurd fraud amounts change nothing: fraud rows aren't used
        stage.transform(probe)["expected_log_amount"],
        stage_huge.transform(probe)["expected_log_amount"],
    )


def test_training_rows_get_out_of_fold_values() -> None:
    df = _claims()
    stage = _stage()
    oof = stage.fit_oof(df)["expected_log_amount"].to_numpy()
    in_sample = stage.transform(df)["expected_log_amount"].to_numpy()
    assert not np.allclose(oof, in_sample)  # a refit-on-all model would leak each row into itself
    legit = df[Y].to_numpy() == 0
    resid_oof = np.log1p(df["claimed_amount"].to_numpy()[legit]) - oof[legit]
    resid_in = np.log1p(df["claimed_amount"].to_numpy()[legit]) - in_sample[legit]
    assert np.std(resid_oof) > np.std(resid_in)  # in-sample residuals look unrealistically good


def test_missing_amount_gives_missing_residual() -> None:
    df = _claims()
    stage = _stage()
    stage.fit_oof(df)
    row = df.head(1).assign(claimed_amount=np.nan)
    assert np.isnan(stage.transform(row)["amount_residual"].iat[0])


def test_seeded() -> None:
    df = _claims()
    a, b = _stage(3).fit_oof(df), _stage(3).fit_oof(df)
    pd.testing.assert_frame_equal(a, b)


def test_helpers_chain_stages() -> None:
    df = _claims()
    stages = [_stage()]
    train = fit_stages(stages, df)
    applied = apply_stages(stages, df.head(5))
    assert stage_outputs(stages) == ["expected_log_amount", "amount_residual"]
    assert set(stage_outputs(stages)) <= set(train.columns) & set(applied.columns)


def test_transform_before_fit_fails() -> None:
    with pytest.raises(AttributeError):
        _stage().transform(_claims().head(2))
