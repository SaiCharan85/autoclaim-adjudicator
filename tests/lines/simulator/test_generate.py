import numpy as np
import pandas as pd
import pytest

from autoclaim.lines.auto.simulator import generate as gen
from autoclaim.lines.auto.simulator.world import World


@pytest.fixture(scope="module")
def claims(world: World, incidents: pd.DataFrame) -> pd.DataFrame:
    return gen.build_claims(world, incidents)


# ---------------------------------------------------------------- weighted sampling


def test_weighted_sampling_has_no_repeats() -> None:
    rng = np.random.default_rng(0)
    idx = gen.weighted_sample_without_replacement(np.ones(100), 60, rng)
    assert len(idx) == len(set(idx)) == 60


def test_weighted_sampling_prefers_heavy_items() -> None:
    rng = np.random.default_rng(0)
    w = np.ones(1000)
    w[:10] = 1000.0
    hits = sum(
        len(set(gen.weighted_sample_without_replacement(w, 10, rng)) & set(range(10)))
        for _ in range(20)
    )
    assert hits > 150  # heavy items dominate (random would give ~2)


def test_weighted_sampling_bounds() -> None:
    rng = np.random.default_rng(0)
    assert len(gen.weighted_sample_without_replacement(np.ones(3), 0, rng)) == 0
    with pytest.raises(ValueError, match="only 3"):
        gen.weighted_sample_without_replacement(np.ones(3), 4, rng)


# ---------------------------------------------------------------- realness guarantees


def test_size_ids_and_unique_records(claims: pd.DataFrame, world: World) -> None:
    assert len(claims) == world.n_claims
    assert claims["claim_id"].is_unique
    assert claims["policy_id"].is_unique
    assert claims["source_record"].is_unique  # each real record used at most once


def test_real_records_are_copied_unchanged(claims: pd.DataFrame, incidents: pd.DataFrame) -> None:
    real = claims[claims["record_origin"] == "crss"].set_index("source_record")
    src = incidents.set_index("source_record").loc[real.index]
    for col in (*gen.CONTEXT_COLS, *gen.INCIDENT_COLS, "cause"):
        pd.testing.assert_series_equal(
            real[col].reset_index(drop=True),
            src[col].reset_index(drop=True),
            check_dtype=False,
            check_names=False,
        )


def test_simulated_claims_borrow_real_context(
    claims: pd.DataFrame, incidents: pd.DataFrame
) -> None:
    sim = claims[claims["record_origin"] == "crss_context"].set_index("source_record")
    src = incidents.set_index("source_record").loc[sim.index]
    for col in gen.CONTEXT_COLS:
        pd.testing.assert_series_equal(
            sim[col].reset_index(drop=True),
            src[col].reset_index(drop=True),
            check_dtype=False,
            check_names=False,
        )


def test_reporting_matches_origin(claims: pd.DataFrame) -> None:
    real = claims["record_origin"] == "crss"
    assert claims.loc[real, "police_report"].all()  # every CRSS crash is police-reported
    unreported_crash_cause = (
        ~real & claims["cause"].isin(gen.CRSS_CAUSES) & (claims["cause"] != "mechanical_breakdown")
    )
    assert not claims.loc[unreported_crash_cause, "police_report"].any()
    assert claims.loc[~claims["police_report"], "police_report_hours"].isna().all()


def test_cap_to_available_demotes_excess() -> None:
    rng = np.random.default_rng(0)
    pool = pd.DataFrame({"cause": ["fire"] * 3})
    cause = np.array(["fire"] * 10)
    real = gen._cap_to_available(np.ones(10, dtype=bool), cause, pool, rng)
    assert real.sum() == 3


def test_exclude_records_and_date_window(world: World, incidents: pd.DataFrame) -> None:
    first = gen.build_claims(world, incidents, n=1000, seed=1)
    window = world.model_copy(update={"loss_date_start": pd.Timestamp("2024-01-01").date()})
    second = gen.build_claims(
        window, incidents, n=500, seed=2, exclude_records=set(first["source_record"])
    )
    assert set(first["source_record"]).isdisjoint(second["source_record"])
    assert second["loss_date"].min() >= pd.Timestamp("2024-01-01")


def test_seeded_and_deterministic(world: World, incidents: pd.DataFrame) -> None:
    a = gen.build_claims(world, incidents, n=500, seed=7)
    b = gen.build_claims(world, incidents, n=500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = gen.build_claims(world, incidents, n=500, seed=8)
    assert not a["source_record"].equals(c["source_record"])


# ---------------------------------------------------------------- policy and dates


def test_dates_are_ordered(claims: pd.DataFrame) -> None:
    assert (claims["policy_start_date"] <= claims["loss_date"]).all()
    assert (claims["loss_date"] <= claims["report_date"]).all()


def test_deductibles_follow_coverage(claims: pd.DataFrame) -> None:
    liab = claims["coverage"] == "liability_only"
    full = claims["coverage"] == "collision_comprehensive"
    assert claims.loc[liab, "collision_deductible"].isna().all()
    assert claims.loc[~liab, "collision_deductible"].notna().all()
    assert claims.loc[~full, "comprehensive_deductible"].isna().all()
    assert not claims.loc[liab, "rideshare_endorsement"].any()


def test_vehicle_value_positive(claims: pd.DataFrame, world: World) -> None:
    assert (claims["vehicle_acv"] >= world.min_acv).all()
    assert (claims["vehicle_age"] >= 0).all()


# ---------------------------------------------------------------- fraud


def test_fraud_rate_near_target(claims: pd.DataFrame, world: World) -> None:
    assert claims["gt_is_fraud"].mean() == pytest.approx(world.fraud.base_rate, abs=0.02)
    assert (claims["fraud_confirmed"] <= claims["gt_is_fraud"]).all()  # only real fraud confirmed


def test_fraud_types_match_their_profiles(claims: pd.DataFrame) -> None:
    f = claims[claims["gt_is_fraud"]]
    give_up = f[f["gt_fraud_type"] == "owner_give_up"]
    assert (give_up["cause"] == "theft").all()
    assert give_up["financed"].all()
    prior = f[f["gt_fraud_type"] == "prior_damage"]
    assert (prior["record_origin"] == "crss_context").all()
    assert prior["cause"].isin(["collision_object", "parked_hit"]).all()
    staged = f[f["gt_fraud_type"] == "staged_collision"]
    assert (staged["record_origin"] == "crss").all()
    assert (staged["area"] == "urban").all()
    assert claims.loc[~claims["gt_is_fraud"], "gt_fraud_type"].isna().all()


def test_amounts(claims: pd.DataFrame) -> None:
    legit = ~claims["gt_is_fraud"]
    pd.testing.assert_series_equal(
        claims.loc[legit, "claimed_amount"], claims.loc[legit, "gt_true_damage"], check_names=False
    )
    inflated = claims["gt_fraud_type"] == "inflated_damage"
    assert (claims.loc[inflated, "claimed_amount"] > claims.loc[inflated, "gt_true_damage"]).all()


def test_coverage_lowers_fraud(claims: pd.DataFrame) -> None:
    rate = claims.groupby("coverage")["gt_is_fraud"].mean()
    assert rate["liability_only"] < rate["collision"] < rate["collision_comprehensive"]


def test_fraud_more_often_soon_after_inception(claims: pd.DataFrame) -> None:
    early = (claims["loss_date"] - claims["policy_start_date"]).dt.days <= 15
    assert early[claims["gt_is_fraud"]].mean() > 4 * early[~claims["gt_is_fraud"]].mean()


# ---------------------------------------------------------------- traps


def test_traps_planted_on_legitimate_claims(claims: pd.DataFrame) -> None:
    legit = ~claims["gt_is_fraud"]
    excluded = claims["driver_role"] == "excluded_driver"
    business = claims["use_at_loss"].isin(gen.BUSINESS_USES) & ~claims["rideshare_endorsement"]
    animal_no_comp = (claims["cause"] == "animal") & (claims["coverage"] == "collision")
    mechanical = claims["cause"] == "mechanical_breakdown"
    for mask in (excluded, business, animal_no_comp, mechanical):
        assert (mask & legit).sum() >= 10


def test_mechanical_claims_are_consistent(claims: pd.DataFrame) -> None:
    mech = claims[claims["cause"] == "mechanical_breakdown"]
    assert not mech["police_report"].any()
    assert (mech["n_vehicles"] == 1).all()
    assert (mech["injury_count"] == 0).all()


def test_kia_hyundai_theft_tilt(world: World, incidents: pd.DataFrame) -> None:
    big = gen.build_claims(world, incidents, n=4000, seed=3)
    theft = big["cause"] == "theft"
    kia = big["vehicle_make"].isin(["Kia", "Hyundai"])
    assert kia[theft].mean() > kia[~theft].mean()


def test_identical_across_processes() -> None:
    """Same seed, different Python hash seeds -> byte-identical dataset (no set-order leaks)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    tests_dir = Path(__file__).parents[2]  # tests/, where fakes.py lives
    code = (
        f"import sys; sys.path.insert(0, r'{tests_dir}');"
        "from fakes import make_incidents;"
        "from autoclaim.lines.auto.simulator.world import load_world;"
        "from autoclaim.lines.auto.simulator.generate import build_claims;"
        "import hashlib;"
        "w = load_world().model_copy(update={'n_claims': 800});"
        "df = build_claims(w, make_incidents(scale=0.3));"
        "print(hashlib.sha256(df.to_csv(index=False).encode()).hexdigest())"
    )
    digests = set()
    for hash_seed in ("1", "2"):
        env = os.environ | {"PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1, "dataset depends on Python's hash seed"
