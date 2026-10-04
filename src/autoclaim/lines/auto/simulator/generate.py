"""Grounded-hybrid claims builder: real CRSS incidents + simulated policy, fraud and traps.

Rules that keep it honest:
- A police-reported collision/animal/hit-and-run/parked/fire claim IS a real CRSS record: its
  date, hour, region, area, weather, light, vehicle, damage extent, towing, injuries, vehicle count
  and at-fault flag are copied unchanged.
- Every other claim borrows its context (date, hour, region, area, weather, light, vehicle) from a
  different real record, so simulated claims look like real ones (no provenance shortcut for a
  model).
- Each real record is used at most once (weighted sampling without replacement), so one crash
  can never land in both train and test.
- Fraud and traps only change simulated fields (amounts, policy terms, reporting, use, driver).
"""

import numpy as np
import pandas as pd

from autoclaim.lines.auto.simulator.world import STATE_REGION, World

CRSS_CAUSES = (
    "collision_vehicle",
    "collision_object",
    "parked_hit",
    "animal",
    "hit_and_run",
    "fire",
)
CONTEXT_COLS = (
    "loss_date",
    "loss_hour",
    "region",
    "area",
    "weather",
    "light",
    "vehicle_make",
    "vehicle_model",
    "vehicle_model_year",
    "body_class",
)
INCIDENT_COLS = (
    "n_vehicles",
    "injury_count",
    "damage_extent",
    "towed",
    "at_fault",
    "vehicle_role",
)
BUSINESS_USES = ("rideshare_active", "delivery_active")
US_COMPREHENSIVE_CAUSES = ("animal", "theft", "vandalism", "hail_weather", "glass", "fire")
UNFILED_BELOW_DEDUCTIBLE = 0.8  # share of below-deductible losses people wouldn't file (assumed)


def weighted_sample_without_replacement(
    weights: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices of k items, P(item) proportional to weight, no repeats (Efraimidis-Spirakis)."""
    if k > len(weights):
        raise ValueError(f"need {k} records but only {len(weights)} are available")
    if k == 0:
        return np.empty(0, dtype=int)
    keys = np.log(rng.random(len(weights))) / np.maximum(weights, 1e-12)
    return np.argpartition(-keys, k - 1)[:k]


def _choice(rng: np.random.Generator, dist: dict, n: int) -> np.ndarray:  # type: ignore[type-arg]
    keys = list(dist)
    p = np.array([dist[k] for k in keys], dtype=float)
    return np.array(keys, dtype=object)[rng.choice(len(keys), size=n, p=p / p.sum())]


def _lognormal(rng: np.random.Generator, median: float, sigma: float, n: int) -> np.ndarray:
    return median * np.exp(sigma * rng.standard_normal(n))


def _cap_to_available(
    real: np.ndarray, cause: np.ndarray, pool: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """If a cause needs more real records than exist, the excess become simulated claims."""
    real = real.copy()
    available = pool["cause"].value_counts()
    for c in CRSS_CAUSES:
        rows = np.flatnonzero(real & (cause == c))
        excess = len(rows) - int(available.get(c, 0))
        if excess > 0:
            real[rng.choice(rows, size=excess, replace=False)] = False
    return real


def _pick_donors(
    pool: pd.DataFrame,
    cause: np.ndarray,
    real: np.ndarray,
    world: World,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pool row index for every claim; each pool row used at most once."""
    donor = np.full(len(cause), -1, dtype=int)
    used = np.zeros(len(pool), dtype=bool)
    weights = pool["weight"].to_numpy()
    pool_cause = pool["cause"].to_numpy()

    def take(rows: np.ndarray, candidates: np.ndarray, w: np.ndarray) -> None:
        picked = candidates[weighted_sample_without_replacement(w, len(rows), rng)]
        donor[rows] = picked
        used[picked] = True

    for c in CRSS_CAUSES:
        rows = np.flatnonzero(real & (cause == c))
        cand = np.flatnonzero((pool_cause == c) & ~used)
        take(rows, cand, weights[cand])

    # Simulated thefts borrow vehicles with the real Kia/Hyundai theft-wave tilt.
    theft = np.flatnonzero(~real & (cause == "theft"))
    cand = np.flatnonzero(~used)
    kia = pool["vehicle_make"].isin(["Kia", "Hyundai"]).to_numpy()[cand]
    year = pool["loss_date"].dt.year.astype(str).to_numpy()[cand]
    mult = np.array([world.kia_hyundai_theft_multiplier.get(y, 1.0) for y in year])
    take(theft, cand, weights[cand] * np.where(kia, mult, 1.0))

    rest = np.flatnonzero(~real & (cause != "theft"))
    cand = np.flatnonzero(~used)
    take(rest, cand, weights[cand])
    return donor


def _simulated_incident(
    cause: np.ndarray, world: World, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Incident fields for claims that are not a real police-reported crash."""
    n = len(cause)
    n_vehicles = np.where(np.isin(cause, ["collision_vehicle"]), 2 + (rng.random(n) < 0.15), 1)
    n_vehicles = np.where(np.isin(cause, ["parked_hit", "hit_and_run"]), 2, n_vehicles)
    inj_p = np.array([world.injury_rate.get(c, world.injury_rate["default"]) for c in cause])
    injuries = np.where(rng.random(n) < inj_p, rng.integers(1, 3, n), 0).astype(float)
    # Unreported damage is mostly minor; comprehensive causes vary.
    extent = np.where(rng.random(n) < 0.7, "minor", "functional").astype(object)
    extent[cause == "theft"] = "unknown"
    extent[cause == "fire"] = "disabling"
    towed = np.where(np.isin(cause, ["fire"]), 1.0, np.where(cause == "theft", np.nan, 0.0))
    fault_p = np.array([world.at_fault_rate.get(c, world.at_fault_rate["default"]) for c in cause])
    at_fault = (rng.random(n) < fault_p).astype(float)
    role = np.where(cause == "parked_hit", "parked", "in_transport").astype(object)
    return {
        "n_vehicles": n_vehicles.astype(int),
        "injury_count": injuries,
        "damage_extent": extent,
        "towed": towed,
        "at_fault": at_fault,
        "vehicle_role": role,
    }


def _states(region: np.ndarray, world: World, rng: np.random.Generator) -> tuple:  # type: ignore[type-arg]
    n = len(region)
    loss_state = np.empty(n, dtype=object)
    for reg in sorted(set(STATE_REGION.values())):  # sorted: set order varies by process
        rows = np.flatnonzero(region == reg)
        states = {s: w for s, w in world.states.items() if STATE_REGION[s] == reg}
        loss_state[rows] = _choice(rng, states, len(rows))
    policy_state = loss_state.copy()
    away = rng.random(n) < world.out_of_state_loss_rate
    others = _choice(rng, world.states, n)
    clash = others == loss_state
    others[clash] = _choice(rng, world.states, int(clash.sum()))
    policy_state[away] = others[away]
    return loss_state, policy_state


def _vehicle_value(df: pd.DataFrame, world: World, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    loss_year = df["loss_date"].dt.year.to_numpy()
    va = world.vehicle_age_years
    sampled_age = np.clip(np.round(rng.normal(va["mean"], va["sd"], n)), 0, va["max"])
    model_year = df["vehicle_model_year"].to_numpy(dtype=float)
    age = np.where(np.isnan(model_year), sampled_age, np.clip(loss_year - model_year, 0, 40))
    base = df["body_class"].map(world.body_new_price).fillna(world.new_vehicle_price.median)
    lux = np.where(df["vehicle_make"].isin(world.luxury_makes), world.luxury_multiplier, 1.0)
    new_price = (
        base.to_numpy() * lux * np.exp(world.new_vehicle_price.sigma * rng.standard_normal(n))
    )
    acv = np.maximum(world.min_acv, new_price * (1 - world.depreciation_per_year) ** age)
    fin = world.financed_rate_by_age
    adas = world.adas_rate_by_age
    return df.assign(
        vehicle_age=age,
        vehicle_acv=np.round(acv, 2),
        financed=rng.random(n) < np.where(age <= fin["new_max_age"], fin["new"], fin["old"]),
        adas=rng.random(n) < np.where(age <= adas["new_max_age"], adas["new"], adas["old"]),
    )


def _true_damage(df: pd.DataFrame, world: World, rng: np.random.Generator) -> np.ndarray:
    n = len(df)
    cause = df["cause"].to_numpy()
    extent = df["damage_extent"].to_numpy()
    by_cause = np.array([world.damage_fraction[c] for c in cause])
    by_extent = np.array([world.damage_by_extent.get(e, (np.nan, np.nan)) for e in extent])
    use_extent = (df["record_origin"].to_numpy() == "crss") & ~np.isnan(by_extent[:, 0])
    median = np.where(use_extent, by_extent[:, 0], by_cause[:, 0])
    sigma = np.where(use_extent, by_extent[:, 1], by_cause[:, 1])
    frac = median * np.exp(sigma * rng.standard_normal(n))
    frac = np.where(df["towed"].to_numpy() == 1.0, frac * world.towed_damage_multiplier, frac)
    frac = np.where(cause == "theft", 1.0, np.minimum(frac, 1.2))
    damage = frac * df["vehicle_acv"].to_numpy()
    glass_adas = (cause == "glass") & df["adas"].to_numpy()
    damage = damage + np.where(glass_adas, world.adas_glass_recalibration_usd, 0.0)
    out: np.ndarray = np.round(np.maximum(damage, 50.0), 2)
    return out


def _rarely_filed_below_deductible(
    df: pd.DataFrame, comprehensive_causes: tuple[str, ...], rng: np.random.Generator
) -> np.ndarray:
    """People rarely file a loss smaller than their deductible. For most such claims, redraw the
    (simulated) repair cost above the deductible instead of dropping the row and its real record."""
    damage = df["gt_true_damage"].to_numpy().copy()
    comp = np.isin(df["cause"].to_numpy(), comprehensive_causes)
    deductible = np.where(
        comp, df["comprehensive_deductible"].to_numpy(), df["collision_deductible"].to_numpy()
    )
    below = (damage < np.nan_to_num(deductible, nan=-1.0)) & (
        rng.random(len(df)) < UNFILED_BELOW_DEDUCTIBLE
    )
    damage[below] = np.round(deductible[below] * rng.uniform(1.1, 3.0, below.sum()), 2)
    return damage


def _reporting(df: pd.DataFrame, world: World, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    cause = df["cause"].to_numpy()
    real = df["record_origin"].to_numpy() == "crss"
    rate = np.array([world.police_report_rate[c] for c in cause])
    # A simulated claim of a CRSS-covered cause is, by construction, the unreported share.
    reported = np.where(
        real, True, np.where(np.isin(cause, CRSS_CAUSES), False, rng.random(n) < rate)
    )
    hours = _lognormal(rng, world.police_report_hours.median, world.police_report_hours.sigma, n)
    w_p = np.array([world.witness_rate.get(c, world.witness_rate["default"]) for c in cause])
    witnesses = np.where(rng.random(n) < w_p, rng.integers(1, 4, n), 0)
    notice = np.floor(_lognormal(rng, world.notice_days.median, world.notice_days.sigma, n))
    return df.assign(
        police_report=reported,
        police_report_hours=np.where(reported, np.round(hours, 1), np.nan),
        witness_count=witnesses.astype(int),
        notice_days=notice.astype(int),
        attorney_involved=(df["injury_count"].fillna(0).to_numpy() > 0) & (rng.random(n) < 0.15),
    )


def _policy(df: pd.DataFrame, world: World, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    coverage = _choice(rng, world.coverage_mix, n)
    coll = _choice(rng, world.collision_deductibles, n).astype(float)
    comp = _choice(rng, world.comprehensive_deductibles, n).astype(float)
    legit_early = rng.random(n) < world.fraud.legit_early_inception_share
    days_since_start = np.where(legit_early, rng.integers(0, 16, n), rng.integers(16, 1461, n))
    da = world.driver_age
    addr = rng.random(n)
    lr = world.fraud.legit_address_change_recent_share
    return df.assign(
        coverage=coverage,
        collision_deductible=np.where(coverage != "liability_only", coll, np.nan),
        comprehensive_deductible=np.where(coverage == "collision_comprehensive", comp, np.nan),
        rideshare_endorsement=(rng.random(n) < world.rideshare_endorsement_rate)
        & (coverage != "liability_only"),
        days_since_start=days_since_start,
        driver_role=_choice(rng, world.driver_role, n),
        driver_age=np.clip(np.round(rng.normal(da["mean"], da["sd"], n)), da["min"], da["max"]),
        use_at_loss=_choice(rng, world.use_at_loss, n),
        address_change_days=np.where(
            addr < lr,
            rng.integers(1, 181, n),
            np.where(addr < lr + 0.10, rng.integers(181, 731, n), np.nan),
        ),
        prior_claims_3y=rng.poisson(0.3, n),
    )


def _fraud(df: pd.DataFrame, world: World, rng: np.random.Generator) -> pd.DataFrame:
    """Latent fraud with a profile-driven type; edits only simulated fields."""
    fw = world.fraud
    n = len(df)
    cause = df["cause"].to_numpy()
    real = df["record_origin"].to_numpy() == "crss"
    give_up = (cause == "theft") & df["financed"].to_numpy()
    prior = ~real & np.isin(cause, ["collision_object", "parked_hit"])
    staged = (
        real
        & (cause == "collision_vehicle")
        & (df["area"].to_numpy() == "urban")
        & (df["injury_count"].fillna(0).to_numpy() >= 1)
    )
    pm = fw.profile_multiplier
    mult = (
        1
        + (pm["give_up"] - 1) * give_up
        + (pm["prior_damage"] - 1) * prior
        + (pm["staged"] - 1) * staged
    )
    p = df["coverage"].map(fw.coverage_multiplier).to_numpy() * mult
    p = np.clip(p * fw.base_rate / p.mean(), 0, 0.95)
    is_fraud = rng.random(n) < p

    u = rng.random(n)
    ftype = np.full(n, "", dtype=object)
    ftype[is_fraud] = "inflated_damage"
    ftype[is_fraud & prior & (u < fw.prior_damage_share)] = "prior_damage"
    ftype[is_fraud & staged & (u < fw.staged_share)] = "staged_collision"
    ftype[is_fraud & give_up] = "owner_give_up"

    out = df.copy()
    true = out["gt_true_damage"].to_numpy()
    claimed = true.copy()
    lo, hi = fw.inflation_factor
    m = ftype == "inflated_damage"
    claimed[m] = true[m] * rng.uniform(lo, hi, m.sum())
    m = ftype == "prior_damage"
    claimed[m] = true[m] * rng.uniform(1.0, 1.3, m.sum())
    out.loc[m, "witness_count"] = 0
    late = m & (rng.random(n) < fw.late_notice_share)
    out.loc[late, "notice_days"] = rng.integers(15, 61, late.sum())
    m = ftype == "staged_collision"
    claimed[m] = true[m] * rng.uniform(1.1, 1.6, m.sum())
    out.loc[m & (rng.random(n) < 0.6), "attorney_involved"] = True
    friendly = m & (rng.random(n) < 0.5)
    out.loc[friendly, "witness_count"] = rng.integers(1, 3, friendly.sum())
    m = ftype == "owner_give_up"
    claimed[m] = out["vehicle_acv"].to_numpy()[m] * rng.uniform(1.0, 1.15, m.sum())
    out.loc[m, "police_report"] = True  # a theft claim needs a report; the fraudster files late
    out.loc[m, "police_report_hours"] = np.round(rng.uniform(24, 96, m.sum()), 1)
    late = m & (rng.random(n) < fw.late_notice_share)
    out.loc[late, "notice_days"] = rng.integers(3, 21, late.sum())

    early = is_fraud & (rng.random(n) < fw.early_inception_share)
    out.loc[early, "days_since_start"] = rng.integers(0, 16, early.sum())
    moved = is_fraud & (rng.random(n) < fw.address_change_recent_share)
    out.loc[moved, "address_change_days"] = rng.integers(1, 181, moved.sum())
    out.loc[is_fraud, "prior_claims_3y"] = rng.poisson(0.9, is_fraud.sum())

    out["claimed_amount"] = np.round(claimed, 2)
    out["gt_is_fraud"] = is_fraud
    out["gt_fraud_type"] = pd.Series(ftype, index=out.index).where(is_fraud)
    out["fraud_confirmed"] = is_fraud & (rng.random(n) < fw.detection_rate)
    return out


def _plant_traps(df: pd.DataFrame, world: World, rng: np.random.Generator) -> pd.DataFrame:
    """Force each coverage trap onto a few legitimate claims (simulated fields only)."""
    out = df.copy()
    n = len(out)
    taken = out["gt_is_fraud"].to_numpy().copy()
    covered = out["coverage"].to_numpy() != "liability_only"
    cause = out["cause"].to_numpy()
    real = out["record_origin"].to_numpy() == "crss"

    def pick(eligible: np.ndarray, rate: float) -> np.ndarray:
        cand = np.flatnonzero(eligible & ~taken)
        k = min(len(cand), round(rate * n))
        chosen = rng.choice(cand, size=k, replace=False) if k else np.empty(0, dtype=int)
        taken[chosen] = True
        return chosen

    t = world.traps
    rows = pick(cause == "animal", t["animal_on_collision_only"])
    out.loc[rows, "coverage"] = "collision"
    out.loc[rows, "comprehensive_deductible"] = np.nan
    out.loc[rows, "collision_deductible"] = out.loc[rows, "collision_deductible"].fillna(500.0)

    rows = pick(
        covered & np.isin(cause, ["collision_vehicle", "collision_object"]),
        t["business_use_no_endorsement"],
    )
    out.loc[rows, "use_at_loss"] = rng.choice(BUSINESS_USES, size=len(rows))
    out.loc[rows, "rideshare_endorsement"] = False

    rows = pick(covered, t["excluded_driver"])
    out.loc[rows, "driver_role"] = "excluded_driver"

    rows = pick(
        covered & ~real & np.isin(cause, ["glass", "hail_weather", "vandalism"]), t["wear_and_tear"]
    )
    out.loc[rows, "cause"] = "mechanical_breakdown"
    out.loc[rows, ["police_report", "attorney_involved"]] = False
    out.loc[rows, "police_report_hours"] = np.nan
    out.loc[rows, ["n_vehicles", "witness_count"]] = [1, 0]
    out.loc[rows, ["injury_count", "at_fault"]] = 0.0
    out.loc[rows, "damage_extent"] = "functional"
    frac = _lognormal(rng, *world.damage_fraction["mechanical_breakdown"], len(rows))
    damage = np.round(np.maximum(frac * out.loc[rows, "vehicle_acv"].to_numpy(), 50.0), 2)
    out.loc[rows, ["gt_true_damage", "claimed_amount"]] = np.column_stack([damage, damage])

    rows = pick(
        covered & (cause == "hit_and_run") & out["police_report"].to_numpy(),
        t["hit_and_run_no_timely_report"],
    )
    out.loc[rows, "police_report_hours"] = np.round(rng.uniform(30, 120, len(rows)), 1)

    rows = pick(covered, t["late_notice"])
    out.loc[rows, "notice_days"] = rng.integers(31, 121, len(rows))
    return out


def build_claims(
    world: World,
    incidents: pd.DataFrame,
    n: int | None = None,
    seed: int | None = None,
    exclude_records: set[str] | None = None,
    comprehensive_causes: tuple[str, ...] = US_COMPREHENSIVE_CAUSES,
) -> pd.DataFrame:
    """Build `n` claims. `exclude_records` keeps a fresh holdout disjoint from an earlier build;
    `comprehensive_causes` decides which deductible a loss is compared against."""
    rng = np.random.default_rng(world.seed if seed is None else seed)
    n = world.n_claims if n is None else n
    start, end = pd.Timestamp(world.loss_date_start), pd.Timestamp(world.loss_date_end)
    pool = incidents[incidents["loss_date"].between(start, end)]
    if exclude_records:
        pool = pool[~pool["source_record"].isin(exclude_records)]
    pool = pool.reset_index(drop=True)

    cause = _choice(rng, world.loss_causes, n)
    rate = np.array([world.police_report_rate[c] for c in cause])
    real = np.isin(cause, CRSS_CAUSES) & (rng.random(n) < rate)
    real = _cap_to_available(real, cause, pool, rng)
    donor = _pick_donors(pool, cause, real, world, rng)
    picked = pool.iloc[donor].reset_index(drop=True)

    sim = _simulated_incident(cause, world, rng)
    df = picked[list(CONTEXT_COLS)].copy()
    df["cause"] = cause
    df["record_origin"] = np.where(real, "crss", "crss_context")
    df["source_record"] = picked["source_record"].to_numpy()
    for col in INCIDENT_COLS:
        df[col] = np.where(real, picked[col].to_numpy(), sim[col])
    df = df.astype({"n_vehicles": int, "injury_count": float, "towed": float, "at_fault": float})

    loss_state, policy_state = _states(df["region"].to_numpy(), world, rng)
    df["loss_state"], df["policy_state"] = loss_state, policy_state
    df = _policy(df, world, rng)
    df = _vehicle_value(df, world, rng)
    df["gt_true_damage"] = _true_damage(df, world, rng)
    df["gt_true_damage"] = _rarely_filed_below_deductible(df, comprehensive_causes, rng)
    df = _reporting(df, world, rng)
    df = _fraud(df, world, rng)
    df = _plant_traps(df, world, rng)

    df["policy_start_date"] = df["loss_date"] - pd.to_timedelta(df["days_since_start"], unit="D")
    df["report_date"] = df["loss_date"] + pd.to_timedelta(df["notice_days"], unit="D")
    df = df.drop(columns=["days_since_start"])
    df.insert(0, "claim_id", [f"CLM-{i:06d}" for i in range(n)])
    df.insert(1, "policy_id", [f"POL-{i:06d}" for i in rng.permutation(n)])
    return df
