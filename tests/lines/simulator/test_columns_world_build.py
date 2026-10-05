from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from autoclaim.config import load_carrier_config
from autoclaim.lines.auto.simulator import build, columns
from autoclaim.lines.auto.simulator.generate import build_claims
from autoclaim.lines.auto.simulator.world import World, load_world, with_overrides

# ---------------------------------------------------------------- columns / leakage tags


def test_fnol_features_pass() -> None:
    columns.assert_fnol_only(["cause", "claimed_amount", "days_policy_to_loss"])


@pytest.mark.parametrize(
    "bad", ["gt_is_fraud", "gt_true_damage", "fraud_confirmed", "record_origin", "claim_id"]
)
def test_non_fnol_columns_rejected(bad: str) -> None:
    with pytest.raises(columns.LeakageError, match=bad):
        columns.assert_fnol_only(["cause", bad])


def test_untagged_column_rejected() -> None:
    with pytest.raises(columns.LeakageError, match="no availability tag"):
        columns.assert_fnol_only(["mystery_feature"])


def test_every_built_column_is_tagged(world: World, incidents: pd.DataFrame) -> None:
    jur = load_carrier_config().active_jurisdiction
    frame = build.with_ground_truth(build_claims(world, incidents, n=300, seed=1), jur)
    for col in frame.columns:
        columns.availability(col)  # raises if untagged


# ---------------------------------------------------------------- world config


def test_world_loads_with_holdouts() -> None:
    w = load_world()
    assert w.holdouts is not None
    assert w.holdouts.fresh.seed != w.holdouts.shift.seed != w.seed
    assert sum(w.loss_causes.values()) == pytest.approx(1.0)


def test_overrides_nested_and_validated() -> None:
    w = with_overrides(load_world(), {"fraud.base_rate": 0.1, "n_claims": 10})
    assert (w.fraud.base_rate, w.n_claims) == (0.1, 10)
    with pytest.raises(KeyError, match="unknown world setting"):
        with_overrides(load_world(), {"fraud.nope": 1})
    with pytest.raises(ValidationError):
        with_overrides(load_world(), {"coverage_mix": {"collision": 0.5}})  # doesn't sum to 1


def test_bad_distribution_rejected(tmp_path: Path) -> None:
    import yaml

    from autoclaim.lines.auto.simulator.world import DEFAULT_WORLD_PATH

    raw = yaml.safe_load(DEFAULT_WORLD_PATH.read_text(encoding="utf-8"))
    raw["use_at_loss"]["personal"] = 0.9
    path = tmp_path / "w.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="use_at_loss"):
        load_world(path)


def test_unknown_state_rejected() -> None:
    data = load_world().model_dump()
    data["states"]["ZZ"] = 0.1
    with pytest.raises(ValidationError, match="census region"):
        World.model_validate(data)


# ---------------------------------------------------------------- build_all / write / load


@pytest.fixture(scope="module")
def built(world: World, incidents: pd.DataFrame) -> dict[str, pd.DataFrame]:
    small = world.model_copy(
        update={
            "holdouts": world.holdouts.model_copy(
                update={
                    "fresh": world.holdouts.fresh.model_copy(update={"n_claims": 300}),
                    "shift": world.holdouts.shift.model_copy(update={"n_claims": 300}),
                }
            )
        }
    )
    return build.build_all(small, incidents, load_carrier_config().active_jurisdiction, n=1500)


def test_build_all_sets_are_disjoint_and_in_window(built: dict[str, pd.DataFrame]) -> None:
    records = [set(f["source_record"]) for f in built.values()]
    assert records[0].isdisjoint(records[1])
    assert records[0].isdisjoint(records[2])
    assert records[1].isdisjoint(records[2])
    for name in ("holdout_fresh", "holdout_shift"):
        assert built[name]["loss_date"].min() >= pd.Timestamp("2024-07-01")


def test_build_all_attaches_ground_truth(built: dict[str, pd.DataFrame]) -> None:
    for frame in built.values():
        assert {"gt_decision", "gt_reasons", "gt_payout", "gt_traps"} <= set(frame.columns)
        assert frame["gt_decision"].isin(["approve", "deny", "escalate"]).all()


def test_shift_holdout_uses_overrides(built: dict[str, pd.DataFrame]) -> None:
    # shift moves no-comprehensive policies 17% -> 28%: big enough to see on 300 claims
    no_comp = {k: (v["coverage"] != "collision_comprehensive").mean() for k, v in built.items()}
    assert no_comp["holdout_shift"] > no_comp["holdout_fresh"]


def test_write_and_load_round_trip(built: dict[str, pd.DataFrame], tmp_path: Path) -> None:
    out = build.write(built, tmp_path, {"seed": 1})
    manifest = (out / "manifest.json").read_text(encoding="utf-8")
    assert '"claims.csv"' in manifest
    assert '"seed": 1' in manifest
    back = build.load("claims", tmp_path)
    assert len(back) == len(built["claims"])
    assert pd.api.types.is_datetime64_any_dtype(back["loss_date"])


def test_load_missing_gives_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"simulate_claims\.py"):
        build.load("claims", tmp_path)


def test_stage_features_allow_appraisal_only_at_stage_two() -> None:
    columns.assert_stage_features(["cause", "log_claim_to_appraisal"], "appraisal")
    columns.assert_stage_features(["cause"], "fnol")
    with pytest.raises(columns.LeakageError, match="appraisal"):
        columns.assert_stage_features(["appraised_amount"], "fnol")
    with pytest.raises(columns.LeakageError):
        columns.assert_stage_features(["fraud_confirmed"], "appraisal")
    with pytest.raises(columns.LeakageError):
        columns.assert_fnol_only(["appraiser_prior_damage"])
