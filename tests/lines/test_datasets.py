import numpy as np
import pandas as pd
import pytest

from autoclaim.config import load_carrier_config
from autoclaim.lines.auto import datasets as ds
from autoclaim.lines.auto.simulator.columns import LeakageError
from autoclaim.lines.auto.simulator.generate import build_claims
from autoclaim.lines.auto.simulator.world import load_world
from autoclaim.ml.frame import AMOUNT, T, Y, feature_list
from autoclaim.ml.split import assert_time_ordered
from fakes import make_incidents

CFG = load_carrier_config().fraud_model


@pytest.fixture(scope="module")
def sim_raw() -> pd.DataFrame:
    world = load_world().model_copy(update={"n_claims": 2000})
    return build_claims(world, make_incidents(scale=0.5))


def test_sim_features_derivations(sim_raw: pd.DataFrame) -> None:
    f = ds.sim_features(sim_raw)
    expected = (sim_raw["loss_date"] - sim_raw["policy_start_date"]).dt.days
    np.testing.assert_array_equal(f["days_policy_to_loss"], expected)
    assert (f["days_policy_to_claim"] >= f["days_policy_to_loss"]).all()
    assert f["claim_to_acv"].iloc[0] == pytest.approx(
        sim_raw["claimed_amount"].iloc[0] / sim_raw["vehicle_acv"].iloc[0]
    )
    assert set(f["police_report"].unique()) <= {0.0, 1.0}
    assert f["address_change_days"].notna().all()  # "no move" is a known value, never NaN
    assert (f["out_of_state"] == (sim_raw["loss_state"] != sim_raw["policy_state"])).all()


def test_sim_features_never_include_ground_truth_or_provenance(sim_raw: pd.DataFrame) -> None:
    cols = ds.sim_features(sim_raw).columns
    assert not [c for c in cols if c.startswith("gt_")]
    for banned in ("fraud_confirmed", "record_origin", "source_record", "claim_id", "loss_date"):
        assert banned not in cols


def test_sim_prepare_contract(sim_raw: pd.DataFrame) -> None:
    frame = ds.sim_prepare(sim_raw)
    assert frame[Y].tolist() == sim_raw["fraud_confirmed"].astype(int).tolist()
    assert frame[AMOUNT].tolist() == sim_raw["claimed_amount"].tolist()
    expected_t = (sim_raw["loss_date"] - pd.Timestamp("1970-01-01")).dt.days
    assert frame[T].tolist() == expected_t.tolist()  # time key = loss date as a day number


def test_sim_spec_split_is_time_ordered_and_guarded(sim_raw: pd.DataFrame) -> None:
    spec = ds.get_spec("sim_us", CFG)
    frame = spec.prepare(sim_raw)
    train, val, test = spec.split(frame)
    assert_time_ordered(train, val, test)
    assert len(train) + len(val) + len(test) == len(frame)
    spec.check_features(feature_list(frame))  # all FNOL: passes
    with pytest.raises(LeakageError):
        spec.check_features(["gt_is_fraud"])


def test_legacy_canonical_mapping(valid_frame: pd.DataFrame) -> None:
    raw = valid_frame.assign(
        Days_Policy_Accident="1 to 7",
        AddressChange_Claim="under 6 months",
        Fault="Policy Holder",
        PoliceReportFiled="No",
        WitnessPresent="No",
        AccidentArea="Rural",
    )
    c = ds.legacy_canonical(raw)
    row = c.iloc[0]
    assert (row.days_policy_to_loss, row.address_change_days) == (4.0, 90.0)
    no_move = ds.legacy_canonical(raw.assign(AddressChange_Claim="no change")).iloc[0]
    assert no_move.address_change_days == ds.NO_RECENT_MOVE_DAYS  # known, not missing
    assert (row.at_fault, row.police_report, row.witness_count, row.area) == (
        1.0,
        0.0,
        0.0,
        "rural",
    )


def test_legacy_split_years(legacy_spec, fraud_std: pd.DataFrame) -> None:
    train, val, test = legacy_spec.split(fraud_std)
    assert set(test[T] // 100) == {1996}
    assert set(pd.concat([train, val])[T] // 100) <= {1994, 1995}
    assert_time_ordered(train, val, test)


def test_unknown_dataset() -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        ds.get_spec(
            "nope",
            CFG.model_copy(update={"datasets": {**CFG.datasets, "nope": CFG.datasets["sim_us"]}}),
        )


def test_sim_appraisal_stage(sim_raw: pd.DataFrame) -> None:
    spec = ds.get_spec("sim_us_appraisal", CFG)
    frame = spec.prepare(sim_raw)
    expected = np.log(sim_raw["claimed_amount"] / sim_raw["appraised_amount"])
    assert np.allclose(frame["log_claim_to_appraisal"], expected)
    features = feature_list(frame)
    spec.check_features(features)  # appraisal features allowed at stage 2 ...
    with pytest.raises(LeakageError):
        ds.get_spec("sim_us", CFG).check_features(features)  # ... never at first notice
    partial = ds.sim_appraisal_features(sim_raw.drop(columns=["appraised_amount"]))
    assert partial["log_claim_to_appraisal"].isna().all()


def test_decades_specs_registered() -> None:
    one = ds.get_spec("sim_decades", CFG)
    two = ds.get_spec("sim_decades_appraisal", CFG)
    one.check_features(["cause", "claimed_amount"])
    two.check_features(["cause", "log_claim_to_appraisal"])
    with pytest.raises(LeakageError):
        one.check_features(["log_claim_to_appraisal"])
