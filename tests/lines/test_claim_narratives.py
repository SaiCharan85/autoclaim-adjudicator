import json
import re

import pandas as pd
import pytest

from autoclaim.config import load_carrier_config
from autoclaim.lines.auto import narratives as nv
from autoclaim.lines.auto.claim import ClaimPackage, people, policy_record, story_facts
from autoclaim.lines.auto.simulator.generate import build_claims
from autoclaim.lines.auto.simulator.oracle import trap_families
from autoclaim.lines.auto.simulator.world import load_world
from autoclaim.llm.cache import LLMCache
from autoclaim.llm.client import LLMClient
from autoclaim.llm.types import ChatRequest, ChatResult
from autoclaim.llm.usage import UsageLedger
from fakes import make_incidents


@pytest.fixture(scope="module")
def claims() -> pd.DataFrame:
    df = build_claims(load_world().model_copy(update={"n_claims": 400}), make_incidents(scale=0.5))
    df["gt_traps"] = trap_families(df, load_carrier_config().active_jurisdiction)
    return df


def test_people_match_driver_role(claims: pd.DataFrame) -> None:
    for _, row in claims.head(80).iterrows():
        insured, listed, excluded, driver = people(row)
        assert people(row)[3] == driver  # deterministic
        assert insured not in excluded and listed[0] not in excluded
        role = row["driver_role"]
        if role == "named_insured":
            assert driver.name == insured
        elif role == "listed_driver":
            assert driver.name in listed
        elif role == "excluded_driver":
            assert driver.name in excluded
        else:
            assert driver.name not in (insured, *listed, *excluded)


def test_policy_record_and_story_hide_truth(claims: pd.DataFrame) -> None:
    row = claims.iloc[0]
    rec = policy_record(row)
    assert rec.vehicle.actual_cash_value == pytest.approx(row["vehicle_acv"])
    assert (rec.vehicle.lienholder is not None) == bool(row["financed"])
    facts = story_facts(row)
    leaked = [k for k in facts if k.startswith("gt_") or k in ("fraud_confirmed", "claimed_amount")]
    assert leaked == [] and facts["driver"]["name"]
    json.dumps(facts, default=str)  # serializable for the prompt


def test_style_card_never_omits_decisive_facts(claims: pd.DataFrame) -> None:
    for _, row in claims.head(100).iterrows():
        cid = str(row["claim_id"])
        card = nv.style_card(cid, story_facts(row))
        assert card == nv.style_card(cid, story_facts(row))
        assert card.leave_out in (None, *nv.OMITTABLE)


def test_select_claims_enriches_traps_and_respects_period(claims: pd.DataFrame) -> None:
    sel = nv.select_claims(claims, "2023-01-01", "2024-01-01", 40, seed=0)
    when = pd.to_datetime(sel["loss_date"])
    assert len(sel) == 40 and when.min() >= pd.Timestamp("2023-01-01")
    assert when.max() < pd.Timestamp("2024-01-01")
    assert (sel["gt_traps"].str.len() > 0).mean() >= 0.3


class BatchWriter:
    """Writes a narrative for every claim id in the prompt, except ids in `skip` when batched."""

    name = "p"

    def __init__(self, skip: set[str]) -> None:
        self.skip = skip
        self.calls = 0

    def chat(self, request: ChatRequest) -> ChatResult:
        self.calls += 1
        ids = re.findall(r'"id": "([^"]+)"', request.messages[-1].content)
        keep = [i for i in ids if len(ids) == 1 or i not in self.skip]
        text = "I hit a deer on the way home and the bumper is cracked, nobody hurt. " * 2
        items = [{"id": i, "narrative": text} for i in keep]
        return ChatResult(json.dumps({"items": items}), 10, 10, 0.0)


def test_generate_builds_packages_and_retries_skipped(claims: pd.DataFrame) -> None:
    cfg = load_carrier_config().models.model_copy(deep=True)
    key = cfg.roles[nv.ROLE].chain[0]
    provider_name = key.split(":")[0]
    rows = claims.head(6)
    writer = BatchWriter(skip={str(rows.iloc[1]["claim_id"])})
    ledger = UsageLedger(":memory:", cfg.catalog, 1.0, 0, sleep=lambda s: None)
    client = LLMClient(cfg, {provider_name: writer}, LLMCache(":memory:"), ledger)
    records, failed = nv.generate(client, rows, batch_size=3)
    assert failed == [] and len(records) == 6 and writer.calls == 3  # 2 batches + 1 retry
    pkg = ClaimPackage.model_validate(records[0]["package"])
    assert pkg.estimate_amount == pytest.approx(rows.iloc[0]["claimed_amount"])
    assert records[0]["generator"] == key
