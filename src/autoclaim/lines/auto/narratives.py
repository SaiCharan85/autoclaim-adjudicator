"""Row-conditioned claimant narratives (Step 4): an LLM writes the first-notice story for a
simulated claim from the facts the claimant knows, in a seeded "style card" (channel, voice,
typos, slang, one optional fact left out). Batched, cached, budgeted like every LLM call.

The generator never sees the fraud label, the coverage decision or any simulation truth, so the
story cannot leak the answer through tone. Facts a story leaves out are recorded, so intake's
`missing_info` can be scored.
"""

import json
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from autoclaim.lines.auto.claim import ClaimPackage, _rng_int, policy_record, story_facts
from autoclaim.llm.client import LLMClient
from autoclaim.llm.types import LLMUnavailableError

ROLE = "synthetic_generator"
log = logging.getLogger(__name__)
CHANNELS = ("phone", "web", "email", "app")
VOICES = ("calm and organized", "upset and rambling", "terse", "chatty with irrelevant details",
          "apologetic", "confused about dates but corrects themself")  # fmt: skip
TYPOS = ("none", "a few typos", "many typos and no capital letters")
# Facts a claimant often forgets; never the cause, driver or use (those decide coverage).
OMITTABLE = ("loss_hour", "weather", "light", "witness_count", "police_report_hours", "area")

INSTRUCTIONS = """You write realistic first-notice-of-loss statements for a US auto insurer's
test data. For each claim you get the facts the claimant knows and a style card. Write what the
claimant says or types, in first person (the policyholder; if someone else drove, the
policyholder reports what that driver told them).
Rules:
- Convey EVERY fact given, in natural words, except the one named in "leave_out" (no hints).
- Never add facts that would change coverage: no extra drivers, uses, causes, injuries or police
  reports beyond those given. Small harmless color (traffic, errands, feelings) is fine.
- Use everyday words, never insurance terms or field names. Examples: rideshare_active -> "I had
  the Uber/Lyft app on"; delivery_active -> "I was doing DoorDash/Instacart"; animal -> "a deer"
  or "a buck"; hit_and_run -> the other driver took off; mechanical_breakdown -> the engine or
  transmission failed by itself; parked_hit -> found it damaged while parked; vandalism -> keyed,
  smashed window; towed=1 -> had it towed; at_fault=1 -> admits causing it, at_fault=0 -> other
  driver's fault or no one's.
- police_report_hours: say roughly when police were called or the report was filed, in everyday
  terms ("right away", "the next morning", "about two days later"), never a decimal number;
  police_report=0: no report was made.
- Never use the words "named insured", "mechanical breakdown", "collision with an object",
  "functional damage", "disabling damage" or "in transport": describe what happened instead.
- Mention the driver's name and relationship exactly as given.
- Dates: use the loss_date (any natural format). Do not mention the claim amount.
- 60-170 words. Follow the style card's channel, voice and typos.
Return one item per claim id."""


class StyleCard(BaseModel):
    channel: str
    voice: str
    typos: str
    leave_out: str | None


class Narrative(BaseModel):
    id: str
    narrative: str = Field(min_length=80, max_length=2500)


class NarrativeBatch(BaseModel):
    items: list[Narrative]


def style_card(claim_id: str, facts: dict[str, Any]) -> StyleCard:
    present = [f for f in OMITTABLE if f in facts]
    leave = present[_rng_int(claim_id, "omit", len(present))] if present else None
    if _rng_int(claim_id, "omit?", 3) == 0:  # a third of stories are complete
        leave = None
    return StyleCard(
        channel=CHANNELS[_rng_int(claim_id, "channel", len(CHANNELS))],
        voice=VOICES[_rng_int(claim_id, "voice", len(VOICES))],
        typos=TYPOS[_rng_int(claim_id, "typos", len(TYPOS))],
        leave_out=leave,
    )


def _jsonable(facts: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in facts.items():
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    return out


def batch_prompt(rows: Sequence[pd.Series]) -> tuple[str, list[StyleCard]]:
    blocks, cards = [], []
    for row in rows:
        facts = story_facts(row)
        card = style_card(str(row["claim_id"]), facts)
        cards.append(card)
        blocks.append(json.dumps({"id": str(row["claim_id"]), "facts": _jsonable(facts),
                                  "style": card.model_dump()}, default=str))  # fmt: skip
    return "\n".join(blocks), cards


def select_claims(
    claims: pd.DataFrame, start: str, end: str, n: int, seed: int, trap_share: float = 0.5
) -> pd.DataFrame:
    """`n` claims with loss_date in [start, end): `trap_share` from claims exercising any of the
    six coverage traps (they are ~1% each in the population, too rare to evaluate otherwise)."""
    when = pd.to_datetime(claims["loss_date"])
    pool = claims[(when >= pd.Timestamp(start)) & (when < pd.Timestamp(end))]
    has_trap = pool["gt_traps"].fillna("").astype(str).str.len() > 0
    n_trap = min(round(n * trap_share), int(has_trap.sum()))
    rng = np.random.default_rng(seed)
    traps = pool[has_trap].sample(n=n_trap, random_state=rng)
    rest = pool[~has_trap].sample(n=n - n_trap, random_state=rng)
    return pd.concat([traps, rest]).sample(frac=1.0, random_state=rng)


def generate(
    client: LLMClient, rows: pd.DataFrame, batch_size: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Packages (with style + generator metadata) and the claim ids that failed."""
    records, failed = [], []
    for start in range(0, len(rows), batch_size):
        chunk = [r for _, r in rows.iloc[start : start + batch_size].iterrows()]
        user, cards = batch_prompt(chunk)
        try:
            res = client.structured(ROLE, INSTRUCTIONS, user, NarrativeBatch)
        except LLMUnavailableError as exc:
            log.warning("narrative batch failed: %s", exc)
            failed += [str(r["claim_id"]) for r in chunk]
            continue
        by_id = {item.id: item.narrative for item in res.value.items}
        for row, card in zip(chunk, cards, strict=True):
            cid = str(row["claim_id"])
            if cid not in by_id:
                failed.append(cid)
                continue
            package = ClaimPackage(
                claim_id=cid,
                channel=card.channel,  # type: ignore[arg-type]
                report_date=pd.Timestamp(row["report_date"]).date(),
                estimate_amount=float(row["claimed_amount"]),
                narrative=by_id[cid],
                policy=policy_record(row),
            )
            records.append({"package": package.model_dump(mode="json"),
                            "style": card.model_dump(), "generator": res.meta.model})  # fmt: skip
    if failed and batch_size > 1:  # a batch may skip a claim: retry those one at a time
        retry = rows[rows["claim_id"].astype(str).isin(failed)]
        more, failed = generate(client, retry, 1)
        records += more
    return records, failed
