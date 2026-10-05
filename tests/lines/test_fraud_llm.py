import math

import pandas as pd
import pytest

from autoclaim.core.rules import RuleHit
from autoclaim.lines.auto import fraud_llm
from autoclaim.lines.auto.fraud_tools import FraudSignals
from autoclaim.ml.explain import ShapReason


def test_describe_claim_skips_unknowns_and_formats_numbers() -> None:
    text = fraud_llm.describe_claim(
        pd.Series({"cause": "theft", "towed": 1.0, "claim_to_acv": 0.123456, "x": math.nan})
    )
    assert text == "cause=theft; towed=1; claim_to_acv=0.123"


def test_describe_signals() -> None:
    sig = FraudSignals(
        model_score=0.4567,
        top_reasons=[ShapReason(feature="claim_to_acv", value="0.91", contribution=0.8)],
        anomaly_score=0.9,
        rules_fired=[RuleHit(rule_id="late", description="d", weight=0.2)],
        rule_score=0.2,
        rules_unchecked=[],
        model_version="v",
    )
    text = fraud_llm.describe_signals(sig)
    assert text == "ml=0.457; anomaly=0.90; shap: claim_to_acv=0.91(+0.80); flags: late"


def test_batch_prompt_ids_and_tools() -> None:
    assert fraud_llm.batch_ids(3) == ["C01", "C02", "C03"]
    assert fraud_llm.batch_ids(100)[-1] == "C100"
    plain = fraud_llm.batch_prompt(["a=1", "b=2"])
    assert plain == "[C01] a=1\n[C02] b=2"
    hybrid = fraud_llm.batch_prompt(["a=1"], ["ml=0.1"])
    assert hybrid == "[C01] a=1\n  tools: ml=0.1"


def test_align_scores_marks_missing_and_ignores_invented() -> None:
    res = fraud_llm.BatchScores(
        scores=[
            fraud_llm.ClaimScore(id="c02", p=0.9, why="x"),
            fraud_llm.ClaimScore(id="C99", p=0.5, why="y"),
        ]
    )
    out = fraud_llm.align_scores(res, 2)
    assert math.isnan(out[0]) and out[1] == 0.9


def test_claim_score_bounds() -> None:
    with pytest.raises(ValueError):
        fraud_llm.ClaimScore(id="C01", p=1.5, why="x")


def test_hybrid_instructions_add_tool_glossary_only_there() -> None:
    assert "ml:" in fraud_llm.INSTRUCTIONS["hybrid"]
    assert "ml:" not in fraud_llm.INSTRUCTIONS["llm"]
    assert fraud_llm.INSTRUCTIONS["llm"].startswith(fraud_llm.INSTRUCTIONS["hybrid"][:200])
