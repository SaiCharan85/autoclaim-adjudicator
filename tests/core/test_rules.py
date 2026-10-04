from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from autoclaim.core.rules import Condition, Rule, RuleSet


def _rule(rid: str, weight: float, *conds: dict) -> Rule:
    return Rule(id=rid, description=rid, weight=weight, all_of=[Condition(**c) for c in conds])


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        ("eq", 2, [False, True, False]),
        ("ne", 2, [True, False, True]),
        ("lt", 2, [True, False, False]),
        ("le", 2, [True, True, False]),
        ("gt", 2, [False, False, True]),
        ("ge", 2, [False, True, True]),
        ("in", [1, 3], [True, False, True]),
        ("not_in", [1, 3], [False, True, False]),
    ],
)
def test_condition_ops(op: str, value: object, expected: list[bool]) -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert Condition(field="x", op=op, value=value).mask(df).tolist() == expected


@pytest.mark.parametrize("op", ["eq", "ne", "lt", "ge", "in", "not_in"])
def test_missing_value_never_fires(op: str) -> None:
    df = pd.DataFrame({"x": [np.nan, None]}, dtype=object)
    value = [1] if op in ("in", "not_in") else 1
    assert not Condition(field="x", op=op, value=value).mask(df).any()


def test_all_of_is_a_conjunction() -> None:
    rs = RuleSet(
        name="t",
        version="1",
        rules=[
            _rule(
                "r",
                0.5,
                {"field": "a", "op": "eq", "value": 1},
                {"field": "b", "op": "eq", "value": "y"},
            )
        ],
    )
    df = pd.DataFrame({"a": [1, 1, 0], "b": ["y", "n", "y"]})
    assert rs.evaluate(df)["r"].tolist() == [True, False, False]


def test_missing_column_does_not_fire_and_is_reported() -> None:
    rs = RuleSet(
        name="t", version="1", rules=[_rule("needs_z", 0.5, {"field": "z", "op": "eq", "value": 1})]
    )
    df = pd.DataFrame({"a": [1, 2]})
    assert not rs.evaluate(df)["needs_z"].any()
    assert rs.missing_fields(set(df.columns)) == {"needs_z": ["z"]}


def test_noisy_or_score() -> None:
    rs = RuleSet(
        name="t",
        version="1",
        rules=[
            _rule("a", 0.5, {"field": "x", "op": "ge", "value": 1}),
            _rule("b", 0.5, {"field": "x", "op": "ge", "value": 2}),
        ],
    )
    scores = rs.score(rs.evaluate(pd.DataFrame({"x": [0, 1, 2]})))
    assert scores.tolist() == pytest.approx([0.0, 0.5, 0.75])


def test_evaluate_record() -> None:
    rs = RuleSet(
        name="t",
        version="1",
        rules=[
            _rule("hit", 0.4, {"field": "x", "op": "eq", "value": "yes"}),
            _rule("miss", 0.4, {"field": "x", "op": "eq", "value": "no"}),
        ],
    )
    fired, score = rs.evaluate_record({"x": "yes"})
    assert [h.rule_id for h in fired] == ["hit"]
    assert score == pytest.approx(0.4)


def test_report_support_precision_lift() -> None:
    rs = RuleSet(
        name="t", version="1", rules=[_rule("r", 0.5, {"field": "x", "op": "eq", "value": 1})]
    )
    df = pd.DataFrame({"x": [1, 1, 0, 0], "y": [1, 0, 0, 0]})
    row = rs.report(df, "y").loc["r"]
    assert row["fires_on"] == 2
    assert row["support"] == 0.5
    assert row["fraud_rate_when_fired"] == 0.5
    assert row["lift"] == 2.0


def test_report_rule_that_never_fires() -> None:
    rs = RuleSet(
        name="t", version="1", rules=[_rule("r", 0.5, {"field": "x", "op": "eq", "value": 9})]
    )
    row = rs.report(pd.DataFrame({"x": [1], "y": [1]}), "y").loc["r"]
    assert row["fires_on"] == 0
    assert pd.isna(row["lift"])


def test_duplicate_ids_rejected() -> None:
    r = _rule("dup", 0.5, {"field": "x", "op": "eq", "value": 1})
    with pytest.raises(ValidationError, match="duplicate"):
        RuleSet(name="t", version="1", rules=[r, r])


def test_membership_op_needs_list() -> None:
    with pytest.raises(ValidationError, match="list"):
        Condition(field="x", op="in", value="a")


@pytest.mark.parametrize("weight", [0, 1.5])
def test_weight_bounds(weight: float) -> None:
    with pytest.raises(ValidationError):
        _rule("r", weight, {"field": "x", "op": "eq", "value": 1})


def test_rule_needs_a_condition() -> None:
    with pytest.raises(ValidationError):
        Rule(id="r", description="r", weight=0.5, all_of=[])


def test_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(
        "name: t\nversion: '1'\nrules:\n"
        "  - {id: a, description: d, weight: 0.3, all_of: [{field: x, op: eq, value: 1}]}\n",
        encoding="utf-8",
    )
    rs = RuleSet.from_yaml(path)
    assert rs.rules[0].fields == {"x"}
