from pathlib import Path

import pandas as pd
import pytest

from autoclaim.datasets import profile
from autoclaim.datasets.schema import validate
from autoclaim.datasets.vehicle_fraud import SCHEMA


@pytest.fixture
def toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "grp": ["a"] * 5 + ["b"] * 5,
            "const": ["k"] * 10,
            "id": list(range(10)),
        }
    )


def test_class_balance(toy: pd.DataFrame) -> None:
    out = profile.class_balance(toy, "y")
    assert out.loc[0, "count"] == 8
    assert out.loc[1, "count"] == 2
    assert out.loc[1, "share"] == 0.2


def test_missingness_combines_nulls_and_sentinels(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[[0, 1, 2], "Age"] = 0
    valid_frame["Make"] = valid_frame["Make"].astype(object)
    valid_frame.loc[5, "Make"] = None
    out = profile.missingness(valid_frame, validate(valid_frame, SCHEMA))
    assert out.loc["Age", "sentinels"] == 3
    assert out.loc["Age", "nulls"] == 0
    assert out.loc["Make", "nulls"] == 1
    assert out.index[0] == "Age"  # sorted by share, descending
    assert "Fault" not in out.index  # clean columns are omitted


def test_cardinality_notes(toy: pd.DataFrame) -> None:
    out = profile.cardinality(toy)
    assert out.loc["const", "note"] == "constant"
    assert out.loc["id", "note"] == "unique (ID-like)"
    assert out.loc["grp", "nunique"] == 2
    assert out.loc["grp", "top_share"] == 0.5


def test_near_constant_flag() -> None:
    df = pd.DataFrame({"c": ["x"] * 96 + ["y"] * 4})
    assert profile.cardinality(df).loc["c", "note"] == "near-constant"


def test_target_rate_by_filters_small_groups(toy: pd.DataFrame) -> None:
    out = profile.target_rate_by(toy, "y", "grp", min_count=1)
    assert out.loc["a", "rate"] == 0.4
    assert out.loc["b", "rate"] == 0.0
    assert out.index[0] == "a"
    assert profile.target_rate_by(toy, "y", "grp", min_count=6).empty


def test_target_rate_spread(toy: pd.DataFrame) -> None:
    out = profile.target_rate_spread(toy, "y", min_count=1)
    assert out.loc["grp", "spread"] == 0.4
    assert out.loc["grp", "highest_category"] == "a"
    assert "const" not in out.index  # single category -> no spread
    assert "y" not in out.index


def test_target_rate_spread_empty_when_nothing_qualifies(toy: pd.DataFrame) -> None:
    assert profile.target_rate_spread(toy, "y", min_count=100).empty


def test_to_markdown_shape() -> None:
    df = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="k"))
    assert profile.to_markdown(df).splitlines() == [
        "| k | a |",
        "|---|---|",
        "| x | 1 |",
        "| y | 2 |",
    ]


def test_to_markdown_keeps_int_columns_int() -> None:
    df = pd.DataFrame({"count": [14497, 923], "share": [0.9401, 0.0599]})
    assert profile.to_markdown(df).splitlines()[2] == "| 0 | 14497 | 0.9401 |"


def test_render_report_sections(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, "Age"] = 0
    text = profile.render_report(valid_frame, SCHEMA, time_column="Year")
    for heading in (
        "## Class balance",
        "## Target rate by `Year`",
        "## Schema findings",
        "## Missing / sentinel values",
        "## Cardinality",
        "## Univariate target-rate spread",
    ):
        assert heading in text
    assert f"**Rows:** {len(valid_frame):,}" in text


def test_render_report_omits_findings_when_clean(valid_frame: pd.DataFrame) -> None:
    assert "## Schema findings" not in profile.render_report(valid_frame, SCHEMA)


def test_main_writes_report(tmp_path: Path, valid_frame: pd.DataFrame) -> None:
    csv = tmp_path / "fraud_oracle.csv"
    valid_frame.to_csv(csv, index=False)
    out = tmp_path / "profile.md"
    assert profile.main(["--csv", str(csv), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Data profile: `vehicle_fraud`")
