import pandas as pd
import pytest

from autoclaim.datasets.schema import ColumnSpec, Kind, TableSchema, validate
from autoclaim.datasets.vehicle_fraud import SCHEMA


def _checks(issues, column):
    return {i.check for i in issues if i.column == column}


def test_valid_frame_has_no_issues(valid_frame: pd.DataFrame) -> None:
    report = validate(valid_frame, SCHEMA)
    assert report.ok
    assert report.errors == [] and report.warnings == []
    assert report.n_rows == len(valid_frame)


def test_missing_column_is_error(valid_frame: pd.DataFrame) -> None:
    report = validate(valid_frame.drop(columns=["Fault"]), SCHEMA)
    assert not report.ok
    assert "missing_column" in _checks(report.errors, "Fault")


def test_unexpected_column_is_warning(valid_frame: pd.DataFrame) -> None:
    report = validate(valid_frame.assign(Extra=1), SCHEMA)
    assert report.ok
    assert "unexpected_column" in _checks(report.warnings, "Extra")


@pytest.mark.parametrize("bad", [2, -1])
def test_non_binary_target_is_error(valid_frame: pd.DataFrame, bad: int) -> None:
    valid_frame.loc[0, "FraudFound_P"] = bad
    report = validate(valid_frame, SCHEMA)
    assert "binary" in _checks(report.errors, "FraudFound_P")


def test_duplicate_key_is_error(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[1, "PolicyNumber"] = valid_frame.loc[0, "PolicyNumber"]
    report = validate(valid_frame, SCHEMA)
    issue = next(i for i in report.errors if i.check == "unique")
    assert issue.n_rows == 2


def test_out_of_range_integer_is_warning_with_count(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[[0, 1, 2], "DriverRating"] = 9
    report = validate(valid_frame, SCHEMA)
    assert report.ok
    issue = next(i for i in report.warnings if i.column == "DriverRating")
    assert (issue.check, issue.n_rows, issue.examples) == ("range", 3, ["9"])


def test_non_integer_value_is_error(valid_frame: pd.DataFrame) -> None:
    frame = valid_frame.astype({"Deductible": object})
    frame.loc[0, "Deductible"] = "four hundred"
    report = validate(frame, SCHEMA)
    assert "integer" in _checks(report.errors, "Deductible")


def test_sentinel_counted_and_excluded_from_range_check(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[[0, 1], "Age"] = 0  # 0 is a sentinel, and also below min_value 16
    report = validate(valid_frame, SCHEMA)
    assert _checks(report.warnings, "Age") == {"sentinel"}
    assert next(i for i in report.warnings if i.column == "Age").n_rows == 2


def test_categorical_sentinel_and_unexpected_value(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, "MonthClaimed"] = "0"
    valid_frame.loc[1, "MonthClaimed"] = "Smarch"
    report = validate(valid_frame, SCHEMA)
    assert _checks(report.warnings, "MonthClaimed") == {"sentinel", "allowed_values"}


def test_unrestricted_categorical_accepts_anything(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, "Make"] = "Nisson"  # the source data has misspelled makes; that's fine
    assert validate(valid_frame, SCHEMA).warnings == []


def test_null_in_non_nullable_column_is_warning() -> None:
    schema = TableSchema(name="t", target="y", columns=(ColumnSpec(name="y", kind=Kind.BINARY),))
    report = validate(pd.DataFrame({"y": [0, 1, None]}), schema)
    assert report.ok
    assert _checks(report.warnings, "y") == {"null"}


def test_nullable_column_allows_nulls() -> None:
    schema = TableSchema(
        name="t",
        target="y",
        columns=(ColumnSpec(name="y", kind=Kind.BINARY, nullable=True),),
    )
    assert validate(pd.DataFrame({"y": [0, None]}), schema).warnings == []


def test_examples_are_deduplicated_and_capped(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[:, "AccidentArea"] = [f"zone{i % 8}" for i in range(len(valid_frame))]
    issue = next(i for i in validate(valid_frame, SCHEMA).warnings if i.column == "AccidentArea")
    assert issue.n_rows == len(valid_frame)
    assert len(issue.examples) == 5 and len(set(issue.examples)) == 5


def test_schema_has_33_columns_and_target_last_known() -> None:
    assert len(SCHEMA.columns) == 33
    assert SCHEMA.target in SCHEMA.column_names
    assert len(set(SCHEMA.column_names)) == 33
