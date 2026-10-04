"""Schema and loader for the Kaggle "Vehicle Insurance Claim Fraud Detection" table.

File: fraud_oracle.csv.

Most columns are pre-binned strings (e.g. VehiclePrice "20000 to 29000"), not raw numbers.
"""

from pathlib import Path

import pandas as pd

from autoclaim.datasets.schema import ColumnSpec, Kind, TableSchema, ValidationReport, validate
from autoclaim.datasets.sources import VEHICLE_FRAUD
from autoclaim.paths import raw_dir

TARGET = "FraudFound_P"

MONTHS = frozenset(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)
WEEKDAYS = frozenset(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
YES_NO = frozenset(["Yes", "No"])


def _cat(
    name: str, allowed: frozenset[str] | None = None, sentinels: frozenset[str] = frozenset()
) -> ColumnSpec:
    return ColumnSpec(name=name, kind=Kind.CATEGORICAL, allowed=allowed, sentinels=sentinels)


def _int(name: str, lo: int, hi: int, sentinels: frozenset[str] = frozenset()) -> ColumnSpec:
    return ColumnSpec(name=name, kind=Kind.INTEGER, min_value=lo, max_value=hi, sentinels=sentinels)


SCHEMA = TableSchema(
    name="vehicle_fraud",
    target=TARGET,
    unique_key="PolicyNumber",
    columns=(
        # when the accident happened: no exact date; Year + Month + WeekOfMonth = coarse time axis
        _cat("Month", MONTHS),
        _int("WeekOfMonth", 1, 5),
        _cat("DayOfWeek", WEEKDAYS),
        # when it was claimed; "0" marks an unknown claim date
        _cat("DayOfWeekClaimed", WEEKDAYS, sentinels=frozenset(["0"])),
        _cat("MonthClaimed", MONTHS, sentinels=frozenset(["0"])),
        _int("WeekOfMonthClaimed", 1, 5),
        _int("Year", 1994, 1996),
        # vehicle
        _cat("Make"),
        _cat("VehicleCategory", frozenset(["Sport", "Sedan", "Utility"])),
        _cat("VehiclePrice"),
        _cat("AgeOfVehicle"),
        _cat("NumberOfCars"),
        # accident circumstances
        _cat("AccidentArea", frozenset(["Urban", "Rural"])),
        _cat("Fault", frozenset(["Policy Holder", "Third Party"])),
        _cat("PoliceReportFiled", YES_NO),
        _cat("WitnessPresent", YES_NO),
        # policyholder; Age 0 is a known placeholder for "unknown"
        _cat("Sex", frozenset(["Male", "Female"])),
        _cat("MaritalStatus", frozenset(["Single", "Married", "Widow", "Divorced"])),
        _int("Age", 16, 100, sentinels=frozenset(["0"])),
        _cat("AgeOfPolicyHolder"),
        _int("DriverRating", 1, 4),
        # policy and claim history
        _int("PolicyNumber", 1, 10_000_000),
        _int("RepNumber", 1, 100),
        _int("Deductible", 0, 10_000),
        _cat("PolicyType"),
        _cat("BasePolicy", frozenset(["Liability", "Collision", "All Perils"])),
        _cat("Days_Policy_Accident"),
        _cat("Days_Policy_Claim"),
        _cat("PastNumberOfClaims"),
        _cat("NumberOfSuppliments"),
        _cat("AddressChange_Claim"),
        _cat("AgentType", frozenset(["External", "Internal"])),
        ColumnSpec(name=TARGET, kind=Kind.BINARY),
    ),
)


def csv_path(data_root: Path | None = None) -> Path:
    base = data_root if data_root is not None else raw_dir(VEHICLE_FRAUD.key)
    return base / VEHICLE_FRAUD.files[0]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the CSV as-is (no cleaning). Raises FileNotFoundError with a hint if not downloaded."""
    path = path if path is not None else csv_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python scripts/download_data.py")
    return pd.read_csv(path)


def load_validated(path: Path | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    df = load_raw(path)
    report = validate(df, SCHEMA)
    if not report.ok:
        details = "; ".join(f"{e.column}:{e.check}({e.n_rows})" for e in report.errors)
        raise ValueError(f"{SCHEMA.name} failed schema validation: {details}")
    return df, report
