"""Vectorized table schema checks.

Errors are structural (missing column, non-binary target, duplicate key): the table cannot be used.
Warnings are value-level anomalies (unexpected category, out-of-range number, sentinel values):
the table is usable, but the finding must be handled in cleaning and documented in
docs/data_notes.md.
"""

from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict


class Kind(StrEnum):
    CATEGORICAL = "categorical"
    INTEGER = "integer"
    BINARY = "binary"


class ColumnSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: Kind
    allowed: frozenset[str] | None = None  # categorical: expected values (None = any)
    min_value: int | None = None  # integer bounds, inclusive
    max_value: int | None = None
    sentinels: frozenset[str] = frozenset()  # values that mean "unknown", e.g. "0" or 0
    nullable: bool = False


class TableSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    columns: tuple[ColumnSpec, ...]
    target: str
    unique_key: str | None = None

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class Issue(BaseModel):
    column: str
    check: str
    n_rows: int
    examples: list[str] = []


class ValidationReport(BaseModel):
    table: str
    n_rows: int
    errors: list[Issue] = []
    warnings: list[Issue] = []

    @property
    def ok(self) -> bool:
        return not self.errors


def _examples(values: pd.Series, k: int = 5) -> list[str]:
    return [str(v) for v in values.drop_duplicates().head(k)]


def _check_column(spec: ColumnSpec, col: pd.Series) -> tuple[list[Issue], list[Issue]]:
    errors: list[Issue] = []
    warnings: list[Issue] = []

    nulls = col.isna()
    if nulls.any() and not spec.nullable:
        warnings.append(Issue(column=spec.name, check="null", n_rows=int(nulls.sum())))
    present = col[~nulls]
    as_str = present.astype(str)

    sentinel_mask = as_str.isin(spec.sentinels)
    if sentinel_mask.any():
        warnings.append(
            Issue(
                column=spec.name,
                check="sentinel",
                n_rows=int(sentinel_mask.sum()),
                examples=_examples(as_str[sentinel_mask]),
            )
        )
    present, as_str = present[~sentinel_mask], as_str[~sentinel_mask]

    if spec.kind is Kind.BINARY:
        # numeric compare: a column containing nulls is read as float (0.0 / 1.0)
        bad = ~pd.to_numeric(present, errors="coerce").isin([0, 1])
        if bad.any():
            errors.append(
                Issue(
                    column=spec.name,
                    check="binary",
                    n_rows=int(bad.sum()),
                    examples=_examples(as_str[bad]),
                )
            )
    elif spec.kind is Kind.INTEGER:
        numeric = pd.to_numeric(present, errors="coerce")
        non_int = numeric.isna() | (numeric % 1 != 0)
        if non_int.any():
            errors.append(
                Issue(
                    column=spec.name,
                    check="integer",
                    n_rows=int(non_int.sum()),
                    examples=_examples(as_str[non_int]),
                )
            )
        numeric = numeric[~non_int]
        out = pd.Series(False, index=numeric.index)
        if spec.min_value is not None:
            out |= numeric < spec.min_value
        if spec.max_value is not None:
            out |= numeric > spec.max_value
        if out.any():
            warnings.append(
                Issue(
                    column=spec.name,
                    check="range",
                    n_rows=int(out.sum()),
                    examples=_examples(numeric[out].astype(str)),
                )
            )
    elif spec.allowed is not None:
        unexpected = ~as_str.isin(spec.allowed)
        if unexpected.any():
            warnings.append(
                Issue(
                    column=spec.name,
                    check="allowed_values",
                    n_rows=int(unexpected.sum()),
                    examples=_examples(as_str[unexpected]),
                )
            )
    return errors, warnings


def validate(df: pd.DataFrame, schema: TableSchema) -> ValidationReport:
    report = ValidationReport(table=schema.name, n_rows=len(df))

    missing = [c for c in schema.column_names if c not in df.columns]
    for name in missing:
        report.errors.append(Issue(column=name, check="missing_column", n_rows=len(df)))
    extra = [c for c in df.columns if c not in schema.column_names]
    for name in extra:
        report.warnings.append(Issue(column=str(name), check="unexpected_column", n_rows=len(df)))

    for spec in schema.columns:
        if spec.name in missing:
            continue
        errors, warnings = _check_column(spec, df[spec.name])
        report.errors.extend(errors)
        report.warnings.extend(warnings)

    if schema.unique_key and schema.unique_key not in missing:
        dupes = df[schema.unique_key].duplicated(keep=False)
        if dupes.any():
            report.errors.append(
                Issue(
                    column=schema.unique_key,
                    check="unique",
                    n_rows=int(dupes.sum()),
                    examples=_examples(df.loc[dupes, schema.unique_key].astype(str)),
                )
            )
    return report
