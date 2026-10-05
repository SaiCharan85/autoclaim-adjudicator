"""Value labels for SAS releases (e.g. NHTSA GES): code -> official label text, per format name.

Labels come from a compiled catalog (formats.sas7bcat, read with pyreadstat) or from PROC FORMAT
source text (*.sas). Each data column names its format in the .sas7bdat metadata, so mapping by
label text is robust to code-scheme changes across years.
"""

import re
from pathlib import Path

import pyreadstat

Labels = dict[str, dict[float, str]]  # format name (upper, no trailing '.') -> code -> label

_VALUE_BLOCK = re.compile(r"\bVALUE\s+(\$?\w+)(.*?);", re.IGNORECASE | re.DOTALL)
_ENTRY = re.compile(r"([\w.]+(?:\s*-\s*[\w.]+)?)\s*=\s*'([^']*)'")


def _norm(name: str) -> str:
    return name.strip().rstrip(".").upper()


def parse_proc_format(text: str) -> Labels:
    """Numeric VALUE blocks of PROC FORMAT source; ranges like 1-3 expand to integers."""
    out: Labels = {}
    for name, body in _VALUE_BLOCK.findall(text):
        if name.startswith("$"):
            continue  # character formats: not used for numeric codes
        labels: dict[float, str] = {}
        for code, label in _ENTRY.findall(body):
            lo, _, hi = code.partition("-")
            try:
                if hi:
                    for v in range(int(float(lo)), int(float(hi)) + 1):
                        labels[float(v)] = label.strip()
                else:
                    labels[float(lo)] = label.strip()
            except ValueError:
                continue  # OTHER, LOW, HIGH ...
        out[_norm(name)] = labels
    return out


def read_labels(directory: Path) -> Labels:
    """All value labels found in a release folder (catalog first; PROC FORMAT text fills gaps)."""
    labels: Labels = {}
    for src in sorted(directory.glob("*.sas")):
        labels |= parse_proc_format(src.read_text(encoding="latin-1"))
    catalog = directory / "formats.sas7bcat"
    if catalog.exists():
        try:
            _, meta = pyreadstat.read_sas7bcat(str(catalog))
            for k, d in meta.value_labels.items():
                numeric = {}
                for c, v in d.items():
                    try:
                        numeric[float(c)] = str(v)
                    except (TypeError, ValueError):
                        continue  # character codes (e.g. state 'AL', '1a'): not used here
                if numeric:
                    labels[_norm(k)] = numeric
        except pyreadstat.ReadstatError:
            pass  # unreadable catalog build: the PROC FORMAT text (if any) stays
    return labels


def column_formats(path: Path) -> dict[str, str]:
    """Data column -> format name, from the .sas7bdat metadata (no rows read)."""
    _, meta = pyreadstat.read_sas7bdat(str(path), metadataonly=True)
    return {col: _norm(fmt) for col, fmt in meta.original_variable_types.items() if fmt}
