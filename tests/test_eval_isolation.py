"""Leakage guard: the harness never imports simulation truth or eval-only code."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
EVAL_ONLY = {"autoclaim.lines.auto.simulator.oracle", "autoclaim.lines.auto.judge_eval",
             "autoclaim.lines.auto.simulator.adjuster"}  # fmt: skip
EVAL_ONLY_PREFIXES = ("autoclaim.finetune",)  # training data builders read truth too
# Data-building code (the simulator) may use the oracle to write ground-truth columns.
ALLOWED_PREFIXES = ("autoclaim.lines.auto.simulator", *EVAL_ONLY_PREFIXES)


def _is_eval_only(name: str) -> bool:
    return name in EVAL_ONLY or name.startswith(EVAL_ONLY_PREFIXES)


def _module(path: Path) -> str:
    return ".".join(path.relative_to(SRC).with_suffix("").parts).removesuffix(".__init__")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


def test_no_harness_module_imports_eval_only_code() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        mod = _module(path)
        if mod in EVAL_ONLY or mod.startswith(ALLOWED_PREFIXES):
            continue
        hits = {name for name in _imports(path) if _is_eval_only(name)}
        if hits:
            offenders.append(f"{mod}: {sorted(hits)}")
    assert not offenders, f"eval-only modules imported by harness code: {offenders}"


def test_guard_sees_the_known_importers() -> None:
    # sanity: the scan does find the simulator's (allowed) use of the oracle
    build = SRC / "autoclaim" / "lines" / "auto" / "simulator" / "build.py"
    assert "autoclaim.lines.auto.simulator.oracle" in _imports(build)
