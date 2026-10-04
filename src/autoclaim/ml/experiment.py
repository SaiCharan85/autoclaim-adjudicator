"""Compare fraud-model variants on VALIDATION only (the locked test set is never loaded here).

A variant = feature stages + model(s) + parameter overrides + optional amount weighting. Each is
trained over the configured seeds; results are paired against a reference variant with a
bootstrap 95% CI. Decision rule (pre-declared in docs/model_card.md): adopt a lever only if it
improves validation recall@budget with a CI that excludes zero (amount weighting: net savings).

Usage: uv run python scripts/fraud_experiments.py [--levers repair_cost weighting ensemble grid]
"""

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from autoclaim.config import FraudModelConfig, load_carrier_config
from autoclaim.datasets.profile import to_markdown
from autoclaim.ml.frame import AMOUNT, WEIGHT, Y, feature_list
from autoclaim.ml.metrics import (
    net_savings_per_1k,
    paired_bootstrap_diff,
    recall_at_budget,
    score_report,
)
from autoclaim.ml.models import make_model
from autoclaim.ml.spec import DatasetSpec
from autoclaim.ml.split import assert_time_ordered, latest_slice
from autoclaim.ml.stages import apply_stages, fit_stages, stage_outputs


@dataclass(frozen=True)
class Variant:
    name: str
    stages: tuple[str, ...] = ()
    models: tuple[str, ...] = ("catboost",)  # more than one = rank-averaged ensemble
    params: dict[str, Any] = field(default_factory=dict)  # CatBoost overrides
    weight_by_amount: bool = False


@dataclass
class VariantResult:
    variant: Variant
    metrics: pd.DataFrame  # one row per seed
    scores: np.ndarray  # seed-averaged validation scores


def _with_weights(frame: pd.DataFrame) -> pd.DataFrame:
    amount = frame[AMOUNT].clip(lower=1.0)
    return frame.assign(**{WEIGHT: (amount / amount.mean()).to_numpy()})


def _score_once(
    variant: Variant, features: list[str], train: pd.DataFrame, val: pd.DataFrame, seed: int
) -> tuple[np.ndarray, float | None]:
    """Validation scores for one seed (+ train-minus-val PR-AUC gap for single models)."""
    fit_set, es = latest_slice(train, 0.2)
    preds, gap = [], None
    for name in variant.models:
        params = variant.params if name == "catboost" else None
        model = make_model(name, features, seed, params)
        model.fit(fit_set, es if model.needs_val else None)
        p = model.predict_proba(val)
        preds.append(pd.Series(p).rank(pct=True).to_numpy() if len(variant.models) > 1 else p)
        if len(variant.models) == 1:
            gap = float(
                average_precision_score(fit_set[Y], model.predict_proba(fit_set))
                - average_precision_score(val[Y], p)
            )
    return np.mean(preds, axis=0), gap


def run_variant(
    variant: Variant,
    frame: pd.DataFrame,
    spec: DatasetSpec,
    cfg: FraudModelConfig,
) -> VariantResult:
    train, val, test = spec.split(frame)
    assert_time_ordered(train, val, test)
    del test  # never used: decisions are validation-only
    stages = [spec.stage_factories[name](cfg.seeds[0]) for name in variant.stages]
    train, val = fit_stages(stages, train), apply_stages(stages, val)
    features = feature_list(frame, exclude=spec.sensitive) + stage_outputs(stages)
    spec.check_features(features)
    if variant.weight_by_amount:
        train = _with_weights(train)
    rows, all_scores = [], []
    for seed in cfg.seeds:
        scores, gap = _score_once(variant, features, train, val, seed)
        m = score_report(val[Y], scores, cfg.review_budget)
        if AMOUNT in val:
            m["net_savings_per_1k"] = net_savings_per_1k(
                val[Y], scores, val[AMOUNT], cfg.review_budget, cfg.siu_review_cost_usd
            )
        m["train_minus_val_pr_auc"] = gap if gap is not None else float("nan")
        rows.append(m)
        all_scores.append(scores)
    return VariantResult(variant, pd.DataFrame(rows), np.mean(all_scores, axis=0))


def compare(
    results: Sequence[VariantResult], val: pd.DataFrame, cfg: FraudModelConfig, reference: str
) -> pd.DataFrame:
    ref = next(r for r in results if r.variant.name == reference)
    y = val[Y].to_numpy()
    recall = partial(recall_at_budget, budget=cfg.review_budget)
    rows = []
    for r in results:
        m = r.metrics

        def ms(col: str, digits: int = 4, frame: pd.DataFrame = m) -> str:
            sd = frame[col].std(ddof=1) if len(frame) > 1 else 0.0
            return f"{frame[col].mean():.{digits}f} +/- {sd:.{digits}f}"

        row: dict[str, object] = {
            "variant": r.variant.name,
            "pr_auc": ms("pr_auc"),
            "recall@budget": ms("recall_at_budget"),
            "train_minus_val_pr_auc": (
                f"{m['train_minus_val_pr_auc'].mean():.3f}"
                if m["train_minus_val_pr_auc"].notna().any()
                else "n/a"
            ),
        }
        if r is ref:
            row["delta_recall_vs_ref"] = "reference"
        else:
            d, lo, hi = paired_bootstrap_diff(y, r.scores, ref.scores, recall, seed=cfg.seeds[0])
            row["delta_recall_vs_ref"] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        if "net_savings_per_1k" in m:
            row["net_savings_per_1k_usd"] = ms("net_savings_per_1k", 0)
            if r is not ref:
                amount = val[AMOUNT].to_numpy()
                d, lo, hi = _paired_savings(y, r.scores, ref.scores, amount, cfg)
                row["delta_savings_vs_ref"] = f"{d:+.0f} [{lo:+.0f}, {hi:+.0f}]"
            else:
                row["delta_savings_vs_ref"] = "reference"
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def _paired_savings(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    amount: np.ndarray,
    cfg: FraudModelConfig,
    n_boot: int = 1000,
) -> tuple[float, float, float]:
    """Paired bootstrap of the net-savings difference (amounts resampled with their claims)."""

    def sav(idx: np.ndarray, s: np.ndarray) -> float:
        return net_savings_per_1k(
            y[idx], s[idx], amount[idx], cfg.review_budget, cfg.siu_review_cost_usd
        )

    rng = np.random.default_rng(cfg.seeds[0])
    full = np.arange(len(y))
    point = sav(full, a) - sav(full, b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(sav(idx, a) - sav(idx, b))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(point), float(lo), float(hi)


LEVERS: dict[str, list[Variant]] = {
    "repair_cost": [Variant("+ repair_cost residual", stages=("repair_cost",))],
    "weighting": [
        Variant("+ repair_cost + amount weights", stages=("repair_cost",), weight_by_amount=True)
    ],
    "ensemble": [
        Variant(
            "+ repair_cost, ensemble cat+lgbm+ebm",
            stages=("repair_cost",),
            models=("catboost", "lightgbm", "ebm"),
        )
    ],
    "grid": [
        Variant(
            f"+ repair_cost, depth {d} l2 {l2}",
            stages=("repair_cost",),
            params={"depth": d, "l2_leaf_reg": l2},
        )
        for d, l2 in ((3, 10), (5, 10), (4, 30), (6, 30))
    ],
}


def main(argv: Sequence[str] | None = None) -> int:
    from autoclaim.lines.auto.datasets import get_spec

    cfg = load_carrier_config().fraud_model
    parser = argparse.ArgumentParser(description="Compare fraud-model variants on validation.")
    parser.add_argument("--dataset", default=cfg.production_dataset, choices=sorted(cfg.datasets))
    parser.add_argument("--levers", nargs="*", default=["repair_cost"], choices=sorted(LEVERS))
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    spec = get_spec(args.dataset, cfg)
    frame = spec.prepare(spec.load_raw())
    variants = [Variant("baseline")] + [v for lever in args.levers for v in LEVERS[lever]]
    start = time.perf_counter()
    results = [run_variant(v, frame, spec, cfg) for v in variants]
    _, val, _ = spec.split(frame)
    table = compare(results, val, cfg, args.reference)
    print(f"{len(variants)} variants in {time.perf_counter() - start:.0f}s (validation only)")
    print(table.to_string())
    if args.report:
        Path(args.report).write_text(to_markdown(table, "variant") + "\n", encoding="utf-8")
    return 0
