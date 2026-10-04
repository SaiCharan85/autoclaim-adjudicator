"""Benchmark the five classifiers on a dataset spec's time-ordered splits.

Default: train -> validation only. Every model/protocol decision is made here.
--final: additionally refit on train+validation and score the locked test set once.
Each number is mean +/- std over the configured seeds; confidence intervals and the paired
"difference vs CatBoost" use the seed-averaged predictions.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from autoclaim.config import FraudModelConfig
from autoclaim.ml.frame import AMOUNT, Y, feature_list
from autoclaim.ml.metrics import (
    bootstrap_ci,
    net_savings_per_1k,
    paired_bootstrap_diff,
    recall_at_budget,
    score_report,
)
from autoclaim.ml.models import MODELS, make_model
from autoclaim.ml.spec import DatasetSpec
from autoclaim.ml.split import assert_time_ordered, latest_slice

ES_FRACTION = 0.2
REFERENCE_MODEL = "catboost"
OVERFIT_GAP = 0.15  # train PR-AUC minus evaluation PR-AUC above this is flagged


@dataclass
class Run:
    model: str
    seed: int
    metrics: dict[str, float]
    fit_seconds: float
    n_trees: int | None
    scores: np.ndarray = field(repr=False)


@dataclass
class Benchmark:
    validation: pd.DataFrame
    test: pd.DataFrame | None
    features: list[str]


def _fit(
    name: str, features: Sequence[str], fit_set: pd.DataFrame, seed: int
) -> tuple[object, float]:
    model = make_model(name, features, seed)
    es = None
    train = fit_set
    if model.needs_val:
        train, es = latest_slice(fit_set, ES_FRACTION)
    start = time.perf_counter()
    model.fit(train, es)
    return model, time.perf_counter() - start


def evaluate(
    name: str,
    features: Sequence[str],
    fit_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    seed: int,
    cfg: FraudModelConfig,
) -> Run:
    model, secs = _fit(name, features, fit_set, seed)
    scores = model.predict_proba(eval_set)  # type: ignore[attr-defined]
    metrics = score_report(eval_set[Y], scores, cfg.review_budget)
    train_scores = model.predict_proba(fit_set)  # type: ignore[attr-defined]
    metrics["train_pr_auc"] = float(average_precision_score(fit_set[Y], train_scores))
    if AMOUNT in eval_set:
        metrics["net_savings_per_1k"] = net_savings_per_1k(
            eval_set[Y], scores, eval_set[AMOUNT], cfg.review_budget, cfg.siu_review_cost_usd
        )
    return Run(name, seed, metrics, secs, model.n_trees, scores)  # type: ignore[attr-defined]


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def _summarize(
    runs: dict[str, list[Run]], y: np.ndarray, cfg: FraudModelConfig, seed: int
) -> pd.DataFrame:
    recall = partial(recall_at_budget, budget=cfg.review_budget)
    mean_scores = {name: np.mean([r.scores for r in rs], axis=0) for name, rs in runs.items()}
    reference = mean_scores.get(REFERENCE_MODEL)
    rows = []
    for name, rs in runs.items():
        m = pd.DataFrame([r.metrics for r in rs])

        def ms(col: str, digits: int = 4, frame: pd.DataFrame = m) -> str:
            std = frame[col].std(ddof=1) if len(frame) > 1 else 0.0
            return f"{frame[col].mean():.{digits}f} +/- {std:.{digits}f}"

        gap = (m["train_pr_auc"] - m["pr_auc"]).mean()
        pr_lo, pr_hi = bootstrap_ci(y, mean_scores[name], _pr_auc, seed=seed)
        rc_lo, rc_hi = bootstrap_ci(y, mean_scores[name], recall, seed=seed)
        row: dict[str, object] = {
            "model": name,
            "pr_auc": ms("pr_auc"),
            "pr_auc_ci": f"[{pr_lo:.3f}, {pr_hi:.3f}]",
            "recall@budget": ms("recall_at_budget"),
            "recall_ci": f"[{rc_lo:.3f}, {rc_hi:.3f}]",
            "roc_auc": ms("roc_auc"),
        }
        if "net_savings_per_1k" in m:
            row["net_savings_per_1k_usd"] = ms("net_savings_per_1k", 0)
        if reference is not None and name != REFERENCE_MODEL:
            d, lo, hi = paired_bootstrap_diff(y, mean_scores[name], reference, recall, seed=seed)
            row[f"delta_recall_vs_{REFERENCE_MODEL}"] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        else:
            row[f"delta_recall_vs_{REFERENCE_MODEL}"] = "n/a"
        row["train_minus_eval_pr_auc"] = f"{gap:.3f}" + (" OVERFIT?" if gap > OVERFIT_GAP else "")
        trees = [r.n_trees for r in rs if r.n_trees is not None]
        row["n_trees"] = int(np.mean(trees)) if trees else "n/a"
        row["fit_s"] = round(float(np.mean([r.fit_seconds for r in rs])), 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def _run_all(
    fit_set: pd.DataFrame,
    eval_set: pd.DataFrame,
    features: list[str],
    spec: DatasetSpec,
    cfg: FraudModelConfig,
    models: Sequence[str],
    ablation_model: str | None,
) -> pd.DataFrame:
    runs: dict[str, list[Run]] = {}
    for name in models:
        runs[name] = [evaluate(name, features, fit_set, eval_set, s, cfg) for s in cfg.seeds]
    if ablation_model is not None and spec.sensitive:
        with_sensitive = features + [c for c in spec.sensitive if c in fit_set]
        label = f"{ablation_model} + {'/'.join(spec.sensitive)}"
        runs[label] = [
            evaluate(ablation_model, with_sensitive, fit_set, eval_set, s, cfg) for s in cfg.seeds
        ]
    return _summarize(runs, eval_set[Y].to_numpy(), cfg, cfg.seeds[0])


def run_benchmark(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    cfg: FraudModelConfig,
    models: Sequence[str] = tuple(MODELS),
    final: bool = False,
    ablation_model: str | None = "catboost",
) -> Benchmark:
    train, val, test = spec.split(frame)
    assert_time_ordered(train, val, test)
    features = feature_list(frame, exclude=spec.sensitive)
    spec.check_features(features)
    validation = _run_all(train, val, features, spec, cfg, models, ablation_model)
    test_table = None
    if final:
        fit_set = pd.concat([train, val])
        test_table = _run_all(fit_set, test, features, spec, cfg, models, ablation_model)
    return Benchmark(validation, test_table, features)
