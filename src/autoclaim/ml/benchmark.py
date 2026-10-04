"""Benchmark the five classifiers on a time split (headline) and stratified CV (stability)."""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from autoclaim.config import FraudModelConfig
from autoclaim.ml.features import TARGET, feature_columns
from autoclaim.ml.metrics import (
    bootstrap_ci,
    paired_bootstrap_diff,
    recall_at_budget,
    score_report,
)
from autoclaim.ml.models import MODELS, make_model
from autoclaim.ml.split import latest_slice, stratified_folds, stratified_holdout, time_split

VAL_FRACTION = 0.2
REFERENCE_MODEL = "catboost"


@dataclass(frozen=True)
class RunResult:
    model: str
    variant: str
    metrics: dict[str, float]
    fit_seconds: float
    n_trees: int | None = None
    y_test: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    scores: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)


def _fit_predict(
    name: str,
    features: Sequence[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    time_ordered: bool,
) -> tuple[np.ndarray, float, int | None]:
    model = make_model(name, features, seed)
    val = None
    if model.needs_val:
        if time_ordered:
            train, val = latest_slice(train, VAL_FRACTION)
        else:
            train, val = stratified_holdout(train, VAL_FRACTION, seed)
    start = time.perf_counter()
    model.fit(train, val)
    elapsed = time.perf_counter() - start
    return model.predict_proba(test), elapsed, model.n_trees


def evaluate_time_split(
    df: pd.DataFrame, name: str, features: Sequence[str], cfg: FraudModelConfig, seed: int
) -> RunResult:
    train, test = time_split(df, cfg.train_years, cfg.test_years)
    scores, secs, n_trees = _fit_predict(name, features, train, test, seed, time_ordered=True)
    y = test[TARGET].to_numpy()
    metrics = score_report(y, scores, cfg.review_budget)
    return RunResult(name, "time", metrics, secs, n_trees, y, scores)


def evaluate_cv(
    df: pd.DataFrame, name: str, features: Sequence[str], cfg: FraudModelConfig, seed: int
) -> RunResult:
    folds = []
    total_secs = 0.0
    for train, test in stratified_folds(df, cfg.cv_folds, seed):
        scores, secs, _ = _fit_predict(name, features, train, test, seed, time_ordered=False)
        folds.append(score_report(test[TARGET], scores, cfg.review_budget))
        total_secs += secs
    frame = pd.DataFrame(folds)
    metrics = {f"{k}_mean": float(v) for k, v in frame.mean().items()}
    metrics |= {f"{k}_std": float(v) for k, v in frame.std(ddof=1).items()}
    return RunResult(name, "cv", metrics, total_secs / cfg.cv_folds)


def run_benchmark(
    df: pd.DataFrame,
    cfg: FraudModelConfig,
    seed: int,
    models: Sequence[str] = tuple(MODELS),
    ablation_model: str | None = "catboost",
) -> pd.DataFrame:
    """One row per model (time-split metrics with bootstrap CIs + CV mean/std), plus a
    sensitive-features ablation. Recall differences are paired against REFERENCE_MODEL."""
    features = feature_columns(df, exclude=cfg.sensitive_features)
    runs: list[tuple[str, RunResult, RunResult]] = []
    for name in models:
        t = evaluate_time_split(df, name, features, cfg, seed)
        runs.append((name, t, evaluate_cv(df, name, features, cfg, seed)))
    if ablation_model is not None and cfg.sensitive_features:
        with_sensitive = feature_columns(df)
        t = evaluate_time_split(df, ablation_model, with_sensitive, cfg, seed)
        cv = evaluate_cv(df, ablation_model, with_sensitive, cfg, seed)
        runs.append((f"{ablation_model} + {'/'.join(cfg.sensitive_features)}", t, cv))

    reference = next((t for n, t, _ in runs if n == REFERENCE_MODEL), None)
    recall = partial(recall_at_budget, budget=cfg.review_budget)
    rows = []
    for name, t, cv in runs:
        row = _row(name, t, cv)
        pr_lo, pr_hi = bootstrap_ci(t.y_test, t.scores, _pr_auc, seed=seed)
        rc_lo, rc_hi = bootstrap_ci(t.y_test, t.scores, recall, seed=seed)
        row["pr_auc_1996_ci"] = f"[{pr_lo:.3f}, {pr_hi:.3f}]"
        row["recall_1996_ci"] = f"[{rc_lo:.3f}, {rc_hi:.3f}]"
        if reference is not None and name != REFERENCE_MODEL:
            d, lo, hi = paired_bootstrap_diff(
                t.y_test, t.scores, reference.scores, recall, seed=seed
            )
            row[f"delta_recall_vs_{REFERENCE_MODEL}"] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        else:
            row[f"delta_recall_vs_{REFERENCE_MODEL}"] = "n/a"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def _row(name: str, t: RunResult, cv: RunResult) -> dict[str, object]:
    m, c = t.metrics, cv.metrics
    return {
        "model": name,
        "pr_auc_1996": round(m["pr_auc"], 4),
        "recall@budget_1996": round(m["recall_at_budget"], 4),
        "roc_auc_1996": round(m["roc_auc"], 4),
        "pr_auc_cv": f"{c['pr_auc_mean']:.4f} +/- {c['pr_auc_std']:.4f}",
        "recall@budget_cv": f"{c['recall_at_budget_mean']:.4f} +/- {c['recall_at_budget_std']:.4f}",
        "n_trees": t.n_trees if t.n_trees is not None else "n/a",
        "fit_s": round(t.fit_seconds, 1),
    }
