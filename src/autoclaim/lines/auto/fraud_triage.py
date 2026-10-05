"""Two-stage fraud triage on the auto simulator: first notice (stage 1) + after the independent
appraisal (stage 2). Models train on CONFIRMED fraud (what an insurer observes); interception is
measured against TRUE fraud (simulation truth, evaluation only).

Validation: both stages fit on the training period only; the stage-2 threshold is fitted here for
the target true-fraud recall and must then be frozen in config (fraud_model.triage).
--final: production models (train + validation), frozen threshold, locked test + holdouts, logged.
"""

import argparse
from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import roc_auc_score

from autoclaim.config import FraudModelConfig, TriageConfig, load_carrier_config
from autoclaim.datasets.profile import to_markdown
from autoclaim.lines.auto.datasets import EPOCH, get_spec
from autoclaim.lines.auto.fraud_benchmark import load_or_train
from autoclaim.ml import fraud_model, models, test_log
from autoclaim.ml.canaries import run_canaries
from autoclaim.ml.frame import WEIGHT, T, Y, feature_list
from autoclaim.ml.models import CatBoostModel
from autoclaim.ml.split import latest_slice, recency_weights, rolling_origin
from autoclaim.ml.triage import (
    TwoStagePolicy,
    bootstrap_policy,
    budget_for_recall,
    fit_policy,
    recall_curve,
)
from autoclaim.paths import REPO_ROOT, models_dir

BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30)
FRAUD_TYPES = ("inflated_damage", "prior_damage", "staged_collision", "owner_give_up")


def score_pool(
    raw: pd.DataFrame, cfg: FraudModelConfig, tri: TriageConfig, final: bool, retrain: bool
) -> tuple[pd.DataFrame, str]:
    """Stage-1 and stage-2 scores + labels for the evaluation pool (validation, or test)."""
    spec1 = get_spec(cfg.production_dataset, cfg)
    spec2 = get_spec(tri.stage2_dataset, cfg)
    f1, f2 = spec1.prepare(raw), spec2.prepare(raw)
    _, v1, t1 = spec1.split(f1)
    _, v2, t2 = spec2.split(f2)
    seed = cfg.seeds[0]
    if final:
        a1 = fraud_model.load()
        a2 = fraud_model.train_final(f2, spec2, cfg, seed)
        fraud_model.save(a2, models_dir() / "fraud_appraisal")
        p1, p2 = t1, t2
    else:
        train2, val2, _ = spec2.split(f2)
        feats = feature_list(f2, exclude=spec2.sensitive)
        spec2.check_features(feats)
        run_canaries(train2, val2, feats, seed=seed)  # raises on leakage
        a1 = load_or_train(f1, spec1, seed, models_dir() / "fraud_val", retrain)
        a2 = load_or_train(f2, spec2, seed, models_dir() / "fraud_appraisal_val", retrain)
        p1, p2 = v1, v2
    r = raw.loc[p1.index]
    pool = pd.DataFrame({
        "s1": a1.predict_proba(p1).to_numpy(),
        "s2": a2.predict_proba(p2.loc[p1.index]).to_numpy(),
        "confirmed": p1[Y].to_numpy().astype(bool),
        "true": r["gt_is_fraud"].to_numpy().astype(bool),
        "fraud_type": r["gt_fraud_type"].fillna("").to_numpy(),
        "amount": r["claimed_amount"].to_numpy(dtype=float),
    }, index=p1.index)  # fmt: skip
    return pool, f"stage1 {a1.version}, stage2 {a2.version}"


def summarize(pool: pd.DataFrame, policy: TwoStagePolicy) -> dict[str, pd.DataFrame]:
    y = pool["true"].to_numpy()
    out: dict[str, pd.DataFrame] = {}
    auc = {
        name: [
            roc_auc_score(y, pool[col]),
            roc_auc_score(pool["confirmed"], pool[col]),
            budget_for_recall(y, pool[col], 0.8),
        ]
        for name, col in (("stage 1 (first notice)", "s1"), ("stage 2 (after appraisal)", "s2"))
    }
    out["models"] = pd.DataFrame(
        auc, index=["roc_auc_true", "roc_auc_confirmed", "review_share_for_80pct_true"]
    ).T.round(3)
    for stage in ("s1", "s2"):
        out[f"curve_{stage}"] = recall_curve(y, pool[stage], BUDGETS, pool["amount"]).round(3)
    rows = {}
    for label, col in (("true fraud", "true"), ("confirmed fraud", "confirmed")):
        rows[label] = policy.evaluate(pool[col], pool["s1"], pool["s2"], pool["amount"])
    out["policy"] = pd.DataFrame(rows).T.round(3)
    ci = bootstrap_policy(policy, pool["true"], pool["s1"], pool["s2"], pool["amount"], seed=0)
    out["policy_ci"] = pd.DataFrame(
        {k: [f"{lo:.3f} - {hi:.3f}"] for k, (lo, hi) in ci.items()}, index=["true fraud 95% CI"]
    )
    tier1, tier2 = policy.referrals(pool["s1"], pool["s2"])
    by_type = {}
    for t in FRAUD_TYPES:
        m = (pool["fraud_type"] == t).to_numpy()
        if m.any():
            by_type[t] = {"n": int(m.sum()), "stage1_recall": tier1[m].mean(),
                          "two_stage_recall": (tier1 | tier2)[m].mean()}  # fmt: skip
    out["by_type"] = pd.DataFrame(by_type).T.round(3)
    return out


def rolling_cv(
    raw: pd.DataFrame,
    cfg: FraudModelConfig,
    tri: TriageConfig,
    fold_starts: Sequence[str],
    half_lives: Sequence[float | None],
    horizon_days: int = 91,
    calib_days: int = 91,
    datasets: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Deployment replayed per quarter, before the locked test window only: fit both stages on
    the past, fit the stage-2 threshold on the last quarter (confirmed labels), score the next."""
    name1, name2 = datasets or (cfg.production_dataset, tri.stage2_dataset)
    spec1, spec2 = get_spec(name1, cfg), get_spec(name2, cfg)
    f1, f2 = spec1.prepare(raw), spec2.prepare(raw)
    boundary = cfg.datasets[name1].test_start
    if boundary is None:
        raise ValueError("rolling CV needs a date-based locked test boundary")
    test_start = (pd.Timestamp(boundary) - EPOCH).days
    f1, f2 = f1[f1[T] < test_start], f2[f2[T] < test_start]
    starts = [(pd.Timestamp(d) - EPOCH).days for d in fold_starts]
    feats1 = feature_list(f1, exclude=spec1.sensitive)
    feats2 = feature_list(f2, exclude=spec2.sensitive)
    seed = cfg.seeds[0]
    rows = []
    for hl in half_lives:
        folds1 = rolling_origin(f1, starts, horizon_days, calib_days)
        folds2 = rolling_origin(f2, starts, horizon_days, calib_days)
        for start, (fit1, cal1, te1), (fit2, cal2, te2) in zip(fold_starts, folds1, folds2,
                                                               strict=True):  # fmt: skip
            scores = {}
            for name, fit, cal, te, feats in (("s1", fit1, cal1, te1, feats1),
                                              ("s2", fit2, cal2, te2, feats2)):  # fmt: skip
                fit = fit.assign(**{WEIGHT: recency_weights(fit[T].to_numpy(), hl)})
                tr, es = latest_slice(fit, 0.2)
                model = CatBoostModel(feats, seed).fit(tr, es.drop(columns=[WEIGHT]))
                scores[name] = (model.predict_proba(cal), model.predict_proba(te))
            policy = fit_policy(cal1[Y], scores["s1"][0], scores["s2"][0], tri.stage1_budget,
                                tri.target_true_recall)  # fmt: skip
            truth = raw.loc[te1.index, "gt_is_fraud"].to_numpy().astype(bool)
            m = policy.evaluate(truth, scores["s1"][1], scores["s2"][1])
            rows.append({
                "half_life_days": "none" if hl is None else int(hl), "test_quarter": start,
                "n_test": len(te1), "true_recall": m["recall"],
                "confirmed_recall": policy.evaluate(te1[Y], scores["s1"][1],
                                                    scores["s2"][1])["recall"],
                "review_rate": m["review_rate"], "precision_true": m["precision"],
                "auc_stage1_true": roc_auc_score(truth, scores["s1"][1]),
                "auc_stage2_true": roc_auc_score(truth, scores["s2"][1]),
            })  # fmt: skip
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def render(
    title: str, versions: str, policy: TwoStagePolicy, tables: dict[str, pd.DataFrame]
) -> str:
    return "\n".join([
        f"# Two-stage fraud triage: {title}", "",
        f"Models: {versions}. Stage 1 refers the top {policy.stage1_budget:.0%} at first notice; "
        f"stage 2 refers claims scoring >= {policy.threshold:.4f} after the appraisal. "
        "Models train on confirmed fraud; recall is measured on TRUE fraud.", "",
        "## Policy (interception)", to_markdown(tables["policy"], "measured on"), "",
        to_markdown(tables["policy_ci"], "bootstrap"), "",
        "## By fraud type", to_markdown(tables["by_type"], "fraud type"), "",
        "## Each stage alone", to_markdown(tables["models"], "model"), "",
        "### Stage 1 recall curve (true fraud)", to_markdown(tables["curve_s1"], "budget"), "",
        "### Stage 2 recall curve (true fraud)", to_markdown(tables["curve_s2"], "budget"), "",
    ])  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    cfg = load_carrier_config().fraud_model
    tri = cfg.triage
    if tri is None:
        raise SystemExit("fraud_model.triage is not configured")
    ap = argparse.ArgumentParser(description="Two-stage fraud triage.")
    ap.add_argument("--final", action="store_true", help="locked test with the frozen threshold")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--rolling", action="store_true", help="rolling-origin CV, pre-test only")
    ap.add_argument("--decades", action="store_true", help="use the 2002-2024 world (yearly folds)")
    ap.add_argument(
        "--max-trees",
        type=int,
        default=None,
        help="cap boosting rounds for rolling CV only (faster comparisons)",
    )
    ap.add_argument(
        "--half-lives",
        nargs="*",
        default=["none", "365"],
        help="recency weighting to compare (days; 'none' = equal weights)",
    )
    args = ap.parse_args(argv)
    if args.final and tri.stage2_threshold is None:
        raise SystemExit("fit on validation first and freeze fraud_model.triage.stage2_threshold")

    if args.rolling:
        if args.max_trees:
            models.MAX_TREES = args.max_trees  # experiment speed-up; never used for production
        hls = [None if h == "none" else float(h) for h in args.half_lives]
        if args.decades:  # yearly folds over the multi-decade world
            names = ("sim_decades", "sim_decades_appraisal")
            raw = get_spec(names[0], cfg).load_raw()
            folds = [f"{y}-01-01" for y in range(2021, 2024)]
            res = rolling_cv(raw, cfg, tri, folds, hls, 365, 365, names)
        else:
            raw = get_spec(cfg.production_dataset, cfg).load_raw()
            folds = ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01",
                     "2024-04-01"]  # fmt: skip
            res = rolling_cv(raw, cfg, tri, folds, hls)
        agg = res.drop(columns=["test_quarter"]).groupby("half_life_days").agg(["mean", "std"])
        agg.columns = ["_".join(map(str, col)) for col in agg.columns]
        stem = "fraud_triage_rolling_decades" if args.decades else "fraud_triage_rolling"
        path = REPO_ROOT / "docs" / f"{stem}.md"
        header = (
            "# Two-stage triage: rolling-origin CV (quarterly, before the locked test)\n\n"
            "Each fold fits both stages on the past, fits the stage-2 threshold on the previous "
            "quarter with CONFIRMED labels, and scores the next quarter against TRUE fraud.\n\n"
        )
        per_fold = to_markdown(res.round(3).set_index("test_quarter"), "test_quarter")
        body = f"{to_markdown(agg.round(3), 'half_life_days')}\n\n{per_fold}\n"
        path.write_text(header + body, encoding="utf-8")
        print(agg.round(3).to_string())
        print(f"report -> {path}")
        return 0
    raw = get_spec(cfg.production_dataset, cfg).load_raw()
    pool, versions = score_pool(raw, cfg, tri, args.final, args.retrain)
    if args.final:
        assert tri.stage2_threshold is not None
        policy = TwoStagePolicy(tri.stage1_budget, tri.stage2_threshold)
        title = "LOCKED TEST (H2 2024)"
    else:
        # fit on CONFIRMED fraud: an insurer never observes the truth; TRUE recall is reported
        policy = fit_policy(pool["confirmed"], pool["s1"], pool["s2"], tri.stage1_budget,
                            tri.target_true_recall)  # fmt: skip
        title = "validation (H1 2024): threshold fitted here"
    tables = summarize(pool, policy)
    print(tables["policy"].to_string())
    print(tables["by_type"].to_string())
    print(tables["models"].to_string())
    name = "fraud_triage_test" if args.final else "fraud_triage_validation"
    path = REPO_ROOT / "docs" / f"{name}.md"
    path.write_text(render(title, versions, policy, tables), encoding="utf-8")
    print(f"stage-2 threshold {policy.threshold:.6f}; report -> {path}")
    if args.final:
        m = tables["policy"].loc["true fraud"]
        test_log.append(
            "sim_us (two-stage triage)",
            f"true-fraud recall {m['recall']:.4f} at review rate "
            f"{m['review_rate']:.4f} (stage-2 threshold {policy.threshold:.4f})",
        )
    return 0
