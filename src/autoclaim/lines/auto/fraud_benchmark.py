"""ML vs LLM-only vs hybrid fraud scoring on the same claims (paired, cached, budgeted).

Protocol (pre-declared in docs/fraud_llm_benchmark.md):
- Claims: a stratified sample (fraud over-sampled so the comparison has enough frauds); every
  prefix of the claim order keeps the same fraud share, so a pilot is the first N claims of the
  full run and is reused from the cache.
- ML arm on validation: CatBoost + Isolation Forest fit on the TRAINING period only (the
  production model also saw validation, which would be in-sample). With --final: the production
  model on the locked test period, logged in docs/test_set_log.md.
- Each LLM arm is pinned to ONE model (no mid-run fallback). When its free daily budget runs out
  the run stops; re-running the next day resumes from the cache at no cost.
- Primary metric: ROC-AUC (unaffected by the over-sampling). Secondary: recall@budget and PR-AUC
  reweighted to the real fraud rate. Differences: paired bootstrap 95% CIs.
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from autoclaim.core.rules import RuleSet
from autoclaim.lines.auto import fraud_llm
from autoclaim.lines.auto.fraud_tools import RULES_PATH, FraudSignals, FraudToolkit
from autoclaim.llm.client import LLMClient, system_prompt
from autoclaim.llm.types import LLMUnavailableError
from autoclaim.ml import fraud_model, test_log
from autoclaim.ml.anomaly import AnomalyScorer
from autoclaim.ml.frame import Y, feature_list
from autoclaim.ml.metrics import (
    class_weights,
    paired_bootstrap_diff,
    weighted_recall_at_budget,
    weighted_report,
)
from autoclaim.ml.models import CatBoostModel
from autoclaim.ml.spec import DatasetSpec
from autoclaim.ml.split import latest_slice
from autoclaim.paths import REPO_ROOT, models_dir

ROLE = "fraud"


# ---------------------------------------------------------------- sampling


def stratified_order(y: pd.Series, n: int, fraud_share: float, seed: int) -> pd.Index:
    """`n` claim ids with ~`fraud_share` frauds, ordered so every prefix keeps that share."""
    rng = np.random.default_rng(seed)
    n_fraud = round(n * fraud_share)
    fraud, legit = y.index[y == 1], y.index[y == 0]
    if n_fraud > len(fraud) or n - n_fraud > len(legit):
        raise ValueError(f"not enough claims for n={n}, fraud_share={fraud_share}")
    picks = [rng.permutation(fraud)[:n_fraud], rng.permutation(legit)[: n - n_fraud]]
    # interleave by position-within-class: claim i of k sits at (i + 0.5) / k of the run
    keyed = [((i + 0.5) / len(p), c, idx) for c, p in enumerate(picks) for i, idx in enumerate(p)]
    keyed.sort(key=lambda t: (t[0], t[1]))
    out: pd.Index = pd.Index([idx for _, _, idx in keyed])
    return out


# ---------------------------------------------------------------- ML arm


def train_only_artifacts(
    frame: pd.DataFrame, spec: DatasetSpec, seed: int
) -> fraud_model.FraudArtifacts:
    """CatBoost (early-stopped on the latest train slice) + Isolation Forest, train period only."""
    train, _, _ = spec.split(frame)
    features = feature_list(frame, exclude=spec.sensitive)
    spec.check_features(features)
    fit_set, es = latest_slice(train, 0.2)
    model = CatBoostModel(features, seed).fit(fit_set, es)
    anomaly = AnomalyScorer(features, seed).fit(train)
    meta = {"dataset": spec.name, "version": f"train-only-seed{seed}", "features": features,
            "categorical_features": model.cats}  # fmt: skip
    return fraud_model.FraudArtifacts(model.model, anomaly, features, model.cats, meta)


def load_or_train(
    frame: pd.DataFrame, spec: DatasetSpec, seed: int, directory: Path, retrain: bool = False
) -> fraud_model.FraudArtifacts:
    """Reuse the saved train-only model so hybrid prompts stay byte-identical (cache hits)."""
    if not retrain and (directory / fraud_model.MODEL_FILE).exists():
        art = fraud_model.load(directory)
        if art.dataset == spec.name:
            return art
    art = train_only_artifacts(frame, spec, seed)
    fraud_model.save(art, directory)
    return art


def tool_signals(
    artifacts: fraud_model.FraudArtifacts, spec: DatasetSpec, raw: pd.DataFrame
) -> list[FraudSignals]:
    toolkit = FraudToolkit(artifacts, RuleSet.from_yaml(RULES_PATH), spec)
    return toolkit.assess_frame(raw)


# ---------------------------------------------------------------- LLM arms


@dataclass
class ArmRun:
    arm: str
    model: str
    scores: list[float] = field(default_factory=list)
    requests: int = 0  # live (non-cached) calls
    tokens: int = 0
    stopped: str | None = None  # why the run stopped early, if it did


def _batches(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def arm_prompts(
    arm: fraud_llm.Arm, facts: Sequence[str], tools: Sequence[str], batch_size: int
) -> list[str]:
    return [
        fraud_llm.batch_prompt(facts[b], tools[b] if arm == "hybrid" else None)
        for b in _batches(len(facts), batch_size)
    ]


def estimate(client: LLMClient, model: str, prompts: dict[str, list[str]]) -> dict[str, float]:
    """Worst-case requests/tokens/minutes before anything is spent (cached calls excluded)."""
    requests = tokens = 0
    for arm, batch_prompts in prompts.items():
        system = system_prompt(fraud_llm.INSTRUCTIONS[arm], fraud_llm.BatchScores)  # type: ignore[index]
        for user in batch_prompts:
            requests += 1
            tokens += client.estimate_tokens(ROLE, system, user)
    lim = client.cfg.catalog[model]
    margin = client.cfg.safety_margin
    minutes = max(
        requests / (lim.rpm * margin) if lim.rpm else 0.0,
        tokens / (lim.tpm * margin) if lim.tpm else 0.0,
    )
    req_left, tok_left = client.ledger.headroom(model)
    return {"requests": requests, "tokens": tokens, "minutes": round(minutes, 1),
            "requests_left_today": req_left, "tokens_left_today": tok_left}  # fmt: skip


def run_arm(
    client: LLMClient, arm: fraud_llm.Arm, model: str, prompts: Sequence[str], sizes: Sequence[int]
) -> ArmRun:
    run = ArmRun(arm=arm, model=model)
    for user, n in zip(prompts, sizes, strict=True):
        if run.stopped:
            run.scores += [math.nan] * n
            continue
        try:
            res = client.structured(
                ROLE, fraud_llm.INSTRUCTIONS[arm], user, fraud_llm.BatchScores, chain=[model]
            )
        except LLMUnavailableError as exc:
            run.stopped = str(exc)
            run.scores += [math.nan] * n
            continue
        run.scores += fraud_llm.align_scores(res.value, n)
        if not res.meta.cached:
            run.requests += 1 + res.meta.validation_retries
            run.tokens += res.meta.prompt_tokens + res.meta.completion_tokens
    return run


# ---------------------------------------------------------------- metrics


def scorable(y: np.ndarray, arms: dict[str, np.ndarray]) -> bool:
    """True when the claims every arm scored include both frauds and non-frauds."""
    ok = np.all([np.isfinite(s) for s in arms.values()], axis=0)
    return bool(ok.any()) and 0 < int(y[ok].sum()) < int(ok.sum())


def summarize(
    y: np.ndarray,
    arms: dict[str, np.ndarray],
    population_rate: float,
    budget: float,
    seed: int,
    n_boot: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-arm metrics and paired differences, on the claims every arm scored."""
    ok = np.all([np.isfinite(s) for s in arms.values()], axis=0)
    y, arms = y[ok], {k: v[ok] for k, v in arms.items()}
    w = class_weights(y, population_rate)
    metrics = pd.DataFrame({name: weighted_report(y, s, w, budget) for name, s in arms.items()}).T
    metrics["n_scored"] = int(ok.sum())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(y), size=(n_boot, len(y)))
    idx = [i for i in draws if 0 < y[i].sum() < len(i)]
    rows = []
    names = list(arms)
    for a_i, a in enumerate(names):
        for b in names[a_i + 1 :]:
            d_auc, lo, hi = paired_bootstrap_diff(y, arms[a], arms[b], roc_auc_score, n_boot, seed)
            rec = [
                weighted_recall_at_budget(y[i], arms[a][i], w[i], budget)
                - weighted_recall_at_budget(y[i], arms[b][i], w[i], budget)
                for i in idx
            ]
            r_lo, r_hi = np.nanquantile(rec, [0.025, 0.975])
            d_rec = weighted_recall_at_budget(y, arms[a], w, budget) - weighted_recall_at_budget(
                y, arms[b], w, budget
            )
            rows.append({
                "comparison": f"{a} - {b}",
                "delta_roc_auc": f"{d_auc:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                "delta_recall@budget": f"{d_rec:+.3f} [{r_lo:+.3f}, {r_hi:+.3f}]",
            })  # fmt: skip
    return metrics, pd.DataFrame(rows).set_index("comparison")


def population_rate(frame: pd.DataFrame, spec: DatasetSpec, final: bool) -> float:
    _, val, test = spec.split(frame)
    return float((test if final else val)[Y].mean())


def benchmark_pool(frame: pd.DataFrame, spec: DatasetSpec, final: bool) -> pd.DataFrame:
    _, val, test = spec.split(frame)
    return test if final else val


# ---------------------------------------------------------------- CLI


def to_md(df: pd.DataFrame, index: str) -> str:
    from autoclaim.datasets.profile import to_markdown

    return to_markdown(df, index)


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    from autoclaim.config import load_carrier_config
    from autoclaim.lines.auto.datasets import get_spec

    cfg = load_carrier_config()
    fcfg = cfg.fraud_model
    ap = argparse.ArgumentParser(description="ML vs LLM-only vs hybrid fraud scoring.")
    ap.add_argument("--dataset", default=fcfg.production_dataset, choices=sorted(fcfg.datasets))
    ap.add_argument("--n", type=int, default=200, help="claims (pilot: 50)")
    ap.add_argument("--fraud-share", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--model", default=cfg.models.roles[ROLE].chain[0])
    ap.add_argument("--arms", nargs="+", default=["llm", "hybrid"], choices=["llm", "hybrid"])
    ap.add_argument("--seed", type=int, default=cfg.harness.seed)
    ap.add_argument("--dry-run", action="store_true", help="estimate requests/tokens/time only")
    ap.add_argument("--final", action="store_true", help="locked test period (logged)")
    ap.add_argument("--retrain", action="store_true", help="refit the train-only ML model")
    args = ap.parse_args(argv)
    if args.model not in cfg.models.catalog:
        ap.error(f"--model {args.model!r} is not in the model catalog")

    spec = get_spec(args.dataset, fcfg)
    raw = spec.load_raw()
    frame = spec.prepare(raw)
    pool = benchmark_pool(frame, spec, args.final)
    order = stratified_order(pool[Y], args.n, args.fraud_share, args.seed)
    if args.final:
        artifacts = fraud_model.load()
        if artifacts.dataset != spec.name:
            raise SystemExit(f"production model is for {artifacts.dataset}, not {spec.name}")
    else:
        artifacts = load_or_train(frame, spec, fcfg.seeds[0], models_dir() / "fraud_val",
                                  args.retrain)  # fmt: skip
    rows = raw.loc[order]
    facts = [fraud_llm.describe_claim(r) for _, r in spec.features_frame(rows).iterrows()]
    signals = tool_signals(artifacts, spec, rows)
    tools = [fraud_llm.describe_signals(s) for s in signals]
    y = pool.loc[order, Y].to_numpy()

    load_dotenv(REPO_ROOT / ".env")
    client = LLMClient.from_config(cfg.models)
    prompts = {arm: arm_prompts(arm, facts, tools, args.batch_size) for arm in args.arms}
    est = estimate(client, args.model, prompts)
    period = "TEST (locked)" if args.final else "validation"
    print(f"{args.n} {period} claims ({int(y.sum())} fraud), LLM {args.model}")
    print(f"ML {artifacts.version}; estimate (worst case, before cache): {est}")
    if args.dry_run:
        return 0

    sizes = [len(range(len(facts))[b]) for b in _batches(len(facts), args.batch_size)]
    runs = [run_arm(client, arm, args.model, prompts[arm], sizes) for arm in args.arms]
    for r in runs:
        status = f"STOPPED: {r.stopped}" if r.stopped else "complete"
        print(f"{r.arm}: {r.requests} live requests, {r.tokens} tokens, {status}")
    scores = {"ml": np.array([s.model_score for s in signals])}
    scores |= {r.arm: np.array(r.scores, dtype=float) for r in runs}
    if not scorable(y, scores):
        print(
            "quota stopped the LLM arms before both classes were scored: nothing to report "
            "yet (and nothing logged); re-run tomorrow, finished batches come from the cache"
        )
        return 2
    rate = population_rate(frame, spec, args.final)
    metrics, diffs = summarize(y, scores, rate, fcfg.review_budget, args.seed)
    print(metrics.round(3).to_string())
    print(diffs.to_string())

    suffix = f"{spec.name}{'_final' if args.final else ''}_n{args.n}"
    out = REPO_ROOT / ".cache" / "runs" / f"fraud_llm_{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": order, "y": y, **scores}).to_csv(out, index=False)
    report = REPO_ROOT / "docs" / f"fraud_llm_benchmark_{suffix}.md"
    header = (
        f"# ML vs LLM vs hybrid: {spec.name}, {period}, {args.n} claims\n\n"
        f"LLM {args.model}; ML {artifacts.version}; fraud share in sample {y.mean():.2f}, "
        f"population rate {rate:.3f} (budget metrics reweighted to it).\n\n"
    )
    tables = f"{to_md(metrics.round(3), 'arm')}\n\n{to_md(diffs, 'comparison')}\n"
    report.write_text(header + tables, encoding="utf-8")
    print(f"report: {report}")
    if args.final and not any(r.stopped for r in runs):
        aucs = ", ".join(f"{a} ROC-AUC {m:.3f}" for a, m in metrics["roc_auc"].items())
        test_log.append(spec.name, f"fraud LLM benchmark n={args.n}: {aucs}")
    return 0
