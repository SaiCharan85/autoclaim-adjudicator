"""Validate the two-stage fraud triage inside the harness (validation period only; no LLM calls).

1. Fit the stage-1 score cut: the score at the top `stage1_budget` share of validation claims.
2. Run the harness's own code path (FraudToolkit with the stage-2 model and the frozen policy) on
   every validation claim, using models trained on the training period only, and compare with
   first-notice-only routing at the router's line.
3. Wiring check: the auto line's fraud node (fraud_record -> toolkit) on a sample of claims built
   from true facts must refer exactly the claims the toolkit refers from the raw records.

Usage: uv run python scripts/validate_two_stage.py [--sample 300] -> docs/fraud_two_stage_harness.md
The locked test is not touched.
"""

import argparse
import sys
from typing import Any, cast

import numpy as np
import pandas as pd

from autoclaim.config import load_carrier_config
from autoclaim.core.judge import build_judge, load_rubric
from autoclaim.finetune.datasets import GoldClient
from autoclaim.lines.auto import judge_eval as je
from autoclaim.lines.auto.build import RUBRIC_PATH, build_retriever, harness_config
from autoclaim.lines.auto.claim import Appraisal
from autoclaim.lines.auto.datasets import get_spec
from autoclaim.lines.auto.facts import derive
from autoclaim.lines.auto.fraud_tools import FraudToolkit, StageTwo, two_stage_refer
from autoclaim.lines.auto.line import AutoLine
from autoclaim.lines.auto.simulator import build as sim_build
from autoclaim.llm.client import LLMClient
from autoclaim.ml import fraud_model
from autoclaim.paths import REPO_ROOT, models_dir

OUT = REPO_ROOT / "docs" / "fraud_two_stage_harness.md"


def rates(refer: np.ndarray, fraud: np.ndarray) -> dict[str, float]:
    return {"reviewed": float(refer.mean()), "fraud_caught": float(refer[fraud].mean()),
            "fraud_could_be_paid": float((~refer[fraud]).mean())}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sample", type=int, default=300, help="claims for the wiring check")
    args = ap.parse_args(argv)
    cfg = load_carrier_config()
    fm, tri = cfg.fraud_model, cfg.fraud_model.triage
    assert tri is not None and tri.stage2_threshold is not None
    split = fm.datasets[fm.production_dataset]
    claims = sim_build.load("claims")
    d = pd.to_datetime(claims["loss_date"]).dt.date
    val = claims[(d >= split.val_start) & (d < split.test_start)].reset_index(drop=True)
    fraud = val["gt_is_fraud"].astype(bool).to_numpy()

    # validation-only models (trained on the training period) for an honest validation
    tk = FraudToolkit.load(
        models_dir() / "fraud_val", stage2_dir=models_dir() / "fraud_appraisal_val"
    )
    feats = tk.spec.features_frame(val)
    for f in tk.artifacts.features:
        if f not in feats:
            feats[f] = float("nan")
    s1 = tk.artifacts.predict_proba(feats[tk.artifacts.features]).to_numpy()
    cut = float(np.quantile(s1, 1 - tri.stage1_budget))
    art2 = fraud_model.load(models_dir() / "fraud_appraisal_val")
    tk.stage2 = StageTwo(art2, get_spec(tri.stage2_dataset, fm), cut, tri.stage2_threshold)
    s2 = tk.stage2_scores(val)
    two = np.array(
        [two_stage_refer(a, b, cut, tri.stage2_threshold) for a, b in zip(s1, s2, strict=True)]
    )
    line_now = float(cfg.model_extra["router"]["fraud_review_score"])  # type: ignore[index]
    one = s1 >= line_now
    r_one, r_two = rates(one, fraud), rates(two, fraud)

    # wiring check through the auto line (true facts + the record's appraisal)
    r, hc = build_retriever(cfg), harness_config(cfg)
    client = cast(LLMClient, GoldClient())
    line = AutoLine(client=client, policy=r.policy, retriever=r, toolkit=tk,
                    judge_impl=build_judge(client, load_rubric(RUBRIC_PATH)),
                    jurisdiction=cfg.active_jurisdiction,
                    fraud_review_score=hc.router.fraud_review_score)  # fmt: skip
    rng = np.random.default_rng(cfg.harness.seed)
    rows = rng.choice(len(val), size=min(args.sample, len(val)), replace=False)
    agree = checked = 0
    for i in rows:
        row = val.iloc[int(i)]
        facts, pkg = je.true_facts(row), je.package(row)
        pkg = pkg.model_copy(update={"appraisal": Appraisal(
            appraised_amount=float(row["appraised_amount"]),
            prior_damage=None if pd.isna(row["appraiser_prior_damage"])
            else bool(row["appraiser_prior_damage"]))})  # fmt: skip
        state: dict[str, Any] = {"claim": pkg.model_dump(mode="json"),
                                 "guardrails": {"narrative": "(validation)"},
                                 "facts": {"extracted": facts.model_dump(mode="json"),
                                           "derived": derive(facts, pkg, cfg.active_jurisdiction)
                                           .model_dump(mode="json")}}  # fmt: skip
        sig = line.fraud(state).update["fraud"]["signals"]  # type: ignore[arg-type]
        checked += 1
        agree += bool(sig["two_stage_referral"]) == bool(two[int(i)])

    pct = lambda x: f"{x:.1%}"  # noqa: E731
    lines = [
        "# Two-stage fraud triage in the harness: validation",
        "",
        f"Validation-period claims {len(val):,} ({int(fraud.sum())} true frauds); models trained "
        "on the training period only (`models/fraud_val`, `models/fraud_appraisal_val`). No LLM "
        "calls; the locked test is not touched.",
        "",
        f"Policy: refer when the first-notice score is in the top {tri.stage1_budget:.0%} "
        f"(score >= {cut:.4f}, fitted here) or the post-appraisal score >= "
        f"{tri.stage2_threshold} (frozen in config).",
        "",
        "| Routing | Claims reviewed | True fraud caught | Fraud that could be auto-paid |",
        "|---|---|---|---|",
        f"| First notice only, line {line_now} | {pct(r_one['reviewed'])} | "
        f"{pct(r_one['fraud_caught'])} | {pct(r_one['fraud_could_be_paid'])} |",
        f"| **Two-stage (with the appraisal)** | {pct(r_two['reviewed'])} | "
        f"**{pct(r_two['fraud_caught'])}** | {pct(r_two['fraud_could_be_paid'])} |",
        "",
        f"Wiring check: the auto line's fraud node referred the same claims as the toolkit on "
        f"{agree} of {checked} sampled claims ({agree / checked:.1%}). Facts come from the "
        "simulator truth here; with live intake, extraction errors can change a score.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"stage1_cut_score to freeze in config: {cut:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
