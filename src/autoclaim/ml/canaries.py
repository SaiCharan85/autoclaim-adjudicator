"""Leakage canaries, run on every training run before any model is trusted.

1. Shuffled labels: permute the training label column (inside the frame, so a label that slipped
   into the features is permuted with it) and train. Over several permutations the mean
   validation ROC-AUC must not sit above chance. A single permutation is too noisy when features
   are strong: the model learns a *random* function of them, which can align with the real labels
   by luck in either direction. A real leak pushes every permutation up, so the test is one-sided
   on the mean.
2. Single-feature scan: no single feature may predict the label almost perfectly (ROC-AUC > 0.9).
   That pattern means the "feature" is really the answer, as in a Kaggle set whose fraud flag was
   a threshold on a risk-score column.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from autoclaim.ml.frame import CategoryVocab, Y, categorical_columns

MAX_SINGLE_FEATURE_AUC = 0.9


class CanaryError(RuntimeError):
    pass


N_PERMUTATIONS = 5


@dataclass
class CanaryReport:
    shuffled_auc: float  # mean over permutations
    shuffled_tolerance: float
    shuffled_aucs: list[float] = field(default_factory=list)
    top_single_feature_auc: dict[str, float] = field(default_factory=dict)
    flagged: list[str] = field(default_factory=list)

    @property
    def shuffled_ok(self) -> bool:
        return self.shuffled_auc - 0.5 <= self.shuffled_tolerance

    @property
    def ok(self) -> bool:
        return not self.flagged and self.shuffled_ok


def chance_tolerance(n_pos: int, n_neg: int) -> float:
    """4 standard errors of ROC-AUC under the null (Hanley-McNeil, A=0.5), at least 0.05."""
    se = math.sqrt((n_pos + n_neg + 1) / (12 * max(n_pos, 1) * max(n_neg, 1)))
    return max(0.05, 4 * se)


def shuffled_label_auc(
    train: pd.DataFrame, val: pd.DataFrame, features: Sequence[str], seed: int
) -> float:
    """Validation ROC-AUC of a model trained on one permutation of the training labels."""
    rng = np.random.default_rng(seed)
    shuffled = train.copy()
    shuffled[Y] = rng.permutation(train[Y].to_numpy())
    cats = categorical_columns(shuffled, features)
    vocab = CategoryVocab.fit(shuffled, cats)
    model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1
    )
    model.fit(vocab.transform(shuffled, features), shuffled[Y])
    scores = np.asarray(model.predict_proba(vocab.transform(val, features)))[:, 1]
    return float(roc_auc_score(val[Y], scores))


def single_feature_aucs(
    train: pd.DataFrame, val: pd.DataFrame, features: Sequence[str]
) -> dict[str, float]:
    """Validation ROC-AUC of each feature alone (direction-free), fitted on train only."""
    cats = set(categorical_columns(train, features))
    prior = train[Y].mean()
    out = {}
    for f in features:
        if f in cats:
            stats = train.groupby(train[f].astype("string"), dropna=False)[Y].agg(["sum", "count"])
            rate = (stats["sum"] + 10 * prior) / (stats["count"] + 10)  # smoothed target mean
            score = val[f].astype("string").map(rate).astype("float64").fillna(prior)
        else:
            col = pd.to_numeric(train[f], errors="coerce")
            score = pd.to_numeric(val[f], errors="coerce").fillna(col.median())
        if score.nunique() < 2:
            continue
        auc = float(roc_auc_score(val[Y], score))
        out[f] = round(max(auc, 1 - auc), 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def run_canaries(
    train: pd.DataFrame, val: pd.DataFrame, features: Sequence[str], seed: int, strict: bool = True
) -> CanaryReport:
    aucs = [shuffled_label_auc(train, val, features, seed + k) for k in range(N_PERMUTATIONS)]
    spread = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
    pos = int(val[Y].sum())
    tolerance = max(
        chance_tolerance(pos, len(val) - pos) / math.sqrt(N_PERMUTATIONS),
        3 * spread / math.sqrt(N_PERMUTATIONS),
        0.03,
    )
    report = CanaryReport(
        shuffled_auc=round(float(np.mean(aucs)), 4),
        shuffled_tolerance=round(tolerance, 4),
        shuffled_aucs=[round(a, 4) for a in aucs],
    )
    single = single_feature_aucs(train, val, features)
    report.top_single_feature_auc = dict(list(single.items())[:8])
    report.flagged = [f for f, a in single.items() if a > MAX_SINGLE_FEATURE_AUC]
    if strict and not report.ok:
        raise CanaryError(
            f"leakage canary failed: mean shuffled AUC {report.shuffled_auc} "
            f"(chance + {report.shuffled_tolerance} allowed; runs {report.shuffled_aucs}), "
            f"near-perfect features {report.flagged}"
        )
    return report
