"""Drift study on the multi-decade world: train on one era, test on every later era.

How fast does a fraud model trained in the past decay on newer claims? Uses only data before the
locked test window. Both stages are measured against TRUE fraud (ROC-AUC, recall at 5%).
"""

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import roc_auc_score

from autoclaim.config import FraudModelConfig
from autoclaim.lines.auto.datasets import EPOCH, get_spec
from autoclaim.ml.frame import T, feature_list
from autoclaim.ml.metrics import recall_at_budget
from autoclaim.ml.models import CatBoostModel
from autoclaim.ml.split import latest_slice

ERAS = {
    "2002-2009": (2002, 2009),
    "2010-2015": (2010, 2015),
    "2016-2019": (2016, 2019),
    "2020-2023": (2020, 2023),
}
TRAIN_WINDOWS = {
    "2002-2009": (2002, 2009),
    "2010-2015": (2010, 2015),
    "2016-2019": (2016, 2019),
    "2002-2019 (all past)": (2002, 2019),
}


def _days(year: int) -> int:
    return int((pd.Timestamp(f"{year}-01-01") - EPOCH).days)


def era_rows(frame: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    return frame[(frame[T] >= _days(years[0])) & (frame[T] < _days(years[1] + 1))]


def drift_matrix(
    raw: pd.DataFrame,
    cfg: FraudModelConfig,
    datasets: Sequence[str] = ("sim_decades", "sim_decades_appraisal"),
    budget: float = 0.05,
) -> pd.DataFrame:
    truth = raw["gt_is_fraud"].astype(bool)
    rows = []
    for name in datasets:
        spec = get_spec(name, cfg)
        frame = spec.prepare(raw)
        feats = feature_list(frame, exclude=spec.sensitive)
        spec.check_features(feats)
        stage = "stage 2 (appraisal)" if name.endswith("_appraisal") else "stage 1 (first notice)"
        for train_name, window in TRAIN_WINDOWS.items():
            tr, es = latest_slice(era_rows(frame, window), 0.2)
            model = CatBoostModel(feats, cfg.seeds[0]).fit(tr, es)
            for test_name, years in ERAS.items():
                if years[0] <= window[1]:
                    continue  # only eras strictly after the training window
                te = era_rows(frame, years)
                y = truth.loc[te.index].to_numpy()
                s = model.predict_proba(te)
                rows.append(
                    {
                        "stage": stage,
                        "train": train_name,
                        "test": test_name,
                        "years_gap": years[0] - window[1],
                        "roc_auc_true": roc_auc_score(y, s),
                        "recall_at_5pct_true": recall_at_budget(y, s, budget),
                    }
                )
                print(rows[-1], flush=True)
    return pd.DataFrame(rows)
