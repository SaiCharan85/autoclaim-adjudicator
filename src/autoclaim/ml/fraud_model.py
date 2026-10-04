"""Train, save and load the production fraud artifacts (CatBoost + Isolation Forest).

Fit on train + validation (everything before the locked test window). The tree count is chosen
by early stopping on the latest slice of that period. Test metrics are computed only with
final=True, so the locked test set is never touched by accident.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import catboost
import joblib
import pandas as pd
import sklearn
from catboost import CatBoostClassifier

from autoclaim.config import FraudModelConfig
from autoclaim.ml.anomaly import AnomalyScorer
from autoclaim.ml.frame import Y, categorical_columns, feature_list, for_catboost
from autoclaim.ml.metrics import score_report
from autoclaim.ml.models import CATBOOST_PARAMS, CatBoostModel
from autoclaim.ml.spec import DatasetSpec
from autoclaim.ml.split import assert_time_ordered, latest_slice
from autoclaim.paths import models_dir

MODEL_FILE = "catboost.cbm"
ANOMALY_FILE = "anomaly.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class FraudArtifacts:
    model: CatBoostClassifier
    anomaly: AnomalyScorer
    features: list[str]
    categorical: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unversioned"))

    @property
    def dataset(self) -> str:
        return str(self.metadata.get("dataset", "unknown"))

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        x = for_catboost(frame, self.features, self.categorical)
        return pd.Series(self.model.predict_proba(x)[:, 1], index=frame.index)

    def test_metrics(self, frame: pd.DataFrame, budget: float) -> dict[str, dict[str, float]]:
        return {
            "catboost": score_report(frame[Y], self.predict_proba(frame), budget),
            "isolation_forest": score_report(frame[Y], self.anomaly.score(frame), budget),
        }


def train_final(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    cfg: FraudModelConfig,
    seed: int,
    final: bool = False,
    data_sha256: str | None = None,
) -> FraudArtifacts:
    train, val, test = spec.split(frame)
    assert_time_ordered(train, val, test)
    fit_set = pd.concat([train, val])
    features = feature_list(frame, exclude=spec.sensitive)
    spec.check_features(features)

    inner, es = latest_slice(fit_set, 0.2)
    probe = CatBoostModel(features, seed).fit(inner, es)
    n_trees = max(1, probe.model.get_best_iteration() + 1)
    cats = categorical_columns(fit_set, features)
    model = CatBoostClassifier(
        iterations=n_trees,
        **CATBOOST_PARAMS,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(for_catboost(fit_set, features, cats), fit_set[Y], cat_features=cats)
    artifacts = FraudArtifacts(model, AnomalyScorer(features, seed).fit(fit_set), features, cats)

    artifacts.metadata = {
        "dataset": spec.name,
        "real_labels": spec.real_labels,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": features,
        "categorical_features": cats,
        "excluded_sensitive_features": list(spec.sensitive),
        "n_fit": len(fit_set),
        "n_test": len(test),
        "n_trees": n_trees,
        "catboost_params": CATBOOST_PARAMS,
        "seed": seed,
        "review_budget": cfg.review_budget,
        "data_sha256": data_sha256,
        "library_versions": {"catboost": catboost.__version__, "sklearn": sklearn.__version__},
        "test_metrics": artifacts.test_metrics(test, cfg.review_budget) if final else None,
    }
    return artifacts


def save(artifacts: FraudArtifacts, directory: Path | None = None) -> Path:
    directory = directory or models_dir() / "fraud"
    directory.mkdir(parents=True, exist_ok=True)
    artifacts.model.save_model(str(directory / MODEL_FILE))
    joblib.dump(artifacts.anomaly, directory / ANOMALY_FILE)
    digest = hashlib.sha256((directory / MODEL_FILE).read_bytes()).hexdigest()[:12]
    artifacts.metadata["version"] = f"catboost-{digest}"
    (directory / METADATA_FILE).write_text(
        json.dumps(artifacts.metadata, indent=2, default=str), encoding="utf-8"
    )
    return directory


def load(directory: Path | None = None) -> FraudArtifacts:
    directory = directory or models_dir() / "fraud"
    if not (directory / MODEL_FILE).exists():
        raise FileNotFoundError(
            f"No fraud model in {directory}. Run: uv run python scripts/train_fraud.py"
        )
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    model = CatBoostClassifier()
    model.load_model(str(directory / MODEL_FILE))
    anomaly: AnomalyScorer = joblib.load(directory / ANOMALY_FILE)
    return FraudArtifacts(
        model,
        anomaly,
        list(metadata["features"]),
        list(metadata["categorical_features"]),
        metadata,
    )
