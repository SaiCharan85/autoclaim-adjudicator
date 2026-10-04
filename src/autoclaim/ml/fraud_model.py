"""Train, save and load the production fraud artifacts (CatBoost + Isolation Forest).

Trained on `train_years` only. The test years stay unseen, so Step 4 can build evaluation claims
from those rows without the fraud score having memorized them.
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
from autoclaim.ml.features import TARGET, categorical_columns, feature_columns, for_catboost
from autoclaim.ml.metrics import score_report
from autoclaim.ml.models import CatBoostModel
from autoclaim.ml.split import latest_slice, time_split
from autoclaim.paths import models_dir

MODEL_FILE = "catboost.cbm"
ANOMALY_FILE = "anomaly.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class FraudArtifacts:
    model: CatBoostClassifier
    anomaly: AnomalyScorer
    features: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unversioned"))

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        proba = self.model.predict_proba(for_catboost(frame, self.features))[:, 1]
        return pd.Series(proba, index=frame.index)


def train_final(
    df: pd.DataFrame, cfg: FraudModelConfig, seed: int, data_sha256: str | None = None
) -> FraudArtifacts:
    """Pick the tree count by early stopping on the latest training slice, then refit on all
    training years with that count, and report metrics on the untouched test years."""
    features = feature_columns(df, exclude=cfg.sensitive_features)
    train, test = time_split(df, cfg.train_years, cfg.test_years)

    inner, val = latest_slice(train, 0.2)
    probe = CatBoostModel(features, seed).fit(inner, val)
    n_trees = max(1, probe.model.get_best_iteration() + 1)

    final = CatBoostClassifier(
        iterations=n_trees,
        learning_rate=0.05,
        depth=6,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
    )
    final.fit(
        for_catboost(train, features), train[TARGET], cat_features=categorical_columns(features)
    )
    artifacts = FraudArtifacts(final, AnomalyScorer(features, seed).fit(train), features)
    test_scores = artifacts.predict_proba(test)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": features,
        "categorical_features": categorical_columns(features),
        "excluded_sensitive_features": list(cfg.sensitive_features),
        "train_years": list(cfg.train_years),
        "test_years": list(cfg.test_years),
        "n_train": len(train),
        "n_test": len(test),
        "n_trees": n_trees,
        "seed": seed,
        "review_budget": cfg.review_budget,
        "data_sha256": data_sha256,
        "test_metrics": {
            "catboost": score_report(test[TARGET], test_scores, cfg.review_budget),
            "isolation_forest": score_report(
                test[TARGET], artifacts.anomaly.score(test), cfg.review_budget
            ),
        },
        "library_versions": {"catboost": catboost.__version__, "sklearn": sklearn.__version__},
    }
    artifacts.metadata = metadata
    return artifacts


def save(artifacts: FraudArtifacts, directory: Path | None = None) -> Path:
    directory = directory or models_dir() / "fraud"
    directory.mkdir(parents=True, exist_ok=True)
    artifacts.model.save_model(str(directory / MODEL_FILE))
    joblib.dump(artifacts.anomaly, directory / ANOMALY_FILE)
    digest = hashlib.sha256((directory / MODEL_FILE).read_bytes()).hexdigest()[:12]
    artifacts.metadata["version"] = f"catboost-{digest}"
    (directory / METADATA_FILE).write_text(
        json.dumps(artifacts.metadata, indent=2), encoding="utf-8"
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
    return FraudArtifacts(model, anomaly, list(metadata["features"]), metadata)
