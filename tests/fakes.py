"""Synthetic data builders shared by tests (no real data needed in CI).

A plain module, not a conftest, so `from fakes import ...` is unambiguous.
"""

import numpy as np
import pandas as pd

from autoclaim.datasets.vehicle_fraud import SCHEMA

N_ROWS = 200


def make_valid_frame(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Schema-valid synthetic stand-in for fraud_oracle.csv (no real data in tests)."""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}
    for spec in SCHEMA.columns:
        if spec.kind == "binary":
            data[spec.name] = (rng.random(n) < 0.1).astype(int)
        elif spec.kind == "integer":
            lo = spec.min_value if spec.min_value is not None else 0
            hi = min(spec.max_value if spec.max_value is not None else lo + 10, lo + 50)
            data[spec.name] = rng.integers(lo, hi + 1, n)
        elif spec.allowed is not None:
            data[spec.name] = rng.choice(sorted(spec.allowed), n)
        else:
            data[spec.name] = rng.choice(["a", "b", "c"], n)
    df = pd.DataFrame(data)
    df["PolicyNumber"] = np.arange(1, n + 1)
    return df


def make_fraud_frame(n: int = 900, seed: int = 1) -> pd.DataFrame:
    """Valid frame with a learnable fraud signal and all three years present."""
    rng = np.random.default_rng(seed)
    df = make_valid_frame(n, seed)
    df["Year"] = rng.choice([1994, 1995, 1996], n)
    at_fault = df["Fault"].eq("Policy Holder")
    rural = df["AccidentArea"].eq("Rural")
    p = 0.03 + 0.35 * (at_fault & rural) + 0.10 * at_fault
    df["FraudFound_P"] = (rng.random(n) < p).astype(int)
    return df


CAUSE_COUNTS = {
    "collision_vehicle": 3000,
    "collision_object": 800,
    "parked_hit": 500,
    "animal": 500,
    "hit_and_run": 600,
    "fire": 60,
}


def make_incidents(seed: int = 0, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for cause, count in CAUSE_COUNTS.items():
        k = int(count * scale)
        dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1096, k), unit="D")
        parts.append(
            pd.DataFrame(
                {
                    "source_record": [f"crss:test:{cause}:{i}" for i in range(k)],
                    "weight": rng.uniform(50, 300, k),
                    "loss_date": dates,
                    "loss_hour": rng.integers(0, 24, k).astype(float),
                    "region": rng.choice(["northeast", "midwest", "south", "west"], k),
                    "area": rng.choice(["urban", "rural"], k, p=[0.76, 0.24]),
                    "cause": cause,
                    "vehicle_role": "parked" if cause == "parked_hit" else "in_transport",
                    "n_vehicles": 1 if cause in ("collision_object", "animal", "fire") else 2,
                    "injury_count": rng.choice([0.0, 1.0, 2.0], k, p=[0.75, 0.2, 0.05]),
                    "damage_extent": rng.choice(["minor", "functional", "disabling", "unknown"], k),
                    "towed": rng.choice([0.0, 1.0], k, p=[0.7, 0.3]),
                    "light": rng.choice(["daylight", "dark"], k),
                    "weather": rng.choice(["clear", "rain"], k),
                    "vehicle_make": rng.choice(["Toyota", "Ford", "Kia", "Hyundai", "BMW"], k),
                    "vehicle_model": rng.choice(["A", "B", None], k),
                    "vehicle_model_year": rng.integers(2005, 2025, k).astype(float),
                    "body_class": rng.choice(["car", "suv", "pickup"], k),
                    "at_fault": rng.choice([0.0, 1.0], k),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
