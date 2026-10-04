"""Build the claims dataset + two holdouts, attach ground truth, write data/sim/ with a manifest.

Usage: uv run python scripts/simulate_claims.py [--n 50000]
"""

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from autoclaim.config import JurisdictionProfile, load_carrier_config
from autoclaim.datasets.download import read_manifest
from autoclaim.datasets.sources import CRSS
from autoclaim.lines.auto.crss import load_incidents
from autoclaim.lines.auto.simulator.generate import build_claims
from autoclaim.lines.auto.simulator.oracle import adjudicate
from autoclaim.lines.auto.simulator.world import (
    DEFAULT_WORLD_PATH,
    World,
    load_world,
    with_overrides,
)
from autoclaim.paths import data_dir, raw_dir

SIM_DIR_NAME = "sim"
FILES = {
    "claims": "claims.csv",
    "holdout_fresh": "holdout_fresh.csv",
    "holdout_shift": "holdout_shift.csv",
}


def sim_dir() -> Path:
    return data_dir() / SIM_DIR_NAME


def with_ground_truth(claims: pd.DataFrame, jur: JurisdictionProfile) -> pd.DataFrame:
    return pd.concat([claims, adjudicate(claims, jur)], axis=1)


def build_all(
    world: World,
    incidents: pd.DataFrame,
    jur: JurisdictionProfile,
    n: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Main dataset plus fresh-seed and shifted-world holdouts from unused real records."""
    causes = tuple(jur.comprehensive_causes)
    main = build_claims(world, incidents, n=n, comprehensive_causes=causes)
    out = {"claims": with_ground_truth(main, jur)}
    if world.holdouts is None:
        return out
    window = world.model_copy(update={"loss_date_start": world.holdouts.test_window_start})
    used = set(main["source_record"])
    for name, spec in (
        ("holdout_fresh", world.holdouts.fresh),
        ("holdout_shift", world.holdouts.shift),
    ):
        w = with_overrides(window, spec.overrides) if spec.overrides else window
        h = build_claims(
            w,
            incidents,
            n=spec.n_claims,
            seed=spec.seed,
            exclude_records=used,
            comprehensive_causes=causes,
        )
        used |= set(h["source_record"])
        out[name] = with_ground_truth(h, jur)
    return out


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(
    datasets: dict[str, pd.DataFrame], directory: Path, provenance: dict[str, object]
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in datasets.items():
        path = directory / FILES[name]
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        files[FILES[name]] = {"rows": len(frame), "sha256": _file_hash(path)}
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    } | provenance
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def load(name: str = "claims", directory: Path | None = None) -> pd.DataFrame:
    path = (directory or sim_dir()) / FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python scripts/simulate_claims.py")
    return pd.read_csv(
        path, parse_dates=["loss_date", "policy_start_date", "report_date"], low_memory=False
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the grounded-hybrid claims dataset.")
    parser.add_argument("--n", type=int, default=None, help="main dataset size (default: config)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    world = load_world()
    config = load_carrier_config()
    incidents = load_incidents()
    datasets = build_all(world, incidents, config.active_jurisdiction, n=args.n)
    crss_hashes = {
        str(y): (m.archive_sha256 if (m := read_manifest(raw_dir(src.key))) else None)
        for y, src in CRSS.items()
    }
    provenance = {
        "world_config_sha256": _file_hash(DEFAULT_WORLD_PATH),
        "jurisdiction": config.jurisdiction,
        "seed": world.seed,
        "crss_archive_sha256": crss_hashes,
        "real_incident_pool": len(incidents),
    }
    out = write(datasets, args.out or sim_dir(), provenance)
    for name, frame in datasets.items():
        real = (frame["record_origin"] == "crss").mean()
        print(
            f"{name:14s} {len(frame):>6,} claims | real-record share {real:.0%} | "
            f"confirmed fraud {frame['fraud_confirmed'].mean():.1%} | "
            f"{frame['loss_date'].min():%Y-%m-%d}..{frame['loss_date'].max():%Y-%m-%d}"
        )
    print(f"-> {out}")
    return 0
