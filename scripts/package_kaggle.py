"""Assemble the Kaggle upload for Step 7.5 (nothing is sent; the push commands are printed).

  .cache/kaggle/dataset/  private dataset: data/finetune/*.jsonl + trainer + pinned requirements
  .cache/kaggle/kernel/   GPU script kernel that installs the requirements and trains

Usage:
  uv run python scripts/package_kaggle.py --user <kaggle-username>
then (your account; runs as a background batch job, ~2 GPU hours on a T4):
  kaggle datasets create -p .cache/kaggle/dataset        (first time; later: datasets version)
  kaggle kernels push -p .cache/kaggle/kernel
  kaggle kernels status <user>/autoclaim-finetune-run
  kaggle kernels output <user>/autoclaim-finetune-run -p models/finetune
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from autoclaim.paths import REPO_ROOT, data_dir

SLUG = "autoclaim-finetune"
KERNEL_SLUG = f"{SLUG}-run"  # a kernel may not reuse the dataset's slug (Kaggle answers 409)
EXPORT_SLUG = f"{SLUG}-export"  # export-only runs keep the training run's outputs intact
FILES = ("intake_train.jsonl", "intake_val.jsonl", "judge_train.jsonl", "judge_val.jsonl")


def dataset_metadata(user: str) -> dict[str, object]:
    return {"title": "autoclaim finetune", "id": f"{user}/{SLUG}",
            "licenses": [{"name": "CC0-1.0"}]}  # fmt: skip


def kernel_metadata(user: str, export_only: bool = False) -> dict[str, object]:
    slug = EXPORT_SLUG if export_only else KERNEL_SLUG
    return {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "export_on_kaggle.py" if export_only else "run_on_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",  # pip install + base model download
        "dataset_sources": [f"{user}/{SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def flatten_adapter(lora: Path, task: str) -> list[tuple[Path, str]]:
    """Adapter files as flat dataset names `<task>_lora__<file>` (datasets have no folders)."""
    return [(p, f"{task}_lora__{p.name}") for p in sorted(lora.iterdir())
            if p.is_file() and p.name != "README.md"]  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--user", required=True, help="your Kaggle username")
    ap.add_argument(
        "--export-only",
        metavar="TASK",
        choices=["intake"],
        help="upload models/finetune/out/TASK/lora; the kernel only exports it",
    )
    args = ap.parse_args()
    src = data_dir() / "finetune"
    missing = [f for f in FILES if not (src / f).exists()]
    if missing:
        print(f"missing {missing}: run scripts/build_finetune_data.py first")
        return 1
    root = REPO_ROOT / ".cache" / "kaggle"
    ds, kn = root / "dataset", root / "kernel"
    for d in (ds, kn):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for f in FILES:
        shutil.copy2(src / f, ds / f)
    for f in ("finetune_qlora.py", "requirements-finetune.txt"):
        shutil.copy2(REPO_ROOT / "kaggle" / f, ds / f)
    export = bool(args.export_only)
    if export:
        lora = REPO_ROOT / "models" / "finetune" / "out" / args.export_only / "lora"
        if not (lora / "adapter_config.json").exists():
            print(f"no adapter at {lora}: download the training run's output first")
            return 1
        for f in flatten_adapter(lora, args.export_only):
            shutil.copy2(f[0], ds / f[1])
    entry = "export_on_kaggle.py" if export else "run_on_kaggle.py"
    shutil.copy2(REPO_ROOT / "kaggle" / entry, kn / entry)
    (ds / "dataset-metadata.json").write_text(json.dumps(dataset_metadata(args.user), indent=1))
    (kn / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(args.user, export), indent=1)
    )
    size = sum(p.stat().st_size for p in ds.iterdir()) / 1e6
    print(f"dataset: {ds} ({size:.1f} MB, private)\nkernel:  {kn} (GPU, internet on)")
    print("push with:\n"
          f"  kaggle datasets create -p {ds}\n  kaggle kernels push -p {kn}\n"
          f"  kaggle kernels status {args.user}/{KERNEL_SLUG}\n"
          f"  kaggle kernels output {args.user}/{KERNEL_SLUG} -p models/finetune")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
