"""Assemble the Kaggle upload for Step 7.5 (nothing is sent; the push commands are printed).

  .cache/kaggle/dataset/  private dataset: data/finetune/*.jsonl + trainer + pinned requirements
  .cache/kaggle/kernel/   GPU script kernel that installs the requirements and trains

Usage:
  uv run python scripts/package_kaggle.py --user <kaggle-username>
then (your account; runs as a background batch job, ~2 GPU hours on a T4):
  kaggle datasets create -p .cache/kaggle/dataset        (first time; later: datasets version)
  kaggle kernels push -p .cache/kaggle/kernel
  kaggle kernels status <user>/autoclaim-finetune
  kaggle kernels output <user>/autoclaim-finetune -p models/finetune
"""

import argparse
import json
import shutil
import sys

from autoclaim.paths import REPO_ROOT, data_dir

SLUG = "autoclaim-finetune"
FILES = ("intake_train.jsonl", "intake_val.jsonl", "judge_train.jsonl", "judge_val.jsonl")


def dataset_metadata(user: str) -> dict[str, object]:
    return {"title": "autoclaim finetune", "id": f"{user}/{SLUG}",
            "licenses": [{"name": "CC0-1.0"}]}  # fmt: skip


def kernel_metadata(user: str) -> dict[str, object]:
    return {
        "id": f"{user}/{SLUG}",
        "title": SLUG,
        "code_file": "run_on_kaggle.py",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--user", required=True, help="your Kaggle username")
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
    shutil.copy2(REPO_ROOT / "kaggle" / "run_on_kaggle.py", kn / "run_on_kaggle.py")
    (ds / "dataset-metadata.json").write_text(json.dumps(dataset_metadata(args.user), indent=1))
    (kn / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(args.user), indent=1))
    size = sum(p.stat().st_size for p in ds.iterdir()) / 1e6
    print(f"dataset: {ds} ({size:.1f} MB, private)\nkernel:  {kn} (GPU, internet on)")
    print("push with:\n"
          f"  kaggle datasets create -p {ds}\n  kaggle kernels push -p {kn}\n"
          f"  kaggle kernels status {args.user}/{SLUG}\n"
          f"  kaggle kernels output {args.user}/{SLUG} -p models/finetune")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
