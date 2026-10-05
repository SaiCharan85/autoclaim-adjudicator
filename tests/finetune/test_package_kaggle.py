import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "package_kaggle.py"
_spec = importlib.util.spec_from_file_location("package_kaggle", SCRIPT)
assert _spec is not None and _spec.loader is not None
pk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pk)


def test_kernel_runs_on_gpu_privately_with_the_dataset_attached() -> None:
    meta = pk.kernel_metadata("alice")
    assert meta["id"] == "alice/autoclaim-finetune-run"  # not the dataset slug
    assert (meta["enable_gpu"], meta["is_private"], meta["enable_internet"]) == ("true",) * 3
    assert meta["kernel_type"] == "script" and meta["language"] == "python"
    assert meta["dataset_sources"] == ["alice/autoclaim-finetune"]
    assert (SCRIPT.parents[1] / "kaggle" / str(meta["code_file"])).exists()


def test_dataset_metadata() -> None:
    meta = pk.dataset_metadata("alice")
    assert meta["id"] == "alice/autoclaim-finetune" and meta["licenses"] == [{"name": "CC0-1.0"}]


def test_only_train_and_validation_files_are_uploaded() -> None:
    assert all(f.endswith(("_train.jsonl", "_val.jsonl")) for f in pk.FILES)
