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


def test_export_only_kernel_is_separate_and_runs_the_export_entry() -> None:
    meta = pk.kernel_metadata("alice", export_only=True)
    assert meta["id"] == "alice/autoclaim-finetune-export"  # training run's outputs untouched
    assert meta["code_file"] == "export_on_kaggle.py" and meta["enable_gpu"] == "true"
    assert (SCRIPT.parents[1] / "kaggle" / "export_on_kaggle.py").exists()


def test_flatten_adapter_names(tmp_path: Path) -> None:
    for name in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
        (tmp_path / name).write_text("x")
    names = [n for _, n in pk.flatten_adapter(tmp_path, "intake")]
    assert names == ["intake_lora__adapter_config.json", "intake_lora__adapter_model.safetensors"]


def test_other_task_sets_get_their_own_kernel_and_files() -> None:
    assert pk.kernel_metadata("alice", tasks=("intake", "judge"))["id"] == (
        "alice/autoclaim-finetune-run"
    )
    meta = pk.kernel_metadata("alice", tasks=("adjudicator",))
    assert meta["id"] == "alice/autoclaim-finetune-adjudicator"  # earlier outputs untouched
    assert pk.files_for(("adjudicator",)) == ("adjudicator_train.jsonl", "adjudicator_val.jsonl")
    assert set(pk.files_for(pk.DEFAULT_TASKS)) == set(pk.FILES)
