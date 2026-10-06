"""The GPU training script's pure helpers, tested on CPU (heavy imports live inside functions)."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "kaggle" / "finetune_qlora.py"
_spec = importlib.util.spec_from_file_location("finetune_qlora", SCRIPT)
assert _spec is not None and _spec.loader is not None
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

IDS = ["faithful_to_clauses", "facts_supported", "numbers_consistent",
       "material_facts_addressed", "outcome_consistent"]  # fmt: skip


def judge_json(no: tuple[str, ...] = ()) -> str:
    return json.dumps({"answers": [{"id": i, "answer": "no" if i in no else "yes", "note": ""}
                                   for i in IDS]})  # fmt: skip


def test_script_imports_without_gpu_libraries() -> None:
    assert ft.BASE_MODEL == "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
    assert ft.RESPONSE_PART == "<|im_start|>assistant\n"


# ---------------------------------------------------------------- folders and data


def test_find_dirs_prefers_flags(tmp_path: Path) -> None:
    data, out = ft.find_dirs(str(tmp_path / "d"), str(tmp_path / "o"))
    assert (data, out) == (tmp_path / "d", tmp_path / "o")


def test_find_dirs_local_defaults() -> None:
    data, out = ft.find_dirs(None, None)
    assert data == Path("data/finetune") and out == Path("finetune_out")


def test_split_prompt_and_read_jsonl(tmp_path: Path) -> None:
    ex = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                       {"role": "assistant", "content": "a"}], "meta": {}}  # fmt: skip
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(ex) + "\n\n", encoding="utf-8")
    (row,) = ft.read_jsonl(p)
    prompt, answer = ft.split_prompt(row)
    assert [m["role"] for m in prompt] == ["system", "user"] and answer == "a"
    with pytest.raises(ValueError):
        ft.split_prompt({"messages": prompt})


@pytest.mark.parametrize(
    ("text", "ok"),
    [('{"a": 1}', True), ('```json\n{"a": 1}\n```', True), ('Sure: {"a": 1} done', True),
     ("no json", False), ("{broken", False), ("[1, 2]", False)],
)  # fmt: skip
def test_extract_json(text: str, ok: bool) -> None:
    assert (ft.extract_json(text) == {"a": 1}) is ok


# ---------------------------------------------------------------- judge scoring


def test_score_judge_perfect_model() -> None:
    refs = [judge_json(), judge_json(no=("facts_supported",))]
    m = ft.score_judge(refs, refs, [None, "invented_fact"])
    assert m == {"n": 2, "json_valid": 1.0, "item_accuracy": 1.0, "catch_rate": 1.0,
                 "false_alarm_rate": 0.0}  # fmt: skip


def test_score_judge_misses_false_alarms_and_garbage() -> None:
    refs = [judge_json(), judge_json(no=("numbers_consistent",)), judge_json()]
    preds = [
        judge_json(no=("outcome_consistent",)),  # false alarm
        judge_json(no=("facts_supported",)),  # flags, but the wrong item: not a catch
        "I cannot answer",
    ]  # invalid -> counts as a flag (no judgment = not a pass)
    m = ft.score_judge(preds, refs, [None, "wrong_amount", None])
    assert m["json_valid"] == pytest.approx(2 / 3)
    assert m["catch_rate"] == 0.0
    assert m["false_alarm_rate"] == 1.0
    assert m["item_accuracy"] == pytest.approx((4 + 3 + 0) / 15)


def test_score_judge_empty() -> None:
    assert ft.score_judge([], [], [])["json_valid"] is None


# ---------------------------------------------------------------- intake scoring


def test_score_intake_fields_and_hours_tolerance() -> None:
    ref = json.dumps({"cause": "hit_and_run", "police_report_hours": 15, "witnesses": 1,
                      "loss_date": "2023-05-15", "use_at_loss": "personal"})  # fmt: skip
    pred = json.dumps({"cause": "hit_and_run", "police_report_hours": 13.0, "witnesses": 2,
                       "loss_date": "2023-05-15", "use_at_loss": "personal"})  # fmt: skip
    m = ft.score_intake([pred, "garbage"], [ref, ref])
    assert m["json_valid"] == 0.5
    assert m["per_field"]["police_report_hours"] == 0.5  # within 3 h
    assert m["per_field"]["witnesses"] == 0.0
    assert m["per_field"]["cause"] == 0.5


def test_score_intake_null_match() -> None:
    ref = json.dumps({"witnesses": None})
    assert ft.score_intake([ref], [ref])["per_field"]["witnesses"] == 1.0


# ---------------------------------------------------------------- export and resume markers


def test_ollama_modelfile() -> None:
    mf = ft.ollama_modelfile("model-q4_k_m.gguf", "judge")
    assert mf.startswith("FROM ./model-q4_k_m.gguf\n")
    assert "PARAMETER temperature 0" in mf and "num_predict 700" in mf


def test_done_markers_are_per_task_and_stage(tmp_path: Path) -> None:
    assert ft.done_marker(tmp_path, "judge", "train") == tmp_path / "judge" / ".done_train"
    assert ft.done_marker(tmp_path, "intake", "gguf") != ft.done_marker(tmp_path, "judge", "gguf")


def test_training_never_uses_test_period_data() -> None:
    # the script only reads <task>_train / <task>_val files: the builder refuses test-period rows
    src = SCRIPT.read_text(encoding="utf-8")
    assert "_train.jsonl" in src and "_val.jsonl" in src and "_test" not in src


def test_find_gguf_looks_inside_unsloths_suffixed_folder(tmp_path: Path) -> None:
    task = tmp_path / "judge"
    (task / "gguf_gguf").mkdir(parents=True)
    small, big = task / "gguf_gguf" / "tiny.gguf", task / "gguf_gguf" / "Qwen3.Q4_K_M.gguf"
    small.write_bytes(b"x")
    big.write_bytes(b"x" * 100)
    assert ft.find_gguf(task) == big  # the real model, not a stray small file
    assert ft.find_gguf(tmp_path / "missing") is None


def test_modelfile_sits_next_to_the_gguf_with_its_real_name(tmp_path: Path) -> None:
    gguf = tmp_path / "gguf_gguf" / "Qwen3-4B.Q4_K_M.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"x")
    mf = ft.write_modelfile(gguf, "intake")
    assert mf.parent == gguf.parent
    text = mf.read_text()
    assert text.startswith("FROM ./Qwen3-4B.Q4_K_M.gguf") and "num_predict 900" in text


def test_cli_has_export_only_and_sampled_training_eval() -> None:
    src = SCRIPT.read_text(encoding="utf-8")
    assert "--export-only" in src and "--train-eval-n" in src
