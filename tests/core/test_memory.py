import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from autoclaim.config import HNSWConfig
from autoclaim.core.memory import (
    ExactIndex,
    FeedbackMemory,
    HNSWMemoryIndex,
    MemoryText,
    few_shot_block,
    make_few_shot,
)

DIM = 64


class BagEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words are similar. Counts calls."""

    name, dim = "bag", DIM

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def embed_documents(self, texts):
        self.calls += 1
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        self.calls += 1
        return self._vec(text)


def describe(state) -> MemoryText:
    return MemoryText(state["claim"]["text"], date.fromisoformat(state["claim"]["date"]))


def state(cid: str, text: str, proposed: str | None, human: str | None, when: str = "2023-03-01",
          reason: str = "reasons: x"):  # fmt: skip
    s = {"claim_id": cid, "claim": {"text": text, "date": when}}
    if proposed:
        s["decision"] = {"outcome": proposed, "reasons": [f"{proposed}_reason"]}
    if human:
        s["human"] = {"outcome": human, "payout": 100.0 if human == "approve" else None,
                      "reason": reason, "adjuster": "oracle"}  # fmt: skip
    return s


def memory(**kw) -> tuple[FeedbackMemory, BagEmbedder]:
    emb = BagEmbedder()
    return FeedbackMemory(emb, describe, **kw), emb


# ---------------------------------------------------------------- indexes


@pytest.mark.parametrize(
    "make",
    [ExactIndex, lambda d: HNSWMemoryIndex(d, HNSWConfig(M=16, ef_construction=40, ef_search=16))],
)
def test_indexes_find_nearest(make) -> None:
    emb = BagEmbedder()
    idx = make(DIM)
    texts = ["deer strike comprehensive", "rideshare collision excluded", "hail damage roof"]
    idx.add(["a", "b", "c"], emb.embed_documents(texts))
    assert len(idx) == 3
    assert idx.search(emb.embed_query("deer strike at night"), 1)[0][0] == "a"
    assert idx.search(emb.embed_query("rideshare collision"), 3)[0][0] == "b"


def test_empty_index_returns_nothing() -> None:
    assert ExactIndex(DIM).search(np.ones(DIM), 3) == []
    assert (
        HNSWMemoryIndex(DIM, HNSWConfig(M=16, ef_construction=40, ef_search=16)).search(
            np.ones(DIM), 3
        )
        == []
    )


# ---------------------------------------------------------------- remembering


def test_remembers_human_reviews_only() -> None:
    mem, _ = memory()
    mem.remember(state("C1", "deer strike", "approve", None))  # auto-decided: nothing to learn
    assert len(mem) == 0
    mem.remember(state("C2", "deer strike", "approve", "deny", reason="reasons: excluded_driver"))
    (ep,) = mem.episodes()
    assert ep.corrected and ep.proposed_outcome == "approve" and ep.human_outcome == "deny"
    assert ep.human_reason == "reasons: excluded_driver" and ep.case_date == date(2023, 3, 1)


def test_confirmation_is_not_a_correction() -> None:
    mem, _ = memory()
    mem.remember(state("C1", "deer strike", "approve", "approve"))
    assert not mem.episodes()[0].corrected


def test_no_proposal_counts_as_correction() -> None:
    mem, _ = memory()
    mem.remember(state("C1", "failsafe case", None, "deny"))
    assert mem.episodes()[0].corrected and mem.episodes()[0].proposed_outcome is None


def test_cutoff_keeps_the_evaluation_period_out() -> None:
    mem, _ = memory(cutoff=date(2024, 7, 1))
    mem.remember(state("OLD", "deer strike", "approve", "deny", when="2024-06-30"))
    mem.remember(state("NEW", "deer strike", "approve", "deny", when="2024-07-01"))
    assert [e.claim_id for e in mem.episodes()] == ["OLD"]
    assert mem.skipped_after_cutoff == 1


def test_undated_case_is_not_remembered_under_a_cutoff() -> None:
    emb = BagEmbedder()
    mem = FeedbackMemory(emb, lambda s: MemoryText("x", None), cutoff=date(2024, 7, 1))
    mem.remember(state("C1", "x", "approve", "deny"))
    assert len(mem) == 0 and mem.skipped_after_cutoff == 1


def test_read_only_memory_never_writes() -> None:
    mem, _ = memory(read_only=True)
    mem.remember(state("C1", "deer strike", "approve", "deny"))
    assert len(mem) == 0


def test_re_review_replaces_the_episode() -> None:
    mem, _ = memory()
    mem.remember(state("C1", "deer strike", "approve", "deny"))
    mem.remember(state("C1", "deer strike", "approve", "approve"))
    (ep,) = mem.episodes()
    assert ep.human_outcome == "approve"
    assert len(mem.index) == 1  # re-indexed, no duplicate vector


# ---------------------------------------------------------------- retrieval


def test_similar_excludes_the_case_itself_and_ranks_corrections_first() -> None:
    mem, _ = memory()
    mem.remember(state("A", "deer strike comprehensive night", "approve", "approve"))
    mem.remember(state("B", "deer strike comprehensive dusk", "approve", "deny"))
    mem.remember(state("C", "hail damage parked", "approve", "approve"))
    found = mem.similar("deer strike comprehensive", k=2, exclude="A")
    assert found[0].claim_id == "B"
    assert "A" not in {e.claim_id for e in found}
    ranked = mem.similar("deer strike comprehensive", k=2)
    assert ranked[0].corrected  # among the near-equals, the correction comes first


def test_similar_on_empty_memory_or_zero_k() -> None:
    mem, _ = memory()
    assert mem.similar("anything", 3) == []
    mem.remember(state("A", "deer strike", "approve", "deny"))
    assert mem.similar("deer", 0) == []


# ---------------------------------------------------------------- persistence


def test_persisted_memory_reloads_without_re_embedding(tmp_path: Path) -> None:
    path = tmp_path / "mem.sqlite3"
    mem, _ = memory(path=path)
    mem.remember(state("A", "deer strike comprehensive", "approve", "deny"))
    mem.remember(state("B", "hail damage parked", "deny", "deny"))
    emb2 = BagEmbedder()
    again = FeedbackMemory(emb2, describe, path=path)
    assert emb2.calls == 0  # vectors come from disk
    assert [e.claim_id for e in again.episodes()] == ["A", "B"]
    assert again.similar("deer strike", 1)[0].claim_id == "A"


def test_hnsw_backend_with_persistence(tmp_path: Path) -> None:
    cfg = HNSWConfig(M=16, ef_construction=40, ef_search=16)
    path = tmp_path / "mem.sqlite3"
    mem, _ = memory(path=path, index_factory=lambda d: HNSWMemoryIndex(d, cfg))
    for i in range(20):
        text = f"case {i} words{i}" + (" deer" if i == 7 else "")
        mem.remember(state(f"C{i:02d}", text, "approve", "deny"))
    again = FeedbackMemory(BagEmbedder(), describe, path=path,
                           index_factory=lambda d: HNSWMemoryIndex(d, cfg))  # fmt: skip
    assert again.similar("words7 deer", 1)[0].claim_id == "C07"


# ---------------------------------------------------------------- few-shot


def test_few_shot_block_format() -> None:
    mem, _ = memory()
    mem.remember(state("A", "deer strike", "approve", "deny", reason="reasons: excluded_driver"))
    block = few_shot_block(mem.episodes())
    assert block == ('- deer strike | harness proposed: approve ["approve_reason"] | '
                     "adjuster (CORRECTED): deny, because reasons: excluded_driver")  # fmt: skip


def test_few_shot_block_shows_payout_and_missing_proposal() -> None:
    mem, _ = memory()
    mem.remember(state("A", "failsafe", None, "approve"))
    assert "harness proposed: none" in few_shot_block(mem.episodes())
    assert "approve payout 100.00" in few_shot_block(mem.episodes())


def test_make_few_shot_uses_the_case_description_and_skips_itself() -> None:
    mem, _ = memory()
    mem.remember(state("A", "deer strike comprehensive", "approve", "deny"))
    provider = make_few_shot(mem, k=2)
    assert provider(state("A", "deer strike comprehensive", None, None)) == ""  # only itself
    assert "deer strike" in provider(state("Z", "deer strike comprehensive", None, None))
    assert make_few_shot(mem, 0)(state("Z", "deer", None, None)) == ""
    empty, _ = memory()
    assert make_few_shot(empty, 3)(state("Z", "deer", None, None)) == ""


def test_empty_memory_is_still_used_by_the_harness(tmp_path: Path) -> None:
    # regression: an empty FeedbackMemory is falsy (len 0); the harness must not drop it
    from test_graph import CFG, FakeLine

    from autoclaim.core.audit import AuditSink
    from autoclaim.core.finalize import FinalizationLedger
    from autoclaim.core.graph import Harness

    mem, _ = memory()
    harness = Harness(FakeLine(), CFG, AuditSink(tmp_path), FinalizationLedger(":memory:"), mem)
    assert harness.memory is mem


def test_creates_missing_folders_for_its_file(tmp_path: Path) -> None:
    path = tmp_path / "new" / "nested" / "mem.sqlite3"
    mem, _ = memory(path=path)
    mem.remember(state("A", "deer strike", "approve", "deny"))
    assert path.exists() and len(FeedbackMemory(BagEmbedder(), describe, path=path)) == 1
