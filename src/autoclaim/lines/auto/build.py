"""Wire the auto harness from config: one call gives a runnable, checkpointed graph."""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from autoclaim.config import CarrierConfig, load_carrier_config
from autoclaim.core.audit import AuditSink
from autoclaim.core.budget import BudgetConfig
from autoclaim.core.finalize import FinalizationLedger
from autoclaim.core.graph import Harness, HarnessConfig
from autoclaim.core.judge import build_judge, load_rubric
from autoclaim.core.lob import LineOfBusiness, MemoryStore
from autoclaim.core.memory import FeedbackMemory, HNSWMemoryIndex, make_few_shot
from autoclaim.core.router import RouterConfig
from autoclaim.lines.auto.fraud_tools import FraudToolkit
from autoclaim.lines.auto.line import AutoLine, FewShot
from autoclaim.lines.auto.policy import POLICY_PATH
from autoclaim.llm.client import LLMClient
from autoclaim.paths import REPO_ROOT, data_dir
from autoclaim.retrieval.bm25 import BM25Index
from autoclaim.retrieval.corpus import load_policy
from autoclaim.retrieval.dense import FastEmbedEmbedder, cached_index
from autoclaim.retrieval.graph import PolicyGraph
from autoclaim.retrieval.retriever import HybridRetriever

RUBRIC_PATH = REPO_ROOT / "config" / "rubrics" / "adjudication_reasoning.yaml"
CACHE = REPO_ROOT / ".cache"


def harness_config(cfg: CarrierConfig) -> HarnessConfig:
    extra = cfg.model_extra or {}
    return HarnessConfig(
        max_retries=cfg.harness.max_retries,
        router=RouterConfig.model_validate(extra["router"]),
        budget=BudgetConfig.model_validate(extra["budgets"]),
    )


def build_retriever(cfg: CarrierConfig) -> HybridRetriever:
    r = cfg.retrieval
    policy = load_policy(POLICY_PATH)
    ids = [c.id for c in policy.clauses]
    docs = [c.document for c in policy.clauses]
    emb = FastEmbedEmbedder(r.embedding_model, CACHE / "fastembed", r.query_prefix)
    short = r.embedding_model.split("/")[-1]
    index = cached_index(ids, docs, emb, r.hnsw, CACHE / "retrieval" /
                         f"{short}-{policy.fingerprint()}.faiss")  # fmt: skip
    return HybridRetriever(policy, BM25Index(ids, docs), (emb, index),
                           PolicyGraph(policy.clauses), r.rrf_k, r.candidates)  # fmt: skip


def build_memory(
    cfg: CarrierConfig, retriever: HybridRetriever, line: AutoLine, read_only: bool = False
) -> FeedbackMemory:
    """Feedback memory on the retriever's (already loaded) local embedder, HNSW-indexed."""
    if cfg.memory is None or retriever.dense is None:
        raise ValueError("feedback memory needs a memory config and the dense retriever's embedder")
    embedder = retriever.dense[0]
    hnsw = cfg.retrieval.hnsw
    return FeedbackMemory(
        embedder,
        line.memory_text,
        path=REPO_ROOT / cfg.memory.path,
        index_factory=lambda dim: HNSWMemoryIndex(dim, hnsw),
        cutoff=cfg.memory.cutoff,
        read_only=read_only,
    )


@dataclass
class AutoHarness:
    harness: Harness
    graph: Any
    line: AutoLine
    client: LLMClient


def build(
    cfg: CarrierConfig | None = None,
    client: LLMClient | None = None,
    memory: MemoryStore | None = None,
    few_shot: FewShot | None = None,
    checkpoint_path: Path | None = None,
    read_only_memory: bool = False,
    state_dir: Path | None = None,
    wrap_line: Callable[[AutoLine], LineOfBusiness] | None = None,
) -> AutoHarness:
    """`read_only_memory`: evaluation runs read the memory built from earlier periods but
    never write to it. `state_dir`: where the finalization ledger, checkpoints and audit log
    live (each evaluation arm gets its own, so arms never see each other's decisions).
    `wrap_line`: an ablation wrapper around the line (evaluation only)."""
    cfg = cfg or load_carrier_config()
    client = client or LLMClient.from_config(cfg.models)
    hcfg = harness_config(cfg)
    retriever = build_retriever(cfg)
    line = AutoLine(
        client=client,
        policy=retriever.policy,
        retriever=retriever,
        toolkit=FraudToolkit.load(),
        judge_impl=build_judge(client, load_rubric(RUBRIC_PATH)),
        jurisdiction=cfg.active_jurisdiction,
        fraud_review_score=hcfg.router.fraud_review_score,
        top_k=cfg.retrieval.top_k,
        graph_hops=cfg.retrieval.graph_hops,
        max_expanded=cfg.retrieval.max_expanded,
        few_shot=few_shot,
    )
    if memory is None and cfg.memory is not None and cfg.memory.enabled:
        memory = build_memory(cfg, retriever, line, read_only=read_only_memory)
    k = cfg.memory.few_shot_k if cfg.memory is not None else 0
    if few_shot is None and isinstance(memory, FeedbackMemory) and k > 0:
        line.few_shot = make_few_shot(memory, k)
    state = state_dir or CACHE / "harness"
    state.mkdir(parents=True, exist_ok=True)
    ledger = FinalizationLedger(state / "finalized.sqlite3")
    audit = AuditSink(state / "audit" if state_dir else data_dir() / "audit")
    lob = wrap_line(line) if wrap_line else line
    harness = Harness(lob, hcfg, audit, ledger, memory)
    path = checkpoint_path or state / "checkpoints.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    return AutoHarness(harness, harness.build(saver), line, client)
