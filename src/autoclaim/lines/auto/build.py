"""Wire the auto harness from config: one call gives a runnable, checkpointed graph."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from autoclaim.config import CarrierConfig, load_carrier_config
from autoclaim.core.audit import AuditSink
from autoclaim.core.budget import BudgetConfig
from autoclaim.core.finalize import FinalizationLedger
from autoclaim.core.graph import Harness, HarnessConfig
from autoclaim.core.judge import LLMJudge, load_rubric
from autoclaim.core.lob import MemoryStore
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
) -> AutoHarness:
    cfg = cfg or load_carrier_config()
    client = client or LLMClient.from_config(cfg.models)
    hcfg = harness_config(cfg)
    retriever = build_retriever(cfg)
    line = AutoLine(
        client=client,
        policy=retriever.policy,
        retriever=retriever,
        toolkit=FraudToolkit.load(),
        judge_impl=LLMJudge(client, load_rubric(RUBRIC_PATH)),
        jurisdiction=cfg.active_jurisdiction,
        fraud_review_score=hcfg.router.fraud_review_score,
        top_k=cfg.retrieval.top_k,
        graph_hops=cfg.retrieval.graph_hops,
        max_expanded=cfg.retrieval.max_expanded,
        few_shot=few_shot,
    )
    ledger = FinalizationLedger(CACHE / "harness" / "finalized.sqlite3")
    harness = Harness(line, hcfg, AuditSink(data_dir() / "audit"), ledger, memory)
    path = checkpoint_path or CACHE / "harness" / "checkpoints.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    return AutoHarness(harness, harness.build(saver), line, client)
