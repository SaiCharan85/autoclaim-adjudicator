"""Benchmark policy retrieval configurations on labeled queries -> docs/retrieval_benchmark.md.

Embedding models run locally (fastembed ONNX, CPU) and download once to .cache/fastembed.
Usage: uv run python scripts/retrieval_benchmark.py [--models BAAI/bge-small-en-v1.5 ...]
"""

import argparse
import sys
from functools import partial

import numpy as np
import pandas as pd

from autoclaim.config import load_carrier_config
from autoclaim.datasets.profile import to_markdown
from autoclaim.lines.auto.policy import POLICY_PATH, QUERIES_PATH
from autoclaim.paths import REPO_ROOT
from autoclaim.retrieval.benchmark import evaluate, hnsw_recall, load_queries
from autoclaim.retrieval.bm25 import BM25Index
from autoclaim.retrieval.corpus import load_policy
from autoclaim.retrieval.dense import FastEmbedEmbedder, cached_index
from autoclaim.retrieval.graph import PolicyGraph
from autoclaim.retrieval.retriever import HybridRetriever

CACHE = REPO_ROOT / ".cache"
DEFAULT_MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"]


def main() -> int:
    cfg = load_carrier_config().retrieval
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--k", type=int, default=cfg.top_k)
    args = ap.parse_args()

    policy = load_policy(POLICY_PATH)
    queries = load_queries(QUERIES_PATH, set(policy.by_id))
    ids = [c.id for c in policy.clauses]
    docs = [c.document for c in policy.clauses]
    bm25 = BM25Index(ids, docs)
    graph = PolicyGraph(policy.clauses)
    rows, ann = [], []

    def run(name: str, r: HybridRetriever, hops: int = 0) -> None:
        search = partial(r.search, k=args.k, hops=hops, max_expanded=cfg.max_expanded)
        rows.append(evaluate(name, search, queries))
        print(rows[-1])

    run("bm25", HybridRetriever(policy, bm25, None, graph, cfg.rrf_k, cfg.candidates))
    for model in args.models:
        short = model.split("/")[-1]
        emb = FastEmbedEmbedder(model, CACHE / "fastembed", cfg.query_prefix)
        path = CACHE / "retrieval" / f"{short}-{policy.fingerprint()}.faiss"
        index = cached_index(ids, docs, emb, cfg.hnsw, path)
        dense = HybridRetriever(policy, None, (emb, index), graph, cfg.rrf_k, cfg.candidates)
        hybrid = HybridRetriever(policy, bm25, (emb, index), graph, cfg.rrf_k, cfg.candidates)
        run(f"dense {short}", dense)
        run(f"hybrid RRF (bm25 + {short})", hybrid)
        for hops in (1, 2):
            run(f"hybrid RRF (bm25 + {short}) + graph {hops}-hop", hybrid, hops)
        qv = np.stack([emb.embed_query(q.query) for q in queries])
        recall = hnsw_recall(index.index, emb.embed_documents(docs), qv, cfg.candidates)
        ann.append({"model": short, "hnsw_recall@20 vs exact": round(recall, 3)})

    table = pd.DataFrame(rows).set_index("config")
    ann_table = pd.DataFrame(ann).set_index("model")
    out = REPO_ROOT / "docs" / "retrieval_benchmark.md"
    out.write_text(
        f"# Policy retrieval benchmark\n\n{len(queries)} labeled queries, {len(ids)} clauses, "
        f"top-k = {args.k}, RRF k = {cfg.rrf_k}, graph additions capped at {cfg.max_expanded}.\n\n"
        f"{to_markdown(table, 'config')}\n\n{to_markdown(ann_table, 'model')}\n",
        encoding="utf-8",
    )
    print(table.to_string())
    print(ann_table.to_string())
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
