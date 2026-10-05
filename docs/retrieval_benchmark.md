# Policy retrieval benchmark

60 labeled queries, 75 clauses, top-k = 5, RRF k = 60, graph additions capped at 6.

| config | direct_recall@k | context_recall | mrr | avg_context | p50_ms |
|---|---|---|---|---|---|
| bm25 | 0.628 | 0.531 | 0.674 | 4.9 | 0.2 |
| dense bge-small-en-v1.5 | 0.692 | 0.615 | 0.719 | 5.0 | 12.8 |
| hybrid RRF (bm25 + bge-small-en-v1.5) | 0.747 | 0.646 | 0.782 | 5.0 | 14.7 |
| hybrid RRF (bm25 + bge-small-en-v1.5) + graph 1-hop | 0.747 | 0.863 | 0.782 | 11.0 | 14.8 |
| hybrid RRF (bm25 + bge-small-en-v1.5) + graph 2-hop | 0.747 | 0.863 | 0.782 | 11.0 | 17.7 |
| dense bge-base-en-v1.5 | 0.789 | 0.679 | 0.802 | 5.0 | 48.2 |
| hybrid RRF (bm25 + bge-base-en-v1.5) | 0.778 | 0.676 | 0.822 | 5.0 | 36.2 |
| hybrid RRF (bm25 + bge-base-en-v1.5) + graph 1-hop | 0.778 | 0.922 | 0.822 | 11.0 | 33.8 |
| hybrid RRF (bm25 + bge-base-en-v1.5) + graph 2-hop | 0.778 | 0.922 | 0.822 | 11.0 | 36.3 |
| dense bge-large-en-v1.5 | 0.722 | 0.631 | 0.782 | 5.0 | 106.7 |
| hybrid RRF (bm25 + bge-large-en-v1.5) | 0.742 | 0.643 | 0.807 | 5.0 | 109.0 |
| hybrid RRF (bm25 + bge-large-en-v1.5) + graph 1-hop | 0.742 | 0.889 | 0.807 | 11.0 | 107.5 |
| hybrid RRF (bm25 + bge-large-en-v1.5) + graph 2-hop | 0.742 | 0.889 | 0.807 | 11.0 | 88.2 |

| model | hnsw_recall@20 vs exact |
|---|---|
| bge-small-en-v1.5 | 1.0 |
| bge-base-en-v1.5 | 1.0 |
| bge-large-en-v1.5 | 1.0 |
