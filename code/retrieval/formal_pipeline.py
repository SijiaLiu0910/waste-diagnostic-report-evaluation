# -*- coding: utf-8 -*-
"""Shared regulatory retrieval pipeline.

Steps:
  1. Expand the original query with five fixed keyword sets, embed with bge-large-zh-v1.5,
     and retrieve from the same corpus of 140 clauses.
  2. Fuse the five rankings with RRF and retain the top K candidates.
  3. Rerank against the original query using Qwen3-Reranker-8B.
  4. Greedily deduplicate in reranked order: skip a clause if its cosine similarity
     to a retained clause is >=0.95; continue through candidates to retain at most five.

The query CLI and batch evaluation (eval_k_compare.py) use this module.
"""
import argparse
import time
import numpy as np
import common

EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
RERANKER = "Qwen/Qwen3-Reranker-8B"
TOP_K_FINAL = 5
DEDUP_SIM = 0.95
FACET_PHRASES = [
    "responsible parties persons responsible for management owners managers",
    "duties shall must not prohibitions",
    "technical standards sanitation requirements collection containers provision cleanliness",
    "management procedures inspection collection and transport removal cleaning",
    "penalty provisions penalties fines orders to rectify",
]


def _norm(arr):
    arr = np.asarray(arr, dtype="float32")
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


def build_doc_embeddings(clauses):
    """Build BGE-large clause embeddings, shared by retrieval and deduplication."""
    raw = common.api_embeddings(EMBED_MODEL, [c["text"] for c in clauses])
    return _norm(raw)


def facet_candidate_order(query, clauses, doc_emb):
    """Retrieve across five facets and return RRF-fused clause indices."""
    sub_queries = [query + ". " + p for p in FACET_PHRASES]
    orders = []
    for sub_emb in common.api_embeddings(EMBED_MODEL, sub_queries):
        order, _ = common.dense_rank(doc_emb, sub_emb)
        orders.append(order)
    return common.rrf(orders)


def greedy_dedup(reranked_idx, doc_emb, top_k=TOP_K_FINAL, thresh=DEDUP_SIM):
    """Return retained indices and skipped tuples (index, closest retained index, similarity)."""
    kept, skipped = [], []
    for i in reranked_idx:
        if kept:
            sims = doc_emb[i] @ doc_emb[np.asarray(kept)].T
            j = int(np.argmax(sims))
            max_sim, match = float(sims[j]), kept[j]
        else:
            max_sim, match = 0.0, None
        if kept and max_sim >= thresh:
            skipped.append((i, match, round(max_sim, 4)))
            continue
        kept.append(i)
        if len(kept) == top_k:
            break
    return kept, skipped


def retrieve(query, K, clauses, doc_emb):
    """Run the complete pipeline; return structured results and reranking duration."""
    fused = facet_candidate_order(query, clauses, doc_emb)
    pool = fused[:K]
    docs = [clauses[i]["text"] for i in pool]
    t0 = time.time()
    res = common.api_rerank(RERANKER, query, docs, top_n=K)
    rerank_sec = round(time.time() - t0, 3)
    reranked = [pool[li] for li, _ in res]

    pre_dedup_top5 = reranked[:TOP_K_FINAL]
    kept, skipped = greedy_dedup(reranked, doc_emb)
    return {
        "pool": pool,
        "reranked": reranked,
        "pre_dedup_top5": pre_dedup_top5,
        "kept": kept,
        "skipped": skipped,
        "rerank_sec": rerank_sec,
    }


def main():
    ap = argparse.ArgumentParser(description="Regulatory retrieval: five facets, RRF, reranking, and deduplication")
    ap.add_argument("command", choices=["query"])
    ap.add_argument("--query", required=True)
    ap.add_argument("--K", type=int, default=60)
    args = ap.parse_args()

    clauses = common.load_clauses()
    codes = [c["code"] for c in clauses]
    doc_emb = build_doc_embeddings(clauses)
    r = retrieve(args.query, args.K, clauses, doc_emb)

    print(f"Candidate pool K={args.K} ({len(r['pool'])} clauses); reranking {r['rerank_sec']} s")
    print("Top five before deduplication:", ", ".join(codes[i] for i in r["pre_dedup_top5"]))
    print("Top five after deduplication:")
    for n, i in enumerate(r["kept"], 1):
        print(f"  {n}. {codes[i]}")
    if r["skipped"]:
        print("Skipped during deduplication:")
        for i, match, sim in r["skipped"]:
            print(f"  - {codes[i]}(similarity to {codes[match]}: {sim} >= {DEDUP_SIM})")


if __name__ == "__main__":
    main()
