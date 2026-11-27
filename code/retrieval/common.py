# -*- coding: utf-8 -*-
"""Load regulatory clauses and queries, access APIs, rank embeddings, and compute retrieval metrics."""
import csv
import json
import os
import re
import urllib.request
import urllib.error

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CLAUSE_CSV = os.environ.get("CLAUSE_CSV", str(ROOT / "private" / "clauses_full.csv"))
TEST_CSV = str(ROOT / "data" / "retrieval_queries" / "queries.csv")
CACHE_DIR = str(ROOT / "results" / "runtime_cache")
RESULT_DIR = str(ROOT / "results" / "workflow_runs" / "retrieval_evaluation")
API_BASE = os.environ.get("SF_API_BASE", "https://api.siliconflow.cn/v1")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def require_api_key():
    if not os.environ.get("SF_KEY") or os.environ.get("SF_KEY", "").startswith("XXXX"):
        raise RuntimeError("Set SF_KEY to your own API key; no credentials are bundled.")


def combine_text(body, ctx):
    body = (body or "").strip()
    ctx = (ctx or "").strip()
    return body if not ctx else body + "\n" + ctx


def split_refs(s):
    return [x for x in re.split(r"[、,，;；\s]+", s or "") if x]


def load_clauses():
    if not os.path.isfile(CLAUSE_CSV):
        raise FileNotFoundError("Complete clause table is not bundled. Request it as described in data/knowledge_base/README.md and set CLAUSE_CSV. For an explicitly labelled demonstration only, set CLAUSE_CSV=data/knowledge_base/clause_examples.csv and ALLOW_SAMPLE_CORPUS=1.")
    with open(CLAUSE_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 140 and os.environ.get("ALLOW_SAMPLE_CORPUS") != "1":
        raise ValueError("Formal retrieval requires 140 clauses. A small example table is only allowed with ALLOW_SAMPLE_CORPUS=1; it cannot reproduce formal evaluation.")
    out = []
    for i, r in enumerate(rows):
        out.append({
            "idx": i,
            "code": r["clause_id"],
            "name": r["document_title"],
            "art": r["article_or_section"],
            "body": r["clause_text"],
            "ctx": r["context"],
            "text": combine_text(r["clause_text"], r["context"]),
        })
    return out


def load_queries():
    """Load ten evaluation queries and their reference clause IDs."""
    with open(TEST_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        refs = set(split_refs(r["reference_clause_ids"]))
        out.append({
            "qid": r["query_id"], "version": "general", "scene": r["setting"],
            "status": r["target_state"], "query": r["query"],
            "refs": refs, "n_ref": len(refs),
        })
    return out


def _post(path, payload, timeout=60):
    require_api_key()
    req = urllib.request.Request(
        API_BASE + path, data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + os.environ["SF_KEY"],
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def api_embeddings(model, texts, batch=8, max_chars=500):
    """Generate and cache API embeddings, retaining the first max_chars characters of each input."""
    cache = os.path.join(CACHE_DIR, "emb_" + model.replace("/", "__") + ".json")
    store = {}
    if os.path.exists(cache):
        store = json.load(open(cache, encoding="utf-8"))
    todo = [t for t in texts if t not in store]
    sent_map = {t: t[:max_chars] for t in todo}
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        d = _post("/embeddings", {"model": model,
                                   "input": [sent_map[t] for t in chunk]})
        for t, item in zip(chunk, d["data"]):
            store[t] = item["embedding"]
        json.dump(store, open(cache, "w", encoding="utf-8"))
    return [store[t] for t in texts]


def api_rerank(model, query, documents, top_n=None):
    payload = {"model": model, "query": query, "documents": documents,
               "return_documents": False}
    if top_n:
        payload["top_n"] = top_n
    d = _post("/rerank", payload, timeout=90)
    return [(x["index"], x["relevance_score"]) for x in d["results"]]


def api_chat(model, messages, temperature=0.2, timeout=90):
    d = _post("/chat/completions", {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": 1500,
    }, timeout=timeout)
    return d["choices"][0]["message"]["content"]


def l2(v):
    import numpy as np
    a = np.asarray(v, dtype="float32")
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def dense_rank(docs_emb, q_emb):
    """Return clause indices ranked by decreasing cosine similarity."""
    import numpy as np
    D = np.asarray(docs_emb, dtype="float32")
    q = np.asarray(q_emb, dtype="float32")
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = D @ q
    return list(np.argsort(-sims)), sims


def score_ranking(order, codes, refs, ks=(5, 10, 20, 40, 60, 80)):
    """Return precision and recall at each k from ranked clause indices and their IDs."""
    metrics = {}
    for k in ks:
        top = [codes[i] for i in order[:k]]
        hit = len(set(top) & refs)
        metrics[f"P@{k}"] = round(hit / k, 4)
        metrics[f"R@{k}"] = round(hit / max(1, len(refs)), 4)
    return metrics


def rrf(rank_lists, c=60):
    """Apply reciprocal rank fusion (RRF) to the ranked lists."""
    import collections
    scores = collections.defaultdict(float)
    for order in rank_lists:
        for rank, idx in enumerate(order, 1):
            scores[idx] += 1.0 / (c + rank)
    return sorted(scores, key=lambda i: -scores[i])
