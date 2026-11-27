# -*- coding: utf-8 -*-
"""Evaluate ten queries at K=20/40/60/80 (40 runs) and save detailed and summary records."""
import csv
import os
import statistics
import collections
import common
import formal_pipeline as fp

KS = [20, 40, 60, 80]


def main():
    if os.environ.get("ALLOW_SAMPLE_CORPUS") == "1":
        raise RuntimeError("Formal parameter comparisons require the full 140-clause corpus, not examples.")
    clauses = common.load_clauses()
    codes = [c["code"] for c in clauses]
    queries = [q for q in common.load_queries() if q["version"] == "general"]
    doc_emb = fp.build_doc_embeddings(clauses)

    detail, summary_rows = [], []
    for K in KS:
        per_k = []
        for q in queries:
            r = fp.retrieve(q["query"], K, clauses, doc_emb)
            pool_codes = [codes[i] for i in r["pool"]]
            pre5 = [codes[i] for i in r["pre_dedup_top5"]]
            kept_codes = [codes[i] for i in r["kept"]]
            skip_txt = "; ".join(
                f"{codes[i]}(similarity to {codes[m]}: {s})" for i, m, s in r["skipped"])
            pool_hits = len(set(pool_codes) & q["refs"])
            top5_hits = len(set(kept_codes) & q["refs"])
            row = {
                "query_id": q["qid"], "K": K, "n_ref": q["n_ref"],
                "Candidate clause IDs": "; ".join(pool_codes),
                "Top five before deduplication": "; ".join(pre5),
                "Top five after deduplication": "; ".join(kept_codes),
                "Returned clauses after deduplication": len(kept_codes),
                "Skipped clauses and similarity": skip_txt,
                "Candidate pool hits": pool_hits,
                "Candidate Recall@K": round(pool_hits / q["n_ref"], 4),
                "Top-five hits": top5_hits,
                "P@5": round(top5_hits / fp.TOP_K_FINAL, 4),
                "R@5": round(top5_hits / q["n_ref"], 4),
                "rerank_seconds": r["rerank_sec"],
            }
            detail.append(row)
            per_k.append(row)
        summary_rows.append({
            "K": K,
            "Candidate Recall@K": round(statistics.mean(r["Candidate Recall@K"] for r in per_k), 4),
            "P@5": round(statistics.mean(r["P@5"] for r in per_k), 4),
            "R@5": round(statistics.mean(r["R@5"] for r in per_k), 4),
            "Mean returned clauses": round(statistics.mean(r["Returned clauses after deduplication"] for r in per_k), 2),
            "mean_rerank_seconds": round(statistics.mean(r["rerank_seconds"] for r in per_k), 3),
        })

    d_out = os.path.join(common.RESULT_DIR, "candidate_pool_details.csv")
    with open(d_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader(); w.writerows(detail)
    s_out = os.path.join(common.RESULT_DIR, "candidate_pool_summary.csv")
    with open(s_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    print("Saved:", d_out, "Rows:", len(detail))
    print("Saved:", s_out)
    print(f"\n{'K':>4}{'Recall@K':>10}{'P@5':>8}{'R@5':>8}{'Returned':>7}{'Seconds':>9}")
    for r in summary_rows:
        print(f"{r['K']:>4}{r['Candidate Recall@K']:>10}{r['P@5']:>8}{r['R@5']:>8}"
              f"{r['Mean returned clauses']:>7}{r['mean_rerank_seconds']:>9}")
    meet = [r for r in summary_rows if r["P@5"] >= 0.75]
    chosen = min(meet, key=lambda r: r["mean_rerank_seconds"]) if meet else None
    print("\nFastest K with P@5 >= 0.75:", chosen["K"] if chosen else "None (report the shortfall)")


if __name__ == "__main__":
    main()
