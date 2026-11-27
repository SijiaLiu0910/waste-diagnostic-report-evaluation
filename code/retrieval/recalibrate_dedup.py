# -*- coding: utf-8 -*-
"""Evaluate deduplication thresholds at K=60, reranking once per query before testing each threshold."""
import csv
import os
import statistics
import common
import formal_pipeline as fp

K = 60
THRESHOLDS = [0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 1.01]


def main():
    if os.environ.get("ALLOW_SAMPLE_CORPUS") == "1":
        raise RuntimeError("Formal parameter comparisons require the full 140-clause corpus, not examples.")
    clauses = common.load_clauses()
    codes = [c["code"] for c in clauses]
    queries = common.load_queries()
    doc_emb = fp.build_doc_embeddings(clauses)

    base = []
    for q in queries:
        r = fp.retrieve(q["query"], K, clauses, doc_emb)
        base.append((q, r["reranked"]))

    rows = []
    for th in THRESHOLDS:
        p5, r5, nskip, returned = [], [], [], []
        for q, reranked in base:
            kept, skipped = fp.greedy_dedup(reranked, doc_emb, thresh=th)
            kept_codes = [codes[i] for i in kept]
            hits = len(set(kept_codes) & q["refs"])
            p5.append(hits / 5)
            r5.append(hits / q["n_ref"])
            nskip.append(len(skipped))
            returned.append(len(kept))
        rows.append({"Deduplication threshold": th,
                     "Final P@5": round(statistics.mean(p5), 4),
                     "Final R@5": round(statistics.mean(r5), 4),
                     "Mean skipped clauses": round(statistics.mean(nskip), 2),
                     "Mean returned clauses": round(statistics.mean(returned), 2)})
    out = os.path.join(common.RESULT_DIR, "deduplication_thresholds.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Saved:", out)
    print(f"{'threshold':>7}{'P@5':>8}{'R@5':>8}{'Skipped':>7}{'Returned':>7}")
    for r in rows:
        print(f"{r['Deduplication threshold']:>7}{r['Final P@5']:>8}{r['Final R@5']:>8}"
              f"{r['Mean skipped clauses']:>7}{r['Mean returned clauses']:>7}")


if __name__ == "__main__":
    main()
