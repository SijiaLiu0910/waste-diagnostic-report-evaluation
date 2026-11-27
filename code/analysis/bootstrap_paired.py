#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate paired differences and confidence intervals using 5,000 sampling-site cluster bootstrap replicates."""
import os
from pathlib import Path
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
BASE = str(Path(__file__).resolve().parents[2])
OUT_DIR = os.path.join(BASE, "results/recomputed/statistics")
SUM_DIR = os.path.join(BASE, "results/evaluation/perception")
os.makedirs(OUT_DIR, exist_ok=True)

LONG_CSV = os.path.join(BASE, "results/evaluation/report_quality/ratings_long.csv")
PERC_CSV = os.path.join(SUM_DIR, "perception_detail.csv")
OUT_CSV = os.path.join(OUT_DIR, "paired_differences_and_intervals.csv")

R = 5000
SEED = 20261122
GROUPS = ["A", "B", "C", "D"]
GI = {g: i for i, g in enumerate(GROUPS)}
GLABEL = {
    "A": "A: MLLM without RAG", "B": "B: MLLM with RAG",
    "C": "C: YOLO without RAG", "D": "D: YOLO with RAG",
}
CLASSES = ["overflow", "no_overflow", "waste"]
CI = {c: i for i, c in enumerate(CLASSES)}
PAIRS = [("B", "A"), ("D", "C"), ("C", "A"), ("D", "B")]
PAIR_LABEL = {("B", "A"): "B−A (with versus without RAG, MLLM)",
              ("D", "C"): "D−C (with versus without RAG, YOLO)",
              ("C", "A"): "C−A (YOLO versus MLLM, without RAG)",
              ("D", "B"): "D−B (YOLO versus MLLM, with RAG)"}
FACTORS = ["Δ_Perception", "Δ_RAG", "Δ_Interaction"]


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
long_df = pd.read_csv(LONG_CSV, encoding="utf-8-sig")
rep = long_df.drop_duplicates(subset=["report_id"]).reset_index(drop=True)

rep["basis_correctness_rate"] = np.where(rep["basis_count"] > 0,
                            rep["correct_basis_count"] / rep["basis_count"], np.nan)
rep["scenario_matching_rate"] = np.where(rep["basis_count"] > 0,
                            rep["matched_basis_count"] / rep["basis_count"], np.nan)

perc = pd.read_csv(PERC_CSV, encoding="utf-8-sig")
block2point = rep.drop_duplicates("observation_block_id").set_index(
    "observation_block_id")["sampling_site_id"].to_dict()

points = sorted(rep["sampling_site_id"].unique())
point_blocks = {p: sorted(rep.loc[rep["sampling_site_id"] == p, "observation_block_id"].unique())
                for p in points}
blocks_all = sorted(rep["observation_block_id"].unique())
BI = {b: i for i, b in enumerate(blocks_all)}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
boot_point_idx = rng.integers(0, len(points), size=(R, len(points)))
BOOT_BLOCK_IDX = []
for i in range(R):
    idxs = []
    for j in boot_point_idx[i]:
        idxs.extend(BI[b] for b in point_blocks[points[j]])
    BOOT_BLOCK_IDX.append(idxs)


def percentile_ci(arr):
    """Return the percentile 95% CI and valid replicate count; use NA if fewer than two replicates are valid."""
    v = np.asarray([x for x in arr if not pd.isna(x)])
    if len(v) < 2:
        return np.nan, np.nan, len(v)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return lo, hi, len(v)


def blocks_to_points(blks):
    return len({block2point[b] for b in blks if b in block2point})


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def group_mean(arr, gi, idxs):
    v = arr[idxs, gi]
    v = v[~pd.isna(v)]
    return v.mean() if len(v) else np.nan


def pair_diff(arr, g1, g2, idxs):
    v1, v2 = arr[idxs, GI[g1]], arr[idxs, GI[g2]]
    mask = ~(pd.isna(v1) | pd.isna(v2))
    d = v1[mask] - v2[mask]
    return d.mean() if len(d) else np.nan


def factor(arr, kind, idxs):
    sub = arr[idxs, :]
    mask = ~pd.isna(sub).any(axis=1)
    s = sub[mask]
    if len(s) == 0:
        return np.nan
    A, B, C, D = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    if kind == "Δ_Perception":
        return ((C + D - A - B) / 2).mean()
    if kind == "Δ_RAG":
        return ((B + D - A - C) / 2).mean()
    return ((D - C) - (B - A)).mean()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def basis_array(metric_col):
    w = rep.pivot_table(index="observation_block_id", columns="group",
                        values=metric_col, aggfunc="first")
    return w.reindex(index=blocks_all, columns=GROUPS).values


zero_basis_by_group = rep[rep["basis_count"] == 0].groupby("group").size().to_dict()
zero_basis_ids = rep.loc[rep["basis_count"] == 0, "report_id"].tolist()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
counts = np.zeros((len(blocks_all), 4, 3, 3))
for _, r in perc.iterrows():
    bi = BI[r["observation_block_id"]]
    gi = GI[r["group"]]
    ci = CI[r["class"]]
    tag = r["presence_outcome"]
    if tag == "TP":
        counts[bi, gi, ci, 0] += 1
    elif tag == "FP":
        counts[bi, gi, ci, 1] += 1
    elif tag == "FN":
        counts[bi, gi, ci, 2] += 1

mae_block = perc.groupby(["observation_block_id", "group"])["absolute_count_error"].mean().unstack()
MAE_ARR = mae_block.reindex(index=blocks_all, columns=GROUPS).values


def perc_metrics_from(csub, mae_vals):
    """Compute macro precision, recall, and F1 from group–class TP/FP/FN counts and return group MAE."""
    out = {}
    for gi, g in enumerate(GROUPS):
        P, Rc, F1 = [], [], []
        for ci in range(3):
            tp, fp, fn = csub[gi, ci]
            if tp + fp > 0:
                P.append(tp / (tp + fp))
            if tp + fn > 0:
                Rc.append(tp / (tp + fn))
            if 2 * tp + fp + fn > 0:
                F1.append(2 * tp / (2 * tp + fp + fn))
        out[g] = {
            "Macro-Precision": np.mean(P) if P else np.nan,
            "Macro-Recall": np.mean(Rc) if Rc else np.nan,
            "Macro-F1": np.mean(F1) if F1 else np.nan,
            "Counting MAE": mae_vals[gi] if not pd.isna(mae_vals[gi]) else np.nan,
        }
    return out


full_perc = perc_metrics_from(counts.sum(axis=0), np.nanmean(MAE_ARR, axis=0))
boot_perc = []
for idxs in BOOT_BLOCK_IDX:
    boot_perc.append(perc_metrics_from(
        counts[idxs].sum(axis=0), np.nanmean(MAE_ARR[idxs], axis=0)))

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
EFF_METRICS = [
    ("total_seconds", "total_seconds", "s"),
    ("api_cost_cny", "api_cost_cny", "CNY"),
    ("perception_seconds", "perception_seconds", "s"),
    ("alignment_seconds", "alignment_seconds", "s"),
    ("retrieval_seconds", "retrieval_seconds", "s"),
    ("generation_seconds", "generation_seconds", "s"),
    ("total_tokens", "total_tokens", "count"),
]
eff_arr = {}
for _, col, _ in EFF_METRICS:
    w = rep.pivot_table(index="observation_block_id", columns="group",
                        values=col, aggfunc="first")
    eff_arr[col] = w.reindex(index=blocks_all, columns=GROUPS).values

RAG_ONLY_GROUPS = {"B", "D"}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
rows = []
cache = {"basis": {}, "eff": {}}


def add_row(section, metric, ctype, contrast, g1, g2, point, lo, hi,
            n_blocks, n_points, n_valid_boot, unit, note=""):
    rows.append({
        "component": section, "metric": metric, "contrast_type": ctype, "contrast": contrast,
        "group_1": g1, "group_2": g2, "estimate": point, "ci95_lower": lo, "ci95_upper": hi,
        "ci_method": "Percentile (2.5%/97.5%)", "valid_blocks": n_blocks,
        "valid_sites": n_points, "valid_bootstrap_replicates": n_valid_boot,
        "unit": unit, "note": note,
    })


def fmt(v, nd=4):
    if v is None or pd.isna(v):
        return "NA"
    return round(float(v), nd)


BASIS_METRICS = [("basis_correctness_rate", "basis_correctness_rate"), ("scenario_matching_rate", "scenario_matching_rate")]
for metric_col, metric_name in BASIS_METRICS:
    arr = basis_array(metric_col)
    cache["basis"][metric_name] = {}
    for g in GROUPS:
        gi = GI[g]
        valid_mask = ~pd.isna(arr[:, gi])
        n_blocks = int(valid_mask.sum())
        n_points = blocks_to_points([blocks_all[i] for i in np.where(valid_mask)[0]])
        boot_vals = [group_mean(arr, gi, idxs) for idxs in BOOT_BLOCK_IDX]
        lo, hi, n_ok = percentile_ci(boot_vals)
        point = group_mean(arr, gi, list(range(len(blocks_all))))
        cache["basis"][metric_name][g] = (point, lo, hi)
        note = f"{zero_basis_by_group.get(g, 0)} reports without a basis; inference concerns the quality of the basis when provided"
        add_row("Regulatory basis", metric_name, "Group estimate", GLABEL[g], g, "",
                fmt(point, 4), fmt(lo, 4), fmt(hi, 4), n_blocks, n_points,
                n_ok, "proportion", note)
    for g1, g2 in PAIRS:
        mask = ~(pd.isna(arr[:, GI[g1]]) | pd.isna(arr[:, GI[g2]]))
        n_blocks = int(mask.sum())
        n_points = blocks_to_points([blocks_all[i] for i in np.where(mask)[0]])
        boot_vals = [pair_diff(arr, g1, g2, idxs) for idxs in BOOT_BLOCK_IDX]
        lo, hi, n_ok = percentile_ci(boot_vals)
        point = pair_diff(arr, g1, g2, list(range(len(blocks_all))))
        add_row("Regulatory basis", metric_name, "Paired difference", PAIR_LABEL[(g1, g2)],
                g1, g2, fmt(point * 100, 3), fmt(lo * 100, 3), fmt(hi * 100, 3),
                n_blocks, n_points, n_ok, "percentage points",
                "Difference = group_1−group_2; only blocks with a basis in both groups are included")
    complete = ~pd.isna(arr).any(axis=1)
    n_blocks = int(complete.sum())
    n_points = blocks_to_points([blocks_all[i] for i in np.where(complete)[0]])
    for kind in FACTORS:
        boot_vals = [factor(arr, kind, idxs) for idxs in BOOT_BLOCK_IDX]
        lo, hi, n_ok = percentile_ci(boot_vals)
        point = factor(arr, kind, list(range(len(blocks_all))))
        add_row("Regulatory basis", metric_name, "Factorial contrast", kind, "", "",
                fmt(point * 100, 3), fmt(lo * 100, 3), fmt(hi * 100, 3),
                n_blocks, n_points, n_ok, "percentage points",
                "Complete blocks across all four groups; contrasts are calculated within blocks and then averaged")

PERC_METRICS = ["Macro-Precision", "Macro-Recall", "Macro-F1", "Counting MAE"]
for metric in PERC_METRICS:
    unit = "count error" if metric == "Counting MAE" else "proportion"
    section = "Counting error" if metric == "Counting MAE" else "Class presence"
    for g in GROUPS:
        boot_vals = [bp[g][metric] for bp in boot_perc]
        lo, hi, n_ok = percentile_ci(boot_vals)
        point = full_perc[g][metric]
        add_row(section, metric, "Group estimate", GLABEL[g], g, "",
                fmt(point, 4), fmt(lo, 4), fmt(hi, 4),
                len(blocks_all), len(points), n_ok, unit,
                "Recompute TP/FP/FN from resampled image–class records, then average metrics across the three classes"
                if metric != "Counting MAE" else "Equally weighted counting errors within blocks")
    for g1, g2 in PAIRS:
        boot_vals = [bp[g1][metric] - bp[g2][metric] for bp in boot_perc]
        lo, hi, n_ok = percentile_ci(boot_vals)
        point = full_perc[g1][metric] - full_perc[g2][metric]
        add_row(section, metric, "Paired difference", PAIR_LABEL[(g1, g2)], g1, g2,
                fmt(point, 4), fmt(lo, 4), fmt(hi, 4),
                len(blocks_all), len(points), n_ok, unit,
                "Difference = group_1−group_2, paired within the same bootstrap replicate")
    for kind in FACTORS:
        def fac(bp, kind=kind):
            A, B, C, D = (bp["A"][metric], bp["B"][metric],
                          bp["C"][metric], bp["D"][metric])
            if any(pd.isna(x) for x in (A, B, C, D)):
                return np.nan
            if kind == "Δ_Perception":
                return (C + D - A - B) / 2
            if kind == "Δ_RAG":
                return (B + D - A - C) / 2
            return (D - C) - (B - A)
        boot_vals = [fac(bp) for bp in boot_perc]
        lo, hi, n_ok = percentile_ci(boot_vals)
        point = fac(full_perc)
        add_row(section, metric, "Factorial contrast", kind, "", "",
                fmt(point, 4), fmt(lo, 4), fmt(hi, 4),
                len(blocks_all), len(points), n_ok, unit,
                "Based on group-level metrics recomputed within the same bootstrap replicate")

for metric_name, col, unit in EFF_METRICS:
    arr = eff_arr[col]
    cache["eff"][metric_name] = {}
    rag_stage = col == "retrieval_seconds"
    nd = 3 if unit == "s" else (5 if unit == "CNY" else 1)
    for g in GROUPS:
        gi = GI[g]
        if rag_stage and g not in RAG_ONLY_GROUPS:
            add_row("Efficiency and cost", metric_name, "Group estimate", GLABEL[g], g, "",
                    "NA", "NA", "NA", 0, 0, 0, unit,
                    "No RAG stage in this group (structural NA)")
            continue
        valid_mask = ~pd.isna(arr[:, gi])
        n_blocks = int(valid_mask.sum())
        n_points = blocks_to_points([blocks_all[i] for i in np.where(valid_mask)[0]])
        boot_vals = [group_mean(arr, gi, idxs) for idxs in BOOT_BLOCK_IDX]
        lo, hi, n_ok = percentile_ci(boot_vals)
        point = group_mean(arr, gi, list(range(len(blocks_all))))
        cache["eff"][metric_name][g] = (point, lo, hi)
        add_row("Efficiency and cost", metric_name, "Group estimate", GLABEL[g], g, "",
                fmt(point, nd), fmt(lo, nd), fmt(hi, nd),
                n_blocks, n_points, n_ok, unit,
                "Observed long durations, retries, and valid zero costs and token counts are retained")
    for g1, g2 in PAIRS:
        if rag_stage and (g1 not in RAG_ONLY_GROUPS or g2 not in RAG_ONLY_GROUPS):
            add_row("Efficiency and cost", metric_name, "Paired difference", PAIR_LABEL[(g1, g2)],
                    g1, g2, "NA", "NA", "NA", 0, 0, 0, unit,
                    "Only B and D include a RAG stage; this contrast is unavailable")
            continue
        mask = ~(pd.isna(arr[:, GI[g1]]) | pd.isna(arr[:, GI[g2]]))
        n_blocks = int(mask.sum())
        n_points = blocks_to_points([blocks_all[i] for i in np.where(mask)[0]])
        boot_vals = [pair_diff(arr, g1, g2, idxs) for idxs in BOOT_BLOCK_IDX]
        lo, hi, n_ok = percentile_ci(boot_vals)
        point = pair_diff(arr, g1, g2, list(range(len(blocks_all))))
        add_row("Efficiency and cost", metric_name, "Paired difference", PAIR_LABEL[(g1, g2)],
                g1, g2, fmt(point, nd), fmt(lo, nd), fmt(hi, nd),
                n_blocks, n_points, n_ok, unit,
                "Difference = group_1−group_2, paired within blocks and then averaged")
    for kind in FACTORS:
        if rag_stage:
            add_row("Efficiency and cost", metric_name, "Factorial contrast", kind, "", "",
                    "NA", "NA", "NA", 0, 0, 0, unit,
                    "Only B and D include a RAG stage; factorial contrasts are unavailable")
            continue
        boot_vals = [factor(arr, kind, idxs) for idxs in BOOT_BLOCK_IDX]
        lo, hi, n_ok = percentile_ci(boot_vals)
        point = factor(arr, kind, list(range(len(blocks_all))))
        add_row("Efficiency and cost", metric_name, "Factorial contrast", kind, "", "",
                fmt(point, nd), fmt(lo, nd), fmt(hi, nd),
                len(blocks_all), len(points), n_ok, unit,
                "Calculate contrasts within blocks and then average")

n_tok_miss = int(rep["total_tokens"].isna().sum())
n_cost_miss = int(rep["api_cost_cny"].isna().sum())
n_sec_miss = int(rep["total_seconds"].isna().sum())
n_tok_zero = int((rep["total_tokens"] == 0).sum())
n_cost_zero = int((rep["api_cost_cny"] == 0).sum())
audit_rows = [
    ("Missing or unverifiable total token records", n_tok_miss, "Missing values are not replaced by zero; no token values are missing in these efficiency records"),
    ("Missing or unverifiable total cost records", n_cost_miss, "Missing values are not replaced by zero; no costs are missing in these efficiency records"),
    ("Missing total duration records", n_sec_miss, ""),
    ("Valid zero-token report totals", n_tok_zero, "Valid original records are retained; YOLO perception does not consume tokens"),
    ("Valid zero-cost report totals", n_cost_zero, "Valid original records are retained"),
    ("Reports with basis_count=0", len(zero_basis_ids), "IDs: " + "、".join(zero_basis_ids)),
]
for name, val, note in audit_rows:
    rows.append({
        "component": "Data checks", "metric": name, "contrast_type": "", "contrast": "",
        "group_1": "", "group_2": "", "estimate": val, "ci95_lower": "NA",
        "ci95_upper": "NA", "ci_method": "—", "valid_blocks": "",
        "valid_sites": "", "valid_bootstrap_replicates": "", "unit": "", "note": note,
    })

result = pd.DataFrame(rows)
result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")


print("Paired bootstrap results saved to", OUT_CSV)
