# Evaluation data and results

## Evaluation records

`report_quality/ratings_long.csv` contains 240 records, one per report and rater. `Rater1` and `Rater2` are anonymous rater IDs. `sampling_site_id` identifies the 15 clustering units, and `observation_block_id` identifies the 30 paired observation blocks.

| Fields | Definition |
|---|---|
| `report_id`, `group`, `perception_pathway`, `rag_condition` | Report identifier and the A–D configurations crossing MLLM or YOLO perception with or without RAG |
| `basis_count`, `correct_basis_count`, `matched_basis_count` | Consensus counts after resolving disagreements; matched items are both correct and applicable to the site |
| `basis_correctness_rate`, `scenario_matching_rate` | Corresponding counts divided by `basis_count`; undefined denominators are recorded as NA |
| `reliability`, `logical_coherence` | Independent integer scores from 1 to 5 |
| `perception_seconds`, `alignment_seconds`, `retrieval_seconds`, `generation_seconds`, `total_seconds` | Stage durations and their sum, in seconds |
| `total_tokens`, `api_cost_cny` | Report-level token usage and API cost in Chinese yuan (CNY) |

Regulatory basis counts and efficiency records are repeated for both raters. Deduplicate by `report_id` before computing report-level metrics. Group regulatory basis rates are unweighted means across reports; ordinal models use the independent ratings. Rater agreement is not calculated for consensus regulatory basis counts.

`perception/perception_detail.csv` contains 1,800 records: 150 images × three classes × four groups. A class is present when `true_count` or `predicted_count` exceeds zero. `group`, `observation_block_id`, `image_id`, and `class` identify each record; `true_presence` and `predicted_presence` indicate presence, `presence_outcome` records TP/FP/FN/TN, and `absolute_count_error` is the absolute difference in counts. Macro metrics weight classes equally. MAE is the mean absolute counting error across images and classes. Counts summed across images are not the number of distinct objects after tracking or cross-image deduplication.

Dataset B metadata use `image_id`, `setting`, `location`, `gis_coordinates`, `capture_time`, `weather`, `zoom`, `aspect_ratio`, `resolution`, and `file_format`. GIS coordinates retain their original longitude–latitude order. Stable image, report, observation block, and clause IDs are preserved throughout the repository.

## Result files

| File | Contents |
|---|---|
| `retrieval/retrieval_evaluation.xlsx` | Candidate pool size, deduplication threshold, and candidate recall curve records |
| `statistics/statistical_results.xlsx` | Group summaries, numerical differences and intervals, overall recommendation score tests, and prespecified contrasts |
| `statistics/statistical_checks.xlsx` | Rating agreement, rater sensitivity analyses, model coefficients, and diagnostics |
| `detection/validation_metrics.csv` and `detection/validation_metrics.json` | Detector validation metrics |
| Summary CSVs in `perception/` and `efficiency_cost/` | Machine-readable perception and stage efficiency summaries |

Blank cells or NA indicate inapplicable or undefined values, not zero. Regulatory basis rates apply only to reports that provide a basis. Linear weighted kappa is the primary rating agreement measure; quadratic weighted kappa is supplementary. Tests retain their original multiple-comparison families and Holm-adjusted p values. Odds ratio (OR) intervals are the corresponding 95% model-based intervals. Numerical contrast tables specify units, effective observation blocks, and cluster bootstrap information.
