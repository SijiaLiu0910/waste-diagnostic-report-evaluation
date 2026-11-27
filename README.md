# Waste diagnostic report evaluation

This repository provides the code, data, and evaluation results for a study of diagnostic report generation for waste management. The experiment crosses two perception pathways with two regulatory knowledge conditions to form four configurations.

| Group | Perception pathway | Regulatory knowledge condition |
|---|---|---|
| A | Multimodal large language model (MLLM) | Model internal knowledge, without retrieval-augmented generation (RAG) |
| B | MLLM | With RAG |
| C | YOLO object detection | Model internal knowledge, without RAG |
| D | YOLO object detection | With RAG |

Each group independently follows the sequence of problem perception, contextual alignment, and diagnostic recommendations. The generation model is `Qwen/Qwen3-VL-32B-Instruct`; retrieval uses `BAAI/bge-large-zh-v1.5` and `Qwen/Qwen3-Reranker-8B`. The complete prompts are provided in the [prompt module](code/workflow/prompts.py).

Repository text and prompts have been translated from Chinese into English. Rerunning the workflow with the English prompts may produce reports that differ from the archived outputs; the evaluation scores refer to the original Chinese reports.

## Repository structure

| Directory or file | Contents |
|---|---|
| `code/training/` | Object detection training, validation, and training configuration |
| `code/workflow/` | Four-group report generation workflow and prompts |
| `code/retrieval/` | Regulatory retrieval and evaluation of candidate pool size and deduplication thresholds |
| `code/analysis/` | Metrics, paired cluster bootstrap, ordinal mixed-effects models, and rater sensitivity analyses |
| `data/dataset_A/` | Training and validation split manifest and three image–annotation examples from web-sourced images |
| `data/dataset_B/` | 150 field-collected images, metadata, and final consensus reference annotations |
| `data/knowledge_base/` | Regulatory document inventory, clause catalog, and text examples |
| `data/retrieval_queries/` | Ten retrieval evaluation queries and reference clause IDs |
| `models/` | Detector weights and model information |
| `results/reports/` | 120 evaluated reports and their identifier mapping |
| `results/evaluation/` | Evaluation data, metric summaries, and statistical results |
| `privacy/` | Image redaction records and data integrity information |
| `MANIFEST_SHA256.csv` | File sizes and SHA-256 checksums |

The result workbooks contain:

- [Retrieval evaluation](results/evaluation/retrieval/retrieval_evaluation.xlsx): candidate pool summaries and details, deduplication thresholds, and candidate recall curves.
- [Main statistical results](results/evaluation/statistics/statistical_results.xlsx): group summaries, numerical comparisons, and recommendation score comparisons.
- [Supplementary statistical checks](results/evaluation/statistics/statistical_checks.xlsx): rating agreement, rater sensitivity analyses, and model diagnostics.

Evaluation records are also provided as CSV files. See the [data dictionary](results/evaluation/data_dictionary.md) for field definitions and units and the [data availability statement](DATA_AVAILABILITY.md) for access details.

## Environment and API configuration

Inference and validation used Python 3.9 and the dependencies in `requirements.txt`. Statistical analyses use a separate environment; Python dependencies are listed in `requirements-statistics.txt` and R package versions in `environment_R.txt`. Separate virtual environments are recommended.

```bash
python -m pip install -r requirements.txt
export SF_KEY=XXXX_YOUR_API_KEY
export YOLO_DEVICE=cpu
```

Set `SF_KEY` to your API key and, if needed, `SF_API_BASE` to the service endpoint. See `.env.example`; this file is not loaded automatically. Set `YOLO_DEVICE` to `cpu`, `mps`, or a CUDA device index as appropriate. `YOLO_PYTHON` optionally specifies the Python interpreter for object detection.

## Generate reports

Run commands from the repository root, specifying an observation block and one or more groups. Available block IDs are listed in the Dataset B metadata.

```bash
python code/workflow/run_site.py --help
python code/workflow/run_site.py S01-A A
python code/workflow/run_site.py S01-A C
```

Groups B and D require the complete 140-clause text table. Set its path using `CLAUSE_CSV`.

```bash
export CLAUSE_CSV=private/clauses_full.csv
python code/workflow/run_site.py S01-A B
python code/workflow/run_site.py S01-A D
```

New outputs are written to `results/workflow_runs/`, separately from the evaluated reports. Generation, embedding, and reranking calls use the API account configured by the user.

## Recompute evaluation and statistical results

```bash
python code/analysis/compute_metrics.py
python code/analysis/bootstrap_paired.py
Rscript code/analysis/clmm_models.R
Rscript code/analysis/rater_sensitivity.R
```

Recomputed results are written to `results/recomputed/` in CSV and other machine-readable formats. They correspond to the relevant sections of the archived workbooks, which retain the precision of the archived estimates and the original multiple-comparison families.

## Train and validate the detector

Training and validation require the complete Dataset A; see the [Dataset A documentation](data/dataset_A/README.md) for its directory structure. Historical training used Ultralytics 8.3.26; inference and validation used 8.4.203. Training settings are provided in `code/training/training_config.yaml`.

```bash
python code/training/train.py --data XXXX_FULL_DATASET/data.yaml --device cpu
python code/training/validate.py --data XXXX_FULL_DATASET/data.yaml --device cpu
```

Training uses `optimizer: auto`, whose selection rules depend on the software version. See the [privacy and integrity documentation](privacy/README.md) for image processing and model integrity details.
