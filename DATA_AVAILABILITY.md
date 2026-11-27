# Data availability

| Material | Available contents |
|---|---|
| Dataset A (web-sourced images) | Fixed training and validation split manifest, file checksums, and three image–annotation examples; the complete image collection is not included |
| Dataset B (field-collected images) | 150 images, metadata including site names and GIS coordinates, and final consensus reference annotations |
| Regulatory knowledge base | Inventory of 16 documents, catalog of 140 clauses, and three complete clause examples |
| Generated reports | All 120 evaluated reports and their report–observation block–group mapping |
| Evaluation and analysis data | 240 independent rating records, consensus regulatory basis counts, 1,800 perception comparison records, and evaluation results |
| Models and code | Detector weights, training configuration, report generation, retrieval, and statistical analysis code |

The complete clause text table was compiled from regulatory documents lawfully published in China. It is available for academic research upon reasonable request to the corresponding author and review of the request. Contact details are provided in the associated paper.

The repository contains the final consensus reference annotations, rather than the initial independent image annotations. Initial annotation agreement therefore cannot be recomputed from the final annotations alone. Full training and validation of Dataset A require the complete images and annotations.

Public images have undergone local blurring and metadata removal; study site names and GIS coordinates are retained. Image processing did not change the substantive report content or the original scores. As some input pixels have changed and cloud API outputs may vary, regenerated reports are not guaranteed to match the archived reports. Timing and cost records reflect the original experimental conditions.
