# Object detection validation results

Validation used the complete set of 699 validation images and the weights in `models/detector.pt`.

| Parameter | Setting |
|---|---|
| Ultralytics version | 8.4.203 |
| Input image size | 640 |
| Confidence threshold | 0.001 |
| Non-maximum suppression (NMS) IoU threshold | 0.60 |

`validation_metrics.csv` and `validation_metrics.json` provide overall and per-class metrics. Overall precision, recall, and F1 are unweighted means of the respective per-class metrics. Overall F1 is not recalculated from overall precision and recall, consistent with Table 5 of the paper.

These validation settings differ from the inference settings for field report generation, which use a confidence threshold of 0.25 and an NMS IoU threshold of 0.70. Recalculation commands are provided in the [root README](../../../README.md).
