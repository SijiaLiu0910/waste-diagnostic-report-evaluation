# Dataset A

Dataset A comprises 3,349 web-sourced images: 2,650 for training and 699 for validation.

| File or directory | Contents |
|---|---|
| `split_manifest.csv` | Image splits and SHA-256 checksums of the original images and annotation files |
| `examples/images/` | Three example images, one per target class |
| `examples/labels/` | Corresponding YOLO-format annotations |
| `data.yaml` | Dataset path and class configuration template |

The examples match the representative images in the paper:

| Example | Original image ID | Split | Target class |
|---|---|---|---|
| `A_example_01.jpg` | `000689.jpg` | Validation | Overflowing waste bin |
| `A_example_02.jpg` | `001862.jpg` | Validation | Waste accumulation |
| `A_example_03.jpg` | `002325.jpg` | Validation | Non-overflowing waste bin |

Detector class IDs are 0 = overflowing waste bin (`overflow`), 1 = waste accumulation (`waste`), and 2 = non-overflowing waste bin (`no_overflow`). Perception JSON and reports list classes in the order `overflow`, `no_overflow`, and `waste`; class identities are unchanged.

Organize the complete training or validation data as follows and supply the corresponding `data.yaml` path at runtime:

```text
dataset_A/
├── train/images/
├── train/labels/
├── val/images/
├── val/labels/
└── data.yaml
```

The three examples illustrate the data and annotation format and do not replace the full training or validation set. See [data availability](../../DATA_AVAILABILITY.md).
