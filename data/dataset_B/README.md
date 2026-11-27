# Dataset B

Dataset B contains 150 field-collected JPEG images from 15 sampling sites and 30 observation blocks, with five images per block.

| File or directory | Contents |
|---|---|
| `images/` | Field-collected images, named by image ID |
| `metadata.csv` | Image IDs, settings, locations, GIS coordinates, capture times, weather, and camera settings |
| `reference_annotations.xml` | Final consensus reference annotations, including target classes and bounding box coordinates |

Image IDs follow `Sxx-A/B-xx`, identifying the sampling site, observation block, and image within the block. Annotation classes are `overflow`, `no_overflow`, and `waste`. Image-level class counts correspond to the perception evaluation data.

Identifiable information has been obscured using soft-edged local Gaussian blur. Image dimensions and annotation coordinates are unchanged. See the [image redaction records](../../privacy/image_redactions.csv) and [data availability statement](../../DATA_AVAILABILITY.md).
