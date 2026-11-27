# Image privacy processing and data integrity

Public images were checked using face, text, and identifier detection followed by manual review. Soft-edged local Gaussian blur was applied to faces, contact information, and potentially identifying vehicle or parcel markings. EXIF, XMP, IPTC, and comment metadata were removed; ICC color profiles were retained. Identifying parcel text in report prose was also anonymized. Site names and GIS coordinates remain in the study tables and reports.

The [image redaction log](image_redactions.csv) lists 42 Dataset B images with 74 processed regions. Region counts describe processed areas, rather than confirmed instances of personal information; some uncertain identifiers were treated conservatively. The three Dataset A examples match the representative manuscript images and received no additional visual redaction.

All 150 Dataset B images were copied from the original JPEGs. For 108 images, only ancillary metadata were removed without re-encoding, preserving decoded pixels. The other 42 were locally blurred after copying and saved as JPEGs; re-encoding may also introduce minor compression differences outside the blurred regions. Dimensions, display orientation, and annotation coordinates were preserved.

All 600 embedded image instances in the 120 reports were synchronized with the public images: 168 use locally blurred images, and the remainder use metadata-cleaned images. Embedded images are display thumbnails. The image updates preserved substantive report content, original ratings, and reference annotations.

Local paths and other metadata were removed from the model file. All 355 state dictionary tensors are identical to those in the original model. API keys are configured through environment variables. File integrity can be checked against `MANIFEST_SHA256.csv` in the repository root.

See [data availability](../DATA_AVAILABILITY.md) for the implications of privacy processing and API variability for regenerated outputs.
