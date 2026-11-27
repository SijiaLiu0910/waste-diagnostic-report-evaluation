# -*- coding: utf-8 -*-
"""Perception, contextual alignment, and diagnostic prompts for the four groups."""

# Shared target class definitions.
LABEL_RULE = (
    "overflow=overflowing waste bin (waste extends above the rim or spills outside); "
    "no_overflow=non-overflowing waste bin (a bin is present without overflow); "
    "waste=waste accumulation (scattered or accumulated waste outside bins)."
)

PERCEPT = """Input
Original image: {image_ref}
Image identifier: {image_id}
Shared annotation rules: {label_rule}

Task
Using the input image and shared annotation rules, count targets in the following three classes:
- overflow: overflowing waste bin
- no_overflow: non-overflowing waste bin
- waste: waste accumulation

Instructions
Count each independent bin as one object. Count an integrated bin with multiple openings as one bin.
Count distinguishable areas of waste accumulation. Count a continuous accumulation without clear internal boundaries as one area; count distinguishable accumulations separately when they are separated by a clear gap.
Record only recognized targets. Enter 0 for any class in which no target is recognized.

Output
Return JavaScript Object Notation (JSON) only. Use the input image identifier for image_id. The objects array must contain all three classes, with class and count for each. count must be a non-negative integer.
The example below illustrates the output structure only. Fill in the identifier and counts according to the actual input and recognition results.

{
  "image_id": "S08-A-03.jpg",
  "objects": [
    {"class": "overflow", "count": 0},
    {"class": "no_overflow", "count": 0},
    {"class": "waste", "count": 1}
  ]
}"""

ALIGN = """Input
Original image: {image_ref}
Site data: {field_data}
Site data include the collection area, setting, sampling site name, specific location, GIS coordinates, weather during collection, and target classes and counts from perception.

Task
Generate a scene description using the image records, perception results, and original images.
First state the sampling site’s location and setting, then describe the surroundings of recognized targets, for example, beside a road, at a market entrance, along a path, or next to a planted area.
Describe identifiable items or waste forms, such as plastic bags, cardboard boxes, or foam food containers.

Instructions
Use the classes and counts supplied by perception. Preserve these values even if you believe they are incorrect.
Retain perception results for each image and class. If a class has a zero count in an image, do not describe that class as detected in that image; even if suspected items are visible, do not describe a zero-count class as “overflow” or “accumulation.”
You may describe recognized targets, their surroundings, and identifiable items in the image, but must not add or change any class assignment on this basis.
If all counts are zero (no waste bins or waste accumulation at the site), describe the surroundings faithfully, state that no waste sorting bins or waste are visible, and do not invent bins or anomalies.
Describe targets only in the attached images. Images that are not attached had no detected targets; do not describe their contents or infer them from the site data.
If detection boxes are shown, the targets confirmed by perception are inside those boxes. Describe only the targets inside the boxes; do not describe items outside the boxes or treat them as detected targets.

Output
Scene description: one paragraph.
Retrieval descriptions: provide a short description for each recognized anomaly type. Each description must include the setting, target state, and relevant field characteristics, for insertion into the shared query template by the program."""

DIAG_RAG = """Input
Site data: {field_data}
Scene description: {scene}
Regulatory evidence: {clauses}

Task
Generate a municipal waste management diagnostic report. Use a two-column “Category–Details” table, with rows in the following order.

1. Weather, location, GIS coordinates, target classes and counts
   Place each item in a separate row and retain the values in the site data.

2. Scene description
   Present the description generated during contextual alignment as one paragraph.

3. Regulatory basis
Summarize supported content from the supplied retrieved clauses in bullet points, with each point covering one dimension. These may address responsible parties, duties and prohibitions, technical and sanitation standards, management procedures, or liability and penalties. Start each point with “- ”.
   Immediately follow each regulatory statement with its source, citing the document title, article or section number, and knowledge base clause ID.
   For liability or penalties, state the conditions under which the provision applies.

4. Severity
Based on the recognized problem’s state, count, visible distribution, and surroundings, select “minor,” “moderate,” or “severe” and justify the rating in one sentence.

5. Impact assessment
   Describe visible impacts and possible short-term impacts in the current setting.

6. Potential causes
Propose possible causes based on field evidence. Identify the evidence supporting each inference, or specify which records or observations would be needed to verify it.

7. Long-term effects
   Describe possible effects if the problem persists or recurs. Use conditional statements.

8. Recommendations
   Provide immediate actions for the current problem and long-term measures to reduce recurrence.

9. Priority
   Select “low,” “medium,” or “high” to inform human judgment, and briefly justify the choice.
   Severity describes the extent of the site problem; priority describes the order or urgency of action.

Instructions
Distinguish field observations from model inferences. Use wording such as “may” or “if the problem persists” for inferences, and link them to specific field evidence or conditions. Retain all six diagnostic fields. If the available material does not support a judgment, briefly explain this in the corresponding field."""

DIAG_NO_RAG = """Input
Site data: {field_data}
Scene description: {scene}

Task
Generate a municipal waste management diagnostic report. Use a two-column “Category–Details” table, with rows in the following order.

1. Weather, location, GIS coordinates, target classes and counts
   Place each item in a separate row and retain the values in the site data.

2. Scene description
   Present the description generated during contextual alignment as one paragraph.

3. Regulatory basis
Use the model’s internal knowledge to write a regulatory basis applicable to the site problem, in bullet points. Each point should cover one dimension, which may address responsible parties, duties and prohibitions, technical and sanitation standards, management procedures, or liability and penalties. Start each point with “- ”.
   Follow each regulatory statement with the corresponding document title and article or section number.
   For liability or penalties, state the conditions under which the provision applies.
If the overflow and accumulation counts are both 0 (neither overflowing waste bins nor waste accumulation were recognized), enter only “No overflowing waste bins or waste accumulation were detected in this run; regulatory retrieval was not performed.” in the regulatory basis field. Do not cite any provisions or add a regulatory basis from the model’s internal knowledge.

4. Severity
Based on the recognized problem’s state, count, visible distribution, and surroundings, select “minor,” “moderate,” or “severe” and justify the rating in one sentence.

5. Impact assessment
   Describe visible impacts and possible short-term impacts in the current setting.

6. Potential causes
Propose possible causes based on field evidence. Identify the evidence supporting each inference, or specify which records or observations would be needed to verify it.

7. Long-term effects
   Describe possible effects if the problem persists or recurs. Use conditional statements.

8. Recommendations
   Provide immediate actions for the current problem and long-term measures to reduce recurrence.

9. Priority
   Select “low,” “medium,” or “high” to inform human judgment, and briefly justify the choice.
   Severity describes the extent of the site problem; priority describes the order or urgency of action.

Instructions
Distinguish field observations from model inferences. Use wording such as “may” or “if the problem persists” for inferences, and link them to specific field evidence or conditions. Retain all six diagnostic fields. If the available material does not support a judgment, briefly explain this in the corresponding field."""
