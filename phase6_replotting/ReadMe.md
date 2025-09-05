# Phase 6: JSON Structuring and Replotting

This is the final phase of the Datasheet Figures Extraction project. This module takes the fully enriched JSON files from the previous phase, parses the structured data, and uses the `matplotlib` library to digitally re-generate the original graphs.

## Installation
... (same as before) ...

## Usage

**Step 1: Provide Input**
Copy the entire output folder from Phase 5 (`text_extraction_output`) and place it inside the `input_data` folder for this project.

**Step 2: Run the script**
```bash
python main.py




Output Explanation
The script follows the "copy-then-enhance" pattern. It first creates a full copy of the Phase 5 output and then adds the final replotted image to each diagram's folder within the new final_output directory.
Final JSON: The diagram_*.json file in the output is the final, structured data artifact, containing all information gathered throughout the pipeline.
Replotted Figure: For each diagram successfully processed, a new reconstructed_plot.png file is created. This image is a digital-native version of the original graph.
Example Final Output Structure:
code
Code
final_output/
└── ...
    └── diagram_1/
        ├── diagram_1.jpg
        ├── diagram_1.json              (The final structured data)
        ├── components/
        ├── ... (other overlays)
        └── reconstructed_plot.png      <-- FINAL DELIVERABLE
Final JSON Schema Definition
The diagram_*.json file consumed and finalized by this phase has the following structure:
code
JSON
{
    "class": "string (e.g., 'Diagram')",
    "bbox": "[x1, y1, x2, y2] (pixel coordinates on the original page)",
    "bbox_normalized": "[x1, y1, x2, y2] (0-1 normalized coordinates)",
    "conf": "float (confidence of the initial diagram detection)",
    "center": "[x, y]",
    "center_normalized": "[x, y]",
    "legend_boxes": [
        {
            "class": "string (e.g., 'legend_box')",
            "bbox": "[x1, y1, x2, y2] (coordinates relative to the diagram image)",
            "bbox_normalized": "[...]",
            "conf": "float"
        }
    ],
    "labels": [
        {
            "class": "string (e.g., 'plot_area', 'x_axis', 'y_axis')",
            "bbox": "[x1, y1, x2, y2] (coordinates relative to the diagram image)",
            "bbox_normalized": "[...]",
            "conf": "float"
        }
    ],
    "lines": [
        [
            [x1, y1], [x2, y2], ...
        ]
    ],
    "ocr_results": [
        {
            "text": "string (the detected text)",
            "bbox": "[x1, y1, x2, y2] (coordinates relative to the diagram image)",
            "conf": "float",
            "words": "[...]",
            "associated_element": "string (e.g., 'x_axis', 'plot_area', 'none')"
        }
    ]
}
