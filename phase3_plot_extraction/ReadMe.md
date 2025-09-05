# Phase 3: Plot Area and Label Extraction Module

This package takes cropped diagram images (from Phase 2) as input and uses two YOLOv5 models to detect and extract fine-grained components like the plot area, axes, and legends.

It updates the JSON metadata file for each diagram and saves the cropped component images to an output directory.

## Pre-requisites

1.  **Copy YOLOv5 Repository:** You must have a local copy of the `yolov5` repository in the root of this project folder (`phase3_plot_extraction/yolov5`).

2.  **Place Model Files:** Place your `legend-detector-best.pt` and `labelChart-detector-best.pt` model files inside the `models` directory.

## Installation
... (same as before) ...

## Usage

### 1. As a Standalone Script

**Step 1: Provide Input**

Copy the entire output directory from Phase 2 (e.g., `detection_output`) and place it inside the `input_diagrams` folder of this project.

**Step 2: Run the script**

Execute `main.py` from your terminal:
```bash
python main.py

### Output Explanation

This script's output is a **complete superset** of its input. It is designed to be non-destructive.

1.  **Full Copy:** First, the script creates a full copy of your input directory (e.g., `input_diagrams/detection_output/`) in the `extraction_output` folder. This ensures that all files from Phase 2, including overlays and pages with no diagrams, are preserved.

2.  **Enhancement:** The script then processes the diagrams *within this new output folder*. For each diagram, it:
    *   Adds `"legend_boxes"` and `"labels"` data to its `diagram_*.json` file.
    *   Creates a new `components` subfolder containing cropped images of the detected plot area, axes, etc.
    *   Creates a new `components_overlay.jpg`.

**Example Final Output Structure:**

If your input included a page with diagrams (`page_1`) and a page without (`page_2`), the final output will contain both:
extraction_output/
└── detection_output/
└── datasheet_A/
├── page_1/
│ ├── diagrams/
│ │ └── diagram_1/
│ │ ├── diagram_1.jpg (Original diagram image)
│ │ ├── diagram_1.json (NOW CONTAINS component data)
│ │ ├── components/ (NEW FOLDER)
│ │ │ └── plot_area/ ...
│ │ └── components_overlay.jpg (NEW FILE)
│ │
│ └── detection_overlay.jpg (Preserved from Phase 2)
│
└── page_2/
└── page_2_original.jpg (Preserved from Phase 2)