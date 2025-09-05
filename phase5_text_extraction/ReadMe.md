# Phase 5: Text Extraction with Google Vision

This package uses the Google Cloud Vision API to perform Optical Character Recognition (OCR) on diagram images. It then associates the extracted text with the component regions (plot area, axes, legends) identified in previous phases.

## CRITICAL: Pre-requisites

**1. Google Cloud Authentication:**
This module **will not work** without Google Cloud credentials.
-   You must have a Google Cloud project with the **Cloud Vision API** enabled.
-   You must create a service account and download its JSON key file.
-   Rename this file to `google_credentials.json`.
-   Place the `google_credentials.json` file in the root directory of this project.

## Installation

**Create a virtual environment (recommended):**

```bash
conda create -n venv python=3.8
conda activate venv
```

2.  **Install the dependencies**
```bash
pip install -r requirements.txt
```


## Usage

**Step 1: Provide Input**
Copy the entire output folder from Phase 4 (`line_extraction_output`) and place it inside the `input_data` folder for this project.

**Step 2: Run the script**
```bash
python main.py
```
### Output Explanation
The script follows the "copy-then-enhance" pattern. It first creates a full copy of the Phase 4 output and then enhances the diagrams within the new text_extraction_output folder.

For each diagram, it performs two actions:
- Updates JSON: It opens the diagram_*.json file and adds a new key, "ocr_results". This key holds a list of all text blocks found, including their content, coordinates, and which component they belong to (e.g., x_axis, legend_box).
- Creates Overlay: It saves a new ocr_overlay.jpg file in the diagram's folder, showing the bounding boxes of the detected text and their associations.

Example Final Output Structure:
```bash
text_extraction_output/
└── ...
    └── diagram_1/
        ├── diagram_1.jpg
        ├── diagram_1.json         <-- NOW CONTAINS "ocr_results" data
        ├── components/
        ├── lines_overlay.jpg
        └── ocr_overlay.jpg        <-- NEW VISUALIZATION
```