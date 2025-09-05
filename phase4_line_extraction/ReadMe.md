# Phase 4: Line Extraction using LineFormer

This package processes `plot_area` images identified in Phase 3. It uses the LineFormer model to detect data curves and digitizes them into a series of coordinates.

## Pre-requisites

1.  **Place Model Files:** Place the LineFormer model (`best_segm_mAP_iter_3679.pth`) and its configuration (`lineformer_config.py`) into the `models` directory.

2.  **Provide Helper Scripts:** The LineFormer helper scripts (`infer.py` and `line_utils.py`) must be placed inside the `line_extractor/` package directory.

## Installation

This package has specific dependencies for the LineFormer model.

1.  **Create a virtual environment (recommended):**
    ```bash
    conda create -n venv python=3.8
    conda activate venv
    ```

2.  **Install this dependencies manually and in order:**
    ```bash
    pip install openmim==0.3.9
    mim install mmcv-full==1.7.2
    mim install "mmengine>=0.7.0"
    cd mmdetection
    pip install -v -e .
    cd phase4_line_extraction
    pip install -r requirements.txt
    ```

## Usage

**Step 1: Provide Input**

Copy the entire output folder from Phase 3 (`extraction_output`) and place it inside the `input_data` folder for this project.

**Step 2: Run the script**
```bash
python main.py
```

### Output Explanation

The script follows the "copy-then-enhance" pattern to create a complete and non-destructive output.

Full Copy: It first creates a full copy of the Phase 3 output in the line_extraction_output folder.

Enhancement: It then finds every plot_area image within this new structure and performs two actions:
- Updates JSON: It locates the parent diagram_*.json file and adds a new key, "lines", containing an array of coordinate data for each detected curve.
- Creates Overlay: It saves a new lines_overlay.jpg file in the same folder as the plot_area image, visually showing the detected lines.

#### Example Output Structure:
The output for a processed diagram will now be fully enriched:

```bash
line_extraction_output/
└── ...
    └── diagram_1/
        ├── diagram_1.jpg
        ├── diagram_1.json         <-- NOW CONTAINS "lines" data
        ├── components/
        │   └── plot_area/
        │       ├── plot_area_1.jpg
        │       └── lines_overlay.jpg  <-- NEW VISUALIZATION
        │
        └── components_overlay.jpg```
```

