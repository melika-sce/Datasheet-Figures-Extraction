
# Phase 2: Graph Detection Module

This package detects graph and figure regions from page images using a configurable YOLO model. It is designed to process image files (e.g., PNG, JPG) and produces cropped images of each detection, JSON metadata, and visual overlays.

The diagrams are sorted in a natural **reading order** (top-to-bottom, left-to-right) by intelligently grouping detections into rows.

## Pre-requisites

This module can operate in two modes, `'yolov5'` or `'yolov11'`. Your setup must match the mode you intend to use.

#### For YOLOv5 Mode:
1.  **Download YOLOv5 Repository:** You must have a local copy of the `yolov5` repository.
    *   Download the ZIP from [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
    *   Unzip it and rename the folder from `yolov5-master` to `yolov5`.
    *   Place the `yolov5` folder in the root of this `phase2_graph_detection` project.
2.  **Place Model File:** Place your YOLOv5-compatible model (e.g., `Diagram-detector-best.pt`) inside the `models` directory.

#### For YOLOv11 Mode:
1.  **Place Model File:** Place your Ultralytics-compatible model (e.g., a YOLOv11 model named `Diagram-detector-yolov11.pt`) inside the `models` directory.

## Installation

1.  **Create a virtual environment (recommended):**
    ```bash
    # Using conda
    conda create -n venv python=3.8
    conda activate venv
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This module can be used as a standalone script or as a library.

### 1. As a Standalone Script

**Step 1: Configure the script**

Open the `main.py` file and edit the user configuration section at the top.

```python
# --- User Configuration ---
# 1. Choose the detection mode: 'yolov5' or 'yolov11'
DETECTION_MODE = 'yolov5'

# 2. Set the input and output directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "detection_output"

# 3. Define paths for your model files
YOLOV5_MODEL_PATH = "models/Diagram-detector-best.pt"
YOLOV5_REPO_PATH = "yolov5"
YOLOV11_MODEL_PATH = "models/Diagram-detector-yolov11.pt"
```

**Step 2: Add your image files**

Place the page images you want to process (e.g., the output from Phase 1) into the `input_images` folder. You can paste the entire output structure from the previous phase; the script will find all images recursively.

**Step 3: Run the script**

Execute the `main.py` script from your terminal:
```Bash
python main.py
```

**Example Output Structure:**

If input_images contains `page_1.JPG` and two diagrams are detected, the detection_output directory will be organized as follows:
```bash
detection_output/
└── page_1/
├── diagrams/
│ ├── diagram_1/
│ │ ├── diagram_1.jpg (Cropped image)
│ │ └── diagram_1.json (Metadata)
│ └── diagram_2/
│ ├── diagram_2.jpg (Cropped image)
│ └── diagram_2.json (Metadata)
│
└── detection_overlay.jpg (Original image with all detections drawn on it)
```
### 2. As a Library
You can import the `GraphDetector` class to integrate graph detection into your own code.

**Example:**
```python
import cv2
from graph_detector.detector import GraphDetector

# --- Option 1: Using YOLOv5 mode ---
try:
    detector_v5 = GraphDetector(
        mode='yolov5',
        model_path="models/Diagram-detector-best.pt",
        yolov5_repo_path="yolov5",
        conf_threshold=0.6
    )
    image = cv2.imread("path/to/your/page_image.png")
    detections_v5 = detector_v5.detect(image)
    print(f"Found {len(detections_v5)} diagrams using YOLOv5.")

except (FileNotFoundError, RuntimeError, ValueError) as e:
    print(f"An error occurred: {e}")


# --- Option 2: Using YOLOv11 mode ---
try:
    detector_v11 = GraphDetector(
        mode='yolov11',
        model_path="models/Diagram-detector-yolov11.pt", # Your YOLOv11 model
        conf_threshold=0.5
    )
    image = cv2.imread("path/to/your/page_image.png")
    detections_v11 = detector_v11.detect(image)
    print(f"Found {len(detections_v11)} diagrams using YOLOv11.")

except (FileNotFoundError, RuntimeError, ValueError) as e:
    print(f"An error occurred: {e}")
```