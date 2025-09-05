# Datasheet Figures Extraction Pipeline

This project is a complete, end-to-end pipeline for extracting figures, plots, and data from PDF datasheets. It processes PDFs, identifies diagrams, extracts graphical components and text, digitizes data lines, and finally, reconstructs the plots digitally.

The entire system is built as a series of modular, lightweight Python packages, orchestrated by a central control script.
## Key Features:
1. Fully Automated: Runs the entire 6-phase process with a single command.
2. Modular by Design: Each phase is a self-contained package that can be tested or run individually.
3. Configurable Detection Engine: Easily switch between YOLOv5 and YOLOv11 (or other Ultralytics models) for graph and component detection right from the main script.
4. Structured JSON Output: Produces a rich, detailed JSON file for each detected diagram, containing all extracted information.
5. Digital Reconstruction: The final output includes a digitally re-plotted version of the original graph using Matplotlib, based on the extracted data.

### 1. Project Structure Setup
Before running the pipeline, you must arrange your project folders and files exactly as shown below. **This is the most critical step.**
```bash
Datasheet_Extraction_Pipeline/
├── pipeline_input_pdfs/
│   └── (PLACE YOUR PDF FILES HERE)
│
├── phase1_pdf_to_image/
│   └── ... (package files)
│
├── phase2_graph_detection/
│   ├── yolov5/                  <-- REQUIRED for YOLOv5 mode
│   ├── models/
│   │   ├── Diagram-detector-best.pt        (YOLOv5 model)
│   │   └── Diagram-detector-yolov11.pt     (YOLOv11 model)
│   └── ... (package files)
│
├── phase3_plot_extraction/
│   ├── yolov5/                  <-- REQUIRED for YOLOv5 mode
│   ├── models/
│   │   ├── legend-detector-best.pt         (YOLOv5 model)
│   │   ├── labelChart-detector-best.pt     (YOLOv5 model)
│   │   ├── legend-detector-yolov11.pt      (YOLOv11 model)
│   │   └── labelChart-detector-yolov11.pt  (YOLOv11 model)
│   └── ... (package files)
│
├── phase4_line_extraction/
│   ├── models/
│   │   ├── best_seg_mAP_iter_3679.pth
│   │   └── lineformer_config.py
│   └── ... (package files)
│
├── phase5_text_extraction/
│   ├── google_credentials.json  <-- PLACE YOUR GCP KEY FILE HERE
│   └── ... (package files)
│
├── phase6_replotting/
│   └── ... (package files)
│
├── requirements.txt             (The master requirements file)
└── run_pipeline.py              (The main orchestrator script)
```
### 2. Installation
Follow these steps in your terminal from the root directory.

#### Step 2.1: Create a Virtual Environment
Using a virtual environment is strongly recommended to avoid package conflicts.

Using Conda:
```Bash
conda create -n venv python=3.8
conda activate venv
```

#### Step 2.2: Install All Dependencies
Install all required Python packages for each phase (phase 4 need extra attention)

## 3. Configuration
Before running, you can configure the pipeline's behavior by editing the user configuration section at the top of the `run_pipeline.py` script.

File: run_pipeline.py
```bash
# ========================================================================
# ---                       USER CONFIGURATION                         ---
# ========================================================================
# 1. Set the master input and output directories for the entire pipeline.
MASTER_INPUT_DIR = "pipeline_input_pdfs"
MASTER_OUTPUT_DIR = "pipeline_output"

# 2. Choose the detection model for Phase 2 (Graph Detection).
#    Options: 'yolov5' or 'yolov11'
GRAPH_DETECTION_MODE = 'yolov5'

# 3. Choose the detection model for Phase 3 (Component Extraction).
#    Options: 'yolov5' or 'yolov11'
COMPONENT_DETECTION_MODE = 'yolov5'
# ========================================================================
```
`MASTER_INPUT_DIR`: The folder where you place your PDFs.
`MASTER_OUTPUT_DIR`: The folder where all results will be saved. 

    Warning: This folder is deleted and recreated every time you run the pipeline.
`GRAPH_DETECTION_MODE`: Switches between yolov5 and yolov11 for the initial diagram finding. Ensure you have the corresponding model file in phase2_graph_detection/models/.
`COMPONENT_DETECTION_MODE`: Switches between yolov5 and yolov11 for finding the plot area, axes, and legends. Ensure you have the corresponding model files in phase3_plot_extraction/models/.

## 4. Running the Full Pipeline
Once setup and configuration are complete, running the entire process is a single command.

Place your PDF files in the `pipeline_input_pdfs` folder.
Activate your virtual environment
From the root `Datasheet-Figures-Extraction` directory, run the script:
```Bash
python run_pipeline.py
```

The script will now execute all 6 phases in sequence, printing the progress of each stage to the console.

## 5. Understanding the Output
After the pipeline finishes, all results will be in the pipeline_output directory, organized into subfolders corresponding to each phase.
```bash
pipeline_output/
├── 1_page_images/
│   └── (Images of each PDF page)
├── 2_diagram_detection/
│   └── (Cropped diagrams from each page)
├── 3_component_extraction/
│   └── (Diagrams enhanced with component data)
├── 4_line_extraction/
│   └── (Diagrams enhanced with line data)
├── 5_text_extraction/
│   └── (Diagrams enhanced with OCR data)
└── 6_final_output/
    └── (The final, complete results)
```    
The most important folder is `6_final_output`. Inside, you will find a complete, mirrored structure of your input, where each detected diagram folder contains:

`diagram_N.jpg`: The cropped diagram image.

`diagram_N.json`: The final, complete JSON file with all extracted data.

`final_combined_vis_N`.jpg: The final reconstructed plot visualization.

Various overlay images from intermediate steps.