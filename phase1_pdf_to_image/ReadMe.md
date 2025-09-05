# Phase 1: PDF to Image Conversion Module

This package converts each page of a PDF document into a high-resolution (300 DPI) image. It can process all PDF files within a specified directory.

## Installation

1.  **Create a virtual conda environment (recommended):**
    ```bash
    conda create -n venv python==3.8
    conda activate venv
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This module can be used in two ways: as a standalone script or as an importable Python library.

### 1. As a Standalone Script

The script is designed to process an entire directory of PDF files.

**Step 1: Configure the script**

Open the `main.py` file and modify the configuration variables at the top:
```python
# --- User Configuration ---
# Please modify the INPUT_DIR and OUTPUT_DIR variables below.

# Set the path to the directory containing your PDF files.
INPUT_DIR = "/home/user/datasheets" # <-- CHANGE THIS

# Set the parent directory where the output image folders will be saved.
OUTPUT_DIR = "converted_images" # <-- CHANGE THIS (optional)
# ------------------------
```

**step 2: Run the script**
Execute the main.py script from your terminal:
```bash
python main.py
```

#### Example Directory Structure:
If your `INPUT_DIR` looks like this:
```bash
datasheets/
├── doc1.pdf
└── doc2.pdf
After running the script, your OUTPUT_DIR (converted_images) will look like this:

converted_images/
├── doc1/
│   ├── page_1.png
│   └── page_2.png
└── doc2/
    ├── page_1.png
    ├── page_2.png
    └── page_3.png
```

### 2. As a Library
The core PDFToImageConverter class is designed to work on a single PDF file, making it easy to integrate into other workflows.

Example:
```bash
from pdf_converter.converter import PDFToImageConverter

# Path to a single PDF and desired output location
pdf_file = "path/to/your/document.pdf"
output_folder = "single_pdf_output"

# Instantiate the converter
converter = PDFToImageConverter()

try:
    # The 'convert' method processes one PDF at a time
    list_of_images = converter.convert(pdf_path=pdf_file, output_dir=output_folder)
    print(f"Successfully converted {len(list_of_images)} pages for {pdf_file}.")

except (FileNotFoundError, RuntimeError) as e:
    print(f"An error occurred: {e}")
```