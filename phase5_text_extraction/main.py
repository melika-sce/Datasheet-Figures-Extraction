import os
import cv2
import json
import sys
import shutil
from text_extractor.ocr import TextExtractorOCR

def main():
    # --- User Configuration ---
    INPUT_DIR = "input_data"
    OUTPUT_DIR = "text_extraction_output"
    CREDENTIALS_FILE = "google_credentials.json"
    # ------------------------

    # --- Setup and Checks ---
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"FATAL: Google credentials file '{CREDENTIALS_FILE}' not found.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' is empty or does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Copying structure from '{INPUT_DIR}' to '{OUTPUT_DIR}'...")
    shutil.copytree(INPUT_DIR, OUTPUT_DIR, dirs_exist_ok=True)
    print("Copy complete.")

    try:
        extractor = TextExtractorOCR(CREDENTIALS_FILE)
    except Exception as e:
        print(f"Failed to initialize OCR extractor: {e}", file=sys.stderr)
        sys.exit(1)
        
    # --- Find and Process Diagrams ---
    diagram_paths = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.lower().startswith('diagram_') and file.lower().endswith('.jpg'):
                diagram_paths.append(os.path.join(root, file))

    if not diagram_paths:
        print("Warning: No diagrams were found to process.")
        sys.exit(0)

    print(f"\nFound {len(diagram_paths)} diagrams to process for OCR.")

    for diag_path in diagram_paths:
        try:
            diag_dir = os.path.dirname(diag_path)
            diag_filename = os.path.basename(diag_path)
            diag_basename = os.path.splitext(diag_filename)[0]
            json_path = os.path.join(diag_dir, f"{diag_basename}.json")

            if not os.path.exists(json_path):
                continue
            
            print(f"--- Processing: {os.path.relpath(diag_path, OUTPUT_DIR)} ---")
            
            diagram_image = cv2.imread(diag_path)
            if diagram_image is None: continue
            
            # Get component regions from the JSON file to associate with text
            with open(json_path, 'r') as f:
                data = json.load(f)
            component_regions = data.get("legend_boxes", []) + data.get("labels", [])
            
            # Perform OCR and association
            raw_ocr_results = extractor.extract_text(diagram_image)
            associated_ocr_results = extractor.associate_text_to_regions(raw_ocr_results, component_regions)
            print(f"Extracted and associated {len(associated_ocr_results)} text blocks.")
            
            # Update the JSON file with the ocr_results
            data["ocr_results"] = associated_ocr_results
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)

            # Create a visual overlay for OCR
            overlay_image = diagram_image.copy()
            for text_det in associated_ocr_results:
                x1, y1, x2, y2 = [int(v) for v in text_det["bbox"]]
                cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{text_det['associated_element']}: {text_det['text']}"
                cv2.putText(overlay_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            overlay_path = os.path.join(diag_dir, "ocr_overlay.jpg")
            cv2.imwrite(overlay_path, overlay_image)

        except Exception as e:
            print(f"An error occurred while processing {diag_path}: {e}", file=sys.stderr)
            continue
            
    print("\n--- Process Finished ---")
    print(f"Complete, enhanced output is in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()