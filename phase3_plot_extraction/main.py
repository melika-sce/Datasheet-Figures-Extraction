import os
import cv2
import json
import sys
import shutil
from plot_extractor.extractor import PlotRegionExtractor

def main():
    """
    Main function to run plot area and label extraction on all diagrams
    found in the input directory, using a selectable model backend.
    """
    # ========================================================================
    # ---                       USER CONFIGURATION                         ---
    # ========================================================================
    # 1. Choose the detection mode: 'yolov5' or 'yolov11'
    DETECTION_MODE = 'yolov11'

    # 2. Set directories
    INPUT_DIR = "input_diagrams"
    OUTPUT_DIR = "extraction_output"
    
    # 3. Define model paths
    #    Paths for YOLOv5 models and the required repository folder
    YOLOV5_REPO_PATH = "yolov5"
    YOLOV5_LEGEND_MODEL = "models/legend-detector-best.pt"
    YOLOV5_LABEL_MODEL = "models/labelChart-detector-best.pt"
    
    #    Paths for YOLOv11/Ultralytics models (must be provided by user)
    YOLOV11_LEGEND_MODEL = "models/best_legend_train_yolov11n.pt" 
    YOLOV11_LABEL_MODEL = "models/best_labelchart_train_yolov11l.pt"
    # ========================================================================

    # --- Script Logic ---
    # Create a complete copy of the previous phase's output first
    print(f"Preparing output directory '{OUTPUT_DIR}'...")
    if not os.path.isdir(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' is empty or does not exist.", file=sys.stderr)
        print("Please place the output from the previous phase into this directory.", file=sys.stderr)
        sys.exit(1)
    
    # This ensures the output is a complete superset of the input
    shutil.copytree(INPUT_DIR, OUTPUT_DIR, dirs_exist_ok=True)
    print("Copied input structure to output directory.")

    # Initialize the extractor based on the chosen mode
    try:
        if DETECTION_MODE == 'yolov5':
            print("Initializing extractor in YOLOv5 mode...")
            extractor = PlotRegionExtractor(
                mode='yolov5',
                legend_model_path=YOLOV5_LEGEND_MODEL,
                label_model_path=YOLOV5_LABEL_MODEL,
                yolov5_repo_path=YOLOV5_REPO_PATH
            )
        elif DETECTION_MODE == 'yolov11':
            print("Initializing extractor in YOLOv11 mode...")
            extractor = PlotRegionExtractor(
                mode='yolov11',
                legend_model_path=YOLOV11_LEGEND_MODEL,
                label_model_path=YOLOV11_LABEL_MODEL
            )
        else:
            raise ValueError(f"Invalid DETECTION_MODE '{DETECTION_MODE}' in main.py.")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"FATAL: Failed to initialize extractor: {e}", file=sys.stderr)
        sys.exit(1)

    # Recursively find all diagram images to process within the OUTPUT directory
    diagram_paths = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.lower().startswith('diagram_') and file.lower().endswith(('.jpg', '.png')):
                diagram_paths.append(os.path.join(root, file))

    if not diagram_paths:
        print("Warning: No diagram images found to process in the output directory.")
        sys.exit(0)

    print(f"\nFound {len(diagram_paths)} diagrams to enhance.")

    for diag_path in diagram_paths:
        try:
            # All paths are now relative to the output directory
            output_diag_dir = os.path.dirname(diag_path)
            diag_filename = os.path.basename(diag_path)
            diag_basename = os.path.splitext(diag_filename)[0]
            
            json_path = os.path.join(output_diag_dir, f"{diag_basename}.json")
            if not os.path.exists(json_path):
                print(f"Warning: JSON file not found for {diag_filename}. Skipping.", file=sys.stderr)
                continue
            
            relative_path = os.path.relpath(diag_path, OUTPUT_DIR)
            print(f"--- Enhancing: {relative_path} ---")
            
            diagram_image = cv2.imread(diag_path)
            if diagram_image is None:
                print(f"Warning: Could not read image {diag_filename}. Skipping.")
                continue

            # Perform the extractions using the initialized extractor
            results = extractor.extract_regions(diagram_image)
            all_detections = results["legend_boxes"] + results["labels"]
            print(f"Found {len(all_detections)} components.")

            # Update the JSON file IN-PLACE within the output directory
            with open(json_path, 'r+') as f:
                data = json.load(f)
                data["legend_boxes"] = results["legend_boxes"]
                data["labels"] = results["labels"]
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()

            # Save cropped images of each component
            overlay_image = diagram_image.copy()
            for det in all_detections:
                cls = det["class"]
                cls_dir = os.path.join(output_diag_dir, "components", cls)
                os.makedirs(cls_dir, exist_ok=True)
                
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                crop = diagram_image[y1:y2, x1:x2]
                
                crop_filename = f"{cls}_{len(os.listdir(cls_dir)) + 1}.jpg"
                cv2.imwrite(os.path.join(cls_dir, crop_filename), crop)

                # Draw bounding box on the overlay image
                cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay_image, f"{cls}: {det['conf']:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the visual overlay
            overlay_path = os.path.join(output_diag_dir, "components_overlay.jpg")
            cv2.imwrite(overlay_path, overlay_path)

        except Exception as e:
            print(f"An error occurred while processing {diag_path}: {e}", file=sys.stderr)
            continue
            
    print("\n--- Process Finished ---")
    print(f"Complete, enhanced output is located in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()