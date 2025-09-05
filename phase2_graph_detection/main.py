import os
import cv2
import json
import sys
from graph_detector.detector import GraphDetector

def main():
    """
    Main function to run graph detection on all images in a directory.
    Saves cropped detections, JSON metadata, and visual overlays.
    This script can be configured to use either YOLOv5 or YOLOv11.
    """
    # ========================================================================
    # ---                       USER CONFIGURATION                         ---
    # ========================================================================
    # 1. Choose the detection mode: 'yolov5' or 'yolov11'
    DETECTION_MODE = 'yolov11'

    # 2. Set the input and output directories
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "detection_output"
    
    # 3. Define paths for your model files
    #    Path for the YOLOv5 model and its required repository folder
    YOLOV5_MODEL_PATH = "models/Diagram-detector-best.pt"
    YOLOV5_REPO_PATH = "yolov5"

    #    Path for the YOLOv11 model (you must provide this model file)
    YOLOV11_MODEL_PATH = "models/Diagram-detector-yolov11.pt"
    
    # 4. Set the detection confidence threshold
    CONFIDENCE = 0.6
    # ========================================================================

    # --- Script Logic ---
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Instantiate the detector based on the chosen mode
    try:
        if DETECTION_MODE == 'yolov5':
            print("Initializing detector in YOLOv5 mode...")
            detector = GraphDetector(
                mode='yolov5',
                model_path=YOLOV5_MODEL_PATH, 
                yolov5_repo_path=YOLOV5_REPO_PATH, 
                conf_threshold=CONFIDENCE
            )
        elif DETECTION_MODE == 'yolov11':
            print("Initializing detector in YOLOv11 mode...")
            detector = GraphDetector(
                mode='yolov11',
                model_path=YOLOV11_MODEL_PATH,
                conf_threshold=CONFIDENCE
            )
        else:
            raise ValueError(f"Invalid DETECTION_MODE '{DETECTION_MODE}' in main.py. Please choose 'yolov5' or 'yolov11'.")

    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"FATAL: Failed to initialize detector: {e}", file=sys.stderr)
        sys.exit(1)

    # Recursively find all image files
    image_paths = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"No image files found in '{os.path.abspath(INPUT_DIR)}' or its subdirectories.", file=sys.stderr)
        print("Please add page images from Phase 1 to this folder and run again.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(image_paths)} image(s) to process.")

    for full_image_path in image_paths:
        try:
            # Preserve folder structure for output
            relative_path = os.path.relpath(full_image_path, INPUT_DIR)
            relative_base = os.path.splitext(relative_path)[0]
            page_output_dir = os.path.join(OUTPUT_DIR, relative_base)
            os.makedirs(page_output_dir, exist_ok=True)
            
            print(f"\n--- Processing: {relative_path} ---")
            
            image = cv2.imread(full_image_path)
            if image is None:
                print(f"Warning: Could not read image {full_image_path}. Skipping.")
                continue

            detections = detector.detect(image)
            print(f"Found {len(detections)} diagrams.")

            # Handle pages with no detections
            if not detections:
                no_detection_filename = f"{os.path.basename(relative_base)}_original.jpg"
                output_path = os.path.join(page_output_dir, no_detection_filename)
                cv2.imwrite(output_path, image)
                continue

            # Create 'diagrams' folder only if needed
            diagrams_base_dir = os.path.join(page_output_dir, "diagrams")
            os.makedirs(diagrams_base_dir, exist_ok=True)

            # Save each detected diagram and its JSON
            for idx, det in enumerate(detections, 1):
                single_diagram_dir = os.path.join(diagrams_base_dir, f"diagram_{idx}")
                os.makedirs(single_diagram_dir, exist_ok=True)
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                crop = image[y1:y2, x1:x2]
                
                crop_path = os.path.join(single_diagram_dir, f"diagram_{idx}.jpg")
                cv2.imwrite(crop_path, crop)
                
                json_path = os.path.join(single_diagram_dir, f"diagram_{idx}.json")
                with open(json_path, 'w') as f:
                    json.dump(det, f, indent=4)
            
            # Create and save a visual overlay
            overlay_image = image.copy()
            for idx, det in enumerate(detections, 1):
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"Diagram {idx}: {det['conf']:.2f}"
                cv2.putText(overlay_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            overlay_path = os.path.join(page_output_dir, "detection_overlay.jpg")
            cv2.imwrite(overlay_path, overlay_image)

            print(f"Results saved in: {os.path.abspath(page_output_dir)}")
            
        except Exception as e:
            image_filename_for_error = os.path.basename(full_image_path)
            print(f"An error occurred while processing {image_filename_for_error}: {e}", file=sys.stderr)
            continue
    
    print("\n--- Process Finished ---")

if __name__ == "__main__":
    main()