import os
import cv2
import json
import sys
from graph_detector.detector import GraphDetector
from title_extractor import TitleExtractor

def main():
    """
    Main function to run graph detection. It recursively finds all images,
    preserves the input folder structure for the output, and handles
    pages with no detections gracefully.
    """
    # --- User Configuration ---
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "detection_output"
    MODEL_PATH = "models/Diagram-detector-best.pt"
    YOLOV5_REPO_PATH = "yolov5"
    CONFIDENCE = 0.6
    # ------------------------

    # --- Script Logic ---
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Model and repo checks (unchanged)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(YOLOV5_REPO_PATH):
        print(f"Error: YOLOv5 repository not found at '{YOLOV5_REPO_PATH}'.", file=sys.stderr)
        sys.exit(1)

    # Recursive image search (unchanged)
    image_paths = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"No image files found in '{os.path.abspath(INPUT_DIR)}'.", file=sys.stderr)
        sys.exit(0)

    try:
        detector = GraphDetector(
            model_path=MODEL_PATH, 
            yolov5_repo_path=YOLOV5_REPO_PATH, 
            conf_threshold=CONFIDENCE
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Failed to initialize detector: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) to process.")
    
    for full_image_path in image_paths:
        try:
            # --- MODIFICATION: Preserve folder structure for output ---
            # Get the relative path from the input directory
            relative_path = os.path.relpath(full_image_path, INPUT_DIR)
            # Remove the extension to create a base for the output folder
            relative_base = os.path.splitext(relative_path)[0]
            # Create the final output path, mirroring the input structure
            page_output_dir = os.path.join(OUTPUT_DIR, relative_base)
            os.makedirs(page_output_dir, exist_ok=True)
            # ---------------------------------------------------------
            
            print(f"\n--- Processing: {relative_path} ---")
            
            image = cv2.imread(full_image_path)
            if image is None:
                print(f"Warning: Could not read image {full_image_path}. Skipping.")
                continue

            detections = detector.detect(image)
            print(f"Found {len(detections)} diagrams.")

            # --- MODIFICATION: Handle pages with no detections ---
            if not detections:
                # Save the original page image as a record
                no_detection_filename = f"{os.path.basename(relative_base)}_original.jpg"
                output_path = os.path.join(page_output_dir, no_detection_filename)
                cv2.imwrite(output_path, image)
                print(f"No diagrams found. Saved original image to: {os.path.abspath(page_output_dir)}")
                continue # Skip to the next image
            # -----------------------------------------------------

            # --- MODIFICATION: Create 'diagrams' folder only if needed ---
            diagrams_base_dir = os.path.join(page_output_dir, "diagrams")
            os.makedirs(diagrams_base_dir, exist_ok=True)
            # -------------------------------------------------------------

            # Loop to save each detected diagram (logic unchanged)
            json_files = []
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
                json_files.append(json_path)
            try:
                title_extractor = TitleExtractor("sunny-mender-439713-t4-d556251ef75b.json")
                title_extractor.update_json_with_titles(full_image_path, json_files)
            except Exception as e:
                print(f"Title extraction failed for {relative_path}: {e}")
                
            
            # Create and save overlay (logic unchanged)
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