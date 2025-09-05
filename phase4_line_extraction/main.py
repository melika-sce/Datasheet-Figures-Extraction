import os
import cv2
import json
import sys
import shutil
import numpy as np
from line_extractor.extractor import LineExtractor

def find_diagram_json(start_path: str):
    """
    Navigates up the directory tree from a starting path to find the
    corresponding diagram's JSON file (diagram_*.json).
    """
    current_path = start_path
    for _ in range(4): # Search up a few levels
        for f in os.listdir(current_path):
            if f.startswith('diagram_') and f.endswith('.json'):
                return os.path.join(current_path, f)
        current_path = os.path.dirname(current_path)
    return None

def main():
    # --- User Configuration ---
    INPUT_DIR = "input_data"
    OUTPUT_DIR = "line_extraction_output"
    MODEL_CONFIG = "models/lineformer_config.py"
    MODEL_CKPT = "models/best_segm_mAP_iter_3679.pth"
    DEVICE = "cpu"
    # ------------------------

    # --- Setup and Checks ---
    if not all(os.path.exists(p) for p in [MODEL_CONFIG, MODEL_CKPT]):
        print("Error: LineFormer model config or checkpoint is missing from 'models'.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' is empty or does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Copying structure from '{INPUT_DIR}' to '{OUTPUT_DIR}'...")
    shutil.copytree(INPUT_DIR, OUTPUT_DIR, dirs_exist_ok=True)
    print("Copy complete.")

    try:
        extractor = LineExtractor(MODEL_CONFIG, MODEL_CKPT, DEVICE)
    except RuntimeError as e:
        print(f"Initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Find and Process Plot Areas ---
    # We now search for the specific 'plot_area' images within the new output directory
    plot_area_paths = []
    for root, _, files in os.walk(OUTPUT_DIR):
        if 'plot_area' in os.path.basename(root):
            for file in files:
                if file.lower().endswith('.jpg'):
                    plot_area_paths.append(os.path.join(root, file))

    if not plot_area_paths:
        print("Warning: No 'plot_area' images found to process.")
        sys.exit(0)

    print(f"\nFound {len(plot_area_paths)} plot areas to process.")

    for plot_path in plot_area_paths:
        try:
            # Find the main JSON file associated with this plot area
            diagram_json_path = find_diagram_json(os.path.dirname(plot_path))
            if not diagram_json_path:
                print(f"Warning: Could not find parent JSON for {plot_path}. Skipping.")
                continue

            print(f"--- Processing: {os.path.relpath(plot_path, OUTPUT_DIR)} ---")
            
            plot_image = cv2.imread(plot_path)
            if plot_image is None: continue

            # Extract the line coordinate data
            lines = extractor.extract_lines(plot_image)
            print(f"Found {len(lines)} lines.")

            # Update the diagram's JSON file with the new 'lines' key
            with open(diagram_json_path, 'r+') as f:
                data = json.load(f)
                data["lines"] = lines
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()

            # Create a visual overlay of the detected lines on the plot area
            overlay_image = plot_image.copy()
            for line in lines:
                color = tuple(np.random.randint(0, 255, 3).tolist())
                points = np.array(line, dtype=np.int32)
                cv2.polylines(overlay_image, [points], isClosed=False, color=color, thickness=2)
            
            overlay_path = os.path.join(os.path.dirname(plot_path), "lines_overlay.jpg")
            cv2.imwrite(overlay_path, overlay_image)

        except Exception as e:
            print(f"An error occurred while processing {plot_path}: {e}", file=sys.stderr)
            continue
            
    print("\n--- Process Finished ---")
    print(f"Complete, enhanced output is in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()