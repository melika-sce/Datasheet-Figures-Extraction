import os
import json
import sys
import shutil
from rebuilder.rebuilder import FigureRebuilder

def main():
    # --- User Configuration ---
    INPUT_DIR = "input_data"
    OUTPUT_DIR = "final_output"
    # ------------------------

    # --- Setup and Checks ---
    if not os.path.isdir(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' is empty or does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Copying structure from '{INPUT_DIR}' to '{OUTPUT_DIR}'...")
    shutil.copytree(INPUT_DIR, OUTPUT_DIR, dirs_exist_ok=True)
    print("Copy complete.")

    rebuilder = FigureRebuilder()
        
    # --- Find and Process Diagrams in the OUTPUT directory ---
    json_paths = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.lower().startswith('diagram_') and file.lower().endswith('.json'):
                json_paths.append(os.path.join(root, file))

    if not json_paths:
        print("Warning: No diagram JSON files were found to process.")
        sys.exit(0)

    print(f"\nFound {len(json_paths)} diagrams to replot.")

    for json_path in json_paths:
        try:
            output_dir = os.path.dirname(json_path)
            json_basename = os.path.splitext(os.path.basename(json_path))[0]
            
            # The corresponding image has the same base name
            image_path = os.path.join(output_dir, f"{json_basename}.jpg")
            if not os.path.exists(image_path):
                # Try .png if .jpg is not found
                image_path = os.path.join(output_dir, f"{json_basename}.png")
                if not os.path.exists(image_path):
                    print(f"Warning: Image for {json_path} not found. Skipping.")
                    continue

            print(f"--- Processing: {os.path.relpath(json_path, OUTPUT_DIR)} ---")
            
            with open(json_path, 'r') as f:
                unified_json_data = json.load(f)
            
            # Call the rebuilder which uses your original, adapted logic
            rebuilder.rebuild(
                unified_json_data=unified_json_data,
                diagram_image_path=image_path,
                output_dir=output_dir
            )

        except Exception as e:
            print(f"An error occurred in main loop for {json_path}: {e}", file=sys.stderr)
            continue
            
    print("\n--- Process Finished ---")
    print(f"Final output is in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()