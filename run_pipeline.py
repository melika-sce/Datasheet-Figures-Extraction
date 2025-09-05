# import os
# import sys
# import shutil
# import json
# import cv2
# import glob

# # --- Add all phase packages to the Python path ---
# # This allows the script to find and import the code from each phase folder.
# sys.path.append(os.path.abspath('phase1_pdf_to_image'))
# sys.path.append(os.path.abspath('phase2_graph_detection'))
# sys.path.append(os.path.abspath('phase3_plot_extraction'))
# sys.path.append(os.path.abspath('phase4_line_extraction'))
# sys.path.append(os.path.abspath('phase5_text_extraction'))
# sys.path.append(os.path.abspath('phase6_replotting'))

# # --- Import the main class from each phase's library ---
# from phase1_pdf_to_image.pdf_converter.converter import PDFToImageConverter
# from phase2_graph_detection.graph_detector.detector import GraphDetector
# from phase3_plot_extraction.plot_extractor.extractor import PlotRegionExtractor
# from phase4_line_extraction.line_extractor.extractor import LineExtractor
# from phase5_text_extraction.text_extractor.ocr import TextExtractorOCR
# from phase6_replotting.rebuilder.rebuilder import FigureRebuilder



# def run_full_pipeline():
#     """Main orchestrator to run all phases sequentially."""
    
#     # ========================================================================
#     # ---                       USER CONFIGURATION                         ---
#     # ========================================================================
#     # 1. Set the path to the folder containing your input PDF files.
#     #    This folder will be created if it doesn't exist.
#     MASTER_INPUT_DIR = "pipeline_input_pdfs"

#     # 2. Set the path for all pipeline outputs. 
#     #    This folder will be DELETED and RECREATED on each run.
#     MASTER_OUTPUT_DIR = "pipeline_output"
#     # ========================================================================

#     base_dir = os.getcwd()

#     try:
#         # --- SETUP: Prepare all directories for the run ---
#         os.makedirs(MASTER_INPUT_DIR, exist_ok=True)
#         if os.path.exists(MASTER_OUTPUT_DIR):
#             print(f"Cleaning up previous run's output from '{MASTER_OUTPUT_DIR}'...")
#             shutil.rmtree(MASTER_OUTPUT_DIR)
#         os.makedirs(MASTER_OUTPUT_DIR)
        
#         # Define paths for each phase's output
#         p1_out = os.path.join(MASTER_OUTPUT_DIR, '1_page_images')
#         p2_out = os.path.join(MASTER_OUTPUT_DIR, '2_diagram_detection')
#         p3_out = os.path.join(MASTER_OUTPUT_DIR, '3_component_extraction')
#         p4_out = os.path.join(MASTER_OUTPUT_DIR, '4_line_extraction')
#         p5_out = os.path.join(MASTER_OUTPUT_DIR, '5_text_extraction')
#         p6_out = os.path.join(MASTER_OUTPUT_DIR, '6_final_output')

#         # === PHASE 1: PDF TO IMAGE ===
#         print("\n" + "="*50 + "\n--- STARTING PHASE 1: PDF to Image ---\n" + "="*50)
#         p1_phase_dir = os.path.join(base_dir, 'phase1_pdf_to_image')
#         converter = PDFToImageConverter()
#         pdf_files = [f for f in os.listdir(MASTER_INPUT_DIR) if f.lower().endswith('.pdf')]
        
#         if not pdf_files:
#             raise FileNotFoundError(f"No PDFs found in '{MASTER_INPUT_DIR}'. Please add files to process.")

#         for pdf in pdf_files:
#             pdf_path = os.path.join(MASTER_INPUT_DIR, pdf)
#             pdf_output_dir = os.path.join(p1_out, os.path.splitext(pdf)[0])
#             print(f"Processing PDF: {pdf}...")
#             converter.convert(pdf_path, pdf_output_dir)
#         print("--- PHASE 1 COMPLETE ---")


#         # === PHASE 2: Graph Detection ===
#         print("\n" + "="*50 + "\n--- STARTING PHASE 2: Graph Detection ---\n" + "="*50)
#         p2_phase_dir = os.path.join(base_dir, 'phase2_graph_detection')
#         detector = GraphDetector(
#             model_path=os.path.join(p2_phase_dir, 'models', 'Diagram-detector-best.pt'),
#             yolov5_repo_path=os.path.join(p2_phase_dir, 'yolov5'),
#             conf_threshold=0.6
#         )
#         for root, _, files in os.walk(p1_out):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg')):
#                     full_image_path = os.path.join(root, file)
#                     relative_path = os.path.relpath(full_image_path, p1_out)
#                     relative_base = os.path.splitext(relative_path)[0]
#                     page_output_dir = os.path.join(p2_out, relative_base)
#                     os.makedirs(page_output_dir, exist_ok=True)
                    
#                     print(f"Detecting graphs in: {file}")
#                     image = cv2.imread(full_image_path)
#                     detections = detector.detect(image)
                    
#                     shutil.copy2(full_image_path, os.path.join(page_output_dir, "original_page.jpg"))

#                     if not detections:
#                         continue
                    
#                     diagrams_base_dir = os.path.join(page_output_dir, "diagrams")
#                     os.makedirs(diagrams_base_dir)
#                     for idx, det in enumerate(detections, 1):
#                         single_diagram_dir = os.path.join(diagrams_base_dir, f"diagram_{idx}")
#                         os.makedirs(single_diagram_dir)
#                         x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
#                         crop = image[y1:y2, x1:x2]
#                         cv2.imwrite(os.path.join(single_diagram_dir, f"diagram_{idx}.jpg"), crop)
#                         with open(os.path.join(single_diagram_dir, f"diagram_{idx}.json"), 'w') as f:
#                             json.dump(det, f, indent=4)
#         print("--- PHASE 2 COMPLETE ---")


#         # === PHASE 3: Plot Area and Label Extraction ===
#         print("\n" + "="*50 + "\n--- STARTING PHASE 3: Component Extraction ---\n" + "="*50)
#         shutil.copytree(p2_out, p3_out, dirs_exist_ok=True)
#         p3_phase_dir = os.path.join(base_dir, 'phase3_plot_extraction')
#         extractor3 = PlotRegionExtractor(
#             legend_model_path=os.path.join(p3_phase_dir, 'models', 'legend-detector-best.pt'),
#             label_model_path=os.path.join(p3_phase_dir, 'models', 'labelChart-detector-best.pt'),
#             yolov5_repo_path=os.path.join(p3_phase_dir, 'yolov5')
#         )
#         for diagram_path in glob.glob(os.path.join(p3_out, '**', 'diagram_*.jpg'), recursive=True):
#             print(f"Extracting components from: {os.path.relpath(diagram_path, p3_out)}")
#             diag_dir = os.path.dirname(diagram_path)
#             json_path = os.path.join(diag_dir, f"{os.path.splitext(os.path.basename(diagram_path))[0]}.json")
#             image = cv2.imread(diagram_path)
#             results = extractor3.extract_regions(image)
#             with open(json_path, 'r+') as f:
#                 data = json.load(f)
#                 data.update(results)
#                 f.seek(0)
#                 json.dump(data, f, indent=4)
#                 f.truncate()
#         print("--- PHASE 3 COMPLETE ---")
        

#         # === PHASE 4: Line Extraction ===
#         print("\n" + "="*50 + "\n--- STARTING PHASE 4: Line Extraction ---\n" + "="*50)
#         shutil.copytree(p3_out, p4_out, dirs_exist_ok=True)
#         p4_phase_dir = os.path.join(base_dir, 'phase4_line_extraction')
#         extractor4 = LineExtractor(
#             config_path=os.path.join(p4_phase_dir, 'models', 'lineformer_config.py'),
#             model_ckpt_path=os.path.join(p4_phase_dir, 'models', 'best_segm_mAP_iter_3679.pth')
#         )
#         for diagram_path in glob.glob(os.path.join(p4_out, '**', 'diagram_*.jpg'), recursive=True):
#             print(f"Extracting lines from: {os.path.relpath(diagram_path, p4_out)}")
#             json_path = os.path.join(os.path.dirname(diagram_path), f"{os.path.splitext(os.path.basename(diagram_path))[0]}.json")
#             with open(json_path, 'r') as f:
#                 data = json.load(f)
#             plot_area = next((l for l in data.get("labels", []) if l["class"] == "plot_area"), None)
#             if plot_area:
#                 image = cv2.imread(diagram_path)
#                 x1, y1, x2, y2 = [int(v) for v in plot_area['bbox']]
#                 plot_crop = image[y1:y2, x1:x2]
#                 lines = extractor4.extract_lines(plot_crop)
#                 data['lines'] = lines
#                 with open(json_path, 'w') as f:
#                     json.dump(data, f, indent=4)
#         print("--- PHASE 4 COMPLETE ---")


#         # === PHASE 5: Text Extraction ===
#         print("\n" + "="*50 + "\n--- STARTING PHASE 5: Text Extraction ---\n" + "="*50)
#         shutil.copytree(p4_out, p5_out, dirs_exist_ok=True)
#         p5_phase_dir = os.path.join(base_dir, 'phase5_text_extraction')
#         extractor5 = TextExtractorOCR(credentials_path=os.path.join(p5_phase_dir, 'google_credentials.json'))
#         for diagram_path in glob.glob(os.path.join(p5_out, '**', 'diagram_*.jpg'), recursive=True):
#             print(f"Extracting text from: {os.path.relpath(diagram_path, p5_out)}")
#             json_path = os.path.join(os.path.dirname(diagram_path), f"{os.path.splitext(os.path.basename(diagram_path))[0]}.json")
#             with open(json_path, 'r') as f:
#                 data = json.load(f)
#             image = cv2.imread(diagram_path)
#             regions = data.get("labels", []) + data.get("legend_boxes", [])
#             ocr_results = extractor5.extract_text(image)
#             data['ocr_results'] = extractor5.associate_text_to_regions(ocr_results, regions)
#             with open(json_path, 'w') as f:
#                 json.dump(data, f, indent=4)
#         print("--- PHASE 5 COMPLETE ---")


#         # === PHASE 6: Replotting ===
#         print("\n" + "="*50 + "\n--- STARTING PHASE 6: Replotting ---\n" + "="*50)
#         shutil.copytree(p5_out, p6_out, dirs_exist_ok=True)
#         rebuilder6 = FigureRebuilder()
#         for json_path in glob.glob(os.path.join(p6_out, '**', 'diagram_*.json'), recursive=True):
#             print(f"Replotting from: {os.path.relpath(json_path, p6_out)}")
#             output_dir = os.path.dirname(json_path)
#             image_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(json_path))[0]}.jpg")
#             with open(json_path, 'r') as f:
#                 data = json.load(f)
#             rebuilder6.rebuild(
#                 unified_json_data=data,
#                 diagram_image_path=image_path,
#                 output_dir=output_dir
#             )
#         print("--- PHASE 6 COMPLETE ---")
        
#         print("\n\nPIPELINE EXECUTION FINISHED SUCCESSFULLY!")
#         print(f"Final results are located in: {os.path.abspath(MASTER_OUTPUT_DIR)}")

#     except Exception as e:
#         print(f"\n\nFATAL ERROR: An error occurred during the pipeline execution.")
#         print(str(e))
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     run_full_pipeline()


import os
import sys
import shutil
import json
import cv2
import glob

# --- Add all phase packages to the Python path ---
# (This part is unchanged)
sys.path.append(os.path.abspath('phase1_pdf_to_image'))
sys.path.append(os.path.abspath('phase2_graph_detection'))
sys.path.append(os.path.abspath('phase3_plot_extraction'))
sys.path.append(os.path.abspath('phase4_line_extraction'))
sys.path.append(os.path.abspath('phase5_text_extraction'))
sys.path.append(os.path.abspath('phase6_replotting'))

# --- Import the main class from each phase's library ---
# (Corrected imports to be more standard)
from pdf_converter.converter import PDFToImageConverter
from graph_detector.detector import GraphDetector
from plot_extractor.extractor import PlotRegionExtractor
from line_extractor.extractor import LineExtractor
from text_extractor.ocr import TextExtractorOCR
from rebuilder.rebuilder import FigureRebuilder


def run_full_pipeline():
    """Main orchestrator to run all phases sequentially."""
    
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

    base_dir = os.getcwd()

    try:
        # --- SETUP: Prepare all directories for the run ---
        os.makedirs(MASTER_INPUT_DIR, exist_ok=True)
        if os.path.exists(MASTER_OUTPUT_DIR):
            print(f"Cleaning up previous run's output from '{MASTER_OUTPUT_DIR}'...")
            shutil.rmtree(MASTER_OUTPUT_DIR)
        os.makedirs(MASTER_OUTPUT_DIR)
        
        # Define paths for each phase's output
        p1_out = os.path.join(MASTER_OUTPUT_DIR, '1_page_images')
        p2_out = os.path.join(MASTER_OUTPUT_DIR, '2_diagram_detection')
        p3_out = os.path.join(MASTER_OUTPUT_DIR, '3_component_extraction')
        p4_out = os.path.join(MASTER_OUTPUT_DIR, '4_line_extraction')
        p5_out = os.path.join(MASTER_OUTPUT_DIR, '5_text_extraction')
        p6_out = os.path.join(MASTER_OUTPUT_DIR, '6_final_output')

        # === PHASE 1: PDF TO IMAGE ===
        print("\n" + "="*50 + "\n--- STARTING PHASE 1: PDF to Image ---\n" + "="*50)
        converter = PDFToImageConverter()
        pdf_files = [f for f in os.listdir(MASTER_INPUT_DIR) if f.lower().endswith('.pdf')]
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in '{MASTER_INPUT_DIR}'. Please add files to process.")
        for pdf in pdf_files:
            pdf_path = os.path.join(MASTER_INPUT_DIR, pdf)
            pdf_output_dir = os.path.join(p1_out, os.path.splitext(pdf)[0])
            print(f"Processing PDF: {pdf}...")
            converter.convert(pdf_path, pdf_output_dir)
        print("--- PHASE 1 COMPLETE ---")


        # === PHASE 2: Graph Detection ===
        print("\n" + "="*50 + f"\n--- STARTING PHASE 2: Graph Detection (Mode: {GRAPH_DETECTION_MODE}) ---\n" + "="*50)
        p2_phase_dir = os.path.join(base_dir, 'phase2_graph_detection')
        detector = GraphDetector(
            mode=GRAPH_DETECTION_MODE,
            model_path=os.path.join(p2_phase_dir, 'models', 'Diagram-detector-best.pt') if GRAPH_DETECTION_MODE == 'yolov5' else os.path.join(p2_phase_dir, 'models', 'Diagram-detector-yolov11.pt'),
            yolov5_repo_path=os.path.join(p2_phase_dir, 'yolov5')
        )
        for root, _, files in os.walk(p1_out):
            for file in files:
                if file.lower().endswith(('.png', '.jpg')):
                    # (This processing logic is unchanged from your provided script)
                    full_image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_image_path, p1_out)
                    relative_base = os.path.splitext(relative_path)[0]
                    page_output_dir = os.path.join(p2_out, relative_base)
                    os.makedirs(page_output_dir, exist_ok=True)
                    print(f"Detecting graphs in: {file}")
                    image = cv2.imread(full_image_path)
                    detections = detector.detect(image)
                    shutil.copy2(full_image_path, os.path.join(page_output_dir, "original_page.jpg"))
                    if not detections: continue
                    diagrams_base_dir = os.path.join(page_output_dir, "diagrams")
                    os.makedirs(diagrams_base_dir)
                    for idx, det in enumerate(detections, 1):
                        single_diagram_dir = os.path.join(diagrams_base_dir, f"diagram_{idx}")
                        os.makedirs(single_diagram_dir)
                        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                        crop = image[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(single_diagram_dir, f"diagram_{idx}.jpg"), crop)
                        with open(os.path.join(single_diagram_dir, f"diagram_{idx}.json"), 'w') as f:
                            json.dump(det, f, indent=4)
        print("--- PHASE 2 COMPLETE ---")


        # === PHASE 3: Plot Area and Label Extraction ===
        print("\n" + "="*50 + f"\n--- STARTING PHASE 3: Component Extraction (Mode: {COMPONENT_DETECTION_MODE}) ---\n" + "="*50)
        shutil.copytree(p2_out, p3_out, dirs_exist_ok=True)
        p3_phase_dir = os.path.join(base_dir, 'phase3_plot_extraction')
        extractor3 = PlotRegionExtractor(
            mode=COMPONENT_DETECTION_MODE,
            legend_model_path=os.path.join(p3_phase_dir, 'models', 'legend-detector-best.pt') if COMPONENT_DETECTION_MODE == 'yolov5' else os.path.join(p3_phase_dir, 'models', 'legend-detector-yolov11.pt'),
            label_model_path=os.path.join(p3_phase_dir, 'models', 'labelChart-detector-best.pt') if COMPONENT_DETECTION_MODE == 'yolov5' else os.path.join(p3_phase_dir, 'models', 'labelChart-detector-yolov11.pt'),
            yolov5_repo_path=os.path.join(p3_phase_dir, 'yolov5')
        )
        for diagram_path in glob.glob(os.path.join(p3_out, '**', 'diagram_*.jpg'), recursive=True):
            # (This processing logic is unchanged from your provided script)
            print(f"Extracting components from: {os.path.relpath(diagram_path, p3_out)}")
            diag_dir = os.path.dirname(diagram_path)
            json_path = os.path.join(diag_dir, f"{os.path.splitext(os.path.basename(diagram_path))[0]}.json")
            image = cv2.imread(diagram_path)
            results = extractor3.extract_regions(image)
            with open(json_path, 'r+') as f:
                data = json.load(f)
                data.update(results)
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
        print("--- PHASE 3 COMPLETE ---")
        

        # === PHASE 4: Line Extraction ===
        print("\n" + "="*50 + "\n--- STARTING PHASE 4: Line Extraction ---\n" + "="*50)
        shutil.copytree(p3_out, p4_out, dirs_exist_ok=True)
        p4_phase_dir = os.path.join(base_dir, 'phase4_line_extraction')
        extractor4 = LineExtractor(
            config_path=os.path.join(p4_phase_dir, 'models', 'lineformer_config.py'),
            model_ckpt_path=os.path.join(p4_phase_dir, 'models', 'best_segm_mAP_iter_3679.pth')
        )
        for diagram_path in glob.glob(os.path.join(p4_out, '**', 'diagram_*.jpg'), recursive=True):
            # (This processing logic is unchanged from your provided script)
            print(f"Extracting lines from: {os.path.relpath(diagram_path, p4_out)}")
            json_path = os.path.join(os.path.dirname(diagram_path), f"{os.path.splitext(os.path.basename(diagram_path))[0]}.json")
            with open(json_path, 'r') as f:
                data = json.load(f)
            plot_area = next((l for l in data.get("labels", []) if l["class"] == "plot_area"), None)
            if plot_area:
                image = cv2.imread(diagram_path)
                x1, y1, x2, y2 = [int(v) for v in plot_area['bbox']]
                plot_crop = image[y1:y2, x1:x2]
                lines = extractor4.extract_lines(plot_crop)
                data['lines'] = lines
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=4)
        print("--- PHASE 4 COMPLETE ---")


        # === PHASE 5: Text Extraction ===
        print("\n" + "="*50 + "\n--- STARTING PHASE 5: Text Extraction ---\n" + "="*50)
        shutil.copytree(p4_out, p5_out, dirs_exist_ok=True)
        p5_phase_dir = os.path.join(base_dir, 'phase5_text_extraction')
        extractor5 = TextExtractorOCR(credentials_path=os.path.join(p5_phase_dir, 'google_credentials.json'))
        for diagram_path in glob.glob(os.path.join(p5_out, '**', 'diagram_*.jpg'), recursive=True):
            # (This processing logic is unchanged from your provided script)
            print(f"Extracting text from: {os.path.relpath(diagram_path, p5_out)}")
            json_path = os.path.join(os.path.dirname(diagram_path), f"{os.path.splitext(os.path.basename(diagram_path))[0]}.json")
            with open(json_path, 'r') as f:
                data = json.load(f)
            image = cv2.imread(diagram_path)
            regions = data.get("labels", []) + data.get("legend_boxes", [])
            ocr_results = extractor5.extract_text(image)
            data['ocr_results'] = extractor5.associate_text_to_regions(ocr_results, regions)
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        print("--- PHASE 5 COMPLETE ---")


        # === PHASE 6: Replotting ===
        print("\n" + "="*50 + "\n--- STARTING PHASE 6: Replotting ---\n" + "="*50)
        shutil.copytree(p5_out, p6_out, dirs_exist_ok=True)
        rebuilder6 = FigureRebuilder()
        for json_path in glob.glob(os.path.join(p6_out, '**', 'diagram_*.json'), recursive=True):
            # (This processing logic is unchanged from your provided script)
            print(f"Replotting from: {os.path.relpath(json_path, p6_out)}")
            output_dir = os.path.dirname(json_path)
            image_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(json_path))[0]}.jpg")
            if not os.path.exists(image_path): image_path = image_path.replace('.jpg', '.png') # Try png
            with open(json_path, 'r') as f:
                data = json.load(f)
            rebuilder6.rebuild(
                unified_json_data=data,
                diagram_image_path=image_path,
                output_dir=output_dir
            )
        print("--- PHASE 6 COMPLETE ---")
        
        print("\n\nPIPELINE EXECUTION FINISHED SUCCESSFULLY!")
        print(f"Final results are located in: {os.path.abspath(MASTER_OUTPUT_DIR)}")

    except Exception as e:
        print(f"\n\nFATAL ERROR: An error occurred during the pipeline execution.")
        print(str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_full_pipeline()