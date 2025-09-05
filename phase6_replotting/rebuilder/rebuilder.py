import logging
import cv2
import matplotlib.pyplot as plt

# Import the logic from your original files, now packaged as modules
from . import _reconstructor_logic as reconstructor
from . import _plotting_logic as plotter

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

class FigureRebuilder:
    """
    Acts as an interface to the original plotting and reconstruction logic,
    adapting the unified JSON format from the pipeline to the format
    expected by the original scripts.
    """

    def rebuild(self, unified_json_data: dict, diagram_image_path: str, output_dir: str):
        """
        Orchestrates the figure rebuilding process using the original logic.

        Args:
            unified_json_data (dict): The single, complete JSON from the pipeline.
            diagram_image_path (str): Path to the original diagram image.
            output_dir (str): The directory where the final plot image will be saved.
        """
        # Step 1: Deconstruct the unified JSON into the two separate data structures
        # that the original scripts expect.
        
        # 'structure_json_path' equivalent
        structure_data = {
            "pdf_name": unified_json_data.get("pdf_name"),
            "page_number": unified_json_data.get("page_number"),
            "diagram_id": unified_json_data.get("diagram_id"),
            "image_width": unified_json_data.get("image_width"),
            "image_height": unified_json_data.get("image_height"),
            "diagram_bbox": unified_json_data.get("diagram_bbox"),
            "legend_boxes": unified_json_data.get("legend_boxes", []),
            "labels": unified_json_data.get("labels", []),
            "lines": unified_json_data.get("lines", [])
        }
        
        # 'ocr_json_path' equivalent
        ocr_data = {
            "ocr_results": unified_json_data.get("ocr_results", [])
        }

        # Step 2: Call the original reconstruction logic from _reconstructor_logic.py
        # I have adapted the original function slightly to accept dicts instead of file paths
        # to avoid writing and reading temporary files.
        try:
            digital_diagram_obj = reconstructor.reconstruct_digital_diagram(ocr_data, structure_data)
            if not digital_diagram_obj:
                logger.error("Failed to reconstruct digital data object. Skipping plot.")
                return
        except Exception as e:
            logger.error(f"Error during reconstruction logic: {e}")
            return

        # Step 3: Call the original plotting logic from _plotting_logic.py
        # This function is also adapted to take data directly.
        try:
            plotter.create_combined_visualization(
                image_path=diagram_image_path,
                diagram_data_for_annotator=structure_data,
                ocr_data_for_annotator=ocr_data,
                digital_diagram_obj=digital_diagram_obj,
                output_folder_path=output_dir
            )
        except Exception as e:
            logger.error(f"Error during plotting logic: {e}")
            # Ensure any open figures are closed on error
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)