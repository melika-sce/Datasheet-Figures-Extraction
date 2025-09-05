import cv2
import numpy as np
import logging

# Import the provided LineFormer scripts as local package modules
from . import infer
from . import line_utils

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

class LineExtractor:
    """
    Extracts curve lines from a plot area image and converts them into
    digital coordinates using the LineFormer model.
    """

    def __init__(self, config_path: str, model_ckpt_path: str, device: str = "cpu"):
        """
        Initializes the LineExtractor by loading the LineFormer model into memory.

        Args:
            config_path (str): Path to the LineFormer model config file.
            model_ckpt_path (str): Path to the LineFormer model checkpoint (.pth) file.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        try:
            # This is the exact model loading call from the mother code
            infer.load_model(config_path, model_ckpt_path, device)
            logger.info("LineFormer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LineFormer model: {e}")
            raise RuntimeError(f"Failed to load LineFormer model: {e}")

    def extract_lines(self, plot_image: np.ndarray) -> list:
        """
        Performs line extraction on a single plot area image.
        This method contains the exact extraction logic from the mother code.

        Args:
            plot_image (np.ndarray): The input image of the plot area in BGR format.

        Returns:
            list: A list of lines, where each line is a list of [x, y] coordinates.
        """
        if plot_image is None or plot_image.size == 0:
            logger.warning("Invalid or empty image provided to LineFormer.")
            return []

        # 1. Image pre-processing (dilation), as in mother code
        kernel = np.ones((2, 2), np.uint8)
        img_dilation = cv2.dilate(plot_image, kernel, iterations=1)
        
        # 2. Get data series from the model
        line_dataseries = infer.get_dataseries(img_dilation, to_clean=False)
        
        # 3. Convert points to a clean array format
        lines = line_utils.points_to_array(line_dataseries)
        
        return lines