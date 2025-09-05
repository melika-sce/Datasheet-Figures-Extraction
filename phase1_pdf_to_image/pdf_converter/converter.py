import fitz  # PyMuPDF
import numpy as np
import cv2
import os
import logging
from typing import List

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

class PDFToImageConverter:
    """
    Handles the conversion of PDF document pages into high-resolution images.
    """

    def convert(self, pdf_path: str, output_dir: str) -> List[np.ndarray]:
        """
        Converts each page of a specified PDF into an image and saves it to a directory.

        This method uses the exact conversion logic from the mother code, rendering
        pages at 300 DPI.

        Args:
            pdf_path (str): The file path to the input PDF document.
            output_dir (str): The directory where the output images will be saved.

        Returns:
            List[np.ndarray]: A list of images, where each image is an OpenCV
                              (numpy.ndarray) object in BGR format.
        
        Raises:
            FileNotFoundError: If the pdf_path does not exist.
            RuntimeError: If the PDF cannot be opened or processed.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found at: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory: {output_dir}. Error: {e}")
            raise

        images = []
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Processing {doc.page_count} pages from '{os.path.basename(pdf_path)}'...")
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                
                # Use a matrix to render at 300 DPI, as in the mother code.
                matrix = fitz.Matrix(300/72, 300/72)
                pix = page.get_pixmap(matrix=matrix)
                
                # Convert pixmap bytes to an OpenCV image.
                img_data = pix.tobytes("png")
                nparr = np.frombuffer(img_data, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                images.append(img_cv)
                
                # Save the image to the output directory.
                image_filename = f"page_{page_num + 1}.png"
                output_path = os.path.join(output_dir, image_filename)
                
                if not cv2.imwrite(output_path, img_cv):
                    logger.error(f"Failed to write image to {output_path}")
                else:
                    logger.info(f"Successfully saved {output_path}")

            doc.close()
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise RuntimeError(f"Failed to convert PDF to images: {e}")
        
        logger.info("PDF to image conversion completed.")
        return images