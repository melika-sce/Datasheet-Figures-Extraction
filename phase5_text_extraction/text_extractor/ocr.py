import os
import cv2
import logging
import numpy as np
from google.cloud import vision
from google.cloud.vision_v1 import types
from typing import List, Dict, Any

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

class TextExtractorOCR:
    """
    Recognizes and extracts text from diagram components using the Google Vision API.
    """

    def __init__(self, credentials_path: str):
        """
        Initializes the OCR client.
        
        Args:
            credentials_path (str): Path to the Google Cloud service account JSON file.
        """
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Google credentials file not found at: {credentials_path}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = vision.ImageAnnotatorClient()

    def _is_point_in_bbox(self, point: tuple, bbox: list) -> bool:
        """Checks if a 2D point is inside a bounding box. From mother code."""
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Performs OCR on an image and returns structured text data.
        This method uses the robust sequential matching logic from the mother code.
        """
        try:
            # Image pre-processing and API call
            if image is None or image.size == 0:
                logger.error("Invalid or empty image provided.")
                return []

            _, encoded = cv2.imencode('.png', image)
            gcp_image = types.Image(content=encoded.tobytes())
            response = self.client.text_detection(image=gcp_image)
            texts = response.text_annotations

            if not texts:
                return []

            # Exact line grouping logic from mother code
            ground_truth_lines = texts[0].description.split('\n')
            all_words = []
            for text in texts[1:]:
                vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
                all_words.append({
                    'text': text.description,
                    'bbox': [
                        min(v[0] for v in vertices), min(v[1] for v in vertices),
                        max(v[0] for v in vertices), max(v[1] for v in vertices)
                    ],
                    'confidence': getattr(text, 'confidence', 0.9)
                })

            results = []
            word_pointer = 0
            for line_text in ground_truth_lines:
                line_text = line_text.strip()
                if not line_text:
                    continue

                words_in_this_line = []
                target_line_no_spaces = line_text.replace(" ", "")
                reconstructed_line = ""

                while word_pointer < len(all_words) and len(reconstructed_line) < len(target_line_no_spaces):
                    current_word = all_words[word_pointer]
                    words_in_this_line.append(current_word)
                    reconstructed_line += current_word['text']
                    word_pointer += 1

                if not words_in_this_line:
                    continue

                line_bboxes = [w['bbox'] for w in words_in_this_line]
                x1, y1, x2, y2 = (min(b[0] for b in line_bboxes), min(b[1] for b in line_bboxes),
                                  max(b[2] for b in line_bboxes), max(b[3] for b in line_bboxes))

                results.append({
                    'text': line_text,
                    'bbox': [x1, y1, x2, y2],
                    'conf': sum(w['confidence'] for w in words_in_this_line) / len(words_in_this_line),
                    'words': words_in_this_line
                })
            return results

        except Exception as e:
            logger.error(f"OCR Error in extract_text: {e}")
            return []

    def associate_text_to_regions(self, ocr_results: list, component_regions: list) -> list:
        """
        Associates each OCR text block to its most likely figure component.
        This is the association logic from the mother code.
        """
        for ocr_det in ocr_results:
            x1, y1, x2, y2 = ocr_det["bbox"]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            associated_element = "none"
            
            # Check if the center of the text is inside any component's bounding box
            for region_det in component_regions:
                if self._is_point_in_bbox(center, region_det["bbox"]):
                    associated_element = region_det["class"]
                    break
            
            ocr_det["associated_element"] = associated_element
        return ocr_results