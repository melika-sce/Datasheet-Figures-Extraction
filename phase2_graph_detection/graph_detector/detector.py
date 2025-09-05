import os
import cv2
import numpy as np
import logging
import torch
import pathlib
import platform
from typing import List, Dict, Any

from ultralytics import YOLO

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

class GraphDetector:
    """
    Detects graph or figure regions from page images using either a YOLOv5 or a modern Ultralytics model.
    """
    # ... __init__, _load_yolov5_model, _load_yolov11_model methods are UNCHANGED ...
    def __init__(self, mode: str, model_path: str, yolov5_repo_path: str = None, conf_threshold: float = 0.6):
        self.mode = mode.lower()
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_names = None
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if self.mode == 'yolov5':
            if not yolov5_repo_path or not os.path.isdir(yolov5_repo_path):
                raise FileNotFoundError(f"YOLOv5 repository not found at: {yolov5_repo_path}")
            self._load_yolov5_model(model_path, yolov5_repo_path)
        elif self.mode == 'yolov11':
            self._load_yolov11_model(model_path)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'yolov5' or 'yolov11'.")

    def _load_yolov5_model(self, model_path, repo_path):
        logger.info("Loading model in YOLOv5 mode...")
        original_posix_path = pathlib.PosixPath
        try:
            if platform.system() == "Windows": pathlib.PosixPath = pathlib.WindowsPath
            model = torch.hub.load(repo_path, 'custom', path=model_path, source='local', verbose=False)
            model.conf = self.conf_threshold
            self.model = model
            self.model_names = model.names
            logger.info(f"YOLOv5 model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5 model: {e}")
        finally:
            pathlib.PosixPath = original_posix_path

    def _load_yolov11_model(self, model_path):
        logger.info("Loading model in YOLOv11/Ultralytics mode...")
        try:
            model = YOLO(model_path)
            self.model = model
            self.model_names = model.names
            logger.info(f"Ultralytics model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Ultralytics model: {e}")

    # --- THIS IS THE NEW, DEFINITIVE SORTING FUNCTION ---
    def _sort_diagrams_by_location(self, detections: List[Dict]) -> List[Dict]:
        """
        Sorts diagrams in a natural reading order (top-to-bottom, left-to-right)
        by grouping them into horizontal rows first.
        """
        if not detections:
            return []

        # 1. Define a dynamic vertical tolerance based on the average diagram height.
        #    This allows diagrams to be in the same "row" even if their centers
        #    are not perfectly aligned.
        try:
            avg_height = sum(d['bbox'][3] - d['bbox'][1] for d in detections) / len(detections)
            Y_TOLERANCE = avg_height * 0.5  # Use 50% of the average height as the tolerance
        except (ZeroDivisionError, KeyError):
            Y_TOLERANCE = 50  # Fallback to a fixed pixel tolerance

        # Helper function to get the center point
        def get_center(det: Dict) -> tuple:
            x1, y1, x2, y2 = det['bbox']
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        # 2. Pre-sort all detections primarily by their vertical position (top-to-bottom).
        detections.sort(key=lambda d: get_center(d)[1])

        # 3. Group detections into rows based on the Y_TOLERANCE.
        rows = []
        if detections:
            current_row = [detections[0]]
            for i in range(1, len(detections)):
                prev_det_center_y = get_center(current_row[-1])[1]
                current_det_center_y = get_center(detections[i])[1]
                
                # If the current diagram is vertically close to the previous one, it's in the same row.
                if abs(current_det_center_y - prev_det_center_y) < Y_TOLERANCE:
                    current_row.append(detections[i])
                else:
                    # Otherwise, the previous row is finished. Add it and start a new one.
                    rows.append(current_row)
                    current_row = [detections[i]]
            rows.append(current_row) # Add the last row

        # 4. Sort each row by its horizontal position (left-to-right) and flatten the list.
        sorted_detections = []
        for row in rows:
            row.sort(key=lambda d: get_center(d)[0])
            sorted_detections.extend(row)
            
        return sorted_detections

    # ... detect() method is UNCHANGED from the last correct version ...
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if image is None or image.size == 0: return []
        img_height, img_width = image.shape[:2]
        temp_detections = []
        try:
            if self.mode == 'yolov5':
                results = self.model(image)
                for *box, conf, cls in results.xyxy[0].cpu().numpy():
                    bbox = [float(v) for v in box]
                    x1, y1, x2, y2 = bbox
                    temp_detections.append({
                        "class": self.model_names[int(cls)],
                        "bbox": bbox, "conf": float(conf),
                        "bbox_normalized": [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height],
                        "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                        "center_normalized": [((x1 + x2) / 2) / img_width, ((y1 + y2) / 2) / img_height]
                    })
            elif self.mode == 'yolov11':
                results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
                results_object = results[0]
                if results_object.boxes:
                    for box in results_object.boxes:
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = bbox
                        temp_detections.append({
                            "class": self.model_names[cls_id],
                            "bbox": bbox, "conf": conf,
                            "bbox_normalized": [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height],
                            "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                            "center_normalized": [((x1 + x2) / 2) / img_width, ((y1 + y2) / 2) / img_height]
                        })
        except Exception as e:
            logger.error(f"Detection inference failed in {self.mode} mode: {e}")
            return []
        return self._sort_diagrams_by_location(temp_detections)
