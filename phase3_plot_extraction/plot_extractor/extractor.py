import os
import torch
import pathlib
import platform
import logging
import numpy as np
from typing import List, Dict, Any

from ultralytics import YOLO

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

class PlotRegionExtractor:
    """
    Isolates plot regions, axes, and legends from a diagram image using two
    dedicated YOLO models, with support for both YOLOv5 and YOLOv11/Ultralytics backends.
    """

    def __init__(self, mode: str, legend_model_path: str, label_model_path: str, yolov5_repo_path: str = None):
        """
        Initializes the extractor by loading the legend and label detection models
        based on the selected mode.
        """
        self.mode = mode.lower()
        if self.mode not in ['yolov5', 'yolov11']:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'yolov5' or 'yolov11'.")

        if self.mode == 'yolov5':
            if not yolov5_repo_path or not os.path.isdir(yolov5_repo_path):
                raise FileNotFoundError(f"YOLOv5 repository not found at: {yolov5_repo_path}")
            self.legend_model = self._load_yolov5_model(legend_model_path, yolov5_repo_path, 0.5)
            self.label_model = self._load_yolov5_model(label_model_path, yolov5_repo_path, 0.5)
        elif self.mode == 'yolov11':
            self.legend_model = self._load_yolov11_model(legend_model_path)
            self.label_model = self._load_yolov11_model(label_model_path)

    def _load_yolov5_model(self, model_path, repo_path, conf_threshold):
        """Private helper to load a single YOLOv5 model."""
        logger.info(f"Loading YOLOv5 model: {os.path.basename(model_path)}")
        original_posix_path = pathlib.PosixPath
        try:
            if platform.system() == "Windows": pathlib.PosixPath = pathlib.WindowsPath
            model = torch.hub.load(repo_path, 'custom', path=model_path, source='local', verbose=False)
            model.conf = conf_threshold
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5 model from {model_path}: {e}")
        finally:
            pathlib.PosixPath = original_posix_path

    def _load_yolov11_model(self, model_path):
        """Private helper to load a single Ultralytics-compatible model."""
        logger.info(f"Loading Ultralytics model: {os.path.basename(model_path)}")
        try:
            return YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Ultralytics model from {model_path}: {e}")

    def _parse_results(self, results, img_width, img_height, model_type: str) -> List[Dict]:
        """
        Parses detection results from either backend into our standard dictionary format.
        Includes specific class ID mappings for custom YOLOv11 models.
        """
        detections = []
        if self.mode == 'yolov5':
            model_names = results.names
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                bbox = [float(v) for v in box]
                raw_class_name = model_names[int(cls)]
                # Apply original hardcoded mapping for the old legend model
                class_name = "legend_box" if str(raw_class_name) == "3" else str(raw_class_name)
                detections.append({
                    "class": class_name, "bbox": bbox, "conf": float(conf),
                    "bbox_normalized": [bbox[0]/img_width, bbox[1]/img_height, bbox[2]/img_width, bbox[3]/img_height]
                })

        elif self.mode == 'yolov11':
            # Define the explicit mappings for your custom YOLOv11 models
            YOLOV11_LABEL_MAP = {0: 'plot_area', 1: 'x_axis', 2: 'y_axis'}
            YOLOV11_LEGEND_MAP = {0: 'legend_box'}
            
            # Choose the correct map based on which model was run
            active_map = YOLOV11_LEGEND_MAP if model_type == 'legend' else YOLOV11_LABEL_MAP
            
            if results[0].boxes:
                for box in results[0].boxes:
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    class_id = int(box.cls[0])
                    
                    # Use the map to get the correct class name from the ID
                    class_name = active_map.get(class_id, f"unknown_id_{class_id}")
                    
                    detections.append({
                        "class": class_name, "bbox": bbox, "conf": float(box.conf[0]),
                        "bbox_normalized": [bbox[0]/img_width, bbox[1]/img_height, bbox[2]/img_width, bbox[3]/img_height]
                    })

        return detections

    def extract_regions(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Performs detection for legends and labels on a single diagram image
        using the configured model backend.
        """
        if image is None or image.size == 0:
            return {"legend_boxes": [], "labels": []}
        img_height, img_width = image.shape[:2]

        try:
            if self.mode == 'yolov5':
                legend_results = self.legend_model(image)
                label_results = self.label_model(image)
            elif self.mode == 'yolov11':
                legend_results = self.legend_model.predict(image, conf=0.5, verbose=False)
                label_results = self.label_model.predict(image, conf=0.5, verbose=False)
        except Exception as e:
            logger.error(f"Inference failed in {self.mode} mode: {e}")
            return {"legend_boxes": [], "labels": []}

        # Pass a hint about which model's results are being parsed
        legend_detections = self._parse_results(legend_results, img_width, img_height, model_type='legend')
        label_detections = self._parse_results(label_results, img_width, img_height, model_type='label')

        return {"legend_boxes": legend_detections, "labels": label_detections}