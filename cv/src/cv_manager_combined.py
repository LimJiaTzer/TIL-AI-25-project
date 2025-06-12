# ── START OF FILE cv_manager.py ──
import os
import io
import logging
import warnings
import time
import torch
from ultralytics import RTDETR, YOLO # For loading respective models
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any

# --- Configure logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

warnings.filterwarnings("ignore", category=UserWarning)

# --- Helper Function for IoU ---
def calculate_iou(box_a_xywh: List[float], box_b_xywh: List[float]) -> float:
    """Calculates IoU for two boxes in [x_min, y_min, width, height] format."""
    x_a_min, y_a_min, w_a, h_a = box_a_xywh
    x_a_max, y_a_max = x_a_min + w_a, y_a_min + h_a
    x_b_min, y_b_min, w_b, h_b = box_b_xywh
    x_b_max, y_b_max = x_b_min + w_b, y_b_min + h_b
    x_inter_min, y_inter_min = max(x_a_min, x_b_min), max(y_a_min, y_b_min)
    x_inter_max, y_inter_max = min(x_a_max, x_b_max), min(y_a_max, y_b_max)
    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min: return 0.0
    intersection_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    area_a, area_b = w_a * h_a, w_b * h_b
    union_area = area_a + area_b - intersection_area
    return 0.0 if union_area == 0 else intersection_area / union_area

class CVManager:
    TARGET_CLASSES = {
        0: "cargo aircraft", 1: "commercial aircraft", 2: "drone", 3: "fighter jet",
        4: "fighter plane", 5: "helicopter", 6: "light aircraft", 7: "missile",
        8: "truck", 9: "car", 10: "tank", 11: "bus", 12: "van",
        13: "cargo ship", 14: "yacht", 15: "cruise ship", 16: "warship", 17: "sailboat",
    }

    def __init__(
        self,
        rtdetr_model_path: str = "/models/best_rtdetr.pt",
        yolo_model_path: str = "/models/best_yolo.pt",
        device: int = -1,
        image_size: Union[int, Tuple[int, int]] = 640,
        warmup_on_init: bool = True,
        # Optimal parameters found from your testing
        rtdetr_conf: float = 0.67,
        rtdetr_iou: float = 0.4,
        yolo_conf: float = 0.55,
        yolo_iou: float = 0.45, # Assuming a default, adjust if you have an optimal one
        ensemble_overlap_iou_threshold: float = 0.5 # IoU to consider boxes the same object
    ):
        self.rtdetr_model_path = rtdetr_model_path
        self.yolo_model_path = yolo_model_path
        self.image_size = image_size
        
        # Store optimal thresholds for each model
        self.rtdetr_conf = rtdetr_conf
        self.rtdetr_iou = rtdetr_iou
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.ensemble_overlap_iou_threshold = ensemble_overlap_iou_threshold
        
        self._setup_device(device)
        self.rtdetr_model = self._load_model(self.rtdetr_model_path, "RTDETR")
        self.yolo_model = self._load_model(self.yolo_model_path, "YOLO")
        
        if warmup_on_init:
            self.warm_up_models()
        logger.info("CVManager (Cascade Ensemble) initialization complete.")

    def _setup_device(self, device_idx_req: int):
        if device_idx_req == -1 or not torch.cuda.is_available():
            self.device_str = "cpu"; self.device_idx_for_torch = -1 
            logger.info("CVManager: Using CPU.")
        else:
            ngpu = torch.cuda.device_count()
            self.device_idx_for_torch = min(device_idx_req, ngpu - 1) if ngpu > 0 else 0
            if ngpu == 0:
                 self.device_str = "cpu"; self.device_idx_for_torch = -1
                 logger.warning("CVManager: CUDA reported available but no GPUs found. Using CPU.")
                 return
            self.device_str = f"cuda:{self.device_idx_for_torch}"
            name = torch.cuda.get_device_name(self.device_idx_for_torch)
            mem = torch.cuda.get_device_properties(self.device_idx_for_torch).total_memory / (1024**3)
            logger.info(f"CVManager: Using GPU {self.device_idx_for_torch} ({name}, {mem:.1f} GB).")
            try:
                torch.cuda.set_device(self.device_idx_for_torch)
                torch.cuda.empty_cache()
            except RuntimeError as e:
                 logger.error(f"CVManager: Failed to set CUDA device or empty cache: {e}. Falling back to CPU.")
                 self.device_str = "cpu"; self.device_idx_for_torch = -1

    def _load_model(self, model_path: str, model_type: str):
        if not model_path or not os.path.exists(model_path):
            logger.error(f"CVManager: Model file not found for {model_type} at {model_path}. Model disabled.")
            return None
        try:
            logger.info(f"CVManager: Loading {model_type} model from {model_path} onto '{self.device_str}'...")
            model_class = RTDETR if model_type == "RTDETR" else YOLO
            model = model_class(model_path)
            model.to(torch.device(self.device_str))
            logger.info(f"CVManager: {model_type} model loaded successfully from {model_path}.")
            return model
        except Exception as e:
            logger.exception(f"CVManager: CRITICAL - Failed to load {model_type} from {model_path}: {e}")
            return None

    def _preprocess_image_source(self, image_source: Union[bytes, str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
        """Preprocesses image source to PIL Image (RGB). Returns None on failure."""
        if isinstance(image_source, bytes):
            try: return Image.open(io.BytesIO(image_source)).convert('RGB')
            except Exception: return None
        elif isinstance(image_source, str):
            try: return Image.open(image_source).convert('RGB')
            except Exception: return None
        elif isinstance(image_source, np.ndarray):
            try: return Image.fromarray(image_source).convert('RGB')
            except Exception: return None
        elif isinstance(image_source, Image.Image):
            return image_source.convert('RGB') if image_source.mode != 'RGB' else image_source
        return None

    def _format_predictions(self, raw_results) -> List[Dict[str, Any]]:
        """Formats raw Ultralytics results into a consistent list of detection dicts."""
        detections = []
        if raw_results and isinstance(raw_results, list) and raw_results[0].boxes is not None:
            for box_data in raw_results[0].boxes:
                xyxy_abs = box_data.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = xyxy_abs
                detections.append({
                    "bbox": [round(x_min, 2), round(y_min, 2), round(x_max - x_min, 2), round(y_max - y_min, 2)],
                    "category_id": int(box_data.cls.item()),
                    "confidence": round(float(box_data.conf.item()), 4)
                })
        return detections

    def cv(self, image_source: Union[bytes, str, np.ndarray, Image.Image], **kwargs) -> List[Dict[str, Any]]:
        start_time = time.monotonic()
        
        pil_image = self._preprocess_image_source(image_source)
        if pil_image is None: logger.error("CVManager.cv: Image preprocessing failed."); return []

        # 1. Get predictions from RT-DETR (primary model) using its optimal thresholds
        rtdetr_detections = []
        if self.rtdetr_model:
            rtdetr_raw_preds = self.rtdetr_model.predict(
                source=pil_image, conf=self.rtdetr_conf, iou=self.rtdetr_iou,
                imgsz=self.image_size, device=self.device_str, verbose=False)
            rtdetr_detections = self._format_predictions(rtdetr_raw_preds)

        # Initialize our final predictions with RT-DETR's results
        final_detections = rtdetr_detections

        # 2. Get predictions from YOLO (secondary model) to find missed objects
        yolo_detections = []
        if self.yolo_model:
            yolo_raw_preds = self.yolo_model.predict(
                source=pil_image, conf=self.yolo_conf, iou=self.yolo_iou,
                imgsz=self.image_size, device=self.device_str, verbose=False)
            yolo_detections = self._format_predictions(yolo_raw_preds)

        # 3. Add YOLO detections only if they are "new"
        yolo_added_count = 0
        for yolo_det in yolo_detections:
            is_new_object = True
            # Check if this YOLO detection overlaps significantly with any existing detection from RT-DETR
            for rtdetr_det in final_detections:
                if calculate_iou(yolo_det["bbox"], rtdetr_det["bbox"]) > self.ensemble_overlap_iou_threshold:
                    is_new_object = False
                    break # It's a duplicate of an RT-DETR detection, so we ignore it
            
            if is_new_object:
                # YOLO found something RT-DETR missed. Add it to the final list.
                final_detections.append(yolo_det)
                yolo_added_count += 1
        
        if yolo_added_count > 0:
            logger.info(f"Ensemble: Added {yolo_added_count} new detections from YOLO model.")

        processing_time = time.monotonic() - start_time
        logger.info(f"CVManager.cv (Cascade Ensemble): Processed in {processing_time:.3f}s. Final detections: {len(final_detections)}.")
        return final_detections

    def batch_cv(self, image_sources: List[Union[bytes, str, np.ndarray, Image.Image]], **kwargs) -> List[List[Dict[str, Any]]]:
        # This simplified batch_cv calls the new cv logic for each image.
        # For higher throughput, you could batch predict for both models first, then loop through the results to apply the ensemble logic.
        start_time = time.monotonic()
        all_batch_results = [self.cv(img_src, **kwargs) for img_src in image_sources]
        processing_time = time.monotonic() - start_time
        logger.info(f"CVManager.batch_cv (Cascade Ensemble): Processed batch of {len(image_sources)} images in {processing_time:.3f}s.")
        return all_batch_results
        
    def warm_up_models(self):
        if (not self.rtdetr_model) and (not self.yolo_model): logger.warning("CVManager: No models for warmup."); return
        logger.info(f"CVManager: Warming up models on device '{self.device_str}'...")
        try:
            h, w = (self.image_size, self.image_size) if isinstance(self.image_size, int) else self.image_size
            dummy_image = np.zeros((int(h), int(w), 3), dtype=np.uint8)
            if self.rtdetr_model: 
                logger.info("Warming up RT-DETR..."); self.rtdetr_model.predict(dummy_image, conf=0.1, verbose=False)
            if self.yolo_model: 
                logger.info("Warming up YOLO..."); self.yolo_model.predict(dummy_image, conf=0.1, verbose=False)
            logger.info("CVManager: Models warmed up successfully.")
        except Exception as e:
            logger.warning(f"CVManager: Model warm-up failed: {e}", exc_info=True)

# ── END OF FILE cv_manager.py ──
