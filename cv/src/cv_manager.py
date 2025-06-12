# ── START OF FILE cv_manager.py ──
import os
import io
import logging
import warnings
import time
import torch
from ultralytics import YOLO
from PIL import Image, ImageFile
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any

# --- Configure logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

warnings.filterwarnings("ignore", category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CVManager:
    TARGET_CLASSES = {
        0: "cargo aircraft", 1: "commercial aircraft", 2: "drone", 3: "fighter jet",
        4: "fighter plane", 5: "helicopter", 6: "light aircraft", 7: "missile",
        8: "truck", 9: "car", 10: "tank", 11: "bus", 12: "van",
        13: "cargo ship", 14: "yacht", 15: "cruise ship", 16: "warship", 17: "sailboat",
    }

    def __init__(
        self,
        model_path: str = "/models/best_yolo.engine", # Default to loading the optimized engine
        device: int = 0,
        confidence_threshold: float = 0.45, # Final confidence threshold
        iou_threshold: float = 0.60,        # NMS threshold for predictions
        image_size: Union[int, Tuple[int, int]] = 1280,
        warmup_on_init: bool = True,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        
        self._setup_device(device)
        self.model = self._load_model()
        
        if warmup_on_init and self.model:
            self.warm_up_model()
        logger.info(f"CVManager (Single Model Mode) initialization complete. Model: {model_path}")

    def _setup_device(self, device_idx_req: int):
        if device_idx_req == -1 or not torch.cuda.is_available():
            self.device_str = "cpu"; self.device_idx_for_torch = -1 
            logger.info("Using CPU.")
        else:
            ngpu = torch.cuda.device_count()
            self.device_idx_for_torch = min(device_idx_req, ngpu - 1) if ngpu > 0 else 0
            self.device_str = f"cuda:{self.device_idx_for_torch}"
            name = torch.cuda.get_device_name(self.device_idx_for_torch)
            mem = torch.cuda.get_device_properties(self.device_idx_for_torch).total_memory / (1024**3)
            logger.info(f"Using GPU {self.device_idx_for_torch} ({name}, {mem:.1f} GB).")

    def _load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}. Model will not be loaded."); return None
        try:
            logger.info(f"Loading model (PT or Engine) from {self.model_path}...")
            # The YOLO class is the universal loader for all Ultralytics models, including engines
            model = YOLO(self.model_path)
            
            # This is the key logic for handling .pt vs .engine files:
            # We only call .to(device) for PyTorch models (.pt).
            # For .engine files, the device is specified at prediction time.
            if self.model_path.endswith('.pt'):
                logger.info(f"Moving .pt model to device '{self.device_str}'...")
                model.to(torch.device(self.device_str))
            else:
                logger.info(f"Loaded an exported model format ({self.model_path.split('.')[-1]}). Device will be specified during prediction.")

            logger.info(f"Model loaded successfully.")
            return model
        except Exception as e:
            logger.exception(f"CRITICAL - Failed to load model: {e}"); return None

    def _preprocess_image_source(self, image_source: Union[bytes, str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
        """Preprocesses image source to a PIL Image (RGB). Returns None on failure."""
        try:
            if isinstance(image_source, bytes): return Image.open(io.BytesIO(image_source)).convert('RGB')
            if isinstance(image_source, str): return Image.open(image_source).convert('RGB')
            if isinstance(image_source, np.ndarray): return Image.fromarray(image_source).convert('RGB')
            if isinstance(image_source, Image.Image): return image_source.convert('RGB') if image_source.mode != 'RGB' else image_source
            return None
        except Exception: return None

    def _format_predictions(self, raw_result_obj) -> List[Dict[str, Any]]:
        """Formats a single Ultralytics Results object into a consistent list of detection dicts."""
        detections = []
        if raw_result_obj and raw_result_obj.boxes is not None:
            for box_data in raw_result_obj.boxes:
                try:
                    xyxy_abs = box_data.xyxy[0].cpu().numpy()
                    x_min, y_min, x_max, y_max = xyxy_abs
                    detections.append({
                        "bbox": [
                            round(float(x_min), 2), 
                            round(float(y_min), 2), 
                            round(float(x_max - x_min), 2), 
                            round(float(y_max - y_min), 2)
                        ],
                        "category_id": int(box_data.cls.item()),
                        "confidence": round(float(box_data.conf.item()), 4)
                    })
                except: continue
        return detections

    def cv(self, image_source: Union[bytes, str, np.ndarray, Image.Image], **kwargs) -> List[Dict[str, Any]]:
        # Single-image method now calls the efficient batch method
        return self.batch_cv([image_source], **kwargs)[0]

    def batch_cv(self, image_sources: List[Union[bytes, str, np.ndarray, Image.Image]], **kwargs) -> List[List[Dict[str, Any]]]:
        start_time = time.monotonic()
        if not image_sources or not self.model: return [[] for _ in image_sources]

        # Use passed-in thresholds or fall back to instance defaults
        conf = kwargs.get("confidence_threshold", self.confidence_threshold)
        iou = kwargs.get("iou_threshold", self.iou_threshold)
        imgsz = kwargs.get("image_size", self.image_size)

        # 1. Preprocess all images and filter out invalid ones
        original_pil_images = [self._preprocess_image_source(src) for src in image_sources]
        valid_indices_map = {i: img for i, img in enumerate(original_pil_images) if img is not None}
        
        if not valid_indices_map:
            logger.warning("No valid images in batch to process."); return [[] for _ in image_sources]

        # 2. Run a single inference call on the batch of valid images
        logger.debug(f"Running prediction on a batch of {len(valid_indices_map)} images...")
        raw_results_list = self.model.predict(
            source=list(valid_indices_map.values()), 
            conf=conf,
            iou=iou,
            imgsz=imgsz, 
            device=self.device_str, 
            verbose=False,
            stream=False # Ensures a list is returned for a batch
        )
        
        # 3. Map results back to the original input order
        final_batch_results = [[] for _ in image_sources]
        raw_results_map = {idx: res for idx, res in zip(valid_indices_map.keys(), raw_results_list)}

        for i in range(len(original_pil_images)):
            if i in raw_results_map:
                final_batch_results[i] = self._format_predictions(raw_results_map[i])
        
        processing_time = time.monotonic() - start_time
        total_objects = sum(len(dets) for dets in final_batch_results)
        logger.info(f"Processed batch of {len(image_sources)} images in {processing_time:.3f}s. Found a total of {total_objects} objects.")
        return final_batch_results
        
    def warm_up_model(self):
        if not self.model: logger.warning("No model to warm up."); return
        logger.info(f"Warming up model on device '{self.device_str}'...")
        try:
            h, w = (self.image_size, self.image_size) if isinstance(self.image_size, int) else self.image_size
            dummy_image = np.zeros((int(h), int(w), 3), dtype=np.uint8)
            # Warm up with a single image, as even dynamic engines might initialize with batch=1
            _ = self.model.predict(dummy_image, conf=0.1, verbose=False, device=self.device_str)
            logger.info("Model warmed up successfully.")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}", exc_info=True)
# ── END OF FILE cv_manager.py ──
