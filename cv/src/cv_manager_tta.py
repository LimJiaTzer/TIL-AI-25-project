# ── START OF FILE cv_manager.py ──
import os
import io
import logging
import warnings
import time
import torch
from ultralytics import RTDETR, YOLO
from PIL import Image, ImageFile
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from ensemble_boxes import weighted_boxes_fusion 

# --- Configure logging ---
logger = logging.getLogger(__name__)
# EDITED: Set to DEBUG to see detailed prediction logs
logger.setLevel(logging.DEBUG) 
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
        model_path: str = "/models/best_yolo.engine", # Can be .pt or .engine
        device: int = 0,
        image_size: Union[int, Tuple[int, int]] = 1280,
        warmup_on_init: bool = True,
        # --- Thresholds for Selective TTA ---
        final_confidence_threshold: float = 0.25, # Final cutoff for any prediction to be returned
        uncertainty_zone_low: float = 0.01,  # Detections above this are considered for the check
        uncertainty_zone_high: float = 0.50, # Detections below this are considered "uncertain"
        # --- Parameters for the full TTA process ---
        tta_iou_threshold: float = 0.4,
        tta_initial_conf_threshold: float = 0.01,
        wbf_iou_threshold: float = 0.55,
        wbf_skip_box_threshold: float = 0.1
    ):
        self.model_path = model_path
        self.image_size = image_size
        self.final_confidence_threshold = final_confidence_threshold
        self.uncertainty_zone_low = uncertainty_zone_low
        self.uncertainty_zone_high = uncertainty_zone_high
        self.tta_iou_threshold = tta_iou_threshold
        self.tta_initial_conf_threshold = tta_initial_conf_threshold
        self.wbf_iou_threshold = wbf_iou_threshold
        self.wbf_skip_box_threshold = wbf_skip_box_threshold
        self._setup_device(device)
        self.model = self._load_model()
        if warmup_on_init and self.model:
            self.warm_up_model()
        logger.info("CVManager (Uncertainty-Triggered TTA) initialization complete.")

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
            logger.error(f"Model file not found at {self.model_path}."); return None
        try:
            logger.info(f"Loading model (PT or Engine) from {self.model_path} onto '{self.device_str}'...")
            model = YOLO(self.model_path)
            if self.model_path.endswith('.pt'): model.to(torch.device(self.device_str))
            logger.info(f"Model loaded successfully.")
            return model
        except Exception as e:
            logger.exception(f"CRITICAL - Failed to load model: {e}"); return None

    def _preprocess_image_source(self, image_source: Union[bytes, str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
        try:
            if isinstance(image_source, bytes): return Image.open(io.BytesIO(image_source)).convert('RGB')
            if isinstance(image_source, str): return Image.open(image_source).convert('RGB')
            if isinstance(image_source, np.ndarray): return Image.fromarray(image_source).convert('RGB')
            if isinstance(image_source, Image.Image): return image_source.convert('RGB') if image_source.mode != 'RGB' else image_source
            return None
        except Exception: return None

    def _format_predictions(self, raw_result_obj) -> List[Dict[str, Any]]:
        detections = []
        if raw_result_obj and raw_result_obj.boxes is not None:
            for box_data in raw_result_obj.boxes:
                try:
                    xyxy_abs = box_data.xyxy[0].cpu().numpy()
                    x_min, y_min, x_max, y_max = xyxy_abs
                    detections.append({
                        # EDITED: Explicitly cast all numerical values to standard Python types
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

    def _run_full_tta(self, original_pil_image: Image.Image) -> List[Dict[str, Any]]:
        """Helper to run the full, batched TTA process on a single uncertain image."""
        image_width, image_height = original_pil_image.size
        tta_transforms = [
            {"name": "original", "transform": lambda img: img},
            {"name": "hflip", "transform": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)},
            {"name": "scale_up", "transform": lambda img: img.resize((int(img.width * 1.2), int(img.height * 1.2)), Image.Resampling.BILINEAR)},
            {"name": "scale_down", "transform": lambda img: img.resize((int(img.width * 0.8), int(img.height * 0.8)), Image.Resampling.BILINEAR)}
        ]
        images_for_tta_batch = [t['transform'](original_pil_image) for t in tta_transforms]
        
        raw_results_list = self.model.predict(
            source=images_for_tta_batch, conf=self.tta_initial_conf_threshold,
            iou=self.tta_iou_threshold, imgsz=self.image_size, device=self.device_str, verbose=False)

        wbf_boxes, wbf_scores, wbf_labels = [], [], []
        if raw_results_list and len(raw_results_list) == len(tta_transforms):
            for i, result_obj in enumerate(raw_results_list):
                aug_image = images_for_tta_batch[i]
                aug_w, aug_h = aug_image.size
                
                boxes_norm_list, scores_list, labels_list = [], [], []
                if result_obj.boxes is not None:
                    for box_data in result_obj.boxes:
                        xyxy = box_data.xyxy[0].cpu().numpy()
                        boxes_norm_list.append([xyxy[0]/aug_w, xyxy[1]/aug_h, xyxy[2]/aug_w, xyxy[3]/aug_h])
                        scores_list.append(float(box_data.conf.item())); labels_list.append(int(box_data.cls.item()))

                if tta_transforms[i]['name'] == "hflip" and boxes_norm_list:
                    for j in range(len(boxes_norm_list)):
                        xmin_n, _, xmax_n, _ = boxes_norm_list[j]
                        boxes_norm_list[j][0], boxes_norm_list[j][2] = 1.0 - xmax_n, 1.0 - xmin_n
                
                if boxes_norm_list:
                    wbf_boxes.append(boxes_norm_list); wbf_scores.append(scores_list); wbf_labels.append(labels_list)

        if not wbf_boxes: return []
        
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            wbf_boxes, wbf_scores, wbf_labels, iou_thr=self.wbf_iou_threshold, skip_box_thr=self.wbf_skip_box_threshold)
        
        final_fused_predictions = []
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            # score and label from WBF are already numpy types
            x_min, y_min, x_max, y_max = box[0] * image_width, box[1] * image_height, box[2] * image_width, box[3] * image_height
            w, h = x_max - x_min, y_max - y_min
            if w > 0 and h > 0:
                final_fused_predictions.append({
                    # EDITED: Explicitly cast all numerical values to standard Python types
                    "bbox": [
                        round(float(x_min), 2), 
                        round(float(y_min), 2), 
                        round(float(w), 2), 
                        round(float(h), 2)
                    ],
                    "category_id": int(label), 
                    "confidence": round(float(score), 4)
                })
        return final_fused_predictions

    def batch_cv(self, image_sources: List[Union[bytes, str, np.ndarray, Image.Image]], **kwargs) -> List[List[Dict[str, Any]]]:
        start_time = time.monotonic()
        if not image_sources or not self.model: return [[] for _ in image_sources]

        original_pil_images = [self._preprocess_image_source(src) for src in image_sources]
        valid_indices = [i for i, img in enumerate(original_pil_images) if img is not None]
        valid_images = [original_pil_images[i] for i in valid_indices]

        if not valid_images:
            logger.warning("No valid images in batch to process."); return [[] for _ in image_sources]

        # 1. Fast Path Inference
        logger.debug(f"Running fast-path inference on {len(valid_images)} images...")
        fast_path_results_raw = self.model.predict(
            source=valid_images, 
            conf=self.uncertainty_zone_low,
            iou=self.tta_iou_threshold,
            imgsz=self.image_size, device=self.device_str, verbose=False)
        
        # 2. Analyze results
        final_batch_results = [[] for _ in image_sources]
        indices_needing_tta = []
        fast_path_results_map = {idx: res for idx, res in zip(valid_indices, fast_path_results_raw)}

        for i in range(len(original_pil_images)):
            if original_pil_images[i] is None: continue
            
            fast_path_result = fast_path_results_map.get(i)
            fast_path_detections = self._format_predictions(fast_path_result)
            
            is_uncertain = False
            if fast_path_detections:
                for det in fast_path_detections:
                    if self.uncertainty_zone_low <= det['confidence'] < self.uncertainty_zone_high:
                        is_uncertain = True; break
            
            if is_uncertain:
                logger.debug(f"Image {i}: Fast path has uncertain detections. Triggering full TTA.")
                indices_needing_tta.append(i)
            else:
                logger.debug(f"Image {i}: Fast path is certain. Keeping high-confidence detections.")
                final_batch_results[i] = [d for d in fast_path_detections if d['confidence'] >= self.final_confidence_threshold]
                # ADDED: Log the final predictions for this "certain" image
                logger.debug(f"Image {i} (Fast Path): Returning final predictions: {final_batch_results[i]}")

        # 3. Conditional TTA on uncertain images
        if indices_needing_tta:
            logger.info(f"Running full TTA on {len(indices_needing_tta)} uncertain images...")
            for i in indices_needing_tta:
                tta_results = self._run_full_tta(original_pil_images[i])
                final_batch_results[i] = [d for d in tta_results if d['confidence'] >= self.final_confidence_threshold]
                # ADDED: Log the final predictions for this "uncertain" image after TTA
                logger.debug(f"Image {i} (TTA Path): Returning final predictions: {final_batch_results[i]}")
        
        processing_time = time.monotonic() - start_time
        total_objects = sum(len(dets) for dets in final_batch_results)
        
        # ADDED: A final summary log at DEBUG level showing counts per image
        detection_counts_per_image = [len(dets) for dets in final_batch_results]
        logger.debug(f"Detection counts per image in batch: {detection_counts_per_image}")
        
        logger.info(f"Processed batch of {len(image_sources)} images in {processing_time:.3f}s. Found a total of {total_objects} objects.")
        return final_batch_results
        
    def warm_up_model(self):
        if not self.model: logger.warning("No model to warm up."); return
        logger.info(f"Warming up model on device '{self.device_str}'...")
        try:
            h, w = (self.image_size, self.image_size) if isinstance(self.image_size, int) else self.image_size
            dummy_image = np.zeros((int(h), int(w), 3), dtype=np.uint8)
            _ = self.model.predict([dummy_image, dummy_image], conf=0.1, verbose=False, device=self.device_str)
            logger.info("Model warmed up successfully.")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}", exc_info=True)
# ── END OF FILE cv_manager.py ──
