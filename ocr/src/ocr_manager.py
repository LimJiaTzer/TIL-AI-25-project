# ocr_manager.py (for DocTR)
import os
import sys
import io
import logging
import time
import torch
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from doctr.models import ocr_predictor
from doctr.io.elements import Document, Page

os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("OCRManagerDocTR")

# Try to import OpenCV
try:
    import cv2
    _cv2_available = True
except ImportError:
    _cv2_available = False
    logger.warning("OCRManagerDocTR: OpenCV (cv2) not found. PIL will be used as the default image loader. For potentially faster image loading, install opencv-python.")


class OCRManager:
    def __init__(
        self,
        detection_arch: str = "linknet_resnet18",
        recognition_arch: str = "crnn_mobilenet_v3_small",
        pretrained: bool = True,
        device_idx: int = 0, # Renamed from 'device' for clarity
        det_batch_size: int = 256,
        reco_batch_size: int = 256,
        warmup_on_init: bool = True,
        use_amp_if_available: bool = True, # Automatic Mixed Precision for GPU
        use_cv2_default: bool = True,      # Default to using OpenCV if available
    ):
        self.det_batch_size = det_batch_size
        self.reco_batch_size = reco_batch_size
        self._setup_device(device_idx)

        self.use_amp = use_amp_if_available and self.torch_device.type == 'cuda'
        self.cv2_default_enabled = use_cv2_default
        self._cv2_active = _cv2_available and self.cv2_default_enabled

        if self._cv2_active:
            logger.info("OCRManagerDocTR: OpenCV (cv2) is available and configured for use.")
        elif use_cv2_default and not _cv2_available:
            logger.warning("OCRManagerDocTR: OpenCV (cv2) was requested by default but is not installed. Falling back to PIL.")
        else:
            logger.info("OCRManagerDocTR: PIL will be used for image loading.")

        logger.info(
            f"OCRManagerDocTR: Initializing with det_arch='{detection_arch}' (bs={det_batch_size}), "
            f"reco_arch='{recognition_arch}' (bs={reco_batch_size}). AMP: {self.use_amp}"
        )

        try:
            logger.info(f"OCRManagerDocTR: Loading DocTR predictor...")
            self.predictor = ocr_predictor(
                det_arch=detection_arch,
                reco_arch=recognition_arch,
                pretrained=pretrained,
                assume_straight_pages=True,
                det_bs=self.det_batch_size,
                reco_bs=self.reco_batch_size,
                # export_as_straight_boxes=True # Consider if polygons are not needed
            )
            self.predictor.to(self.torch_device)
            logger.info(f"OCRManagerDocTR: DocTR predictor loaded and moved to {self.torch_device}.")
        except Exception as e:
            logger.exception(f"OCRManagerDocTR: CRITICAL - Failed to load DocTR predictor: {e}")
            raise

        if warmup_on_init:
            self.warm_up_model()
        logger.info("OCRManagerDocTR initialization complete.")

    def _setup_device(self, device_idx_req: int):
        if device_idx_req >= 0 and torch.cuda.is_available():
            self.torch_device = torch.device(f"cuda:{device_idx_req}")
            name = torch.cuda.get_device_name(self.torch_device)
            mem = torch.cuda.get_device_properties(self.torch_device).total_memory / (1024**3)
            logger.info(f"OCRManagerDocTR: Target GPU {device_idx_req} ({name}, {mem:.1f} GB)")
            torch.cuda.empty_cache()
        else:
            if device_idx_req >= 0 and not torch.cuda.is_available():
                logger.warning(f"OCRManagerDocTR: GPU {device_idx_req} requested but CUDA not available. Falling back to CPU.")
            self.torch_device = torch.device("cpu")
            logger.info("OCRManagerDocTR: Using CPU.")

    def _pil_to_numpy(self, pil_image: Image.Image) -> np.ndarray:
        return np.array(pil_image)

    def _load_and_preprocess_image_pil(self, image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return self._pil_to_numpy(pil_image)
        except Exception as e:
            logger.debug(f"OCRManagerDocTR: Failed to load image with PIL: {e}")
            return None

    def _load_and_preprocess_image_cv2(self, image_bytes: bytes) -> Optional[np.ndarray]:
        if not _cv2_available: # Should not happen if called appropriately
            return self._load_and_preprocess_image_pil(image_bytes)
        try:
            img_np_encoded = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(img_np_encoded, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.debug("OCRManagerDocTR: cv2.imdecode failed to decode image.")
                return None
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception as e:
            logger.debug(f"OCRManagerDocTR: Failed to load image with OpenCV: {e}")
            return None

    def _get_image_loader_fn(self, use_cv2_override: Optional[bool] = None):
        use_cv2 = self.cv2_default_enabled if use_cv2_override is None else use_cv2_override
        if use_cv2 and _cv2_available:
            return self._load_and_preprocess_image_cv2
        elif use_cv2 and not _cv2_available:
            logger.warning("OCRManagerDocTR: CV2 was requested for this operation but is not available. Falling back to PIL.")
            return self._load_and_preprocess_image_pil
        return self._load_and_preprocess_image_pil


    def _extract_text_from_page(self, page_obj: Page) -> str:
        """Extracts plain text from a single DocTR Page object."""
        page_lines = []
        if not hasattr(page_obj, 'blocks'):
            logger.warning(f"OCRManagerDocTR: Page object missing 'blocks' attribute. Type: {type(page_obj)}")
            return ""
        for block in page_obj.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                page_lines.append(line_text)
        return "\n".join(page_lines)

    def _run_predictor(self, images_np_list: List[np.ndarray]) -> Document:
        if self.use_amp:
            with torch.cuda.amp.autocast():
                return self.predictor(images_np_list)
        else:
            return self.predictor(images_np_list)

    def ocr(self, image_bytes: bytes, use_cv2: Optional[bool] = None) -> str:
        if not hasattr(self, 'predictor'):
            logger.error("OCRManagerDocTR.ocr: DocTR predictor not initialized.")
            return ""

        start_time = time.monotonic()
        text = ""
        load_fn = self._get_image_loader_fn(use_cv2)

        try:
            img_np = load_fn(image_bytes)
            if img_np is None:
                logger.error("OCRManagerDocTR.ocr: Failed to load image.")
                return ""

            with torch.inference_mode():
                result_doc = self._run_predictor([img_np])

            if isinstance(result_doc, Document) and result_doc.pages:
                text = self._extract_text_from_page(result_doc.pages[0])
            else:
                logger.warning(f"OCRManagerDocTR.ocr: Predictor did not return a Document with pages. Got: {type(result_doc)}")
        except Exception as e:
            logger.error(f"OCRManagerDocTR.ocr: OCR failed: {e}", exc_info=True)
            text = ""

        end_time = time.monotonic()
        logger.info(f"OCRManagerDocTR.ocr: Processed in {end_time - start_time:.3f}s. Result: '{text[:100].replace(chr(10), ' ')}...'")
        return text

    def batch_ocr(self, image_bytes_list: List[bytes], use_cv2: Optional[bool] = None) -> List[str]:
        if not hasattr(self, 'predictor'):
            logger.error("OCRManagerDocTR.batch_ocr: DocTR predictor not initialized.")
            return [""] * len(image_bytes_list) if image_bytes_list else []

        start_time = time.monotonic()
        original_batch_size = len(image_bytes_list)
        if original_batch_size == 0:
            return []

        load_fn = self._get_image_loader_fn(use_cv2)
        
        # Results array, initialized to empty strings for all original inputs
        all_texts = [""] * original_batch_size
        
        # Store (original_index, np_image_data) for successfully loaded images
        # This list will be used to create the batch for DocTR and map results back
        loaded_images_data = [] # Stores (original_idx, np_array)

        # Parallel image loading and preprocessing
        # max_workers can be tuned. os.cpu_count() is a common choice.
        # For I/O bound tasks, more workers than cores can be beneficial.
        with ThreadPoolExecutor(max_workers=min(max(os.cpu_count() or 1, 4), original_batch_size)) as executor:
            future_to_original_idx = {
                executor.submit(load_fn, img_bytes): i
                for i, img_bytes in enumerate(image_bytes_list)
            }

            for future in as_completed(future_to_original_idx):
                original_idx = future_to_original_idx[future]
                try:
                    img_np = future.result()
                    if img_np is not None:
                        loaded_images_data.append((original_idx, img_np))
                    else:
                        logger.warning(f"OCRManagerDocTR.batch_ocr: Failed to load image at original index {original_idx} (loader returned None).")
                except Exception as e:
                    logger.warning(f"OCRManagerDocTR.batch_ocr: Exception while loading image at original index {original_idx}: {e}")
        
        # Sort by original index to maintain some order if DocTR predictor is sensitive to it (unlikely but good practice)
        # More importantly, this makes it easier to map results back if we process in sub-batches later.
        # For now, we pass all valid images at once.
        loaded_images_data.sort(key=lambda x: x[0])
        
        images_to_process_for_doctr = [item[1] for item in loaded_images_data]
        original_indices_for_processed = [item[0] for item in loaded_images_data]

        num_valid_images = len(images_to_process_for_doctr)
        logger.info(f"OCRManagerDocTR.batch_ocr: Image loading complete. Valid items: {num_valid_images}/{original_batch_size}")

        if num_valid_images == 0:
            logger.warning("OCRManagerDocTR.batch_ocr: No valid images to process.")
            return all_texts

        try:
            logger.debug(f"OCRManagerDocTR.batch_ocr: Calling predictor with a list of {num_valid_images} images.")
            with torch.inference_mode():
                result_document: Document = self._run_predictor(images_to_process_for_doctr)
            logger.debug(f"OCRManagerDocTR.batch_ocr: Predictor output type: {type(result_document)}")

            if not isinstance(result_document, Document):
                logger.error(f"OCRManagerDocTR.batch_ocr: Predictor returned unexpected type: {type(result_document)}")
                return all_texts # Or handle more gracefully by marking these as errors

            num_pages_returned = len(result_document.pages)
            if num_pages_returned != num_valid_images:
                logger.error(
                    f"OCRManagerDocTR.batch_ocr: Mismatch! Number of pages in result Document ({num_pages_returned}) "
                    f"does not match number of input images processed ({num_valid_images})."
                )
                # Attempt to process what we can, matching page by page
                # This implies some images might not have corresponding results
            
            for i, page_object in enumerate(result_document.pages):
                if i < len(original_indices_for_processed): # Safety check for less pages returned
                    original_idx = original_indices_for_processed[i]
                    all_texts[original_idx] = self._extract_text_from_page(page_object)
                else:
                    # This case should ideally not happen if num_pages_returned <= num_valid_images
                    logger.warning(f"OCRManagerDocTR.batch_ocr: Received more pages from DocTR ({num_pages_returned}) "
                                   f"than expected valid images ({num_valid_images}). Extra page {i} skipped.")


        except Exception as e:
            logger.error(f"OCRManagerDocTR.batch_ocr: OCR failed for batch: {e}", exc_info=True)
            # Existing entries in all_texts for successfully loaded but failed-in-OCR images will remain ""

        end_time = time.monotonic()
        logger.info(f"OCRManagerDocTR.batch_ocr: Processed batch of {original_batch_size} ({num_valid_images} valid) in {end_time - start_time:.3f}s.")
        return all_texts


    def warm_up_model(self):
        if not hasattr(self, 'predictor'):
            logger.warning("OCRManagerDocTR.warm_up_model: DocTR predictor not available, skipping warmup.")
            return

        logger.info(f"OCRManagerDocTR: Warming up DocTR predictor on device: {self.torch_device}...")
        try:
            # Using a slightly larger, more realistic image size for warmup might be better
            # e.g., (256, 256, 3) or even (512, 512, 3) if memory allows and typical images are larger.
            dummy_image_np = np.zeros((128, 128, 3), dtype=np.uint8)
            with torch.inference_mode():
                _ = self._run_predictor([dummy_image_np])
            logger.info("OCRManagerDocTR: DocTR predictor warmed up successfully.")
            if self.torch_device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"OCRManagerDocTR: Model warm-up failed: {e}", exc_info=True)

    def get_gpu_memory_stats(self) -> Dict[str, Union[str, float]]:
        if self.torch_device.type != 'cuda' or not torch.cuda.is_available():
            return {"status": "No GPU in use or CUDA unavailable"}
        try:
            # Ensure device_idx_to_query is an int
            if self.torch_device.index is not None:
                device_idx_to_query = self.torch_device.index
            else: # This case should not happen if self.torch_device is a CUDA device
                device_idx_to_query = torch.cuda.current_device()
            
            props = torch.cuda.get_device_properties(device_idx_to_query)
            total_gb = props.total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated(device_idx_to_query) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(device_idx_to_query) / (1024**3) # AKA cached
            free_inside_reserved_gb = reserved_gb - allocated_gb
            # More accurate free memory (total - reserved)
            # actual_free_gb = total_gb - reserved_gb
            # Or use nvidia-smi for system-wide free, but that's external.
            # Let's report what PyTorch manages.
            
            util_percent = (100 * allocated_gb / total_gb) if total_gb > 0 else 0.0 # Based on allocated
            
            return {
                "status": "OK",
                "device_name": props.name,
                "total_gb": round(total_gb, 2),
                "allocated_gb": round(allocated_gb, 2), # Memory directly used by tensors
                "reserved_gb": round(reserved_gb, 2),  # Total memory managed by PyTorch's caching allocator
                "free_in_reserved_gb": round(free_inside_reserved_gb, 2), # Cached but not currently used by tensors
                "util_pytorch_allocated_percent": round(util_percent, 2)
            }
        except Exception as e:
            logger.error(f"OCRManagerDocTR: Could not get GPU memory stats: {e}", exc_info=False)
            return {"status": "Error", "message": str(e)}

if __name__ == '__main__':
    # Example Usage (requires some dummy image files or real ones)
    logger.info("Starting OCRManagerDocTR example...")

    # Create some dummy image bytes (replace with actual image loading for real test)
    def create_dummy_jpeg_bytes(width=200, height=100) -> bytes:
        img = Image.new('RGB', (width, height), color = 'red')
        # Add some text to make OCR actually do something
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 40) # Try to load a common font
        except IOError:
            font = ImageFont.load_default()
        d.text((10,10), "Hello", fill=(255,255,0), font=font)
        d.text((10,50), "DocTR", fill=(0,255,255), font=font)
        
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='JPEG')
        return byte_arr.getvalue()

    dummy_image_bytes1 = create_dummy_jpeg_bytes(300, 150)
    dummy_image_bytes2 = create_dummy_jpeg_bytes(250, 120)
    invalid_image_bytes = b"this is not an image"
    dummy_image_bytes3 = create_dummy_jpeg_bytes(400, 200)


    # --- Test with CPU ---
    logger.info("\n--- Testing with CPU ---")
    try:
        # Initialize OCRManager for CPU, disable AMP (not applicable), enable CV2 if available
        ocr_manager_cpu = OCRManager(
            device_idx=-1, 
            warmup_on_init=True, 
            use_amp_if_available=False, # AMP is for CUDA
            use_cv2_default=True # Try CV2 if installed
        )
        
        # Test single OCR
        text_single_cpu = ocr_manager_cpu.ocr(dummy_image_bytes1)
        logger.info(f"CPU - Single OCR Result 1: '{text_single_cpu}'")

        # Test batch OCR
        image_list_cpu = [dummy_image_bytes1, invalid_image_bytes, dummy_image_bytes2, dummy_image_bytes3]
        texts_batch_cpu = ocr_manager_cpu.batch_ocr(image_list_cpu)
        for i, text in enumerate(texts_batch_cpu):
            logger.info(f"CPU - Batch OCR Result {i}: '{text.replace(chr(10), ' ')}'")
        
        del ocr_manager_cpu # Release resources
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error during CPU test: {e}", exc_info=True)


    # --- Test with GPU (if available) ---
    if torch.cuda.is_available():
        logger.info("\n--- Testing with GPU ---")
        try:
            ocr_manager_gpu = OCRManager(
                device_idx=0, 
                warmup_on_init=True,
                use_amp_if_available=True, # Enable AMP
                use_cv2_default=True      # Try CV2 if installed
            )
            logger.info(f"GPU Memory Stats after init: {ocr_manager_gpu.get_gpu_memory_stats()}")

            # Test single OCR
            text_single_gpu = ocr_manager_gpu.ocr(dummy_image_bytes1)
            logger.info(f"GPU - Single OCR Result 1: '{text_single_gpu}'")
            logger.info(f"GPU Memory Stats after single OCR: {ocr_manager_gpu.get_gpu_memory_stats()}")

            # Test batch OCR
            image_list_gpu = [dummy_image_bytes1, dummy_image_bytes2, invalid_image_bytes, dummy_image_bytes3, dummy_image_bytes1]
            texts_batch_gpu = ocr_manager_gpu.batch_ocr(image_list_gpu)
            for i, text in enumerate(texts_batch_gpu):
                logger.info(f"GPU - Batch OCR Result {i}: '{text.replace(chr(10), ' ')}'")
            logger.info(f"GPU Memory Stats after batch OCR: {ocr_manager_gpu.get_gpu_memory_stats()}")

            del ocr_manager_gpu # Release resources
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during GPU test: {e}", exc_info=True)
    else:
        logger.info("\nCUDA not available, skipping GPU tests.")

    logger.info("OCRManagerDocTR example finished.")