import io
import logging
import os
import re
import warnings
from typing import Any, List, Optional, Union

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
import torchaudio
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from omegaconf import OmegaConf, open_dict
from torchaudio import sox_effects

# --- Environment Setup ---
# Set HuggingFace Hub to offline mode to prevent unexpected network calls.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_METRICS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- Logging Configuration ---
# Set up a clear and informative logging format.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ASRManager")

# --- Suppress Known Warnings ---
# Filter out common, non-critical warnings from PyTorch Lightning and NeMo
# to keep the logs clean during operation.
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="nemo")
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.",
)
warnings.filterwarnings(
    "ignore",
    message="SoxEffectsChain is deprecated.*",
    category=UserWarning,
    module="torchaudio.backend.sox_io_backend",
)


class ASRManager:
    """
    Manages loading a NeMo ASR model and performing transcription on audio data.

    This class encapsulates device setup, model loading from a .nemo file,
    audio preprocessing (resampling and mono conversion), and transcription logic.
    It is designed for efficient, single-file transcription on a GPU.
    """

    # --- Class-level constants for better configuration management ---
    DEFAULT_MODEL_PATH = "/models/my_model/parakeet-tdt-0.6b-v2.nemo"
    DEFAULT_TARGET_SR = 16_000

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device_id: int = 0,  # GPU device index, -1 for CPU
        dtype: torch.dtype = torch.float32,  # torch.float32 for stability, torch.float16 for speed
    ):
        """
        Initializes the ASR manager by setting up the device and loading the model.
        """
        self.torch_device = self._setup_device(device_id)
        self.dtype = dtype
        self.model = self._initialize_model(model_path)
        self.target_sr = self.model.cfg.get('sample_rate', self.DEFAULT_TARGET_SR)
        self._clean_re = re.compile(r"[^\w\s]")  # Regex for cleaning transcriptions

    def _setup_device(self, device_id: int) -> torch.device:
        """
        Configures and returns the torch.device (GPU or CPU) for computation.
        """
        if device_id == -1 or not torch.cuda.is_available():
            logger.info("Using CPU for ASR.")
            return torch.device("cpu")

        num_gpus = torch.cuda.device_count()
        if device_id >= num_gpus:
            logger.warning(
                f"GPU index {device_id} is out of range ({num_gpus} GPUs available). Defaulting to GPU 0."
            )
            device_id = 0

        device = torch.device(f"cuda:{device_id}")
        name = torch.cuda.get_device_name(device_id)
        mem = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        logger.info(f"Using GPU {device_id} ({name}, {mem:.1f} GB) for ASR.")
        return device

    def _initialize_model(self, model_path: str) -> nemo_asr.models.ASRModel:
        """
        Loads the NeMo ASR model from the specified path and prepares it for inference.
        """
        logger.info(f"Loading NeMo ASR model from {model_path}...")
        try:
            # First, restore the configuration to modify it for inference.
            # This avoids issues with optimizer states or dataset configs from training.
            cfg = nemo_asr.models.ASRModel.restore_from(
                restore_path=model_path, return_config=True, map_location=self.torch_device
            )
            with open_dict(cfg):  # Allow modifications to the Hydra config
                cfg.train_ds = None
                cfg.validation_ds = None
                cfg.test_ds = None
                if hasattr(cfg, 'optim'):
                    cfg.optim = None

            # Now, restore the model with the modified config. `strict=False` helps
            # with models trained on different NeMo versions.
            model = nemo_asr.models.ASRModel.restore_from(
                restore_path=model_path,
                override_config_path=cfg,
                map_location=self.torch_device,
                strict=False,
            )

            if self.dtype == torch.float16 and self.torch_device.type == 'cuda':
                logger.info("Converting NeMo model to float16 (half precision).")
                model = model.half()

            model.eval()  # Set the model to evaluation mode
            logger.info("NeMo ASR model loaded successfully and set to eval mode.")
            return model

        except Exception as e:
            logger.error(f"Failed to load NeMo model from {model_path}.", exc_info=True)
            raise  # Re-raise to halt startup if the model is critical

    def _load_and_resample(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Loads audio from bytes, converts to mono, resamples to the target rate,
        and returns a 1D NumPy array of float32. Returns None on failure.
        """
        try:
            buf = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(buf, format=None)

            # Apply effects: convert to mono and resample.
            effects = [
                ["channels", "1"],
                ["rate", str(self.target_sr)],
            ]
            resampled_waveform, _ = sox_effects.apply_effects_tensor(
                tensor=waveform.to(torch.float32),  # Ensure float32 for processing
                sample_rate=sample_rate,
                effects=effects,
            )
            return resampled_waveform.squeeze().cpu().numpy()

        except Exception:
            logger.error("Failed to load or resample audio.", exc_info=True)
            return None

    def _parse_transcription_result(self, result: List[Any]) -> str:
        """
        Parses the complex output from NeMo's transcribe method to extract plain text.
        Handles various return types like Hypothesis objects, strings, or nested lists.
        """
        if not result or not isinstance(result, list):
            logger.warning("Transcription returned an empty or invalid result.")
            return ""

        # The result is often a list containing one primary transcription.
        # This transcription itself can be a Hypothesis, a string, or a list (for beam search).
        first_item = result[0]
        
        # Case 1: The item is a Hypothesis object (common for RNN-T models)
        if isinstance(first_item, Hypothesis):
            return first_item.text

        # Case 2: The item is already a string
        if isinstance(first_item, str):
            return first_item

        # Case 3: The item is a list (e.g., from beam search)
        if isinstance(first_item, list) and first_item:
            nested_item = first_item[0]
            if isinstance(nested_item, Hypothesis):
                return nested_item.text
            if isinstance(nested_item, str):
                return nested_item

        logger.warning(f"Unexpected transcription result format: {type(first_item)}")
        return ""

    def _clean_text(self, text: str) -> str:
        """
        Applies basic text cleaning: lowercasing, removing punctuation, and stripping whitespace.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower().replace("-", " ")
        return self._clean_re.sub("", text).strip()

    def asr(self, audio_bytes: bytes) -> str:
        """
        Performs end-to-end ASR on a byte string of audio data.
        """
        samples_np = self._load_and_resample(audio_bytes)
        if samples_np is None:
            logger.warning("Audio preprocessing failed; returning empty transcription.")
            return ""
        
        if samples_np.size == 0:
            logger.info("Received empty audio signal; returning empty transcription.")
            return ""

        transcribed_text = ""
        try:
            with torch.inference_mode():  # Ensure no gradients are computed
                results = self.model.transcribe(
                    audio=[samples_np],
                    batch_size=1,
                )
                transcribed_text = self._parse_transcription_result(results)
        
        except RuntimeError as e:
            # Provide specific feedback for common CUDA errors
            if "CUDA error" in str(e) and "illegal memory access" in str(e):
                logger.error("CUDA illegal memory access during transcription.", exc_info=True)
            else:
                logger.error("A runtime error occurred during transcription.", exc_info=True)
        except Exception:
            logger.error("An unexpected error occurred during transcription.", exc_info=True)

        return self._clean_text(transcribed_text)
