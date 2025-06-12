# ── START OF FILE ──
import os

# Force every HF library to stay offline
os.environ["HF_HUB_OFFLINE"]     = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_METRICS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import io
import re
import logging
import warnings
import sys
import time
import torch
import torchaudio
import torchaudio.transforms as T # Added for STFT/ISTFT/Resample
from torchaudio import sox_effects
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from packaging import version
from typing import List, Dict, Optional, Union
import numpy as np

# ——— Configure logging ———————————————————————————————————————————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ASRManager")
# Suppress known HF warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.utils.generic"
)
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.",
)

class ASRManager:
    """End-to-end ASR manager using Hugging Face Whisper + output cleaning,
    with optional spectral subtraction for noise reduction, fast C++ audio I/O,
    and optional torch.compile."""

    def __init__(
        self,
        model_path: str = "/models/my_model",
        device: int = 0,
        dtype: torch.dtype = torch.float16,
        chunk_length_s: float = 32.0,
        stride_length_s: float = 1.5,
        batch_size: int = 128,
        use_flash_attention: bool = True,
        use_bettertransformer: bool = True,
        use_compile: bool = True,
        compile_mode: str = "reduce-overhead",
        warmup_on_init: bool = True,
        # --- Spectral Subtraction Parameters ---
        use_spectral_subtraction: bool = False, # Set to True to enable
        noise_profile_path: Optional[str] = None, # Path to a WAV file containing only noise
        n_fft: int = 2048, # FFT window size for spectral subtraction STFT
        hop_length: int = 512, # Hop length for spectral subtraction STFT
        noise_reduction_amount: float = 1.0, # Factor for noise subtraction (1.0 = full, >1.0 = over-subtraction)
        spectral_floor: float = 0.02, # Factor for spectral flooring
    ):
        # Whisper's target sampling rate
        self.sr = 16_000
        # Pre-compile regex for cleaning
        self._clean_re = re.compile(r"[^\w\s]")

        # Choose CPU vs GPU
        if device == -1 or not torch.cuda.is_available():
            self.device_str = "cpu"
            self.device_idx = -1
            logger.info("ASRManager: using CPU")
        else:
            ngpu = torch.cuda.device_count()
            idx = min(device, ngpu - 1)
            self.device_str = f"cuda:{idx}"
            self.device_idx = idx
            name = torch.cuda.get_device_name(idx)
            mem = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            logger.info(f"ASRManager: using GPU {idx} ({name}, {mem:.1f} GB)")

        # Store parameters
        self.use_bettertransformer = use_bettertransformer
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.dtype = dtype

        # Spectral Subtraction Setup
        self.use_spectral_subtraction = use_spectral_subtraction
        self.mean_noise_magnitude = None
        self.spectrogram_transform = None
        self.inverse_spectrogram_transform = None
        self.noise_reduction_amount = noise_reduction_amount
        self.spectral_floor_val = spectral_floor # Renamed to avoid conflict
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Load our LOCAL model and processor
        self.processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True, torch_dtype=self.dtype)

        if self.use_spectral_subtraction:
            logger.info("ASRManager: Spectral subtraction enabled.")
            if noise_profile_path and os.path.exists(noise_profile_path):
                logger.info(f"ASRManager: Loading noise profile from {noise_profile_path}")
                try:
                    # Initialize transforms needed for noise processing
                    self.spectrogram_transform = T.Spectrogram(
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        power=None, # Get complex output
                        center=True,
                        pad_mode="reflect",
                        normalized=False,
                    ).to(self.device_str)

                    self.inverse_spectrogram_transform = T.InverseSpectrogram(
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        center=True,
                        normalized=False,
                    ).to(self.device_str)

                    self.mean_noise_magnitude = self._load_and_process_noise_profile(
                        noise_profile_path
                    )
                    logger.info("ASRManager: Noise profile processed successfully.")
                except Exception as e:
                    logger.error(f"ASRManager: Failed to load or process noise profile '{noise_profile_path}': {e}. Spectral subtraction will be disabled.")
                    self.use_spectral_subtraction = False
                    self.mean_noise_magnitude = None
                    self.spectrogram_transform = None
                    self.inverse_spectrogram_transform = None
            else:
                logger.warning("ASRManager: 'use_spectral_subtraction' is True, but 'noise_profile_path' is missing or invalid. Spectral subtraction will be disabled.")
                self.use_spectral_subtraction = False

        # Build model_kwargs for Whisper pipeline
        model_kwargs = {}
        #if self.device_idx >= 0 and use_flash_attention:
            #try:
                #from transformers.utils import is_flash_attn_2_available
                #if is_flash_attn_2_available():
                    #model_kwargs["use_flash_attention_2"] = True
                    #logger.info("ASRManager: flash attention enabled")
            #except ImportError:
                #pass

        # Clear & optimize GPU memory
        if self.device_idx >= 0:
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            # Enable mixed precision for Tesla T4
            if torch.cuda.get_device_name(self.device_idx).find("T4") >= 0:
                logger.info("ASRManager: Tesla T4 detected, optimizing for this GPU")
                if self.dtype != torch.float16:
                    logger.info("ASRManager: Forcing FP16 for T4 GPU")
                    self.dtype = torch.float16

        # Load the Hugging Face pipeline
        logger.info("ASRManager: loading Whisper pipeline...")
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            feature_extractor=self.processor.feature_extractor,
            tokenizer=self.processor.tokenizer,
            device=self.device_idx, # pipeline expects integer index or -1
            #torch_dtype=self.dtype,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            model_kwargs=model_kwargs,
        )
        logger.info("ASRManager: pipeline ready.")

        # Apply BetterTransformer optimization if available
        if self.device_idx >= 0 and self.use_bettertransformer:
            try:
                from optimum.bettertransformer import BetterTransformer
                self.transcriber.model = BetterTransformer.transform(
                    self.transcriber.model
                )
                logger.info("ASRManager: BetterTransformer optimization applied")
            except (ImportError, Exception) as e:
                logger.warning(f"ASRManager: BetterTransformer optimization failed ({e})")

        # Optional: compile the model to reduce Python overhead
        if (
            self.device_idx >= 0
            and self.use_compile
            and hasattr(torch, "compile")
            and version.parse(torch.__version__) >= version.parse("2.0.0")
            and sys.version_info < (3, 12) # Dynamo not supported on Python 3.12+
        ):
            try:
                self.transcriber.model = torch.compile(
                    self.transcriber.model,
                    mode=self.compile_mode,
                    fullgraph=False, # Often more stable
                )
                logger.info(
                    f"ASRManager: model compiled with torch.compile (mode={self.compile_mode})"
                )
            except Exception as e:
                logger.warning(f"ASRManager: torch.compile failed ({e})")

        # Additional memory optimization
        if self.device_idx >= 0:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "memory_stats"):
                logger.info(
                    f"Initial GPU memory: {self.get_gpu_memory_stats()['util_percent']}% utilized"
                )

        # Warm up the model to reduce initial inference latency
        if warmup_on_init: # Warmup regardless of device to compile if needed
            self.warmup_model()

    def _clean_text(self, text: str) -> str:
        """Lowercase, replace hyphens, strip punctuation."""
        if not isinstance(text, str):
             logger.warning(f"ASRManager: Received non-string input for cleaning: {type(text)}. Returning empty string.")
             return ""
        text = text.lower().replace("-", " ")
        return self._clean_re.sub("", text).strip()

    def _load_and_resample(
        self, audio_bytes: bytes
    ) -> Optional[torch.Tensor]:
        """
        Load WAV bytes, convert to mono, resample to 16 kHz,
        and return a Tensor[1, T] of float32 on the configured device.
        Returns None if loading fails.
        """
        try:
            buf = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(buf)

            # Ensure waveform is float32 for SoX effects
            waveform = waveform.to(dtype=torch.float32)

            # Apply effects for mono conversion and resampling
            # Use sox_effects for potentially better quality resampling than T.Resample
            waveform, sr_out = sox_effects.apply_effects_tensor(
                tensor=waveform,
                sample_rate=sample_rate,
                effects=[
                    ["channels", "1"],       # mono
                    ["rate", str(self.sr)], # resample to target rate
                    ["norm"]                 # Normalize volume slightly, can help consistency
                ],
                channels_first=True, # Input/Output shape [C, T]
            )
            # Move to the target device *after* processing
            return waveform.to(self.device_str)
        except Exception as e:
             logger.error(f"ASRManager: Failed to load/resample audio: {e}")
             return None

    def _load_and_process_noise_profile(self, noise_file_path: str) -> Optional[torch.Tensor]:
        """Loads a noise file, resamples, calculates STFT, and returns mean magnitude."""
        try:
            noise_waveform, noise_sr = torchaudio.load(noise_file_path)
            noise_waveform = noise_waveform.to(self.device_str)

            # Resample if necessary
            if noise_sr != self.sr:
                logger.info(f"ASRManager: Resampling noise profile from {noise_sr} Hz to {self.sr} Hz")
                resampler = T.Resample(noise_sr, self.sr).to(self.device_str)
                noise_waveform = resampler(noise_waveform)

            # Ensure mono
            if noise_waveform.shape[0] > 1:
                noise_waveform = torch.mean(noise_waveform, dim=0, keepdim=True)

            # Calculate STFT
            noise_stft = self.spectrogram_transform(noise_waveform) # Uses the transform created in __init__
            noise_magnitude = torch.abs(noise_stft)

            # Calculate mean magnitude across time dimension
            mean_noise_magnitude = torch.mean(noise_magnitude, dim=2, keepdim=True)
            return mean_noise_magnitude

        except Exception as e:
            logger.error(f"ASRManager: Error processing noise profile '{noise_file_path}': {e}")
            return None

    def _spectral_subtraction(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applies spectral subtraction to the input waveform tensor using the
        pre-calculated noise profile.

        Args:
            waveform: Input waveform tensor [1, T] on the correct device.

        Returns:
            Cleaned waveform tensor [1, T] on the correct device.
        """
        if not self.use_spectral_subtraction or self.mean_noise_magnitude is None or self.spectrogram_transform is None or self.inverse_spectrogram_transform is None:
            # Skip if not enabled or if noise profile failed to load
            return waveform

        original_length = waveform.shape[1] # Needed for ISTFT padding removal

        # 1. Calculate STFT of the input waveform
        noisy_stft = self.spectrogram_transform(waveform)
        noisy_magnitude = torch.abs(noisy_stft)

        # Get unit phasor (phase information)
        # Add small epsilon to prevent division by zero
        noisy_unit_phasor = noisy_stft / (noisy_magnitude + 1e-8)

        # 2. Subtract Noise Estimate (Magnitude Subtraction)
        # Ensure mean_noise_magnitude is broadcastable across the time dimension
        cleaned_magnitude = noisy_magnitude - (self.noise_reduction_amount * self.mean_noise_magnitude)

        # 3. Apply Spectral Floor (Rectification)
        noise_floor_value = self.mean_noise_magnitude * self.spectral_floor_val
        cleaned_magnitude = torch.maximum(cleaned_magnitude, noise_floor_value)
        # Ensure non-negative just in case flooring wasn't perfect
        cleaned_magnitude = torch.clamp(cleaned_magnitude, min=0.0)

        # 4. Reconstruct Complex Spectrogram
        cleaned_stft = cleaned_magnitude * noisy_unit_phasor

        # 5. Inverse Transform
        # Pass original length to handle potential padding removal by ISTFT
        cleaned_waveform = self.inverse_spectrogram_transform(cleaned_stft, length=original_length)

        return cleaned_waveform


    def asr(self, audio_bytes: bytes) -> str:
        """
        Transcribe a single WAV (bytes), optionally apply spectral subtraction,
        and return cleaned text. Guaranteed to return a str.
        """
        # 1) Load & resample -> Tensor[1, T] on target device
        waveform = self._load_and_resample(audio_bytes)
        if waveform is None:
             return "" # Return empty string if loading failed

        # 2) Apply spectral subtraction if enabled
        if self.use_spectral_subtraction:
            try:
                waveform = self._spectral_subtraction(waveform)
            except Exception as e:
                 logger.warning(f"ASRManager: Spectral subtraction failed during single ASR: {e}, skipping.")

        # 3) Convert tensor to NumPy array for the HF pipeline
        # Ensure it's on CPU and squeezed
        samples_np = waveform.squeeze(0).cpu().numpy()
        # If model is float16 and on GPU, cast numpy array to float16
        if self.device_idx >= 0 and self.dtype == torch.float16:
            samples_np = samples_np.astype(np.float16)

        # 4) Run the pipeline with optimized settings
        text = ""
        try:
            # Use autocast for mixed precision on GPU if dtype is float16/bfloat16
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(self.device_idx >= 0 and self.dtype in [torch.float16, torch.bfloat16])):
                result = self.transcriber({"raw": samples_np, "sampling_rate": self.sr})
                # Handle potential variations in pipeline output format
                if isinstance(result, dict) and "text" in result:
                    text = result["text"]
                elif isinstance(result, str):
                     text = result
                elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "text" in result[0]:
                     # Sometimes pipeline might return list for single item
                     text = result[0]["text"]
                else:
                     logger.warning(f"ASRManager: Unexpected output format from transcriber: {type(result)}")

        except Exception as e:
             logger.error(f"ASRManager: Transcription failed: {e}")
             # Ensure text is a string even on failure
             text = ""


        # 5) Clean & return
        return self._clean_text(text)


    def batch_asr(self, audio_bytes_list: List[bytes]) -> List[str]:
        """Batch-transcribe a list of WAV bytes, optionally applying spectral
        subtraction, returning cleaned strings."""
        batch_inputs = [] # List to hold dictionaries for the pipeline

        # Pre-process all audio files
        for i, b in enumerate(audio_bytes_list):
            # Load and keep as tensor on target device
            wf = self._load_and_resample(b)
            if wf is None:
                 logger.warning(f"ASRManager: Skipping item {i} in batch due to loading error.")
                 # Add a placeholder to maintain output list size correspondence
                 batch_inputs.append(None) # Will be filtered later
                 continue

            # Apply spectral subtraction if enabled
            if self.use_spectral_subtraction:
                 try:
                      wf = self._spectral_subtraction(wf)
                 except Exception as e:
                      logger.warning(f"ASRManager: Spectral subtraction failed for item {i} in batch: {e}, skipping subtraction for this item.")

            # Convert cleaned tensor to NumPy array for the pipeline
            samples_np = wf.squeeze(0).cpu().numpy()
            # If model is float16 and on GPU, cast numpy array to float16
            if self.device_idx >= 0 and self.dtype == torch.float16:
                samples_np = samples_np.astype(np.float16)
            batch_inputs.append({"raw": samples_np, "sampling_rate": self.sr})

        # Filter out any None entries from loading failures
        valid_batch_inputs = [item for item in batch_inputs if item is not None]
        if not valid_batch_inputs:
             logger.warning("ASRManager: No valid audio data in the batch after processing.")
             # Return list of empty strings matching original input size
             return [""] * len(audio_bytes_list)

        # Run batch transcription
        results = []
        try:
            # Determine optimal batch size dynamically (simplified here, could be more complex)
            # For now, we rely on the pipeline's internal batching mechanism configured in __init__

            # Use autocast for mixed precision on GPU if dtype is float16/bfloat16
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(self.device_idx >= 0 and self.dtype in [torch.float16, torch.bfloat16])):
                 # The pipeline handles batching internally based on batch_size in __init__
                 results = self.transcriber(valid_batch_inputs)

        except Exception as e:
            logger.error(f"ASRManager: Batch transcription failed: {e}")
            # Return empty strings for all items in the original batch on failure
            return [""] * len(audio_bytes_list)

        # Process results and map back to original batch structure
        cleaned_texts: List[str] = []
        result_idx = 0
        for item in batch_inputs: # Iterate through the original structure including None placeholders
            if item is None:
                cleaned_texts.append("") # Append empty string for failed loads
            else:
                if result_idx < len(results):
                    r = results[result_idx]
                    # Handle potential variations in pipeline output format
                    txt = ""
                    if isinstance(r, dict) and "text" in r:
                        txt = r["text"]
                    elif isinstance(r, str):
                         txt = r
                    else:
                         logger.warning(f"ASRManager: Unexpected item format in batch results: {type(r)}")
                    cleaned_texts.append(self._clean_text(txt))
                    result_idx += 1
                else:
                    # Should not happen if transcription didn't error, but handle defensively
                    logger.error("ASRManager: Mismatch between input batch size and result size.")
                    cleaned_texts.append("")


        # Aggressive memory cleanup after batch processing
        if self.device_idx >= 0:
            torch.cuda.empty_cache()

        return cleaned_texts

    def get_gpu_memory_stats(self) -> Dict:
        """Report current GPU memory usage."""
        if self.device_idx < 0:
            return {"error": "No GPU in use"}
        try:
            props = torch.cuda.get_device_properties(self.device_idx)
            total = props.total_memory / (1024**3)
            alloc = torch.cuda.memory_allocated(self.device_idx) / (1024**3)
            resv = torch.cuda.memory_reserved(self.device_idx) / (1024**3)
            util = 0.0 if total == 0 else (100 * resv / total) # Avoid division by zero
            return {
                "device_name": props.name,
                "total_gb": round(total, 2),
                "allocated_gb": round(alloc, 2),
                "reserved_gb": round(resv, 2),
                "util_percent": round(util, 2),
            }
        except Exception as e:
             logger.error(f"ASRManager: Could not get GPU memory stats: {e}")
             return {"error": f"Could not get GPU memory stats: {e}"}


    def warmup_model(self, duration_seconds: float = 1.0) -> None:
        """Warm up the model (and spectral subtraction if enabled) with synthetic audio."""
        logger.info("ASRManager: Warming up model...")
        try:
            # Create synthetic audio (silence) for warmup
            num_samples = int(duration_seconds * self.sr)
            # Create on CPU first, then move
            silence_cpu = torch.zeros(1, num_samples, dtype=torch.float32)
            silence = silence_cpu.to(self.device_str)

            # Warm up spectral subtraction if enabled
            cleaned_silence_np = silence.squeeze(0).cpu().numpy() # Default if SS is off
            if self.use_spectral_subtraction and self.mean_noise_magnitude is not None:
                 logger.info("ASRManager: Warming up spectral subtraction...")
                 with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(self.device_idx >= 0 and self.dtype in [torch.float16, torch.bfloat16])):
                      # Run subtraction part
                      cleaned_silence = self._spectral_subtraction(silence)
                      # Convert result for pipeline warmup
                      cleaned_silence_np = cleaned_silence.squeeze(0).cpu().numpy()
                 logger.info("ASRManager: Spectral subtraction warmup done.")


            # Warm up the main transcriber pipeline
            logger.info("ASRManager: Warming up Whisper pipeline...")
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(self.device_idx >= 0 and self.dtype in [torch.float16, torch.bfloat16])):
                _ = self.transcriber({"raw": cleaned_silence_np, "sampling_rate": self.sr})

            # Synchronize and clear cache if on GPU
            if self.device_idx >= 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info("ASRManager: Model warm-up complete")
        except Exception as e:
            logger.warning(f"ASRManager: Model warm-up failed: {e}")
