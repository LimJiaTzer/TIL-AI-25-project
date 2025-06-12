import os
import argparse
from transformers import AutoProcessor, AutoModelForCTC, VisionEncoderDecoderModel, TrOCRProcessor# Add other model types as needed

# --- Model Configuration ---
# You can pre-define common models here or pass them via command line
# Format: "user_friendly_name": ("model_identifier_on_hf", ModelClass, ProcessorClass)
# For TrOCR, the model is often VisionEncoderDecoderModel and the processor is TrOCRProcessor
# For CTC-based ASR models (like older Wav2Vec2), it's often AutoModelForCTC and AutoProcessor

KNOWN_MODELS = {
    "trocr-large-printed": (
        "microsoft/trocr-large-printed",
        VisionEncoderDecoderModel,
        TrOCRProcessor
    ),
    "trocr-base-printed": (
        "microsoft/trocr-base-printed",
        VisionEncoderDecoderModel,
        TrOCRProcessor
    ),
    "GOT-OCR-2.0-hf": (
        "stepfun-ai/GOT-OCR-2.0-hf",
        VisionEncoderDecoderModel,
        TrOCRProcessor
    ),
}

def download_and_save_model(model_name_or_path: str, output_dir: str, model_class=None, processor_class=None):
    """
    Downloads a Hugging Face model and its processor/tokenizer and saves them
    to the specified directory.

    Args:
        model_name_or_path (str): The Hugging Face model identifier (e.g., "microsoft/trocr-base-handwritten")
                                  or a path to a local model.
        output_dir (str): The directory where the model and processor will be saved.
        model_class: The Hugging Face model class (e.g., VisionEncoderDecoderModel).
        processor_class: The Hugging Face processor class (e.g., TrOCRProcessor).
    """
    if not model_class or not processor_class:
        print(f"Model class or processor class not specified for {model_name_or_path}. Attempting with AutoClasses.")
        # Fallback to AutoClasses if specific ones aren't provided, might not always work perfectly
        # For complex models like TrOCR, it's better to specify.
        from transformers import AutoModel, AutoProcessor as GenericAutoProcessor
        model_class = AutoModel
        processor_class = GenericAutoProcessor

    try:
        print(f"Attempting to download/load processor for '{model_name_or_path}' using {processor_class.__name__}...")
        processor = processor_class.from_pretrained(model_name_or_path)
        print(f"Processor for '{model_name_or_path}' loaded successfully.")

        print(f"Attempting to download/load model '{model_name_or_path}' using {model_class.__name__}...")
        model = model_class.from_pretrained(model_name_or_path)
        print(f"Model '{model_name_or_path}' loaded successfully.")

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: '{output_dir}'")

        # Save the processor and model
        print(f"Saving processor to '{output_dir}'...")
        processor.save_pretrained(output_dir)
        print("Processor saved.")

        print(f"Saving model to '{output_dir}'...")
        model.save_pretrained(output_dir)
        print("Model saved.")

        print(f"\nSuccessfully downloaded and saved '{model_name_or_path}' to '{output_dir}'.")
        print("The following files should be present (or similar):")
        for item in os.listdir(output_dir):
            print(f"  - {item}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the model identifier is correct and you have an internet connection (if downloading).")
        print("Also, ensure the specified model_class and processor_class are correct for the model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a Hugging Face model.")
    parser.add_argument(
        "model_key",
        type=str,
        help=f"The key for the model to download from the predefined list: {', '.join(KNOWN_MODELS.keys())}. "
             f"Or, a full Hugging Face model identifier (e.g., 'openai-community/gpt2')."
    )
    parser.add_argument(
        "--output_folder_name",
        type=str,
        default=None,
        help="Optional: Name of the subfolder to create in the current directory for saving the model. "
             "If not provided, uses the model_key (sanitized) or the last part of a HF identifier."
    )
    parser.add_argument(
        "--force_generic_classes",
        action="store_true",
        help="Force using AutoModel and AutoProcessor (less reliable for specific architectures like TrOCR)."
    )


    args = parser.parse_args()

    model_to_download = args.model_key
    model_identifier_on_hf = None
    SpecificModelClass = None
    SpecificProcessorClass = None

    if args.force_generic_classes:
        from transformers import AutoModel, AutoProcessor as GenericAutoProcessor
        model_identifier_on_hf = model_to_download
        SpecificModelClass = AutoModel
        SpecificProcessorClass = GenericAutoProcessor
        print("Forcing use of generic AutoModel and AutoProcessor.")
    elif model_to_download in KNOWN_MODELS:
        model_identifier_on_hf, SpecificModelClass, SpecificProcessorClass = KNOWN_MODELS[model_to_download]
        print(f"Using predefined configuration for '{model_to_download}':")
        print(f"  Identifier: {model_identifier_on_hf}")
        print(f"  Model Class: {SpecificModelClass.__name__}")
        print(f"  Processor Class: {SpecificProcessorClass.__name__}")
    else:
        # Assume it's a direct Hugging Face identifier and attempt with generic Auto classes
        print(f"'{model_to_download}' not in predefined list. Assuming it's a HF identifier.")
        print("Attempting with generic AutoModel and AutoProcessor. This might not be optimal for all models.")
        from transformers import AutoModel, AutoProcessor as GenericAutoProcessor
        model_identifier_on_hf = model_to_download
        SpecificModelClass = AutoModel
        SpecificProcessorClass = GenericAutoProcessor


    # Determine output directory
    if args.output_folder_name:
        folder_name = args.output_folder_name
    else:
        # Sanitize model_key or use last part of HF identifier for folder name
        folder_name = model_to_download.split('/')[-1].replace('.', '_')

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(script_dir, folder_name)

    print(f"\nTarget Hugging Face Identifier: {model_identifier_on_hf}")
    print(f"Chosen output directory: {save_directory}\n")

    download_and_save_model(model_identifier_on_hf, save_directory, SpecificModelClass, SpecificProcessorClass)