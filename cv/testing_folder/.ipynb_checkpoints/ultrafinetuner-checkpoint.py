import os
import torch
from ultralytics import RTDETR

# --- Configuration ---
# === PATHS ===
# This is the base directory where your 'data.yaml' and 'runs' folder are located.
# It should be the same as YOLO_DATASET_BASE_DIR from your previous script.
MODEL_AND_DATA_BASE_DIR = r"../models" # Or the absolute path if preferred

# Path to your existing data.yaml file
DATA_YAML_PATH = os.path.join(MODEL_AND_DATA_BASE_DIR, "data.yaml")

# === MODEL & TRAINING ===
# Specify the RT-DETR model you want to train.
# Options include: 'rtdetr-l.pt' (Large), 'rtdetr-x.pt' (Extra-Large)
# The .pt file will be downloaded automatically if not present.
RTDETR_MODEL_TO_TRAIN = "rtdetr-x.pt"

# Training Hyperparameters (adjust as needed)
EPOCHS = 200  # Same as your previous script, or adjust
# RT-DETR can be more memory-intensive.
# If you get Out-Of-Memory errors, REDUCE BATCH_SIZE significantly (e.g., 8, 4, or even 2).
BATCH_SIZE = 3 # START WITH A LOWER VALUE THAN YOLO and monitor GPU memory
IMG_SIZE = 1280 # Standard input size

# Name for the specific training run directory for this RT-DETR model.
# This will create a new folder like '../models/runs/rtdetr-l_finetune_experiment'
# ensuring it doesn't overwrite your YOLO model.
EXPERIMENT_NAME = f"{os.path.splitext(RTDETR_MODEL_TO_TRAIN)[0]}_finetune_experiment"

# --- Main Training Script ---
def train_rtdetr_model():
    print("--- Starting RT-DETR Model Fine-tuning ---")

    # 1. Check if data.yaml exists
    print(f"\n[1/3] Checking for data.yaml at: {DATA_YAML_PATH}")
    if not os.path.exists(DATA_YAML_PATH):
        print(f"ERROR: data.yaml not found at {DATA_YAML_PATH}")
        print("Please ensure your data.yaml from the previous YOLO training is present.")
        return
    print("    data.yaml found.")

    # 2. Initialize the RT-DETR model
    print(f"\n[2/3] Initializing RT-DETR model: {RTDETR_MODEL_TO_TRAIN}")
    try:
        model = RTDETR(RTDETR_MODEL_TO_TRAIN)
    except Exception as e:
        print(f"Error initializing YOLO model with {RTDETR_MODEL_TO_TRAIN}: {e}")
        print("Ensure the model name is correct and you have internet for download if it's the first time.")
        return

    # 3. Determine training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Training on device: {device}")
    if device == 'cpu':
        print("    Warning: No GPU detected by PyTorch. Training on CPU will be very slow.")
    elif not torch.cuda.is_available():
         print("    Warning: PyTorch reports CUDA not available. Training on CPU.")


    # 4. Start Training
    print(f"\n[3/3] Starting training...")
    print(f"    Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Image Size: {IMG_SIZE}")
    print(f"    Training run will be saved under: {os.path.join(MODEL_AND_DATA_BASE_DIR, 'runs', EXPERIMENT_NAME)}")

    try:
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=device,
            project=os.path.join(MODEL_AND_DATA_BASE_DIR, "runs"), # Base directory for all runs
            name=EXPERIMENT_NAME, # Specific folder name for this experiment
            patience = 10,
            # --- EXTRA AUGMENTATIONS TO IMPROVE ACCURACY ---
            # NOTE: These override the default values. Feel free to experiment.

            # Advanced augmentations that combine multiple images
            mosaic=1.0,     # (Probability: 0.0 to 1.0) Enables mosaic augmentation. Highly recommended.
            mixup=0.15,     # (Probability: 0.0 to 1.0) Enables MixUp. A value like 0.1-0.2 is a good start.
            copy_paste=0.1, # (Probability: 0.0 to 1.0) Enables Copy-Paste. Excellent for small/rare objects.

            # More aggressive geometric augmentations
            degrees=15.0,     # Random rotation range in degrees (+/- 15)
            translate=0.15,   # Random translation range (+/- 15% of image size)
            scale=0.5,        # Random scaling range (+/- 50%). Helps with objects of different sizes.
            shear=10.0,       # Random shear angle in degrees (+/- 10)

            # More aggressive color space augmentations
            hsv_h=0.015,  # Hue augmentation intensity
            hsv_s=0.7,    # Saturation augmentation intensity
            hsv_v=0.4,    # Value (brightness) augmentation intensity

            # Flip augmentations (horizontal flip is usually good, vertical depends on the object)
            fliplr=0.5,     # Horizontal flip probability (default is 0.5, good to keep)
            flipud=0.0,       # Vertical flip probability. Set to > 0 only if your objects can realistically appear upside down.
        )

        print("\n--- RT-DETR Fine-tuning complete! ---")
        # The best model is typically saved as 'best.pt' in the run directory
        print(f"    Trained model and results saved in directory: {results.save_dir}")
        print(f"    The best model weights are likely located at: {os.path.join(results.save_dir, 'weights', 'best.pt')}")

    except Exception as e:
        print(f"\nAn error occurred during RT-DETR training: {e}")
        print("    Common issues:")
        print("    - Out of Memory: Try reducing BATCH_SIZE significantly.")
        print("    - Ensure 'ultralytics' and PyTorch (with CUDA if using GPU) are installed correctly.")
        print(f"    - Verify the integrity of your dataset specified in {DATA_YAML_PATH}.")

# --- Run the script ---
if __name__ == "__main__":
    train_rtdetr_model()