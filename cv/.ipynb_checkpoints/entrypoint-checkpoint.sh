#!/bin/bash
set -e

echo "--- Entrypoint Script Started ---"

# --- Configuration ---
PT_MODEL_PATH="/models/best_yolo.pt"
ENGINE_MODEL_PATH="/models/best_yolo.engine"
IMAGE_SIZE=1280
# EDITED: Define a max batch size for the dynamic engine
MAX_BATCH_SIZE=8

# --- Debugging: List contents of /models directory ---
echo "Listing contents of /models directory before check:"
ls -l /models || echo "/models directory does not exist or is empty."

# --- Engine Generation Logic ---
# Check if the .engine file already exists
if [ -f "$ENGINE_MODEL_PATH" ]; then
    echo "INFO: TensorRT engine found at $ENGINE_MODEL_PATH. Skipping generation."
else
    echo "INFO: TensorRT engine not found. Attempting to generate from $PT_MODEL_PATH..."
    
    # Check if the .pt file exists before trying to convert
    if [ -f "$PT_MODEL_PATH" ]; then
        echo "INFO: Found source .pt model. Starting export with dynamic=True and max batch size=${MAX_BATCH_SIZE}..."
        
        # EDITED: Added the 'batch' argument to create a truly dynamic engine
        python3 -c "from ultralytics import YOLO; print('--- Python Export Script Started ---'); model = YOLO('$PT_MODEL_PATH'); model.export(format='engine', dynamic=True, imgsz=$IMAGE_SIZE, batch=$MAX_BATCH_SIZE); print('--- Python Export Script Finished ---')"
        
        echo "INFO: Python export command finished."
        
        # Check if the engine was created successfully
        if [ -f "$ENGINE_MODEL_PATH" ]; then
            echo "SUCCESS: YOLO TensorRT engine with dynamic batching generated successfully."
        else
            echo "CRITICAL: YOLO TensorRT engine generation FAILED. The .engine file was not created. Check Python logs above."
            exit 1
        fi
    else
        echo "CRITICAL: Source YOLO .pt model not found at $PT_MODEL_PATH. Cannot generate engine."
        exit 1
    fi
fi

# --- Debugging: List contents of /models directory again ---
echo "Listing contents of /models directory after check/generation:"
ls -l /models || echo "/models directory does not exist or is empty."


# --- Start the Uvicorn Server ---
# The "$@" allows us to pass the CMD from the Dockerfile to this script as arguments
echo "--- Starting Uvicorn server... ---"
exec "$@"

