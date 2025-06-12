"""Runs the CV server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import base64
from typing import Any

from fastapi import FastAPI, Request

from .cv_manager import CVManager
# In cv_server.py (or your main app script)
import logging




app = FastAPI()
manager = CVManager()
# Configure your CVManager's logger specifically
cv_manager_logger = logging.getLogger("CVManager")
cv_manager_logger.setLevel(logging.DEBUG) # Or logging.INFO

@app.post("/cv")
async def cv(request: Request) -> dict[str, list[list[dict[str, Any]]]]:
    """Performs CV object detection on image frames.

    Args:
        request: The API request. Contains a list of images, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `dict`s containing your CV model's predictions, in the same order as
        which appears in `request`. See `cv/README.md` for the expected format.
    """

    try:
        inputs_json = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    if "instances" not in inputs_json or not isinstance(inputs_json["instances"], list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'instances' list in JSON payload.")

    image_bytes_list: List[bytes] = []
    for instance in inputs_json["instances"]:
        if "b64" not in instance or not isinstance(instance["b64"], str):
            # Handle missing or invalid b64 string for an instance
            # Option 1: Raise an error for the whole batch
            # raise HTTPException(status_code=400, detail=f"Missing or invalid 'b64' field for an instance: {instance}")
            # Option 2: Add a placeholder or skip (less ideal for batch consistency if server expects all)
            # For now, let's assume valid inputs as per original logic, but error checking is good.
            # For this example, we'll try to decode and catch errors.
            image_bytes_list.append(b"") # Add empty bytes if b64 is missing/invalid, batch_cv should handle it
            print(f"Warning: Instance missing 'b64' field or it's not a string: {instance.get('key', 'N/A')}") 
            continue 

        try:
            # Reads the base-64 encoded image and decodes it into bytes.
            img_bytes = base64.b64decode(instance["b64"])
            image_bytes_list.append(img_bytes)
        except (TypeError, base64.binascii.Error) as e:
            image_bytes_list.append(b"")
            print(f"Warning: Invalid base64 data for instance {instance.get('key', 'N/A')}: {e}") # Log problematic instance
            continue


    if not image_bytes_list:
        # This case occurs if all instances in the request had issues.
        return {"predictions": [[] for _ in inputs_json["instances"]]}
    try:
        all_predictions = manager.batch_cv(image_bytes_list)
        if len(all_predictions) != len(inputs_json["instances"]):
            print(f"Warning: Mismatch between number of input instances ({len(inputs_json['instances'])}) and prediction sets ({len(all_predictions)}).")

    except Exception as e:
        # Handle exceptions from the CVManager's batch processing
        print(f"Error during batch CV processing: {e}") # Log the error server-side
        raise HTTPException(status_code=500, detail=f"Error during object detection: {str(e)}")


    return {"predictions": all_predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for your model."""
    return {"message": "health ok"}
