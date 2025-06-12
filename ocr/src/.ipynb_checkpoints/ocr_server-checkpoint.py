"""Runs the OCR server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import base64

from fastapi import FastAPI, Request
from ocr_manager import OCRManager

app = FastAPI()
manager = OCRManager()


@app.post("/ocr")
async def ocr(request: Request) -> dict[str, list[str]]:
    """Performs OCR on image files using batch processing.

    Args:
        request: The API request. Contains a list of image files, encoded in
            base-64. # Updated docstring

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` OCR texts, in the same order as which appears in `request`.
    """
    try:
        inputs_json = await request.json()

        # Collect all image bytes in a batch
        image_bytes_list = [] 
        for instance in inputs_json.get("instances", []):
            # Reads the base-64 encoded image data and decodes it into bytes # Updated comment
            if "b64" in instance:
                image_bytes = base64.b64decode(instance["b64"]) 
                image_bytes_list.append(image_bytes)
        
        # Process all image files in a single batch for better performance
        if image_bytes_list:
            predictions = manager.batch_ocr(image_bytes_list) 
        else:
            predictions = []

        return {"predictions": predictions}
    except Exception as e:
        import logging 
        logging.error(f"Error in OCR endpoint: {str(e)}")
        return {"predictions": [], "error": str(e)}

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for your model."""
    return {"message": "health ok"}
