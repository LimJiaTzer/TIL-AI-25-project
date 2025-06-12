"""Runs the ASR server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import base64
from fastapi import FastAPI, Request
from asr_manager import ASRManager
import logging # It's good practice to have imports at the top

app = FastAPI()
manager = ASRManager()


@app.post("/asr")
async def asr(request: Request) -> dict[str, list[str]]:
    """Performs ASR on audio files by processing them one by one.

    Args:
        request: The API request. Contains a list of audio files, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` transcriptions, in the same order as which appears in `request`.
        If an error occurs processing any file, the whole request may fail
        returning an empty prediction list and an error.
    """
    try:
        inputs_json = await request.json()

        audio_bytes_list = []
        for instance in inputs_json.get("instances", []):
            # Reads the base-64 encoded audio and decodes it into bytes
            if "b64" in instance:
                audio_bytes = base64.b64decode(instance["b64"])
                audio_bytes_list.append(audio_bytes)
        
        predictions = []
        if audio_bytes_list:
            for single_audio_bytes in audio_bytes_list:
                # Call manager.asr() for each audio file.
                # manager.asr() is expected to be a synchronous method.
                prediction = manager.asr(single_audio_bytes)
                predictions.append(prediction)
        
        # If audio_bytes_list was empty, predictions will also be an empty list.

        return {"predictions": predictions}
    except Exception as e:
        # Using the logging instance from the top of the file.
        # Added exc_info=True for more detailed error logging.
        logging.error(f"Error in ASR endpoint: {str(e)}", exc_info=True)
        return {"predictions": [], "error": str(e)}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for the server."""
    return {"message": "health ok"}
