import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_parking_occupancy_model.keras"
MODEL_INFO_PATH = BASE_DIR / "model_info.json"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def load_model():
    import tensorflow as tf

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model_info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    expected_size = int(model_info["image_size"])
    expected_input = (None, expected_size, expected_size, int(model_info["input_channels"]))
    if tuple(model.input_shape) != expected_input or tuple(model.output_shape) != (None, 1):
        raise ValueError("The model input or output shape does not match model_info.json")

    return model, model_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.model_info = load_model()
    yield


app = FastAPI(
    title="Car Parking Occupancy API",
    description="Classifies a parking-space image as vacant or occupied.",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def preprocess_image(content: bytes, image_size: int) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large")
            image = image.convert("RGB").resize(
                (image_size, image_size),
                Image.Resampling.BILINEAR,
            )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid image") from exc

    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


@app.get("/")
def root():
    return {"message": "Car Parking Occupancy API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    content = file.file.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller")

    try:
        batch = preprocess_image(content, app.state.model_info["image_size"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction = np.asarray(app.state.model.predict(batch, verbose=0)).reshape(-1)
    if prediction.size != 1 or not np.isfinite(prediction[0]):
        raise RuntimeError("The model returned an invalid prediction")

    occupied_probability = float(prediction[0])
    if not 0 <= occupied_probability <= 1:
        raise RuntimeError("The model returned a probability outside [0, 1]")

    threshold = float(app.state.model_info["threshold"])
    status = (
        app.state.model_info["classes"]["1"]
        if occupied_probability >= threshold
        else app.state.model_info["classes"]["0"]
    )
    return {
        "predicted_status": status,
        "occupied_probability": occupied_probability,
        "confidence": max(occupied_probability, 1 - occupied_probability),
    }
