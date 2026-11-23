import io
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import main


class FakeModel:
    def predict(self, batch, verbose=0):
        if batch.shape != (1, 160, 160, 3):
            raise AssertionError(f"Unexpected model input shape: {batch.shape}")
        return np.array([[0.8]], dtype=np.float32)


def make_image():
    image = Image.new("RGB", (64, 48), color=(40, 100, 160))
    content = io.BytesIO()
    image.save(content, format="PNG")
    return content.getvalue()


class PredictionEndpointTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.model_info = {
            "image_size": 160,
            "threshold": 0.5,
            "classes": {"0": "VACANT", "1": "OCCUPIED"},
        }
        self.load_model = patch.object(
            main,
            "load_model",
            return_value=(self.model, self.model_info),
        )
        self.load_model.start()
        self.addCleanup(self.load_model.stop)
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def test_predict_returns_status_probability_and_confidence(self):
        response = self.client.post(
            "/predict",
            files={"file": ("parking.png", make_image(), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "predicted_status": "OCCUPIED",
                "occupied_probability": 0.800000011920929,
                "confidence": 0.800000011920929,
            },
        )

    def test_predict_rejects_non_image_upload(self):
        response = self.client.post(
            "/predict",
            files={"file": ("not-image.txt", b"not an image", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "The uploaded file is not a valid image")
