from pathlib import Path
import os
import unittest
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import api

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = next((PROJECT_ROOT / "rice" / "test" / "0_NOR").glob("*.png"))


class TestApi(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)
        self._original_key = os.environ.get("RICE_API_KEY")
        os.environ.pop("RICE_API_KEY", None)

    def tearDown(self) -> None:
        if self._original_key is None:
            os.environ.pop("RICE_API_KEY", None)
        else:
            os.environ["RICE_API_KEY"] = self._original_key

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_empty_file_rejected(self):
        response = self.client.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")},
        )
        self.assertEqual(response.status_code, 400)

    def test_invalid_image_rejected(self):
        response = self.client.post(
            "/predict",
            files={"file": ("bad.png", b"not-an-image", "image/png")},
        )
        self.assertEqual(response.status_code, 400)

    def test_api_key_required_when_configured(self):
        os.environ["RICE_API_KEY"] = "secret"
        buffer = BytesIO()
        Image.new("RGB", (16, 16), color=(200, 200, 200)).save(buffer, format="PNG")
        buffer.seek(0)
        response = self.client.post(
            "/predict",
            files={"file": ("sample.png", buffer.getvalue(), "image/png")},
        )
        self.assertEqual(response.status_code, 401)

    def test_successful_predict_endpoint(self):
        with open(SAMPLE_IMAGE, "rb") as f:
            response = self.client.post(
                "/predict",
                files={"file": (SAMPLE_IMAGE.name, f.read(), "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_class", data)
        self.assertIn("confidence", data)
        self.assertIn("probabilities", data)
        self.assertEqual(data["feature_count"], 1342)


if __name__ == "__main__":
    unittest.main()
