import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.inference.rice_gate import gate_model_available, validate_rice_or_reject


class TestInputValidation(unittest.TestCase):
    def test_blank_frame_is_rejected(self):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = validate_rice_or_reject(frame)
        self.assertFalse(result.is_valid)

    def test_obvious_non_rice_object_is_rejected(self):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.rectangle(frame, (45, 75), (180, 150), (220, 40, 40), -1)
        result = validate_rice_or_reject(frame)
        self.assertFalse(result.is_valid)

    def test_setup_message_when_gate_not_trained(self):
        if gate_model_available():
            self.skipTest("Gate model already trained on this machine.")
        result = validate_rice_or_reject(np.zeros((224, 224, 3), dtype=np.uint8) + 120)
        self.assertFalse(result.is_valid)
        self.assertIn("train_rice_gate.py", result.reason)

    def test_valid_rice_image_when_gate_trained(self):
        if not gate_model_available():
            self.skipTest("Train the gate with datasets/rice_gate before running acceptance test.")
        project_root = Path(__file__).resolve().parents[1]
        sample = next((project_root / "rice" / "test" / "0_NOR").glob("*.png"))
        result = validate_rice_or_reject(Image.open(sample).convert("RGB"))
        self.assertTrue(result.is_valid, result.reason)


if __name__ == "__main__":
    unittest.main()
