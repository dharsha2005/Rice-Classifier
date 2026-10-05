from pathlib import Path
import unittest
import numpy as np
from PIL import Image

from src.inference.agronomic_explainer import generate_agronomic_diagnosis, translate_feature_name
from src.inference.multi_grain import compute_commercial_grade, detect_and_crop_grains
from src.inference.saliency import generate_activation_heatmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = next((PROJECT_ROOT / "rice" / "test" / "0_NOR").glob("*.png"))


class TestExtendedFeatures(unittest.TestCase):
    def test_translate_feature_name(self):
        meta_shape = translate_feature_name("shape_eccentricity")
        self.assertIn("Slenderness", meta_shape["term"])
        self.assertEqual(meta_shape["category"], "Morphology")

        meta_color = translate_feature_name("color_lab_b_mean")
        self.assertEqual(meta_color["category"], "Color")

        meta_deep = translate_feature_name("deep_feature_42")
        self.assertEqual(meta_deep["category"], "Deep Learned Representation")

    def test_generate_agronomic_diagnosis(self):
        diag = generate_agronomic_diagnosis(
            predicted_class="1_F&S",
            confidence=0.91,
            positive_features=[("color_lab_b_mean", 0.45), ("shape_eccentricity", 0.20)],
            negative_features=[],
        )
        self.assertIn("Fusarium", diag["class_name"])
        self.assertIn("summary", diag)
        self.assertGreater(len(diag["key_agronomic_factors"]), 0)

    def test_saliency_heatmap_generation(self):
        res = generate_activation_heatmap(SAMPLE_IMAGE)
        self.assertIn("original_rgb", res)
        self.assertIn("heatmap_rgb", res)
        self.assertIn("overlay_rgb", res)
        self.assertEqual(res["overlay_rgb"].shape, (224, 224, 3))
        self.assertGreaterEqual(res["peak_intensity"], 0.0)

    def test_compute_commercial_grade(self):
        # High quality sample
        grade_premium = compute_commercial_grade(100, {"0_NOR": 95, "5_BN": 3, "1_F&S": 2})
        self.assertIn("Grade 1", grade_premium["commercial_grade"])
        self.assertEqual(grade_premium["sound_kernel_pct"], 95.0)

        # Off-grade sample
        grade_poor = compute_commercial_grade(100, {"0_NOR": 50, "5_BN": 30, "1_F&S": 20})
        self.assertIn("Substandard", grade_poor["commercial_grade"])


if __name__ == "__main__":
    unittest.main()
