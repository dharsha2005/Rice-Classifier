import unittest
from pathlib import Path

import numpy as np

from src.inference.feature_schema import FEATURE_GROUP_COUNTS, FEATURE_SCHEMA, HANDCRAFTED_FEATURE_SCHEMA
from src.inference.predict import (
    CLASS_LABELS,
    create_hybrid_features,
    decode_class_id,
    decode_proba_column,
    load_scaler,
    load_xgboost_model,
    predict_rice,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_ROOT = PROJECT_ROOT / "rice" / "test"


class TestPredictPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_xgboost_model()
        cls.scaler = load_scaler()

    def test_class_label_decoding(self):
        for index, label in enumerate(CLASS_LABELS):
            self.assertEqual(decode_class_id(index), label)
            self.assertEqual(decode_proba_column(index, self.model), label)

    def test_feature_schema_counts(self):
        self.assertEqual(len(HANDCRAFTED_FEATURE_SCHEMA), 62)
        self.assertEqual(FEATURE_GROUP_COUNTS["deep"], 1280)
        self.assertEqual(len(FEATURE_SCHEMA), 1342)
        self.assertEqual(getattr(self.scaler, "n_features_in_", None), 1342)

    def test_hybrid_vector_shape_and_probabilities(self):
        sample = next((TEST_IMAGE_ROOT / "0_NOR").glob("*.png"))
        vector = create_hybrid_features(sample)
        self.assertEqual(vector.shape, (1342,))
        result = predict_rice(sample)
        probs = np.array(list(result["probabilities"].values()), dtype=float)
        self.assertEqual(len(probs), 8)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertIn(result["predicted_class"], CLASS_LABELS)

    def test_predictions_are_not_all_identical(self):
        predictions = set()
        for label in CLASS_LABELS:
            folder = TEST_IMAGE_ROOT / label
            image = next(folder.glob("*.png"))
            predictions.add(predict_rice(image)["predicted_class"])
        self.assertGreater(len(predictions), 1, "All classes collapsed to one prediction.")


if __name__ == "__main__":
    unittest.main()
