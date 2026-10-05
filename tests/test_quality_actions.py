import unittest

from src.quality.config import (
    ACTION_CONTINUE_PROCESSING,
    ACTION_MANUAL_INSPECTION,
    ACTION_RE_SCREEN_REPROCESS,
    ACTION_SEPARATE_AND_INSPECT,
    BATCH_REPROCESS_THRESHOLD,
    BATCH_REVIEW_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)
from src.quality.recommendations import (
    attach_quality_recommendation,
    calculate_batch_quality,
    get_quality_action,
    get_review_status,
)


class TestQualityActions(unittest.TestCase):
    def test_normal_class_continue_processing(self):
        action = get_quality_action("0_NOR", 0.91)
        self.assertEqual(action["recommended_action"], ACTION_CONTINUE_PROCESSING)
        self.assertFalse(action["review_required"])
        self.assertIn("normal", action["reason"].lower())
        self.assertFalse(action["disposal_eligible"])

    def test_defect_class_separate_and_inspect(self):
        action = get_quality_action("2_SD", 0.7563)
        self.assertEqual(action["recommended_action"], ACTION_SEPARATE_AND_INSPECT)
        self.assertEqual(action["classification_status"], "Defect Class Detected")
        self.assertNotIn("unsafe", action["reason"].lower())
        self.assertNotIn("discard", action["reason"].lower())
        self.assertFalse(action["disposal_eligible"])

    def test_low_probability_manual_inspection(self):
        action = get_quality_action("0_NOR", 0.42)
        self.assertEqual(action["recommended_action"], ACTION_MANUAL_INSPECTION)
        self.assertTrue(action["review_required"])
        defect_uncertain = get_quality_action("2_SD", LOW_CONFIDENCE_THRESHOLD - 0.01)
        self.assertEqual(defect_uncertain["recommended_action"], ACTION_MANUAL_INSPECTION)

    def test_review_status_uses_config_threshold(self):
        self.assertTrue(get_review_status(LOW_CONFIDENCE_THRESHOLD - 0.001)["review_required"])
        self.assertFalse(get_review_status(LOW_CONFIDENCE_THRESHOLD)["review_required"])

    def test_does_not_auto_recommend_disposal(self):
        for label in ["1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_UN", "7_IM"]:
            action = get_quality_action(label, 0.99)
            self.assertNotEqual(action["recommended_action"], "DISPOSAL_ALTERNATE_USE")
            self.assertFalse(action["disposal_eligible"])

    def test_attach_does_not_change_predicted_class(self):
        original = {"predicted_class": "2_SD", "confidence": 0.81, "probabilities": {}}
        enriched = attach_quality_recommendation(original)
        self.assertEqual(enriched["predicted_class"], "2_SD")
        self.assertEqual(original["predicted_class"], "2_SD")
        self.assertIn("recommended_action", enriched)

    def test_batch_low_defect_rate_continue(self):
        rows = [{"predicted_class": "0_NOR", "confidence": 0.9} for _ in range(98)]
        rows.extend([{"predicted_class": "2_SD", "confidence": 0.88} for _ in range(2)])
        summary = calculate_batch_quality(rows)
        self.assertEqual(summary["total_samples"], 100)
        self.assertEqual(summary["defect_count"], 2)
        self.assertAlmostEqual(summary["defect_rate"], 0.02)
        self.assertLess(summary["defect_rate"], BATCH_REVIEW_THRESHOLD)
        self.assertEqual(summary["recommended_action"], ACTION_CONTINUE_PROCESSING)

    def test_batch_mid_defect_rate_additional_screening(self):
        rows = [{"predicted_class": "0_NOR", "confidence": 0.9} for _ in range(90)]
        rows.extend([{"predicted_class": "2_SD", "confidence": 0.88} for _ in range(10)])
        summary = calculate_batch_quality(rows)
        self.assertGreaterEqual(summary["defect_rate"], BATCH_REVIEW_THRESHOLD)
        self.assertLess(summary["defect_rate"], BATCH_REPROCESS_THRESHOLD)
        self.assertEqual(summary["recommended_action"], "ADDITIONAL_SCREENING")

    def test_batch_high_defect_rate_reprocess(self):
        rows = [{"predicted_class": "0_NOR", "confidence": 0.9} for _ in range(80)]
        rows.extend([{"predicted_class": "2_SD", "confidence": 0.88} for _ in range(20)])
        summary = calculate_batch_quality(rows)
        self.assertGreater(summary["defect_rate"], BATCH_REPROCESS_THRESHOLD)
        self.assertEqual(summary["recommended_action"], ACTION_RE_SCREEN_REPROCESS)
        self.assertIn("decision-support", summary["threshold_note"].lower())

    def test_batch_counts_low_probability_reviews(self):
        rows = [
            {"predicted_class": "0_NOR", "confidence": 0.9},
            {"predicted_class": "2_SD", "confidence": 0.3},
            {"Predicted Class": "ERROR", "Confidence": 0.0},
        ]
        summary = calculate_batch_quality(rows)
        self.assertEqual(summary["total_samples"], 2)
        self.assertEqual(summary["review_required_count"], 1)
        self.assertEqual(summary["class_distribution"]["2_SD"], 1)


if __name__ == "__main__":
    unittest.main()
