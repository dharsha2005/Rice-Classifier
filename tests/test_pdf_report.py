import unittest

from src.reports.pdf_report import build_prediction_pdf


class TestPdfReport(unittest.TestCase):
    def test_builds_non_empty_pdf(self):
        history = [
            {
                "timestamp": "2026-01-01 12:00:00",
                "image_name": "sample.png",
                "source": "single",
                "predicted_class": "0_NOR",
                "confidence": 0.92,
                "needs_review": False,
                "review_required": False,
                "recommended_action": "CONTINUE_PROCESSING",
                "operator_action": "NORMAL_AFTER_INSPECTION",
                "operator_notes": "checked",
            }
        ]
        pdf_bytes = build_prediction_pdf(
            history,
            [],
            batch_quality={"total_samples": 10, "normal_count": 9, "defect_count": 1, "defect_rate": 0.1, "review_required_count": 0, "recommended_action": "ADDITIONAL_SCREENING", "threshold_note": "Decision-support thresholds"},
            batch_operator_action="REPROCESS",
            batch_operator_notes="screen again",
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 500)


if __name__ == "__main__":
    unittest.main()
