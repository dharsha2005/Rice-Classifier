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
            }
        ]
        pdf_bytes = build_prediction_pdf(history, [])
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 500)


if __name__ == "__main__":
    unittest.main()
