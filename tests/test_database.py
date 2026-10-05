import tempfile
import unittest
from pathlib import Path

from src.storage.database import PredictionStore


class TestPredictionStore(unittest.TestCase):
    def test_save_load_and_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            store = PredictionStore(database_path=db_path)
            record = {
                "timestamp": "2026-01-01 12:00:00",
                "image_name": "sample.png",
                "source": "single",
                "predicted_class": "0_NOR",
                "confidence": 0.91,
                "needs_review": False,
            }
            store.save_prediction(record)
            loaded = store.recent_predictions(limit=5)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["predicted_class"], "0_NOR")

            store.save_correction(
                {
                    "timestamp": record["timestamp"],
                    "image_name": record["image_name"],
                    "original_prediction": "2_SD",
                    "corrected_label": "0_NOR",
                    "confidence": 0.4,
                    "source": "single",
                }
            )
            self.assertEqual(len(store.recent_corrections()), 1)

            store.clear()
            self.assertEqual(store.recent_predictions(), [])
            self.assertEqual(store.recent_corrections(), [])


if __name__ == "__main__":
    unittest.main()
