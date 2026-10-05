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
            self.assertIsNone(loaded[0]["operator_action"])

            store.save_operator_decision(loaded[0]["id"], "CONFIRMED_DEFECT", "visual check")
            updated = store.get_prediction(loaded[0]["id"])
            self.assertEqual(updated["predicted_class"], "0_NOR")
            self.assertEqual(updated["operator_action"], "CONFIRMED_DEFECT")
            self.assertEqual(updated["operator_notes"], "visual check")

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


    def test_migrates_existing_database_without_deleting_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "legacy.db"
            import sqlite3

            connection = sqlite3.connect(db_path)
            connection.execute(
                """
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    needs_review INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO predictions (timestamp, image_name, source, predicted_class, confidence, needs_review)
                VALUES ('2026-01-01 12:00:00', 'old.png', 'single', '2_SD', 0.8, 0)
                """
            )
            connection.commit()
            connection.close()

            store = PredictionStore(database_path=db_path)
            loaded = store.recent_predictions()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["predicted_class"], "2_SD")
            self.assertIn("recommended_action", loaded[0])
            self.assertIn("operator_action", loaded[0])


if __name__ == "__main__":
    unittest.main()
