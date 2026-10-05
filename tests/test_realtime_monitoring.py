import unittest

from src.realtime.monitoring import PredictionMonitor


class TestPredictionMonitor(unittest.TestCase):
    def test_tracks_predictions_and_average_confidence(self):
        monitor = PredictionMonitor()
        monitor.record_prediction("0_NOR", 0.93)
        monitor.record_prediction("2_SD", 0.68)

        self.assertEqual(monitor.total_predictions, 2)
        self.assertAlmostEqual(monitor.average_confidence, 0.805, places=3)
        self.assertEqual(monitor.latest_label, "2_SD")
        self.assertIn("2_SD", monitor.risk_summary())


if __name__ == "__main__":
    unittest.main()
