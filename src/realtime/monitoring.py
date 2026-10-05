from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PredictionMonitor:
    """Simple in-memory realtime monitor for tracking predictions and confidence."""

    predictions: List[Dict[str, float | str]] = field(default_factory=list)

    @property
    def total_predictions(self) -> int:
        return len(self.predictions)

    @property
    def latest_label(self) -> Optional[str]:
        if not self.predictions:
            return None
        return str(self.predictions[-1]["label"])

    @property
    def average_confidence(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(float(item["confidence"]) for item in self.predictions) / len(self.predictions)

    def record_prediction(self, label: str, confidence: float) -> None:
        self.predictions.append({"label": label, "confidence": float(confidence)})

    def risk_summary(self) -> Dict[str, str | float]:
        if not self.predictions:
            return {"status": "No predictions yet", "average_confidence": 0.0}

        lowest = min(self.predictions, key=lambda item: float(item["confidence"]))
        high_confidence = sum(1 for item in self.predictions if float(item["confidence"]) >= 0.75)
        summary = {
            "status": "Healthy" if highest_confidence(self.predictions) >= 0.75 else "Needs review",
            "average_confidence": self.average_confidence,
            "latest_label": self.latest_label,
            "lowest_confidence_label": str(lowest["label"]),
            "high_confidence_count": high_confidence,
        }
        if self.latest_label is not None:
            summary[str(self.latest_label)] = str(self.latest_label)
        return summary


def highest_confidence(predictions: List[Dict[str, float | str]]) -> float:
    if not predictions:
        return 0.0
    return max(float(item["confidence"]) for item in predictions)
