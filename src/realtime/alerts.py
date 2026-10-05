from __future__ import annotations

from typing import Dict, List


def quality_alert(confidence: float, label: str) -> str:
    if confidence >= 0.90:
        return "High confidence detection. Ready for operational use."
    if confidence >= 0.75:
        return "Moderate confidence. Human review recommended for borderline quality checks."
    return "Low confidence. Manual inspection is recommended before final grading."


def batch_summary(predictions: List[Dict[str, float | str]]) -> Dict[str, object]:
    if not predictions:
        return {"total": 0, "highest_confidence": 0.0, "review_required": 0, "dominant_class": "N/A"}

    labels = [str(item["label"]) for item in predictions]
    highest_confidence = max(float(item["confidence"]) for item in predictions)
    low_confidence = sum(1 for item in predictions if float(item["confidence"]) < 0.75)
    dominant = max(set(labels), key=labels.count)

    return {
        "total": len(predictions),
        "highest_confidence": highest_confidence,
        "review_required": low_confidence,
        "dominant_class": dominant,
    }


def class_guidance(label: str) -> str:
    guidance = {
        "0_NOR": "Normal rice. Keep stock quality with standard moisture control.",
        "1_F&S": "Fusarium and spot damage detected. Inspect storage conditions.",
        "2_SD": "Seed damage suspected. Sort and rescreen before milling.",
        "3_MY": "Mycotoxin risk risk area. Increase quarantine sampling.",
        "4_AP": "Aflatoxin risk: escalate quality review.",
        "5_BN": "Broken grain detected. Consider milling yield optimization.",
        "6_UN": "Undergraded rice. Review nutritional and appearance criteria.",
        "7_IM": "Immature grain identified. Separate from premium stock.",
    }
    return guidance.get(label, "General grading review recommended.")
