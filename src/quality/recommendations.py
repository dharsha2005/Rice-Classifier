"""Rule-based quality-action helpers applied after ML classification.

The trained model only predicts a class and an uncalibrated probability.
These functions never mark rice as unsafe, edible, or disposable.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Mapping

from src.quality.config import (
    ACTION_ADDITIONAL_SCREENING,
    ACTION_CONTINUE_PROCESSING,
    ACTION_DISPLAY_NAMES,
    ACTION_MANUAL_INSPECTION,
    ACTION_RE_SCREEN_REPROCESS,
    ACTION_SEPARATE_AND_INSPECT,
    BATCH_REPROCESS_THRESHOLD,
    BATCH_REVIEW_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    NORMAL_CLASS_LABEL,
)


def _as_probability(value: Any) -> float:
    return float(value)


def is_normal_class(predicted_class: str) -> bool:
    return str(predicted_class) == NORMAL_CLASS_LABEL


def get_review_status(
    model_probability: float,
    *,
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Return whether the uncalibrated model probability requires human review."""
    probability = _as_probability(model_probability)
    review_required = probability < float(threshold)
    return {
        "review_required": review_required,
        "needs_review": review_required,
        "threshold": float(threshold),
        "status_label": "Review required" if review_required else "Ready",
        "message": (
            "The prediction is uncertain. Human inspection is recommended before taking further action."
            if review_required
            else "The current prediction is above the decision-support probability threshold."
        ),
    }


def get_quality_action(
    predicted_class: str,
    model_probability: float,
    *,
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Map a single-sample ML prediction to a rule-based operator recommendation.

    Disposal is never returned automatically from the model prediction.
    """
    label = str(predicted_class)
    probability = _as_probability(model_probability)
    review = get_review_status(probability, threshold=threshold)
    normal = is_normal_class(label)

    if review["review_required"]:
        action_code = ACTION_MANUAL_INSPECTION
        classification_status = "Uncertain prediction — human verification recommended"
        reason = "The prediction is uncertain. Human inspection is recommended before taking further action."
        next_step = "Send the sample for human inspection before taking further action."
        supporting_text = reason
    elif normal:
        action_code = ACTION_CONTINUE_PROCESSING
        classification_status = "Normal class"
        reason = "The sample is classified as the normal class."
        next_step = "Continue the standard processing workflow."
        supporting_text = "The sample is classified as a normal rice class. Continue the standard processing workflow."
    else:
        action_code = ACTION_SEPARATE_AND_INSPECT
        classification_status = "Defect Class Detected"
        reason = "A visual defect class has been detected. Further inspection is recommended."
        next_step = "Send the sample/batch for manual quality inspection or sorting."
        supporting_text = "A visual defect class has been detected. Separate this sample/batch for further quality inspection."

    return {
        "predicted_class": label,
        "model_probability": probability,
        "review_required": review["review_required"],
        "needs_review": review["needs_review"],
        "classification_status": classification_status,
        "recommended_action": action_code,
        "recommended_action_label": ACTION_DISPLAY_NAMES[action_code],
        "reason": reason,
        "next_step": next_step,
        "supporting_text": supporting_text,
        "disposal_eligible": False,
        "threshold_note": (
            "Decision-support thresholds / configurable operational thresholds. "
            "These are not validated food-safety limits. "
            "The model probability is an uncalibrated model probability."
        ),
    }


def calculate_batch_quality(
    predictions: Iterable[Mapping[str, Any]],
    *,
    probability_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    review_threshold: float = BATCH_REVIEW_THRESHOLD,
    reprocess_threshold: float = BATCH_REPROCESS_THRESHOLD,
) -> Dict[str, Any]:
    """Summarize a batch of ML predictions and return a rule-based batch recommendation."""
    rows = []
    for item in predictions:
        predicted_class = str(item.get("predicted_class") or item.get("Predicted Class") or "")
        if not predicted_class or predicted_class == "ERROR":
            continue
        probability = item.get("model_probability", item.get("confidence", item.get("Confidence")))
        if probability is None:
            continue
        probability = float(probability)
        if probability > 1.0:
            probability = probability / 100.0
        rows.append({"predicted_class": predicted_class, "model_probability": probability})

    total = len(rows)
    class_counts = Counter(item["predicted_class"] for item in rows)
    normal_count = int(class_counts.get(NORMAL_CLASS_LABEL, 0))
    defect_count = total - normal_count
    defect_rate = (defect_count / total) if total else 0.0
    review_count = sum(
        1 for item in rows if get_review_status(item["model_probability"], threshold=probability_threshold)["review_required"]
    )

    if total == 0:
        action_code = ACTION_MANUAL_INSPECTION
        action_label = "Manual Inspection"
        reason = "No successful classifications are available for this batch. Human inspection is recommended."
    elif defect_rate >= float(reprocess_threshold):
        action_code = ACTION_RE_SCREEN_REPROCESS
        action_label = ACTION_DISPLAY_NAMES[ACTION_RE_SCREEN_REPROCESS]
        reason = (
            "The batch contains a relatively high proportion of defect-class predictions. "
            "Further screening or reprocessing is recommended."
        )
    elif defect_rate >= float(review_threshold):
        action_code = ACTION_ADDITIONAL_SCREENING
        action_label = ACTION_DISPLAY_NAMES[ACTION_ADDITIONAL_SCREENING]
        reason = (
            "The batch defect-class rate is above the additional-screening decision-support threshold. "
            "Additional screening is recommended."
        )
    else:
        action_code = ACTION_CONTINUE_PROCESSING
        action_label = ACTION_DISPLAY_NAMES[ACTION_CONTINUE_PROCESSING]
        reason = (
            "The batch defect-class rate is within the continue-processing decision-support range. "
            "Continue the standard processing workflow."
        )

    return {
        "total_samples": total,
        "normal_count": normal_count,
        "defect_count": defect_count,
        "defect_rate": defect_rate,
        "class_distribution": dict(class_counts),
        "review_required_count": review_count,
        "recommended_action": action_code,
        "recommended_action_label": action_label,
        "reason": reason,
        "review_threshold": float(review_threshold),
        "reprocess_threshold": float(reprocess_threshold),
        "probability_threshold": float(probability_threshold),
        "threshold_note": (
            "Decision-support thresholds / configurable operational thresholds. "
            "These are not validated food-safety limits and are not a model-issued "
            "food-safety determination."
        ),
    }


def attach_quality_recommendation(
    result: Mapping[str, Any],
    *,
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Return a copy of a prediction dict plus recommendation fields. Does not alter predicted_class."""
    probability = float(result.get("confidence", result.get("model_probability", 0.0)))
    action = get_quality_action(str(result["predicted_class"]), probability, threshold=threshold)
    enriched = dict(result)
    enriched["confidence"] = probability
    enriched["model_probability"] = probability
    enriched["review_required"] = action["review_required"]
    enriched["needs_review"] = action["needs_review"]
    enriched["recommended_action"] = action["recommended_action"]
    enriched["recommended_action_label"] = action["recommended_action_label"]
    enriched["classification_status"] = action["classification_status"]
    enriched["quality_reason"] = action["reason"]
    enriched["quality_next_step"] = action["next_step"]
    return enriched
