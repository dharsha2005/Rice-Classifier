"""Centralized decision-support thresholds for post-classification recommendations.

These are configurable application-level decision-support thresholds and are not
validated food-safety limits. They do not change the trained ML model or its
class labels. Operators should follow applicable quality-control procedures
for any final disposition.
"""

from __future__ import annotations

# Existing raw class label used by the trained model for the normal class.
NORMAL_CLASS_LABEL = "0_NOR"

# Uncalibrated model-probability cutoff for the Manual Inspection recommendation.
# Values below this threshold are treated as uncertain / review-required.
# These are configurable application-level decision-support thresholds and are
# not validated food-safety limits.
LOW_CONFIDENCE_THRESHOLD = 0.60

# Batch defect-prediction rate at or above which additional screening is recommended.
# These are configurable application-level decision-support thresholds and are
# not validated food-safety limits.
BATCH_REVIEW_THRESHOLD = 0.05

# Batch defect-prediction rate at or above which re-screen / reprocess is recommended.
# These are configurable application-level decision-support thresholds and are
# not validated food-safety limits.
BATCH_REPROCESS_THRESHOLD = 0.15

ACTION_CONTINUE_PROCESSING = "CONTINUE_PROCESSING"
ACTION_SEPARATE_AND_INSPECT = "SEPARATE_AND_INSPECT"
ACTION_RE_SCREEN_REPROCESS = "RE_SCREEN_REPROCESS"
ACTION_ADDITIONAL_SCREENING = "ADDITIONAL_SCREENING"
ACTION_MANUAL_INSPECTION = "MANUAL_INSPECTION"
ACTION_DISPOSAL_ALTERNATE_USE = "DISPOSAL_ALTERNATE_USE"

ACTION_DISPLAY_NAMES = {
    ACTION_CONTINUE_PROCESSING: "Continue Processing",
    ACTION_SEPARATE_AND_INSPECT: "Separate & Inspect",
    ACTION_RE_SCREEN_REPROCESS: "Re-screen / Reprocess",
    ACTION_ADDITIONAL_SCREENING: "Additional Screening Recommended",
    ACTION_MANUAL_INSPECTION: "Manual Inspection",
    ACTION_DISPOSAL_ALTERNATE_USE: "Human-confirmed Disposal / Alternate Use",
}

OPERATOR_CONFIRMED_DEFECT = "CONFIRMED_DEFECT"
OPERATOR_NORMAL_AFTER_INSPECTION = "NORMAL_AFTER_INSPECTION"
OPERATOR_REPROCESS = "REPROCESS"
OPERATOR_ALTERNATE_USE = "ALTERNATE_USE"
OPERATOR_DISPOSAL = "DISPOSAL"
OPERATOR_NOT_SURE = "NOT_SURE"

OPERATOR_DECISION_OPTIONS = [
    (OPERATOR_CONFIRMED_DEFECT, "Confirmed Defect"),
    (OPERATOR_NORMAL_AFTER_INSPECTION, "Normal After Inspection"),
    (OPERATOR_REPROCESS, "Reprocess"),
    (OPERATOR_ALTERNATE_USE, "Alternate Use"),
    (OPERATOR_DISPOSAL, "Disposal"),
    (OPERATOR_NOT_SURE, "Not Sure"),
]
