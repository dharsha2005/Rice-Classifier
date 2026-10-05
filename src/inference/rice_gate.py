"""Binary rice-vs-non-rice gate for webcam/API inputs before eight-class defect classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import cv2
import joblib
import numpy as np
from PIL import Image

from src.inference.predict import load_efficientnet, preprocess_for_inference
from src.features.efficientnet_features import image_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RICE_GATE_DIR = PROJECT_ROOT / "datasets" / "rice_gate"
RICE_DIR = RICE_GATE_DIR / "rice"
NON_RICE_DIR = RICE_GATE_DIR / "non_rice"
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "rice_gate" / "rice_gate_model.joblib"
METADATA_PATH = PROJECT_ROOT / "results" / "models" / "rice_gate" / "rice_gate_metadata.json"
DEFAULT_ACCEPTANCE_THRESHOLD = 0.55

SETUP_MESSAGE = (
    "Rice/non-rice validator is not trained yet. Add real images to "
    f"{RICE_DIR} and {NON_RICE_DIR}, then run: "
    "python src/models/train_rice_gate.py"
)


@dataclass(frozen=True)
class RiceGateResult:
    is_valid: bool
    reason: str
    rice_probability: float = 0.0
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD
    validator_available: bool = False


def gate_is_configured() -> bool:
    """Return True when both class folders exist (non-rice may still be empty until the user adds images)."""
    return RICE_DIR.is_dir() and NON_RICE_DIR.is_dir()


def gate_model_available() -> bool:
    return MODEL_PATH.exists() and METADATA_PATH.exists()


def load_gate_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_gate_bundle() -> tuple[Any, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "classifier" in bundle and "scaler" in bundle:
        return bundle["scaler"], bundle["classifier"]
    if hasattr(bundle, "predict_proba"):
        return None, bundle
    raise TypeError("Saved rice gate artifact must be a classifier or {scaler, classifier} bundle.")


class _GateModelAdapter:
    """Apply saved scaler + classifier as a single predict_proba interface."""

    def __init__(self, scaler: Any, classifier: Any) -> None:
        self.scaler = scaler
        self.classifier = classifier
        self.classes_ = getattr(classifier, "classes_", np.array([0, 1]))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features) if self.scaler is not None else features
        return self.classifier.predict_proba(scaled)


def _coerce_rgb(image_input: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image_input, Image.Image):
        return np.asarray(image_input.convert("RGB"), dtype=np.uint8)
    rgb = np.asarray(image_input)
    if rgb.ndim == 2:
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    return rgb.astype(np.uint8, copy=False)


def _blank_frame_reason(rgb: np.ndarray) -> str | None:
    if rgb.size == 0:
        return "The image is empty."
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if float(gray.max()) == 0.0:
        return "The image has insufficient visual detail (blank or uniform frame)."
    return None


def extract_gate_features(image_input: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
    """1280-d EfficientNet embedding from the same grain preprocessing used for defect classification."""
    processed = preprocess_for_inference(image_input)
    standardized = processed["standardized_grain"]
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_efficientnet().to(device)
    tensor = image_to_tensor(standardized, image_size=224).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = backbone(tensor).cpu().numpy().reshape(1, -1)
    if features.shape[1] != 1280:
        raise ValueError(f"Expected 1280 gate features, got {features.shape[1]}")
    return features.astype(np.float32, copy=False)


def validate_rice_or_reject(
    image_input: Union[Image.Image, np.ndarray, str, Path],
    *,
    acceptance_threshold: float | None = None,
) -> RiceGateResult:
    """Accept only genuine rice images when a trained gate model exists; otherwise reject with setup instructions."""
    rgb = _coerce_rgb(image_input) if not isinstance(image_input, (str, Path)) else None
    if rgb is not None:
        blank_reason = _blank_frame_reason(rgb)
        if blank_reason is not None:
            return RiceGateResult(False, blank_reason, validator_available=gate_model_available())

    if not gate_model_available():
        return RiceGateResult(
            False,
            SETUP_MESSAGE,
            validator_available=False,
            acceptance_threshold=acceptance_threshold or DEFAULT_ACCEPTANCE_THRESHOLD,
        )

    metadata = load_gate_metadata()
    threshold = float(
        acceptance_threshold if acceptance_threshold is not None else metadata.get("acceptance_threshold", DEFAULT_ACCEPTANCE_THRESHOLD)
    )
    scaler, classifier = load_gate_bundle()
    model = _GateModelAdapter(scaler, classifier)
    features = extract_gate_features(image_input)
    probabilities = model.predict_proba(features)[0]
    classes = [int(value) for value in getattr(model, "classes_", [0, 1])]
    if 1 not in classes:
        raise ValueError("Rice gate model must use class label 1 for rice.")
    rice_index = classes.index(1)
    rice_probability = float(probabilities[rice_index])
    if rice_probability < threshold:
        return RiceGateResult(
            False,
            f"Image rejected: rice confidence {rice_probability * 100:.1f}% is below threshold {threshold * 100:.1f}%.",
            rice_probability=rice_probability,
            acceptance_threshold=threshold,
            validator_available=True,
        )
    return RiceGateResult(
        True,
        "Rice image accepted by binary validator.",
        rice_probability=rice_probability,
        acceptance_threshold=threshold,
        validator_available=True,
    )


__all__ = [
    "RICE_DIR",
    "NON_RICE_DIR",
    "SETUP_MESSAGE",
    "RiceGateResult",
    "gate_is_configured",
    "gate_model_available",
    "validate_rice_or_reject",
    "extract_gate_features",
]
