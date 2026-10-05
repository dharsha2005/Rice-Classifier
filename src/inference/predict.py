from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import joblib
import numpy as np
import pandas as pd
import shap
import torch
from PIL import Image

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None

from src.features.efficientnet_features import image_to_tensor, load_efficientnet_b0
from src.features.handcrafted_features import extract_handcrafted_features
from src.inference.feature_schema import DEEP_FEATURE_SCHEMA, FEATURE_SCHEMA, HANDCRAFTED_FEATURE_SCHEMA
from src.preprocessing.preprocess import RicePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "hybrid_xgboost" / "hybrid_xgboost_model.joblib"
SCALER_PATH = PROJECT_ROOT / "results" / "models" / "hybrid_xgboost" / "hybrid_scaler.joblib"
TRAIN_HANDCRAFTED_PATH = PROJECT_ROOT / "results" / "features" / "train_handcrafted_features.csv"
CLASS_LABELS = ["0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_UN", "7_IM"]


def load_feature_contract() -> List[str]:
    """Load the exact handcrafted feature order from the saved training CSV."""
    if not TRAIN_HANDCRAFTED_PATH.exists():
        raise FileNotFoundError(f"Missing handcrafted training CSV: {TRAIN_HANDCRAFTED_PATH}")
    columns = list(pd.read_csv(TRAIN_HANDCRAFTED_PATH, nrows=0).columns)
    contract = [column for column in columns if column not in {"image_path", "split", "class_id", "class_name"}]
    if len(contract) != 62:
        raise ValueError(f"Expected 62 handcrafted features, found {len(contract)}")
    if contract != HANDCRAFTED_FEATURE_SCHEMA:
        raise ValueError("Handcrafted feature order mismatch against the verified training schema.")
    return contract


def decode_class_id(class_id: int) -> str:
    """Map a numeric class id (0-7) to the verified label string."""
    if 0 <= int(class_id) < len(CLASS_LABELS):
        return CLASS_LABELS[int(class_id)]
    raise ValueError(f"Unrecognized class id: {class_id}")


def decode_proba_column(column_index: int, model: Any) -> str:
    """Map a predict_proba column index to a class label using model.classes_ order."""
    if model is None or not hasattr(model, "classes_"):
        return decode_class_id(column_index)
    classes = [int(value) for value in model.classes_]
    if not 0 <= column_index < len(classes):
        raise ValueError(f"Probability column index out of range: {column_index}")
    return decode_class_id(classes[column_index])


def decode_prediction(prediction: Any, model: Any | None = None) -> str:
    """Map a model output (label id or predict_proba column index) to the verified class label."""
    if isinstance(prediction, str):
        if prediction in CLASS_LABELS:
            return prediction
        return prediction
    numeric = int(prediction)
    if model is not None and hasattr(model, "classes_"):
        classes = [int(value) for value in model.classes_]
        if numeric in classes:
            return decode_class_id(numeric)
        if 0 <= numeric < len(classes):
            return decode_proba_column(numeric, model)
    if 0 <= numeric < len(CLASS_LABELS):
        return CLASS_LABELS[numeric]
    raise ValueError(f"Unrecognized class index: {prediction}")


def load_xgboost_model() -> Any:
    """Load the saved XGBoost model without retraining."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing final XGBoost model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError(f"Saved model at {MODEL_PATH} is not a valid XGBoostClassifier")
    return model


if st is not None:
    load_xgboost_model = st.cache_resource(load_xgboost_model)


def load_scaler() -> Any:
    """Load the saved StandardScaler without fitting it again."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    if not hasattr(scaler, "transform"):
        raise TypeError(f"Saved scaler at {SCALER_PATH} does not support transform().")
    return scaler


if st is not None:
    load_scaler = st.cache_resource(load_scaler)


def load_efficientnet() -> torch.nn.Module:
    """Load the same ImageNet-pretrained EfficientNet-B0 backbone used for training feature extraction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_dim, _ = load_efficientnet_b0(device)
    if feature_dim != 1280:
        raise ValueError(f"Expected EfficientNet-B0 embedding dimension 1280, got {feature_dim}")
    model.eval()
    return model.to(device)


if st is not None:
    load_efficientnet = st.cache_resource(load_efficientnet)


def _coerce_image_to_bgr(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Convert supported image inputs to a uint8 BGR array."""
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image does not exist: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            return np.asarray(image)[:, :, ::-1].copy()
    if isinstance(image_input, Image.Image):
        rgb = image_input.convert("RGB")
        return np.asarray(rgb)[:, :, ::-1].copy()
    if isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.ndim == 2:
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if arr.shape[-1] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        if arr.shape[-1] == 3:
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        raise ValueError(f"Unsupported image array shape: {arr.shape}")
    raise TypeError(f"Unsupported image input type: {type(image_input)!r}")


def _default_preprocessor() -> RicePreprocessor:
    return RicePreprocessor(
        target_size=(224, 224),
        blur_ksize=(5, 5),
        morph_ksize=(5, 5),
        min_grain_area=500.0,
        margin_ratio=0.05,
        fill_scale=0.88,
    )


def preprocess_for_inference(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """Run the exact training preprocessing pipeline and return the standardized grain representation."""
    preprocessor = _default_preprocessor()
    if isinstance(image_input, (str, Path)) and Path(image_input).exists():
        result = preprocessor.process_image(
            Path(image_input),
            label_id=-1,
            class_name="UNKNOWN",
            relative_path="inference",
        )
    else:
        image_bgr = _coerce_image_to_bgr(image_input)
        if image_bgr.size == 0:
            raise ValueError("Input image is empty.")
        result = preprocessor.process_bgr(image_bgr, image_name="inference", relative_path="inference")

    if not getattr(result, "success", True):
        raise ValueError(getattr(result, "error_message", "Preprocessing failed."))

    standardized = getattr(result, "standardized_grain", None)
    if standardized is None or standardized.shape[:2] != (224, 224):
        raise ValueError("Preprocessing did not produce a 224x224 standardized grain image.")
    if standardized.ndim != 3 or standardized.shape[2] != 3:
        raise ValueError(f"Expected standardized grain image with shape (224, 224, 3), got {standardized.shape}")

    return {
        "original_bgr": result.original_bgr,
        "binary_mask": result.binary_mask,
        "contour": result.contour,
        "segmented_full": result.segmented_full,
        "segmented_cropped": result.segmented_cropped,
        "standardized_grain": standardized,
        "preprocessed_rgb": cv2.cvtColor(standardized, cv2.COLOR_BGR2RGB),
    }


def _validate_handcrafted_vector(vector: np.ndarray, *, check_order: bool = True) -> np.ndarray:
    if vector.shape != (62,):
        raise ValueError(f"Expected handcrafted feature vector shape (62,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("Handcrafted feature vector contains NaN or infinite values.")
    if check_order:
        expected_names = HANDCRAFTED_FEATURE_SCHEMA
        if len(expected_names) != len(vector):
            raise ValueError(f"Expected {len(expected_names)} handcrafted features, found {len(vector)}")
    return vector.astype(np.float64, copy=False)


def _handcrafted_vector_from_processed(processed: Dict[str, Any]) -> np.ndarray:
    original_bgr = processed["original_bgr"]
    mask = processed["binary_mask"]
    contour = processed["contour"]
    if mask is None or contour is None:
        raise ValueError("Preprocessing did not produce a valid grain mask and contour.")
    feature_dict = extract_handcrafted_features(original_bgr, mask, contour)
    missing = [name for name in HANDCRAFTED_FEATURE_SCHEMA if name not in feature_dict]
    if missing:
        raise ValueError(f"Missing handcrafted features: {missing[:10]}")
    vector = np.asarray([float(feature_dict[name]) for name in HANDCRAFTED_FEATURE_SCHEMA], dtype=np.float64)
    return _validate_handcrafted_vector(vector, check_order=True)


def _deep_vector_from_processed(processed: Dict[str, Any]) -> np.ndarray:
    standardized = processed["standardized_grain"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet().to(device)
    tensor = image_to_tensor(standardized, image_size=224).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = model(tensor).cpu().numpy().reshape(-1)
    if features.shape[0] != 1280:
        raise ValueError(f"Expected 1280 deep features, got {features.shape[0]}")
    if not np.isfinite(features).all():
        raise ValueError("Deep feature vector contains NaN or infinite values.")
    return features.astype(np.float32, copy=False)


def extract_handcrafted_features_for_inference(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Extract 62 handcrafted features in the exact training order and validate every step."""
    return _handcrafted_vector_from_processed(preprocess_for_inference(image_input))


def extract_deep_features_for_inference(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Extract the 1280-dimensional EfficientNet-B0 embedding from the same preprocessing output used in training."""
    return _deep_vector_from_processed(preprocess_for_inference(image_input))


def create_hybrid_features(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Create the 1342-dimensional hybrid feature vector in the training order: handcrafted then deep."""
    processed = preprocess_for_inference(image_input)
    handcrafted = _handcrafted_vector_from_processed(processed)
    deep = _deep_vector_from_processed(processed)

    if handcrafted.shape != (62,):
        raise ValueError(f"Expected handcrafted feature vector shape (62,), got {handcrafted.shape}")
    if deep.shape != (1280,):
        raise ValueError(f"Expected deep feature vector shape (1280,), got {deep.shape}")

    hybrid = np.concatenate([handcrafted.astype(np.float64), deep.astype(np.float64)], axis=0)
    if hybrid.shape != (1342,):
        raise ValueError(f"Expected hybrid feature vector shape (1342,), got {hybrid.shape}")
    if not np.isfinite(hybrid).all():
        raise ValueError("Hybrid feature vector contains NaN or infinite values.")
    if list(FEATURE_SCHEMA[:62]) != HANDCRAFTED_FEATURE_SCHEMA:
        raise ValueError("Handcrafted feature order does not match the verified schema.")
    if len(FEATURE_SCHEMA) != 1342:
        raise ValueError(f"Expected 1342 total fused features, found {len(FEATURE_SCHEMA)}")
    return hybrid.astype(np.float64, copy=False)


def _validate_probability_vector(probabilities: np.ndarray, *, expected_classes: int = 8) -> np.ndarray:
    if probabilities.shape != (expected_classes,):
        raise ValueError(f"Probability vector shape mismatch: expected ({expected_classes},), got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability vector contains NaN or infinite values.")
    if np.any(probabilities < 0.0):
        raise ValueError("Probability vector contains negative values.")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Probability vector sum is not approximately 1.0: {probabilities.sum()}")
    return probabilities.astype(np.float64, copy=False)


def predict_rice(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """Run the complete saved pipeline with exact preprocessing, feature fusion, scaling, and XGBoost inference."""
    hybrid_vector = create_hybrid_features(image_input)
    scaler = load_scaler()
    scaled = scaler.transform(hybrid_vector.reshape(1, -1))
    if scaled.shape != (1, 1342):
        raise ValueError(f"Scaled feature shape mismatch: expected (1, 1342), got {scaled.shape}")

    model = load_xgboost_model()
    probabilities = model.predict_proba(scaled)[0]
    probabilities = _validate_probability_vector(probabilities, expected_classes=8)

    predicted_column = int(np.argmax(probabilities))
    predicted_class = decode_proba_column(predicted_column, model)
    confidence = float(probabilities[predicted_column])
    probability_map = {
        decode_proba_column(index, model): float(probabilities[index]) for index in range(len(probabilities))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_map,
        "feature_count": 1342,
        "handcrafted_feature_count": 62,
        "deep_feature_count": 1280,
    }


def explain_prediction(image_input: Union[str, Path, Image.Image, np.ndarray], top_k: int = 10) -> Dict[str, Any]:
    """Attempt a local SHAP explanation for the current prediction, if the runtime supports it safely."""
    model = load_xgboost_model()
    scaler = load_scaler()
    hybrid_vector = create_hybrid_features(image_input)
    scaled = scaler.transform(hybrid_vector.reshape(1, -1))

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled)
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise ValueError(
            "Local SHAP explanation is not available for this multiclass XGBoost model in the current runtime. "
            "The global SHAP results are not a substitute for local explanation on this uploaded image."
        ) from exc

    if isinstance(shap_values, list):
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = np.moveaxis(values, 0, -1)
        class_shap = values[0]
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            class_shap = values[0]
        elif values.ndim == 2:
            class_shap = values.reshape(-1, values.shape[-1])
        else:
            raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

    probabilities = model.predict_proba(scaled)[0]
    predicted_column = int(np.argmax(probabilities))
    class_values = class_shap[:, predicted_column]
    ranked = np.argsort(np.abs(class_values))[::-1][:max(1, top_k)]
    positive = sorted(
        [(FEATURE_SCHEMA[i], float(class_values[i])) for i in ranked if class_values[i] > 0],
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:top_k]
    negative = sorted(
        [(FEATURE_SCHEMA[i], float(class_values[i])) for i in ranked if class_values[i] <= 0],
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:top_k]

    return {
        "predicted_class": decode_proba_column(predicted_column, model),
        "top_positive_features": positive,
        "top_negative_features": negative,
        "message": "Local explanation for this prediction.",
    }


__all__ = [
    "CLASS_LABELS",
    "decode_class_id",
    "decode_proba_column",
    "decode_prediction",
    "load_feature_contract",
    "load_xgboost_model",
    "load_scaler",
    "load_efficientnet",
    "preprocess_for_inference",
    "extract_handcrafted_features_for_inference",
    "extract_deep_features_for_inference",
    "create_hybrid_features",
    "predict_rice",
    "explain_prediction",
]
