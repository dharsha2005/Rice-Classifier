from __future__ import annotations

from io import BytesIO
import os
from typing import List

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference.agronomic_explainer import generate_agronomic_diagnosis
from src.inference.multi_grain import inspect_bulk_rice
from src.inference.predict import explain_prediction, predict_rice
from src.inference.rice_gate import gate_model_available, validate_rice_or_reject
from src.inference.saliency import generate_activation_heatmap
from src.quality.recommendations import attach_quality_recommendation, calculate_batch_quality

app = FastAPI(
    title="Rice Quality & Defect Assessment API",
    version="2.0.0",
    description=(
        "Production-ready API using the verified hybrid EfficientNet-B0 and XGBoost pipeline. "
        "Supports single-grain classification, batch inference, visual explainability, and multi-grain bulk grading."
    ),
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def require_api_key(api_key: str | None) -> None:
    """Require a configured API key for prediction requests in deployed environments."""
    expected_key = os.getenv("RICE_API_KEY")
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")


def _read_and_validate_image(file: UploadFile, image_bytes: bytes) -> Image.Image:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only PNG and JPEG rice images are supported.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image must be smaller than 15 MB.")

    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "rice-quality-assessment",
        "version": "2.0.0",
        "gate_model_available": "yes" if gate_model_available() else "no",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, object]:
    """Single grain quality and defect classification."""
    require_api_key(api_key)
    image_bytes = await file.read()
    image = _read_and_validate_image(file, image_bytes)

    gate_info = "verified"
    if gate_model_available():
        gate = validate_rice_or_reject(image)
        if not gate.is_valid:
            raise HTTPException(status_code=422, detail=gate.reason)
    else:
        gate_info = "gate_bypassed_uncalibrated"

    try:
        result = predict_rice(image)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction could not be completed: {exc}") from exc

    quality = attach_quality_recommendation(result)
    return {
        "filename": file.filename or "uploaded_image",
        "predicted_class": quality["predicted_class"],
        "confidence": quality["confidence"],
        "probability": quality["model_probability"],
        "probabilities": result["probabilities"],
        "review_required": quality["review_required"],
        "recommended_action": quality["recommended_action"],
        "feature_count": result["feature_count"],
        "model": "Hybrid EfficientNet-B0 + XGBoost (1,342 features)",
        "gate_status": gate_info,
    }


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, object]:
    """Batch classification for multiple grain images in a single call."""
    require_api_key(api_key)
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch.")

    batch_results = []
    for f in files:
        try:
            content = await f.read()
            img = _read_and_validate_image(f, content)
            res = predict_rice(img)
            quality = attach_quality_recommendation(res)
            batch_results.append({
                "filename": f.filename or "unknown",
                "predicted_class": quality["predicted_class"],
                "confidence": quality["confidence"],
                "probability": quality["model_probability"],
                "review_required": quality["review_required"],
                "recommended_action": quality["recommended_action"],
                "status": "success",
            })
        except Exception as exc:
            batch_results.append({
                "filename": f.filename or "unknown",
                "predicted_class": "ERROR",
                "confidence": 0.0,
                "probability": 0.0,
                "review_required": True,
                "recommended_action": "MANUAL_INSPECTION",
                "status": f"failed: {exc}",
            })

    return {
        "total_processed": len(batch_results),
        "results": batch_results,
        "batch_quality": calculate_batch_quality(batch_results),
    }


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, object]:
    """Explain prediction using SHAP attributions and agronomic domain diagnosis."""
    require_api_key(api_key)
    image_bytes = await file.read()
    image = _read_and_validate_image(file, image_bytes)

    try:
        prediction = predict_rice(image)
        explanation = explain_prediction(image)
        agronomic = generate_agronomic_diagnosis(
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"],
            positive_features=explanation["top_positive_features"],
            negative_features=explanation["top_negative_features"],
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Explanation failed: {exc}") from exc

    return {
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "agronomic_diagnosis": agronomic,
        "top_positive_features": explanation["top_positive_features"],
        "top_negative_features": explanation["top_negative_features"],
    }


@app.post("/multi-grain")
async def multi_grain(
    file: UploadFile = File(...),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, object]:
    """Multi-grain bulk inspection and commercial quality grading."""
    require_api_key(api_key)
    image_bytes = await file.read()
    image = _read_and_validate_image(file, image_bytes)

    try:
        bulk_summary = inspect_bulk_rice(image)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Multi-grain processing failed: {exc}") from exc

    return {
        "filename": file.filename or "bulk_sample",
        "total_grains_detected": bulk_summary["total_grains"],
        "commercial_grade": bulk_summary["metrics"]["commercial_grade"],
        "grade_status": bulk_summary["metrics"]["grade_status"],
        "sound_kernel_pct": bulk_summary["metrics"]["sound_kernel_pct"],
        "broken_rice_pct": bulk_summary["metrics"]["broken_rice_pct"],
        "diseased_defect_pct": bulk_summary["metrics"]["diseased_defect_pct"],
        "immature_pct": bulk_summary["metrics"]["immature_pct"],
        "market_recommendation": bulk_summary["metrics"]["market_recommendation"],
        "class_breakdown": bulk_summary["class_counts"],
    }
