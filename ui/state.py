from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

import streamlit as st

from src.quality.recommendations import attach_quality_recommendation
from src.realtime.monitoring import PredictionMonitor
from src.storage.database import PredictionStore

LABELS = ["0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_UN", "7_IM"]
ROOT = Path(__file__).resolve().parents[1]


def initialize_state() -> None:
    if "store" not in st.session_state:
        st.session_state.store = PredictionStore()
    if "monitor" not in st.session_state:
        st.session_state.monitor = PredictionMonitor()
    if "history" not in st.session_state:
        st.session_state.history = st.session_state.store.recent_predictions()
        for item in st.session_state.history:
            st.session_state.monitor.record_prediction(item["predicted_class"], item["confidence"])
    if "review_log" not in st.session_state:
        st.session_state.review_log = st.session_state.store.recent_corrections()
    if "review_queue" not in st.session_state:
        st.session_state.review_queue = [item for item in st.session_state.history if item.get("needs_review")]
    for key, default in {
        "single_result": None,
        "single_input_id": None,
        "explanation": None,
        "batch_results": None,
        "batch_quality": None,
        "bulk_result": None,
        "review_image": None,
        "last_prediction_id": None,
        "webcam_result": None,
        "webcam_input_id": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


def image_id(data: bytes, source: str) -> str:
    return f"{source}:{hashlib.sha256(data).hexdigest()}"


def record_prediction(result: dict[str, Any], image_name: str, source: str) -> dict[str, Any]:
    enriched = attach_quality_recommendation(result)
    item = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image_name,
        "source": source,
        "predicted_class": enriched["predicted_class"],
        "confidence": float(enriched["confidence"]),
        "model_probability": float(enriched["model_probability"]),
        "needs_review": bool(enriched["needs_review"]),
        "review_required": bool(enriched["review_required"]),
        "recommended_action": enriched["recommended_action"],
        "operator_action": None,
        "operator_notes": None,
    }
    item["id"] = st.session_state.store.save_prediction(item)
    st.session_state.last_prediction_id = item["id"]
    st.session_state.history.insert(0, item)
    st.session_state.history = st.session_state.history[:50]
    st.session_state.monitor.record_prediction(item["predicted_class"], item["confidence"])
    if item["needs_review"]:
        st.session_state.review_queue.insert(0, item)
        st.session_state.review_queue = st.session_state.review_queue[:20]
    return item


def save_operator_decision(prediction_id: int, operator_action: str, operator_notes: str = "") -> None:
    st.session_state.store.save_operator_decision(prediction_id, operator_action, operator_notes)
    for item in st.session_state.history:
        if item.get("id") == prediction_id:
            item["operator_action"] = operator_action
            item["operator_notes"] = operator_notes
            break


def save_correction(item: dict[str, Any], corrected_label: str) -> None:
    correction = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": item["image_name"],
        "original_prediction": item["predicted_class"],
        "corrected_label": corrected_label,
        "confidence": item["confidence"],
        "source": item["source"],
    }
    st.session_state.store.save_correction(correction)
    st.session_state.review_log.insert(0, correction)


def clear_history() -> None:
    st.session_state.store.clear()
    st.session_state.history = []
    st.session_state.review_queue = []
    st.session_state.review_log = []
    st.session_state.monitor = PredictionMonitor()
