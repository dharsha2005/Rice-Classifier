from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from src.inference.agronomic_explainer import (
    FEATURE_AGRONOMIC_MAP,
    generate_agronomic_diagnosis,
    translate_feature_name,
)
from src.inference.input_validation import SETUP_MESSAGE, validate_rice_image
from src.inference.multi_grain import compute_commercial_grade, inspect_bulk_rice
from src.inference.predict import explain_prediction, predict_rice, preprocess_for_inference
from src.inference.rice_gate import gate_model_available
from src.inference.saliency import generate_activation_heatmap
from src.realtime.alerts import class_guidance, quality_alert
from src.reports.pdf_report import build_prediction_pdf
from src.quality.config import OPERATOR_DECISION_OPTIONS
from src.quality.recommendations import calculate_batch_quality, get_quality_action
from src.storage.database import DATABASE_PATH
from ui.state import LABELS, ROOT, image_id, record_prediction, save_correction, save_operator_decision
from ui.style import page_heading, render_hero


def _probability_table(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Class": label, "Probability": value * 100} for label, value in result["probabilities"].items()]
    ).sort_values("Probability", ascending=False)


def _quality_card_class(action_code: str) -> str:
    mapping = {
        "CONTINUE_PROCESSING": "",
        "SEPARATE_AND_INSPECT": "inspect",
        "MANUAL_INSPECTION": "manual",
        "RE_SCREEN_REPROCESS": "reprocess",
        "ADDITIONAL_SCREENING": "inspect",
        "DISPOSAL_ALTERNATE_USE": "human",
    }
    return mapping.get(action_code, "")


def render_quality_recommendation(predicted_class: str, model_probability: float) -> dict[str, Any]:
    action = get_quality_action(predicted_class, model_probability)
    css = _quality_card_class(action["recommended_action"])
    st.markdown("#### Quality Action Recommendation")
    st.markdown(
        f"""
        <div class="qa-card {css}">
            <div class="qa-kicker">Decision-support recommendation</div>
            <div class="qa-action">Recommended Action: {action["recommended_action_label"]}</div>
            <p><strong>Predicted Class:</strong> {action["predicted_class"]}<br/>
            <strong>Model Probability:</strong> {action["model_probability"]:.2%} (uncalibrated)<br/>
            <strong>Classification Status:</strong> {action["classification_status"]}</p>
            <p><strong>Reason:</strong> {action["reason"]}<br/>
            <strong>Next Step:</strong> {action["next_step"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(action["threshold_note"])
    return action


def render_operator_decision(prediction_id: int | None, key_prefix: str) -> None:
    st.markdown("#### Human Decision")
    st.caption("Operator decisions are stored separately from the ML prediction and never overwrite it.")
    labels = [label for _, label in OPERATOR_DECISION_OPTIONS]
    codes = {label: code for code, label in OPERATOR_DECISION_OPTIONS}
    choice = st.selectbox("Operator Decision", labels, key=f"operator_choice_{key_prefix}")
    notes = st.text_area("Notes", key=f"operator_notes_{key_prefix}", placeholder="Optional inspection notes")
    if choice in {"Disposal", "Alternate Use"}:
        st.warning(
            "Final Action: Disposal / Alternate Use is available only after manual quality inspection. "
            "This is a human confirmation, not an ML determination. "
            "Final disposition should follow applicable quality-control procedures."
        )
    if st.button("Save Decision", type="primary", key=f"operator_save_{key_prefix}"):
        if prediction_id is None:
            st.error("No saved prediction id is available for this sample.")
            return
        save_operator_decision(int(prediction_id), codes[choice], notes or "")
        st.success("Operator decision saved. The original ML prediction was left unchanged.")


def render_result(
    result: dict[str, Any],
    image: Image.Image,
    key_prefix: str,
    show_explain: bool = True,
    prediction_id: int | None = None,  # kept for call-site compatibility; operator form is rendered separately
) -> None:
    _ = prediction_id
    label = result["predicted_class"]
    confidence = result["confidence"]
    action = get_quality_action(label, confidence)
    cards = st.columns(3)
    cards[0].metric("Predicted class", label)
    cards[1].metric("Model probability", f"{confidence:.2%}")
    cards[2].metric("Review status", "Review" if action["review_required"] else "Ready")
    st.warning(quality_alert(confidence, label))
    st.info(class_guidance(label))
    render_quality_recommendation(label, confidence)
    probability_df = _probability_table(result)
    st.dataframe(probability_df, width="stretch", hide_index=True)
    st.bar_chart(probability_df.set_index("Class")["Probability"])
    if show_explain and st.button("Explain Prediction (SHAP + Agronomic)", key=f"explain_{key_prefix}"):
        try:
            st.session_state.explanation = explain_prediction(image)
        except Exception as exc:
            st.session_state.explanation = None
            st.warning("SHAP explanation is unavailable in this runtime.")
            st.caption(str(exc))


def dashboard() -> None:
    render_hero("Rice Quality & Defect Assessment", "A research-oriented workspace for explainable grain inspection.")
    history = st.session_state.history
    monitor = st.session_state.monitor
    st.markdown("#### Operational overview")
    cards = st.columns(4)
    cards[0].metric("Total predictions", len(history))
    cards[1].metric("Review queue", len(st.session_state.review_queue))
    cards[2].metric("Average confidence", f"{monitor.average_confidence:.2%}")
    cards[3].metric("Model status", "Ready" if (ROOT / "results/models/hybrid_xgboost/hybrid_xgboost_model.joblib").exists() else "Missing")
    st.markdown("#### Quick actions")
    actions = st.columns(4)
    actions[0].page_link("Analyze Rice", label="Analyze Rice", icon=":material/search:")
    actions[1].page_link("Batch Analysis", label="Batch Analysis", icon=":material/grid_view:")
    actions[2].page_link("Review Queue", label="Review Queue", icon=":material/rule:")
    actions[3].page_link("Reports", label="Reports", icon=":material/description:")
    st.markdown("#### Recent predictions")
    if history:
        st.dataframe(pd.DataFrame(history[:8]), width="stretch", hide_index=True)
    else:
        st.info("No predictions yet. Start with Analyze Rice.")


def analyze_rice() -> None:
    page_heading("Analyze Rice", "Upload a grain image, inspect segmentation, saliency heatmap, and run hybrid classification.")
    uploaded = st.file_uploader("Upload single grain image", type=["png", "jpg", "jpeg"], key="analyze_upload")
    if uploaded is None:
        st.info("Upload an image to reveal the Analyze Rice action.")
        return
    data = uploaded.getvalue()
    current_id = image_id(data, "upload")
    image = Image.open(uploaded).convert("RGB")
    if st.session_state.single_input_id != current_id:
        st.session_state.single_result = None
        st.session_state.explanation = None
        st.session_state.single_input_id = current_id

    # Visual Triple: Original, Standardized Segmented Grain, and Visual Attention Heatmap
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(image, caption="Original image", width="stretch")
    with c2:
        try:
            processed = preprocess_for_inference(image)
            st.image(processed["preprocessed_rgb"], caption="Segmented grain (224x224)", width="stretch")
        except Exception as exc:
            st.warning(f"Preprocessing preview unavailable: {exc}")
    with c3:
        try:
            saliency = generate_activation_heatmap(image)
            st.image(saliency["overlay_rgb"], caption="Deep Visual Saliency Map", width="stretch")
        except Exception as exc:
            st.caption("Visual saliency available upon analysis.")

    if st.button("Analyze Rice", type="primary", key="analyze_page_button"):
        try:
            st.session_state.single_result = predict_rice(image)
            st.session_state.explanation = None
            record_prediction(st.session_state.single_result, uploaded.name, "single")
        except Exception as exc:
            st.session_state.single_result = None
            st.error(f"Prediction failed: {exc}")

    if st.session_state.single_result is not None:
        st.markdown("#### Prediction result")
        render_result(
            st.session_state.single_result,
            image,
            "analyze_page",
            prediction_id=st.session_state.get("last_prediction_id"),
        )

        if st.session_state.explanation is not None:
            explanation = st.session_state.explanation
            agronomic = generate_agronomic_diagnosis(
                predicted_class=st.session_state.single_result["predicted_class"],
                confidence=st.session_state.single_result["confidence"],
                positive_features=explanation["top_positive_features"],
                negative_features=explanation["top_negative_features"],
            )

            with st.container(border=True):
                st.markdown(f"### Agronomic Quality Profile: {agronomic['class_name']}")
                st.markdown(f"**Commercial Impact:** `{agronomic['grade_impact']}`")
                st.markdown(f"**Probable Agricultural Cause:** {agronomic['probable_cause']}")
                st.markdown(f"**Recommended Handling Action:** {agronomic['recommended_action']}")
                st.info(agronomic["summary"])

            st.markdown("#### Local feature attribution breakdown")
            p_df = pd.DataFrame(
                [
                    {
                        "Feature": item[0],
                        "Agronomic Meaning": translate_feature_name(item[0])["term"],
                        "Category": translate_feature_name(item[0])["category"],
                        "SHAP Impact": item[1],
                    }
                    for item in explanation["top_positive_features"]
                ]
            )
            n_df = pd.DataFrame(
                [
                    {
                        "Feature": item[0],
                        "Agronomic Meaning": translate_feature_name(item[0])["term"],
                        "Category": translate_feature_name(item[0])["category"],
                        "SHAP Impact": item[1],
                    }
                    for item in explanation["top_negative_features"]
                ]
            )
            left_col, right_col = st.columns(2)
            with left_col:
                st.success("Positive factors (supporting prediction)")
                st.dataframe(p_df, width="stretch", hide_index=True)
            with right_col:
                st.error("Negative factors (opposing prediction)")
                st.dataframe(n_df, width="stretch", hide_index=True)

        render_operator_decision(st.session_state.get("last_prediction_id"), "analyze_page")


def multi_grain_page() -> None:
    page_heading("Multi-Grain Bulk Inspection", "Assess bulk samples (petri dishes, grading trays) and calculate official commercial grades.")
    uploaded = st.file_uploader("Upload multi-grain sample image", type=["png", "jpg", "jpeg"], key="bulk_grain_upload")

    if uploaded is None:
        st.info("Upload a bulk photo containing multiple rice kernels to run automated multi-grain segmentation and commercial grading.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded bulk rice sample", width="stretch")

    if st.button("Run Multi-Grain Inspection", type="primary", key="btn_run_bulk"):
        with st.spinner("Segmenting kernels and evaluating hybrid features..."):
            try:
                bulk_summary = inspect_bulk_rice(image)
                st.session_state.bulk_result = bulk_summary
            except Exception as exc:
                st.session_state.bulk_result = None
                st.error(f"Bulk inspection failed: {exc}")

    if st.session_state.bulk_result is not None:
        summary = st.session_state.bulk_result
        metrics = summary["metrics"]

        st.markdown("### Commercial Quality Grading Result")
        with st.container(border=True):
            st.markdown(f"## {metrics['commercial_grade']}")
            st.caption(f"Status: **{metrics['grade_status']}** · Sample Size: **{summary['total_grains']} kernels**")
            st.info(metrics["market_recommendation"])

        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Sound Kernels (HRY)", f"{metrics['sound_kernel_pct']:.1f}%")
        kpi_cols[1].metric("Broken Kernels", f"{metrics['broken_rice_pct']:.1f}%")
        kpi_cols[2].metric("Diseased / Damaged", f"{metrics['diseased_defect_pct']:.1f}%")
        kpi_cols[3].metric("Immature / Chalky", f"{metrics['immature_pct']:.1f}%")

        st.markdown("#### Annotated Multi-Grain Inspection View")
        st.image(summary["annotated_rgb"], caption="Annotated Rice Grains (Colored by Defect Class)", width="stretch")

        st.markdown("#### Individual Kernel Inspection Gallery")
        grain_df = pd.DataFrame(
            [
                {
                    "Kernel ID": g["grain_id"],
                    "Predicted Class": g["predicted_class"],
                    "Confidence": f"{g['confidence'] * 100:.1f}%",
                    "Bounding Box (x,y,w,h)": str(g["bbox"]),
                }
                for g in summary["grain_results"]
            ]
        )
        st.dataframe(grain_df, width="stretch", hide_index=True)
        st.download_button(
            "Download Bulk Inspection CSV",
            grain_df.to_csv(index=False).encode(),
            "rice_bulk_inspection.csv",
            "text/csv",
            key="btn_download_bulk_csv",
        )


def batch_analysis() -> None:
    page_heading("Batch Analysis", "Process multiple samples together and export a clean inspection table.")
    files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="batch_page_upload")
    if not files:
        st.info("Choose multiple images to begin batch analysis.")
        return
    if st.button("Analyze Batch", type="primary", key="batch_page_analyze"):
        results = []
        progress = st.progress(0)
        for index, item in enumerate(files, start=1):
            try:
                result = predict_rice(Image.open(item).convert("RGB"))
                saved = record_prediction(result, item.name, "batch")
                action = get_quality_action(result["predicted_class"], result["confidence"])
                results.append(
                    {
                        "Image": item.name,
                        "Predicted Class": result["predicted_class"],
                        "Confidence": round(result["confidence"] * 100, 2),
                        "Needs Review": action["review_required"],
                        "Recommended Action": action["recommended_action_label"],
                        "prediction_id": saved["id"],
                    }
                )
            except Exception as exc:
                st.warning(f"Could not classify {item.name}: {exc}")
                results.append(
                    {
                        "Image": item.name,
                        "Predicted Class": "ERROR",
                        "Confidence": 0.0,
                        "Needs Review": True,
                        "Recommended Action": "Manual Inspection",
                        "prediction_id": None,
                    }
                )
            progress.progress(index / len(files))
        st.session_state.batch_results = results
        st.session_state.batch_quality = calculate_batch_quality(results)
    if st.session_state.batch_results:
        result_df = pd.DataFrame(st.session_state.batch_results)
        display_df = result_df.drop(columns=["prediction_id"], errors="ignore")
        st.dataframe(display_df, width="stretch", hide_index=True)
        st.metric("Images needing review", int(display_df["Needs Review"].sum()) if "Needs Review" in display_df else 0)
        quality = st.session_state.batch_quality or calculate_batch_quality(st.session_state.batch_results)
        _render_batch_quality_review(quality)
        st.download_button("Download batch CSV", display_df.to_csv(index=False).encode(), "rice_batch_results.csv", "text/csv", key="batch_page_csv")
        if "Confidence" in display_df.columns:
            st.bar_chart(display_df.set_index("Image")["Confidence"])


def _render_batch_quality_review(quality: dict[str, Any]) -> None:
    st.markdown("#### Batch Quality Review")
    kpis = st.columns(5)
    kpis[0].metric("Total samples", quality["total_samples"])
    kpis[1].metric("Normal (0_NOR)", quality["normal_count"])
    kpis[2].metric("Defect-class count", quality["defect_count"])
    kpis[3].metric("Defect rate", f"{quality['defect_rate']:.1%}")
    kpis[4].metric("Review required", quality["review_required_count"])
    css = _quality_card_class(quality["recommended_action"])
    st.markdown(
        f"""
        <div class="qa-card {css}">
            <div class="qa-kicker">Configurable operational thresholds</div>
            <div class="qa-action">Recommended Action: {quality["recommended_action_label"]}</div>
            <p>{quality["reason"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(quality["threshold_note"])
    distribution = quality.get("class_distribution") or {}
    if distribution:
        dist_df = pd.DataFrame(
            [{"Class": label, "Count": count} for label, count in sorted(distribution.items())]
        )
        st.bar_chart(dist_df.set_index("Class")["Count"])
    st.markdown("#### Batch Human Decision")
    st.caption("Batch operator decisions are recorded separately from individual ML predictions.")
    labels = [label for _, label in OPERATOR_DECISION_OPTIONS]
    codes = {label: code for code, label in OPERATOR_DECISION_OPTIONS}
    choice = st.selectbox("Operator Decision", labels, key="batch_operator_choice")
    notes = st.text_area("Notes", key="batch_operator_notes")
    if choice in {"Disposal", "Alternate Use"}:
        st.warning(
            "Final Action: Disposal / Alternate Use is available only after manual quality inspection. "
            "The ML model does not mark rice as unsafe."
        )
    if st.button("Save Batch Decision", type="primary", key="batch_operator_save"):
        st.session_state.store.save_batch_review(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_samples": quality["total_samples"],
                "normal_count": quality["normal_count"],
                "defect_count": quality["defect_count"],
                "defect_rate": quality["defect_rate"],
                "review_required_count": quality["review_required_count"],
                "recommended_action": quality["recommended_action"],
                "operator_action": codes[choice],
                "operator_notes": notes or "",
            }
        )
        st.session_state.batch_operator_action = codes[choice]
        st.session_state.batch_operator_notes = notes or ""
        st.success("Batch operator decision saved. Individual ML predictions were not changed.")


def prediction_history() -> None:
    page_heading("Prediction History", "Search and review durable predictions stored by the application.")
    history_df = pd.DataFrame(st.session_state.history)
    if history_df.empty:
        st.info("No prediction history yet.")
        return
    query = st.text_input("Search by image, class, or source", key="history_search")
    if query:
        mask = history_df.astype(str).apply(lambda column: column.str.contains(query, case=False, na=False)).any(axis=1)
        history_df = history_df[mask]
    st.dataframe(history_df, width="stretch", hide_index=True)


def review_queue() -> None:
    page_heading("Review Queue", "Resolve low-confidence results without changing the prediction pipeline.")
    queue = st.session_state.review_queue
    if not queue:
        st.success("The review queue is clear.")
        return
    for index, item in enumerate(queue[:10]):
        with st.container(border=True):
            st.markdown(f"**{item['image_name']}**  ·  {item['predicted_class']}  ·  {item['confidence']:.2%}")
            selected = st.selectbox("Corrected class", LABELS, index=LABELS.index(item["predicted_class"]) if item["predicted_class"] in LABELS else 0, key=f"review_class_{index}_{item['image_name']}")
            if st.button("Save correction", key=f"review_save_{index}_{item['image_name']}"):
                save_correction(item, selected)
                st.success("Correction saved.")


def shap_page() -> None:
    page_heading("SHAP / Explainability", "Review the latest local explanation, global feature importance, and agronomic dictionary.")
    explanation = st.session_state.explanation
    if explanation is None:
        st.info("Run an analysis and choose Explain Prediction on the Analyze Rice page first.")
    else:
        st.caption("SHAP explains why the model made this prediction. It does not decide edible, unsafe, disposal, or reprocessing.")
        st.markdown("#### Local explanation")
        st.caption("Positive contribution supports the prediction; negative contribution opposes it.")
        positive = pd.DataFrame(
            [
                {
                    "Feature": item[0],
                    "Agronomic Meaning": translate_feature_name(item[0])["term"],
                    "Category": translate_feature_name(item[0])["category"],
                    "SHAP value": item[1],
                }
                for item in explanation["top_positive_features"]
            ]
        )
        negative = pd.DataFrame(
            [
                {
                    "Feature": item[0],
                    "Agronomic Meaning": translate_feature_name(item[0])["term"],
                    "Category": translate_feature_name(item[0])["category"],
                    "SHAP value": item[1],
                }
                for item in explanation["top_negative_features"]
            ]
        )
        left, right = st.columns(2)
        with left:
            st.success("Positive contribution")
            st.dataframe(positive, width="stretch", hide_index=True)
        with right:
            st.error("Negative contribution")
            st.dataframe(negative, width="stretch", hide_index=True)

    st.markdown("#### Global feature group importance")
    path = ROOT / "results/models/hybrid_xgboost/hybrid_feature_group_importance.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path), width="stretch", hide_index=True)
    else:
        st.info("No saved global importance artifact is available.")

    with st.expander("Agronomic Feature Glossary (Physical Interpretation)"):
        glossary_df = pd.DataFrame(
            [
                {"Technical Feature": k, "Agronomic Meaning": v["term"], "Category": v["category"], "Physical Definition": v["description"]}
                for k, v in FEATURE_AGRONOMIC_MAP.items()
            ]
        )
        st.dataframe(glossary_df, width="stretch", hide_index=True)


def reports() -> None:
    page_heading("Reports", "Download durable prediction and review records for project documentation.")
    history = st.session_state.history
    if not history and not st.session_state.review_log:
        st.info("Run a prediction before generating reports.")
        return
    history_df = pd.DataFrame(history)
    if not history_df.empty:
        st.download_button("Download prediction history CSV", history_df.to_csv(index=False).encode(), "rice_prediction_history.csv", "text/csv", key="reports_history_csv")
    if st.session_state.review_log:
        review_df = pd.DataFrame(st.session_state.review_log)
        st.download_button("Download review corrections CSV", review_df.to_csv(index=False).encode(), "rice_review_corrections.csv", "text/csv", key="reports_review_csv")
    try:
        pdf = build_prediction_pdf(
            history,
            st.session_state.review_log,
            batch_quality=st.session_state.get("batch_quality"),
            batch_operator_action=st.session_state.get("batch_operator_action"),
            batch_operator_notes=st.session_state.get("batch_operator_notes"),
        )
        st.download_button("Download complete PDF report", pdf, "rice_quality_report.pdf", "application/pdf", key="reports_pdf")
    except RuntimeError as exc:
        st.warning(str(exc))


def webcam() -> None:
    page_heading("Webcam", "Capture a grain sample and validate it before running rice-defect classification.")
    
    if gate_model_available():
        st.success("Rice/Non-Rice Strict Validator: **Active (Enforced)**")
    else:
        st.warning(SETUP_MESSAGE)

    camera = st.camera_input("Capture rice image", key="webcam_page_camera")
    if camera is None:
        st.info("Position a single rice grain in front of your camera against a contrasting background and snap a photo.")
        return

    image = Image.open(camera).convert("RGB")

    gate = validate_rice_image(image)
    if not gate.validator_available:
        st.error("Capture blocked until the rice/non-rice validator is trained.")
    elif not gate.is_valid:
        st.error(f"❌ {gate.reason}")
        st.warning(
            "💡 **Rice Grain Not Detected:** The camera detected a person, background, hand, or non-rice object. "
            "Only genuine rice grains are accepted. "
            "Please hold a single rice grain close to the camera against a contrasting/dark background."
        )
    else:
        st.success(f"✅ Rice grain detected (Rice confidence: {gate.rice_probability:.1%}). Ready for grading.")
        st.image(image, caption="Accepted rice capture", width="stretch")
        current_id = image_id(camera.getvalue(), "camera")
        if st.session_state.webcam_input_id != current_id:
            st.session_state.webcam_result = None
            st.session_state.webcam_input_id = current_id
        if st.button("Analyze Camera Rice", type="primary", key="webcam_analyze"):
            try:
                result = predict_rice(image)
                saved = record_prediction(result, "camera_capture.jpg", "camera")
                st.session_state.explanation = None
                st.session_state.webcam_result = result
                st.session_state.last_prediction_id = saved["id"]
            except Exception as exc:
                st.session_state.webcam_result = None
                st.error(f"Analysis failed: {exc}")
        if st.session_state.webcam_result is not None:
            render_result(st.session_state.webcam_result, image, "webcam", prediction_id=st.session_state.get("last_prediction_id"))
            render_operator_decision(st.session_state.get("last_prediction_id"), "webcam")



def model_performance() -> None:
    page_heading("Model Performance", "Saved evaluation artifacts presented as a compact research dashboard.")
    comparison_path = ROOT / "results/models/model_comparison_final.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.bar_chart(comparison.set_index("Model")[["Accuracy", "Macro F1", "Weighted F1"]].mul(100))
    for title, path in [("Per-class metrics", ROOT / "results/models/hybrid_xgboost/hybrid_per_class_metrics.csv"), ("Ablation results", ROOT / "results/models/hybrid_xgboost/ablation_results.csv"), ("Confusion matrix", ROOT / "results/models/hybrid_xgboost/hybrid_confusion_matrix.csv")]:
        if path.exists():
            with st.expander(title):
                st.dataframe(pd.read_csv(path), width="stretch", hide_index=True)


def system_status() -> None:
    page_heading("System / API Status", "Operational visibility using existing project artifacts and runtime checks.")
    model_path = ROOT / "results/models/hybrid_xgboost/hybrid_xgboost_model.joblib"
    scaler_path = ROOT / "results/models/hybrid_xgboost/hybrid_scaler.joblib"
    statuses = [
        ("Model artifact", model_path.exists()),
        ("Scaler artifact", scaler_path.exists()),
        ("Database", DATABASE_PATH.exists()),
        ("API module", (ROOT / "api.py").exists()),
        ("SHAP import", _shap_available()),
        ("Webcam gate", gate_model_available()),
    ]
    cols = st.columns(3)
    for index, (label, available) in enumerate(statuses):
        cols[index % 3].metric(label, "Ready" if available else "Setup required")
    st.info("Feature dimension: 1342 (62 Handcrafted + 1280 EfficientNet-B0)")


def _shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def about_project() -> None:
    render_hero("About the Project", "A concise view of the research contribution and application scope.", "Final-year project")
    st.markdown("""
    **Objective**  
    Classify rice quality and defect categories from images using a reproducible hybrid computer-vision pipeline.

    **Methodology & Hybrid Architecture**  
    Images are segmented and standardized to 224x224 pixels, then characterized with **62 domain-specific handcrafted features** (14 geometric morphology, 12 GLCM texture, and 36 color statistics across RGB, HSV, and LAB spaces) fused with **1,280 deep embeddings** from an EfficientNet-B0 backbone. The resulting 1,342-dimensional vector is classified using an optimized **XGBoost** model.

    **Explainable AI (XAI)**  
    - **Local Attribution:** SHAP TreeExplainer calculates exact Shapley value attributions for each prediction.
    - **Visual Saliency Maps:** Convolutional activation mapping highlights the physical pixel areas driving the deep feature responses.
    - **Agronomic Translation:** Technical features are mapped directly to post-harvest grain quality parameters (chalkiness, yellowness, slenderness, fissuring).

    **Industrial Capabilities**  
    - **Single Grain Inspection:** Fine-grained 8-class defect diagnostics.
    - **Multi-Grain Bulk Inspection:** Simultaneous kernel segmentation, defect mapping, and Codex/ISO commercial grading (Grade 1 Premium, Grade 2 Standard, Grade 3 Fair, Off-Grade).
    - **Audit & Review Workflow:** Low-probability predictions can enter a human-in-the-loop review queue stored in SQLite. A separate rule-based Quality Action Recommendation layer then suggests Continue Processing, Separate & Inspect, or Manual Inspection. Disposal / alternate use requires human confirmation.
    - **Headless REST API:** Full FastAPI backend for conveyor belt and automated sorter integration.
    """)
