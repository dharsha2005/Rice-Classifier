from __future__ import annotations

import streamlit as st

from ui.pages import (
    about_project,
    analyze_rice,
    batch_analysis,
    dashboard,
    model_performance,
    multi_grain_page,
    prediction_history,
    reports,
    review_queue,
    shap_page,
    system_status,
    webcam,
)
from ui.state import initialize_state
from ui.style import apply_theme

st.set_page_config(
    page_title="Rice Quality Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
initialize_state()

with st.sidebar:
    st.markdown("## RICE / VISION")
    st.caption("Quality intelligence workspace")
    st.divider()
    st.caption("Hybrid model · 1,342 features")
    st.caption("Final year project build")

pages = {
    "Workspace": [
        st.Page(dashboard, title="Dashboard", icon=":material/dashboard:", default=True),
        st.Page(analyze_rice, title="Analyze Rice", icon=":material/search:"),
        st.Page(multi_grain_page, title="Multi-Grain Inspection", icon=":material/grain:"),
        st.Page(batch_analysis, title="Batch Analysis", icon=":material/grid_view:"),
        st.Page(prediction_history, title="Prediction History", icon=":material/history:"),
        st.Page(review_queue, title="Review Queue", icon=":material/rule:"),
    ],
    "Interpretation": [
        st.Page(shap_page, title="SHAP / Explainability", icon=":material/insights:"),
        st.Page(model_performance, title="Model Performance", icon=":material/monitoring:"),
    ],
    "Operations": [
        st.Page(reports, title="Reports", icon=":material/description:"),
        st.Page(webcam, title="Webcam", icon=":material/photo_camera:"),
        st.Page(system_status, title="System / API Status", icon=":material/health_and_safety:"),
    ],
    "Project": [
        st.Page(about_project, title="About Project", icon=":material/info:"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
