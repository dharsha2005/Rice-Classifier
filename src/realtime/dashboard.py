from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st


def render_monitor_cards(monitor: Any) -> None:
    if monitor is None:
        return

    summary = monitor.risk_summary()
    stats = [
        ("Total predictions", monitor.total_predictions),
        ("Average confidence", f"{summary.get('average_confidence', 0.0):.2%}"),
        ("Latest class", summary.get("lowest_confidence_label", "N/A")),
        ("Status", summary.get("status", "No data")),
    ]

    columns = st.columns(4)
    for index, (label, value) in enumerate(stats):
        with columns[index]:
            st.metric(label, value)


def render_prediction_history(monitor: Any, limit: int = 8) -> None:
    if monitor is None or not monitor.predictions:
        st.info("No prediction history yet. Run a few classifications to monitor realtime quality.")
        return

    history = pd.DataFrame(monitor.predictions[-limit:])
    history.columns = ["Label", "Confidence"]
    history["Confidence"] = history["Confidence"].map(lambda value: f"{float(value):.2%}")
    st.dataframe(history, use_container_width=True)


def render_alert_panel(label: str, confidence: float) -> None:
    from src.realtime.alerts import class_guidance, quality_alert

    st.markdown("### Quality Control Alert")
    st.warning(quality_alert(confidence, label))
    st.info(class_guidance(label))
