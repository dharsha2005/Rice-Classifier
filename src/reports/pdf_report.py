from __future__ import annotations

from io import BytesIO
from typing import Any, Iterable, Mapping, Optional


def _action_label(code: object) -> str:
    from src.quality.config import ACTION_DISPLAY_NAMES

    if code is None or str(code).strip() == "":
        return ""
    text = str(code)
    return ACTION_DISPLAY_NAMES.get(text, text.replace("_", " ").title())


def build_prediction_pdf(
    history: Iterable[Mapping[str, object]],
    review_log: Iterable[Mapping[str, object]] = (),
    batch_quality: Optional[Mapping[str, Any]] = None,
    batch_operator_action: Optional[str] = None,
    batch_operator_notes: Optional[str] = None,
) -> bytes:
    """Build a compact audit report from prediction and manual-review records."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF export requires reportlab. Install dependencies from requirements.txt.") from exc

    history_rows = list(history)
    review_rows = list(review_log)
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Rice Quality Assessment Report", styles["Title"]),
        Paragraph("Hybrid EfficientNet-B0 + handcrafted features + XGBoost", styles["Normal"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Total predictions: {len(history_rows)}", styles["Normal"]),
        Paragraph(f"Manual corrections: {len(review_rows)}", styles["Normal"]),
        Paragraph(
            "Model Prediction, Recommended Action, and Human Final Decision are recorded as separate fields. "
            "Recommended actions use configurable operational / decision-support thresholds and are not "
            "validated food-safety limits. Model probability is an uncalibrated model probability.",
            styles["Normal"],
        ),
        Spacer(1, 0.15 * inch),
    ]

    prediction_data = [["Timestamp", "Image", "Source", "Class", "Confidence", "Review"]]
    for item in history_rows:
        prediction_data.append(
            [
                str(item.get("timestamp", "")),
                str(item.get("image_name", ""))[:28],
                str(item.get("source", "")),
                str(item.get("predicted_class", "")),
                f"{float(item.get('confidence', item.get('model_probability', 0.0))) * 100:.2f}%",
                "Yes" if item.get("needs_review", item.get("review_required", False)) else "No",
            ]
        )

    if len(prediction_data) == 1:
        prediction_data.append(["No prediction history", "", "", "", "", ""])

    prediction_table = Table(prediction_data, repeatRows=1, colWidths=[1.05 * inch, 1.4 * inch, 0.6 * inch, 0.65 * inch, 0.7 * inch, 0.45 * inch])
    prediction_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf3f8")]),
            ]
        )
    )
    story.extend([Paragraph("Prediction History", styles["Heading2"]), prediction_table])

    quality_data = [["Image", "Model Prediction", "Model Probability", "Review", "Recommended Action", "Human Final Decision", "Operator Notes"]]
    for item in history_rows:
        quality_data.append(
            [
                str(item.get("image_name", ""))[:22],
                str(item.get("predicted_class", "")),
                f"{float(item.get('model_probability', item.get('confidence', 0.0))) * 100:.2f}%",
                "Yes" if item.get("review_required", item.get("needs_review", False)) else "No",
                _action_label(item.get("recommended_action")),
                str(item.get("operator_action") or ""),
                str(item.get("operator_notes") or "")[:40],
            ]
        )
    if len(quality_data) == 1:
        quality_data.append(["No quality-action records", "", "", "", "", "", ""])

    quality_table = Table(
        quality_data,
        repeatRows=1,
        colWidths=[1.05 * inch, 0.85 * inch, 0.85 * inch, 0.5 * inch, 1.2 * inch, 1.05 * inch, 1.3 * inch],
    )
    quality_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#075b5d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e8f4f0")]),
            ]
        )
    )
    story.extend(
        [
            Spacer(1, 0.2 * inch),
            Paragraph("Quality Action Recommendation", styles["Heading2"]),
            Paragraph(
                "Recommended Action is a rule-based operator aid. It is not a food-safety determination.",
                styles["Normal"],
            ),
            quality_table,
        ]
    )

    if batch_quality:
        story.extend(
            [
                Spacer(1, 0.2 * inch),
                Paragraph("Batch Quality Review", styles["Heading2"]),
                Paragraph(f"Total samples: {batch_quality.get('total_samples', '')}", styles["Normal"]),
                Paragraph(f"Normal-class count: {batch_quality.get('normal_count', '')}", styles["Normal"]),
                Paragraph(f"Defect-class count: {batch_quality.get('defect_count', '')}", styles["Normal"]),
                Paragraph(f"Batch defect rate: {float(batch_quality.get('defect_rate', 0.0)) * 100:.2f}%", styles["Normal"]),
                Paragraph(f"Review-required count: {batch_quality.get('review_required_count', '')}", styles["Normal"]),
                Paragraph(
                    f"Recommended Action: {_action_label(batch_quality.get('recommended_action') or batch_quality.get('recommended_action_label'))}",
                    styles["Normal"],
                ),
                Paragraph(f"Human Final Decision: {batch_operator_action or 'Not recorded'}", styles["Normal"]),
                Paragraph(f"Operator notes: {batch_operator_notes or ''}", styles["Normal"]),
                Paragraph(str(batch_quality.get("threshold_note", "")), styles["Normal"]),
            ]
        )

    if review_rows:
        story.append(Spacer(1, 0.2 * inch))
        review_data = [["Timestamp", "Image", "Original", "Corrected", "Confidence"]]
        for item in review_rows:
            review_data.append(
                [
                    str(item.get("timestamp", "")),
                    str(item.get("image_name", ""))[:30],
                    str(item.get("original_prediction", "")),
                    str(item.get("corrected_label", "")),
                    f"{float(item.get('confidence', 0.0)) * 100:.2f}%",
                ]
            )
        review_table = Table(review_data, repeatRows=1, colWidths=[1.1 * inch, 1.7 * inch, 1.0 * inch, 1.0 * inch, 0.8 * inch])
        review_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7f1d1d")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.extend([Paragraph("Manual Review Corrections", styles["Heading2"]), review_table])

    document.build(story)
    return buffer.getvalue()
