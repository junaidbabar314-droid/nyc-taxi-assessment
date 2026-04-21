"""
Export utilities for generating downloadable assessment reports.

Provides CSV and PDF export functions for each of the three analysis
modules plus a combined governance summary. Follows progressive-disclosure
principles: the dashboard shows summaries, while exports provide the
full detail for archival and audit purposes (Few, 2006).

References:
    Few, S. (2006) Information Dashboard Design. Analytics Press.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import pandas as pd


# ─── CSV exports ────────────────────────────────────────────────────────

def export_quality_csv(quality_results: dict) -> bytes:
    """
    Serialise quality assessment results to CSV bytes.

    Parameters:
        quality_results: Dict returned by get_quality_metrics().

    Returns:
        UTF-8 encoded CSV bytes suitable for st.download_button.
    """
    rows = []

    # Overall score
    rows.append({
        "category": "overall",
        "metric": "overall_score",
        "value": quality_results.get("overall_score", "N/A"),
    })

    # Sub-dimension scores
    for dim in ("completeness", "accuracy", "consistency", "timeliness"):
        score = quality_results.get("metrics", {}).get(dim, "N/A")
        rows.append({"category": dim, "metric": f"{dim}_score", "value": score})

    # Field-level scores if available
    field_scores = quality_results.get("field_scores")
    if field_scores is not None and isinstance(field_scores, pd.DataFrame):
        for _, row in field_scores.iterrows():
            for col in field_scores.columns:
                if col.lower() not in ("field", "column"):
                    rows.append({
                        "category": "field_detail",
                        "metric": f"{row.iloc[0]}_{col}",
                        "value": row[col],
                    })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def export_privacy_csv(privacy_results: dict) -> bytes:
    """
    Serialise privacy assessment results to CSV bytes.

    Parameters:
        privacy_results: Dict returned by get_privacy_assessment().

    Returns:
        UTF-8 encoded CSV bytes.
    """
    rows = []

    scalar_keys = [
        "overall_risk_score", "risk_level", "uniqueness_percentage",
        "avg_entropy", "linkage_rate",
    ]
    for key in scalar_keys:
        val = privacy_results.get(key, "N/A")
        rows.append({"category": "summary", "metric": key, "value": val})

    # PII fields
    pii = privacy_results.get("pii_fields", [])
    for field in pii:
        rows.append({"category": "pii", "metric": "pii_field", "value": field})

    # K-anonymity summary
    k_anon = privacy_results.get("k_anonymity_summary")
    if isinstance(k_anon, dict):
        for k, v in k_anon.items():
            rows.append({"category": "k_anonymity", "metric": str(k), "value": v})

    # Sensitivity analysis
    sens = privacy_results.get("sensitivity_analysis")
    if isinstance(sens, pd.DataFrame):
        for _, row in sens.iterrows():
            for col in sens.columns:
                rows.append({
                    "category": "sensitivity",
                    "metric": f"{row.iloc[0]}_{col}",
                    "value": row[col],
                })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def export_security_csv(security_results: dict) -> bytes:
    """
    Serialise security compliance results to CSV bytes.

    Parameters:
        security_results: Dict returned by get_security_checklist().

    Returns:
        UTF-8 encoded CSV bytes.
    """
    rows = []

    rows.append({
        "category": "summary",
        "metric": "overall_compliance",
        "value": security_results.get("overall_compliance", "N/A"),
    })

    # Compliance scores by framework
    scores = security_results.get("compliance_scores", {})
    if isinstance(scores, dict):
        for framework, score in scores.items():
            rows.append({
                "category": "framework_score",
                "metric": framework,
                "value": score,
            })

    # Checklist results
    checklist = security_results.get("checklist_results")
    if isinstance(checklist, pd.DataFrame):
        for _, row in checklist.iterrows():
            for col in checklist.columns:
                rows.append({
                    "category": "checklist",
                    "metric": f"{row.iloc[0]}_{col}" if len(checklist.columns) > 1 else str(row.iloc[0]),
                    "value": row[col],
                })

    # Gap summary
    gaps = security_results.get("gap_summary", [])
    if isinstance(gaps, list):
        for i, gap in enumerate(gaps):
            rows.append({"category": "gap", "metric": f"gap_{i+1}", "value": gap})
    elif isinstance(gaps, str):
        rows.append({"category": "gap", "metric": "gap_summary", "value": gaps})

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def generate_summary_pdf(
    quality: Optional[dict] = None,
    privacy: Optional[dict] = None,
    security: Optional[dict] = None,
) -> bytes:
    """
    Generate a combined governance summary PDF.

    Uses reportlab if available; falls back to a text-based PDF-like
    representation if reportlab is not installed.

    Parameters:
        quality:  Results dict from Module 1 (or None).
        privacy:  Results dict from Module 2 (or None).
        security: Results dict from Module 3 (or None).

    Returns:
        PDF file bytes.
    """
    try:
        return _generate_pdf_reportlab(quality, privacy, security)
    except ImportError:
        # Fallback: generate a plain-text report as bytes
        return _generate_text_report(quality, privacy, security)


def _generate_pdf_reportlab(
    quality: Optional[dict],
    privacy: Optional[dict],
    security: Optional[dict],
) -> bytes:
    """Generate PDF using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=18, spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"],
        fontSize=14, spaceAfter=8, textColor=colors.HexColor("#2c3e50"),
    )
    body_style = styles["Normal"]

    elements = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Title
    elements.append(Paragraph(
        "NYC Taxi Data Governance Assessment Report", title_style
    ))
    elements.append(Paragraph(f"Generated: {now}", body_style))
    elements.append(Spacer(1, 12))

    # Summary table
    summary_data = [["Module", "Key Metric", "Value"]]

    if quality:
        summary_data.append([
            "Data Quality",
            "Overall Score",
            f"{quality.get('overall_score', 'N/A'):.1f}%"
            if isinstance(quality.get("overall_score"), (int, float))
            else str(quality.get("overall_score", "N/A")),
        ])
    if privacy:
        summary_data.append([
            "Privacy Risk",
            "Risk Level",
            str(privacy.get("risk_level", "N/A")),
        ])
    if security:
        summary_data.append([
            "Security Compliance",
            "Overall Compliance",
            f"{security.get('overall_compliance', 'N/A'):.1f}%"
            if isinstance(security.get("overall_compliance"), (int, float))
            else str(security.get("overall_compliance", "N/A")),
        ])

    if len(summary_data) > 1:
        elements.append(Paragraph("Executive Summary", heading_style))
        t = Table(summary_data, colWidths=[120, 120, 100])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))

    # Quality detail
    if quality:
        elements.append(Paragraph("Data Quality Assessment", heading_style))
        metrics = quality.get("metrics", {})
        for dim in ("completeness", "accuracy", "consistency", "timeliness"):
            val = metrics.get(dim, "N/A")
            val_str = f"{val:.1f}%" if isinstance(val, (int, float)) else str(val)
            elements.append(Paragraph(f"  {dim.title()}: {val_str}", body_style))
        if quality.get("summary_text"):
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(quality["summary_text"], body_style))
        elements.append(Spacer(1, 12))

    # Privacy detail
    if privacy:
        elements.append(Paragraph("Privacy Risk Assessment", heading_style))
        elements.append(Paragraph(
            f"  Risk Level: {privacy.get('risk_level', 'N/A')}", body_style
        ))
        elements.append(Paragraph(
            f"  Uniqueness: {privacy.get('uniqueness_percentage', 'N/A')}%", body_style
        ))
        elements.append(Paragraph(
            f"  Linkage Rate: {privacy.get('linkage_rate', 'N/A')}", body_style
        ))
        if privacy.get("summary_text"):
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(privacy["summary_text"], body_style))
        elements.append(Spacer(1, 12))

    # Security detail
    if security:
        elements.append(Paragraph("Security Compliance Assessment", heading_style))
        if security.get("summary_text"):
            elements.append(Paragraph(security["summary_text"], body_style))
        elements.append(Spacer(1, 12))

    # Footer
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "MSc IT (Data Analysis) - University of the West of Scotland",
        ParagraphStyle("Footer", parent=body_style, fontSize=8, textColor=colors.grey),
    ))

    doc.build(elements)
    return buffer.getvalue()


def _generate_text_report(
    quality: Optional[dict],
    privacy: Optional[dict],
    security: Optional[dict],
) -> bytes:
    """Fallback plain-text report when reportlab is unavailable."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append("=" * 60)
    lines.append("NYC TAXI DATA GOVERNANCE ASSESSMENT REPORT")
    lines.append(f"Generated: {now}")
    lines.append("=" * 60)
    lines.append("")

    if quality:
        lines.append("--- DATA QUALITY ---")
        lines.append(f"Overall Score: {quality.get('overall_score', 'N/A')}")
        for dim in ("completeness", "accuracy", "consistency", "timeliness"):
            val = quality.get("metrics", {}).get(dim, "N/A")
            lines.append(f"  {dim.title()}: {val}")
        lines.append("")

    if privacy:
        lines.append("--- PRIVACY RISK ---")
        lines.append(f"Risk Level: {privacy.get('risk_level', 'N/A')}")
        lines.append(f"Risk Score: {privacy.get('overall_risk_score', 'N/A')}")
        lines.append(f"Uniqueness: {privacy.get('uniqueness_percentage', 'N/A')}%")
        lines.append("")

    if security:
        lines.append("--- SECURITY COMPLIANCE ---")
        lines.append(f"Overall Compliance: {security.get('overall_compliance', 'N/A')}")
        lines.append("")

    lines.append("-" * 60)
    lines.append("MSc IT (Data Analysis) - University of the West of Scotland")

    return "\n".join(lines).encode("utf-8")
