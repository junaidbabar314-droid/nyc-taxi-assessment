"""
Security Compliance Assessment page — Module 3 integration.

Presents security compliance results against NIST CSF, GDPR, and
ISO 27001 frameworks. Displays checklist status, gap analysis, and
remediation priorities. Follows progressive-disclosure design
(Shneiderman, 1996) with summary metrics first and detail on demand.

Author : Iqra Aziz (B01802319)
Module : Jannat Rafique (B01798960) — Security Assessor

References:
    NIST (2018) Framework for Improving Critical Infrastructure
        Cybersecurity, Version 1.1. National Institute of Standards
        and Technology.
    Few, S. (2006) Information Dashboard Design. Analytics Press.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ─── Path setup ─────────────────────────────────────────────────────────
_PHASE2_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE2_ROOT))

from dashboard.utils.styling import (
    apply_custom_css, SECURITY_RED, SECURITY_RED_DARK, WARNING_ORANGE,
    QUALITY_GREEN, NEUTRAL_GREY, PRIVACY_BLUE, PLOTLY_TEMPLATE, CHART_FONT,
    CHART_MARGIN, COMPLIANCE_COLOURS, score_colour,
)
from dashboard.utils.export_reports import export_security_csv
from config import DATA_DIR, NIST_FUNCTIONS

# ─── Page styling (no set_page_config — only Home.py sets it) ──────────
apply_custom_css()

# ─── Module import ──────────────────────────────────────────────────────
_module_available = False
try:
    from module3_security.security_assessor import get_security_checklist
    _module_available = True
except ImportError:
    pass

# ─── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Security Compliance")
    run = st.button(
        "Run Security Assessment",
        type="primary",
        use_container_width=True,
    )
    clear = st.button(
        "Clear cached results",
        use_container_width=True,
        help="Discard any cached results and re-run the assessment from scratch.",
    )
    if clear:
        st.session_state.pop("security_results", None)
        st.rerun()
    st.info(
        "Security assessment analyses file-level controls "
        "(encryption, permissions) and framework compliance."
    )


# ─── Helper: NIST radar chart ──────────────────────────────────────────

def _nist_radar(scores: dict) -> go.Figure:
    """Radar chart for the five NIST CSF functions."""
    functions = NIST_FUNCTIONS
    values = [scores.get(f, 0) for f in functions]
    values_closed = values + [values[0]]
    funcs_closed = functions + [functions[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=funcs_closed,
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.2)",
        line=dict(color=SECURITY_RED, width=2),
        marker=dict(size=8, color=SECURITY_RED),
        name="NIST CSF",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%"),
            bgcolor="rgba(0,0,0,0)",
        ),
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=dict(l=60, r=60, t=40, b=40),
        height=400,
        showlegend=False,
        title="NIST CSF Compliance Radar",
    )
    return fig


# ─── Helper: compliance heatmap ─────────────────────────────────────────

def _compliance_heatmap(matrix_df: pd.DataFrame) -> go.Figure | None:
    """Heatmap of compliance scores across frameworks and controls.

    Handles two shapes:
      1. LONG format (real module):
         columns = [Framework, Requirement, Control_ID, Status, Gap, Remediation]
         Rows are per-framework, per-control records with Status string
         (PASS/PARTIAL/FAIL/N/A).
      2. WIDE numeric format (older fallback):
         first column = label, remaining columns = numeric scores.
    """
    if matrix_df is None or not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
        return None

    cols = matrix_df.columns.tolist()

    status_to_score = {"PASS": 100, "PARTIAL": 50, "FAIL": 0, "N/A": None}
    is_long = {"Framework", "Control_ID", "Status"}.issubset(set(cols))

    if is_long:
        # Pivot Control_ID (rows) x Framework (cols) using numeric Status.
        df = matrix_df.copy()
        df["Score"] = df["Status"].map(status_to_score)
        df = df.dropna(subset=["Score"])
        if df.empty:
            return None
        pivot = (df.pivot_table(index="Control_ID",
                                columns="Framework",
                                values="Score",
                                aggfunc="mean")
                   .sort_index())
        # Ensure consistent framework column order where available.
        preferred = [c for c in ["NIST CSF 2.0", "GDPR", "ISO 27001"]
                     if c in pivot.columns]
        other = [c for c in pivot.columns if c not in preferred]
        pivot = pivot[preferred + other]
        labels = pivot.index.astype(str).tolist()
        data_cols = pivot.columns.tolist()
        z_data = pivot.fillna(-1).values  # -1 renders as grey 'N/A'
        text = [
            [f"{int(v)}%" if v >= 0 else "n/a" for v in row]
            for row in z_data
        ]
    else:
        # Wide numeric fallback.
        if len(cols) < 2:
            return None
        labels = matrix_df.iloc[:, 0].astype(str).tolist()
        data_cols = cols[1:]
        z_data = matrix_df[data_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        text = [[f"{v:.0f}%" for v in row] for row in z_data]

    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=data_cols,
        y=labels,
        colorscale=[[0, SECURITY_RED], [0.5, WARNING_ORANGE], [1, QUALITY_GREEN]],
        zmin=0, zmax=100,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="%"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=dict(l=150, r=40, t=40, b=60),
        height=max(300, len(labels) * 35),
        title="Compliance Matrix",
    )
    return fig


# ─── Helper: gap analysis bar chart ────────────────────────────────────

def _gap_analysis_chart(scores: dict) -> go.Figure | None:
    """Horizontal bar chart showing compliance gaps (100 - score)."""
    if not scores or not isinstance(scores, dict):
        return None

    frameworks = list(scores.keys())
    compliance = [float(v) if isinstance(v, (int, float)) else 0 for v in scores.values()]
    gaps = [100 - c for c in compliance]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=frameworks, x=compliance,
        orientation="h",
        name="Compliant",
        marker_color=QUALITY_GREEN,
        text=[f"{c:.0f}%" for c in compliance],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        y=frameworks, x=gaps,
        orientation="h",
        name="Gap",
        marker_color=SECURITY_RED,
        text=[f"{g:.0f}%" for g in gaps],
        textposition="inside",
        opacity=0.7,
    ))
    fig.update_layout(
        barmode="stack",
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=dict(l=120, r=40, t=50, b=40),
        height=max(250, len(frameworks) * 60),
        title="Gap Analysis by Framework",
        xaxis=dict(range=[0, 100], title="Percentage"),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ─── Helper: framework comparison chart ─────────────────────────────────

def _framework_comparison(scores: dict) -> go.Figure | None:
    """Grouped bar chart comparing GDPR, ISO, NIST scores."""
    if not scores:
        return None

    frameworks = list(scores.keys())
    values = [float(v) if isinstance(v, (int, float)) else 0 for v in scores.values()]
    colours = [PRIVACY_BLUE, WARNING_ORANGE, SECURITY_RED, QUALITY_GREEN, NEUTRAL_GREY]

    fig = go.Figure(go.Bar(
        x=frameworks,
        y=values,
        marker_color=colours[:len(frameworks)],
        text=[f"{v:.0f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=CHART_MARGIN,
        height=350,
        title="Framework Compliance Comparison",
        yaxis=dict(range=[0, 110], title="Compliance %"),
        xaxis_title="Framework",
    )
    return fig


# ─── Helper: checklist table with colour coding ────────────────────────

def _display_checklist(checklist_df: pd.DataFrame):
    """Render checklist with colour-coded status column."""
    if checklist_df is None or not isinstance(checklist_df, pd.DataFrame) or checklist_df.empty:
        st.info("Checklist data not available.")
        return

    # Try to identify the status column
    status_col = None
    for c in checklist_df.columns:
        if "status" in c.lower() or "result" in c.lower() or "pass" in c.lower():
            status_col = c
            break

    if status_col:
        def _colour_status(val):
            val_str = str(val).lower()
            if val_str in ("pass", "compliant", "yes", "true"):
                return f"background-color: {QUALITY_GREEN}; color: white"
            elif val_str in ("partial", "warning"):
                return f"background-color: {WARNING_ORANGE}; color: white"
            elif val_str in ("fail", "non-compliant", "no", "false"):
                return f"background-color: {SECURITY_RED}; color: white"
            return ""

        styled = checklist_df.style.map(_colour_status, subset=[status_col])
        st.dataframe(styled, use_container_width=True, height=400)
    else:
        st.dataframe(checklist_df, use_container_width=True, height=400)


# ─── Fallback: basic security assessment ────────────────────────────────

def _fallback_security() -> dict:
    """
    Generate a basic security assessment when Module 3 is unavailable.

    Checks file-level properties (existence, size) and produces a
    representative compliance structure for the dashboard to render.
    """
    import os

    data_dir = Path(DATA_DIR)
    files = list(data_dir.glob("*.parquet")) if data_dir.exists() else []

    # File permission checks
    perm_rows = []
    for f in files[:12]:  # limit to 12 files
        readable = os.access(f, os.R_OK)
        writable = os.access(f, os.W_OK)
        perm_rows.append({
            "File": f.name,
            "Readable": readable,
            "Writable": writable,
            "Status": "Warning" if writable else "Pass",
        })
    permission_df = pd.DataFrame(perm_rows) if perm_rows else pd.DataFrame()

    # Encryption check (Parquet files are not encrypted by default)
    enc_rows = []
    for f in files[:12]:
        enc_rows.append({
            "File": f.name,
            "Encrypted": False,
            "Format": "Apache Parquet",
            "Status": "Fail",
        })
    encryption_df = pd.DataFrame(enc_rows) if enc_rows else pd.DataFrame()

    # NIST CSF scores (simulated baseline for unprotected data)
    nist_scores = {
        "Identify": 65,
        "Protect": 30,
        "Detect": 20,
        "Respond": 15,
        "Recover": 25,
    }

    # Framework compliance scores
    compliance_scores = {
        "NIST CSF": np.mean(list(nist_scores.values())),
        "GDPR": 35,
        "ISO 27001": 28,
    }

    overall = np.mean(list(compliance_scores.values()))

    # Checklist
    checklist_items = [
        {"Control": "Data-at-rest encryption", "Framework": "NIST/ISO", "Status": "Fail",
         "Notes": "Parquet files are not encrypted"},
        {"Control": "Access control lists", "Framework": "NIST/ISO", "Status": "Partial",
         "Notes": "OS-level permissions only"},
        {"Control": "Data classification", "Framework": "GDPR", "Status": "Partial",
         "Notes": "No formal classification scheme"},
        {"Control": "Audit logging", "Framework": "NIST", "Status": "Fail",
         "Notes": "No audit trail for data access"},
        {"Control": "Data minimisation", "Framework": "GDPR", "Status": "Pass",
         "Notes": "Zone IDs used instead of GPS coordinates"},
        {"Control": "Pseudonymisation", "Framework": "GDPR", "Status": "Pass",
         "Notes": "No direct identifiers in dataset"},
        {"Control": "Incident response plan", "Framework": "NIST", "Status": "Fail",
         "Notes": "No documented plan"},
        {"Control": "Backup and recovery", "Framework": "ISO", "Status": "Partial",
         "Notes": "Source data available from NYC TLC"},
        {"Control": "Network security", "Framework": "NIST", "Status": "Partial",
         "Notes": "Local storage only"},
        {"Control": "Data retention policy", "Framework": "GDPR/ISO", "Status": "Fail",
         "Notes": "No formal retention policy"},
    ]
    checklist_df = pd.DataFrame(checklist_items)

    # Compliance matrix
    matrix_data = {
        "Control Area": ["Access Control", "Encryption", "Audit", "Data Protection", "Incident Response"],
        "NIST": [50, 20, 15, 60, 15],
        "GDPR": [40, 25, 20, 70, 20],
        "ISO 27001": [35, 15, 10, 55, 10],
    }
    compliance_matrix = pd.DataFrame(matrix_data)

    # Gap summary
    gaps = [
        "No data-at-rest encryption for Parquet files",
        "No formal audit logging mechanism",
        "Missing incident response plan",
        "No data retention or disposal policy",
    ]

    # Remediation priorities
    remediation = pd.DataFrame([
        {"Priority": 1, "Control": "Data-at-rest encryption", "Effort": "Medium",
         "Impact": "High", "Framework": "NIST/ISO/GDPR"},
        {"Priority": 2, "Control": "Audit logging", "Effort": "Medium",
         "Impact": "High", "Framework": "NIST"},
        {"Priority": 3, "Control": "Incident response plan", "Effort": "Low",
         "Impact": "Medium", "Framework": "NIST"},
        {"Priority": 4, "Control": "Data retention policy", "Effort": "Low",
         "Impact": "Medium", "Framework": "GDPR"},
    ])

    # Fallback priority counts derived from remediation DataFrame
    _high_fb = int((remediation["Impact"] == "High").sum()) if not remediation.empty else 0
    _med_fb = int((remediation["Impact"] == "Medium").sum()) if not remediation.empty else 0
    _low_fb = int((remediation["Impact"] == "Low").sum()) if not remediation.empty else 0
    _total_fb = _high_fb + _med_fb + _low_fb
    fb_counts = {"High": _high_fb, "Medium": _med_fb, "Low": _low_fb}
    fb_pct = {
        p: round(100 * c / _total_fb, 1) if _total_fb else 0.0
        for p, c in fb_counts.items()
    }

    return {
        "overall_compliance": round(overall, 2),
        "encryption_results": encryption_df,
        "permission_results": permission_df,
        "checklist_results": checklist_df,
        "compliance_matrix": compliance_matrix,
        "compliance_scores": compliance_scores,
        "gap_summary": gaps,
        "gap_counts": fb_counts,
        "gap_percentages": fb_pct,
        "total_gaps": _total_fb,
        "remediation_priorities": remediation,
        "nist_scores": nist_scores,
        "summary_text": (
            f"**Fallback assessment** (Module 3 not available): "
            f"Analysed {len(files)} data files. Overall compliance: {overall:.1f}%. "
            f"Key gaps: no encryption, no audit logging, no incident response plan."
        ),
    }


# ─── Main page ──────────────────────────────────────────────────────────

def main():
    st.markdown("# Security Compliance Assessment")
    st.markdown(
        "Evaluates security posture against **NIST CSF**, **GDPR**, and "
        "**ISO 27001** frameworks. Includes encryption verification, "
        "permission auditing, and compliance gap analysis. "
        "Based on Jannat Rafique's (B01798960) security assessor module."
    )
    st.markdown("---")

    results = st.session_state.get("security_results")

    if run:
        with st.spinner("Running security compliance assessment..."):
            try:
                if _module_available:
                    results = get_security_checklist(data_dir=DATA_DIR)
                else:
                    st.warning(
                        "Module 3 (security_assessor) not yet available. "
                        "Showing fallback assessment."
                    )
                    results = _fallback_security()
            except Exception as exc:
                st.error(f"Security assessment failed: {exc}")
                st.warning("Falling back to basic security analysis.")
                results = _fallback_security()

        st.session_state["security_results"] = results

    if results is None:
        st.info(
            "No security assessment results yet. Click **Run Security Assessment** "
            "in the sidebar to begin."
        )
        return

    # ── Overall compliance metric ───────────────────────────────────────
    overall = results.get("overall_compliance", 0)
    col_metric, col_space = st.columns([1, 2])
    with col_metric:
        if isinstance(overall, (int, float)):
            colour = score_colour(overall)
            st.markdown(
                f'<div style="background:{colour}; color:white; padding:16px; '
                f'border-radius:10px; text-align:center;">'
                f'<p style="margin:0; font-size:0.85rem; text-transform:uppercase; '
                f'letter-spacing:0.05em; font-weight:600;">Overall Compliance</p>'
                f'<p style="margin:0; font-size:2.5rem; font-weight:700;">'
                f'{overall:.1f}%</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.metric("Overall Compliance", "N/A")

    st.markdown("---")

    # ── Charts in tabs ──────────────────────────────────────────────────
    tab_nist, tab_heatmap, tab_gap, tab_compare = st.tabs([
        "NIST CSF Radar", "Compliance Matrix", "Gap Analysis", "Framework Comparison",
    ])

    # NIST radar — prefer per-function breakdown from the checklist.
    # Real module returns compliance_scores with key "nist_compliance_pct";
    # fallback dict uses "NIST CSF".  Handle both.
    nist_scores = results.get("nist_scores", {})
    if not nist_scores:
        checklist = results.get("checklist_results")
        if (
            checklist is not None
            and hasattr(checklist, "columns")
            and "NIST_Function" in getattr(checklist, "columns", [])
            and "Assessment_Result" in checklist.columns
        ):
            for fn in NIST_FUNCTIONS:
                subset = checklist[checklist["NIST_Function"] == fn]
                if len(subset):
                    passes = (subset["Assessment_Result"] == "PASS").sum()
                    partials = (subset["Assessment_Result"] == "PARTIAL").sum()
                    nist_scores[fn] = round(
                        (passes + 0.5 * partials) / len(subset) * 100, 1
                    )
    if not nist_scores:
        cs = results.get("compliance_scores", {})
        nist_val = cs.get("nist_compliance_pct", cs.get("NIST CSF", 0))
        if isinstance(nist_val, (int, float)) and nist_val:
            nist_scores = {f: nist_val for f in NIST_FUNCTIONS}

    with tab_nist:
        if nist_scores:
            fig_nist = _nist_radar(nist_scores)
            st.plotly_chart(fig_nist, use_container_width=True, key="sec_nist")
        else:
            st.info("NIST CSF scores not available.")

    with tab_heatmap:
        fig_hm = _compliance_heatmap(results.get("compliance_matrix"))
        if fig_hm:
            st.plotly_chart(fig_hm, use_container_width=True, key="sec_heatmap")
        else:
            st.info("Compliance matrix not available.")

    # Build a CLEAN framework-scores dict for the gap/comparison charts.
    # The real module's `compliance_scores` dict holds 8 keys (3 framework
    # percentages + overall + 4 gap structures); passing it raw makes the
    # charts render non-numeric dict entries as zero bars.  Filter to the
    # three framework percentages only.
    cs = results.get("compliance_scores", {}) or {}
    framework_scores = {
        "GDPR":      cs.get("gdpr_compliance_pct", cs.get("GDPR")),
        "ISO 27001": cs.get("iso_compliance_pct", cs.get("ISO 27001")),
        "NIST CSF":  cs.get("nist_compliance_pct", cs.get("NIST CSF")),
    }
    framework_scores = {k: v for k, v in framework_scores.items()
                        if isinstance(v, (int, float))}

    with tab_gap:
        fig_gap = _gap_analysis_chart(framework_scores)
        if fig_gap:
            st.plotly_chart(fig_gap, use_container_width=True, key="sec_gap")
        else:
            st.info("Compliance scores not available for gap analysis.")

    with tab_compare:
        fig_comp = _framework_comparison(framework_scores)
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True, key="sec_compare")
        else:
            st.info("Framework scores not available.")

    st.markdown("---")

    # ── Checklist table ─────────────────────────────────────────────────
    st.markdown("### Compliance Checklist")
    with st.expander("View Full Checklist", expanded=True):
        _display_checklist(results.get("checklist_results"))

    # ── Gap summary ─────────────────────────────────────────────────────
    gaps = results.get("gap_summary", [])
    if gaps:
        st.markdown("### Identified Gaps")
        if isinstance(gaps, dict):
            # Real module: {"High": [control_ids], "Medium": [...], "Low": [...]}
            for priority in ("High", "Medium", "Low"):
                ids = gaps.get(priority, [])
                if ids:
                    icon = {"High": "error", "Medium": "warning", "Low": "info"}[priority]
                    label = f"{priority}-priority gaps ({len(ids)}): {', '.join(ids)}"
                    getattr(st, icon)(label)
        elif isinstance(gaps, list):
            for gap in gaps:
                st.warning(f"Gap: {gap}")
        elif isinstance(gaps, str):
            st.warning(gaps)

    # ── Remediation priorities ──────────────────────────────────────────
    pct = results.get("gap_percentages", {}) or {}
    counts = results.get("gap_counts", {}) or {}
    total_gaps = results.get("total_gaps", 0) or 0

    if total_gaps:
        st.markdown("### Gaps by Remediation Priority")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "High",
            f"{pct.get('High', 0):.1f}%",
            f"{counts.get('High', 0)} of {total_gaps} gaps",
        )
        c2.metric(
            "Medium",
            f"{pct.get('Medium', 0):.1f}%",
            f"{counts.get('Medium', 0)} of {total_gaps} gaps",
        )
        c3.metric(
            "Low",
            f"{pct.get('Low', 0):.1f}%",
            f"{counts.get('Low', 0)} of {total_gaps} gaps",
        )

    remediation = results.get("remediation_priorities")
    if remediation is not None and isinstance(remediation, pd.DataFrame) and not remediation.empty:
        st.markdown("### Remediation Priorities")
        st.dataframe(remediation, use_container_width=True, hide_index=True)

    # ── Encryption & permission results ─────────────────────────────────
    st.markdown("---")
    col_enc, col_perm = st.columns(2)
    with col_enc:
        enc = results.get("encryption_results")
        if enc is not None and isinstance(enc, pd.DataFrame) and not enc.empty:
            with st.expander("Encryption Check Results"):
                st.dataframe(enc, use_container_width=True, hide_index=True)
    with col_perm:
        perm = results.get("permission_results")
        if perm is not None and isinstance(perm, pd.DataFrame) and not perm.empty:
            with st.expander("Permission Audit Results"):
                st.dataframe(perm, use_container_width=True, hide_index=True)

    # ── Summary ─────────────────────────────────────────────────────────
    if results.get("summary_text"):
        with st.expander("Assessment Summary"):
            st.markdown(results["summary_text"])

    # ── Download ────────────────────────────────────────────────────────
    st.markdown("---")
    csv_bytes = export_security_csv(results)
    st.download_button(
        label="Download Compliance Report (CSV)",
        data=csv_bytes,
        file_name="security_compliance_report.csv",
        mime="text/csv",
        use_container_width=True,
    )


main()
