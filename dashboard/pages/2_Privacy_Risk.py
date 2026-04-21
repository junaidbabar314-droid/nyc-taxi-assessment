"""
Privacy Risk Assessment page — Module 2 integration.

Displays privacy risk analysis results including re-identification risk,
k-anonymity, uniqueness, entropy, and linkage attack simulation.  Uses
zone-based analysis (LocationIDs 1-265), not GPS coordinates.

Author : Iqra Aziz (B01802319)
Module : Sami Ullah (B01750598) — Privacy Assessor

References:
    Sweeney, L. (2002) 'k-anonymity: A model for protecting privacy',
        International Journal of Uncertainty, Fuzziness and Knowledge-Based
        Systems, 10(5), pp. 557-570.
    El Emam, K. and Dankar, F.K. (2008) 'Protecting privacy using
        k-anonymity', Journal of the American Medical Informatics
        Association, 15(5), pp. 627-637.
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
    apply_custom_css, PRIVACY_BLUE, PRIVACY_BLUE_DARK, WARNING_ORANGE,
    SECURITY_RED, QUALITY_GREEN, NEUTRAL_GREY, PLOTLY_TEMPLATE, CHART_FONT,
    CHART_MARGIN, risk_colour, RISK_COLOURS,
)
from dashboard.utils.data_loader import cached_load_month
from dashboard.utils.export_reports import export_privacy_csv
from config import (
    AVAILABLE_YEARS, MONTHS, PRIVACY_RISK_CRITICAL, PRIVACY_RISK_HIGH,
    PRIVACY_RISK_MEDIUM,
)

# ─── Page styling (no set_page_config — only Home.py sets it) ──────────
apply_custom_css()

# ─── Module import ──────────────────────────────────────────────────────
_module_available = False
try:
    from module2_privacy.privacy_assessor import get_privacy_assessment
    _module_available = True
except ImportError:
    pass

# ─── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Privacy Risk Assessment")
    year = st.selectbox("Year", AVAILABLE_YEARS, key="pr_year")
    month = st.selectbox(
        "Month", list(MONTHS),
        format_func=lambda m: f"{m:02d}",
        key="pr_month",
    )
    temporal_res = st.selectbox(
        "Temporal Resolution",
        options=["H", "D", "W"],
        format_func=lambda x: {"H": "Hourly", "D": "Daily", "W": "Weekly"}[x],
        index=0,
        key="pr_temporal",
        help="Temporal granularity for uniqueness and k-anonymity analysis.",
    )
    run = st.button("Run Privacy Assessment", type="primary", use_container_width=True)


# ─── Helper: risk level badge ──────────────────────────────────────────

def _risk_badge(level: str):
    """Display a colour-coded risk level badge."""
    colour = risk_colour(level)
    st.markdown(
        f'<div style="background:{colour}; color:white; padding:12px 24px; '
        f'border-radius:8px; text-align:center; font-size:1.5rem; '
        f'font-weight:700; letter-spacing:0.05em;">'
        f'{level.upper()} RISK</div>',
        unsafe_allow_html=True,
    )


# ─── Helper: k-anonymity bar chart ─────────────────────────────────────

def _k_anonymity_chart(k_summary: dict | pd.DataFrame) -> go.Figure | None:
    """Bar chart showing distribution of k-anonymity group sizes."""
    if k_summary is None:
        return None

    if isinstance(k_summary, pd.DataFrame):
        if k_summary.empty:
            return None
        cols = k_summary.columns.tolist()
        x_col = cols[0]
        y_col = cols[1] if len(cols) > 1 else cols[0]
        x = k_summary[x_col].astype(str).tolist()
        y = pd.to_numeric(k_summary[y_col], errors="coerce").fillna(0).tolist()
    elif isinstance(k_summary, dict):
        # Real module shape: {..., "k_distribution": {"k=1 (unique)": n,
        # "k=2-4 (very small)": n, ...}, "min_k", "max_k", "mean_k", ...}
        # Fallback shape: {"k=1 (unique)": count, "k=2": count, ...}
        dist = k_summary.get("k_distribution") if "k_distribution" in k_summary else k_summary
        if not isinstance(dist, dict) or not dist:
            return None
        x = [str(k) for k in dist.keys()]
        y = [float(v) if isinstance(v, (int, float)) else 0 for v in dist.values()]
    else:
        return None

    colours = [SECURITY_RED if float(v) < 5 else WARNING_ORANGE if float(v) < 10
               else QUALITY_GREEN for v in y]

    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker_color=colours,
        text=[f"{v:.0f}" for v in y],
        textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=CHART_MARGIN,
        height=350,
        title="K-Anonymity Distribution",
        xaxis_title="K Value / Group",
        yaxis_title="Count / Percentage",
    )
    return fig


# ─── Helper: resolution comparison ─────────────────────────────────────

def _resolution_comparison_chart(res_df: pd.DataFrame) -> go.Figure | None:
    """Bar chart comparing uniqueness across temporal resolutions."""
    if res_df is None or not isinstance(res_df, pd.DataFrame) or res_df.empty:
        return None

    cols = res_df.columns.tolist()
    x_col = cols[0]
    y_col = cols[1] if len(cols) > 1 else cols[0]

    fig = go.Figure(go.Bar(
        x=res_df[x_col].astype(str),
        y=pd.to_numeric(res_df[y_col], errors="coerce").fillna(0),
        marker_color=PRIVACY_BLUE,
        text=pd.to_numeric(res_df[y_col], errors="coerce").apply(
            lambda v: f"{v:.1f}%" if pd.notna(v) else "N/A"
        ),
        textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=CHART_MARGIN,
        height=350,
        title="Uniqueness by Temporal Resolution",
        xaxis_title="Resolution",
        yaxis_title="Uniqueness %",
    )
    return fig


# ─── Helper: sensitivity analysis chart ────────────────────────────────

def _sensitivity_chart(sens_df: pd.DataFrame) -> go.Figure | None:
    """Line chart showing how risk score changes with weight variations."""
    if sens_df is None or not isinstance(sens_df, pd.DataFrame) or sens_df.empty:
        return None

    cols = sens_df.columns.tolist()
    if len(cols) < 2:
        return None

    x_col = cols[0]
    fig = go.Figure()
    colours = [PRIVACY_BLUE, WARNING_ORANGE, SECURITY_RED, QUALITY_GREEN]

    for i, col in enumerate(cols[1:]):
        fig.add_trace(go.Scatter(
            x=sens_df[x_col],
            y=pd.to_numeric(sens_df[col], errors="coerce"),
            mode="lines+markers",
            name=col,
            line=dict(color=colours[i % len(colours)], width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=CHART_MARGIN,
        height=350,
        title="Sensitivity Analysis (Weight Robustness)",
        xaxis_title=x_col,
        yaxis_title="Risk Score",
    )
    return fig


# ─── Helper: entropy distribution ──────────────────────────────────────

def _entropy_chart(field_scores: dict) -> go.Figure | None:
    """Bar chart of per-field entropy or risk scores."""
    if not field_scores or not isinstance(field_scores, dict):
        return None

    fields = list(field_scores.keys())
    values = [float(v) if isinstance(v, (int, float)) else 0 for v in field_scores.values()]

    fig = go.Figure(go.Bar(
        x=fields,
        y=values,
        marker_color=PRIVACY_BLUE,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=dict(l=40, r=20, t=50, b=80),
        height=350,
        title="Field Risk / Entropy Scores",
        xaxis_title="Field",
        yaxis_title="Score",
        xaxis=dict(tickangle=45),
    )
    return fig


# ─── Fallback: basic privacy analysis ──────────────────────────────────

def _fallback_privacy(df: pd.DataFrame, temporal_resolution: str = "H") -> dict:
    """
    Compute basic privacy metrics when Module 2 is unavailable.

    Analyses quasi-identifier uniqueness using zone IDs (not GPS) and
    temporal rounding to the specified resolution.
    """
    quasi_ids = []
    if "PULocationID" in df.columns:
        quasi_ids.append("PULocationID")
    if "DOLocationID" in df.columns:
        quasi_ids.append("DOLocationID")
    if "tpep_pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        df = df.copy()
        freq_map = {"H": "h", "D": "D", "W": "W"}
        df["_pickup_rounded"] = dt.dt.floor(freq_map.get(temporal_resolution, "h"))
        quasi_ids.append("_pickup_rounded")
    if "passenger_count" in df.columns:
        quasi_ids.append("passenger_count")

    if not quasi_ids:
        return {
            "overall_risk_score": 0,
            "risk_level": "Low",
            "pii_fields": [],
            "uniqueness_percentage": 0,
            "k_anonymity_summary": {},
            "avg_entropy": 0,
            "linkage_rate": 0,
            "sensitivity_analysis": None,
            "field_risk_scores": {},
            "resolution_comparison": None,
            "summary_text": "Insufficient quasi-identifiers for analysis.",
        }

    # Group by quasi-identifiers and compute k-anonymity
    subset = df[quasi_ids].dropna()
    if len(subset) == 0:
        unique_pct = 0
        k_values = {}
    else:
        groups = subset.groupby(quasi_ids).size()
        unique_count = (groups == 1).sum()
        unique_pct = (unique_count / len(groups)) * 100

        # K-anonymity distribution
        k_values = {
            "k=1 (unique)": int((groups == 1).sum()),
            "k=2": int((groups == 2).sum()),
            "k=3-5": int(((groups >= 3) & (groups <= 5)).sum()),
            "k=6-10": int(((groups >= 6) & (groups <= 10)).sum()),
            "k>10": int((groups > 10).sum()),
        }

    # Risk level classification
    if unique_pct >= PRIVACY_RISK_CRITICAL:
        risk_level = "Critical"
    elif unique_pct >= PRIVACY_RISK_HIGH:
        risk_level = "High"
    elif unique_pct >= PRIVACY_RISK_MEDIUM:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    risk_score = min(100, unique_pct * 1.2)

    # PII field classification (zone-based data has no direct PII)
    pii_fields = ["tpep_pickup_datetime", "tpep_dropoff_datetime",
                  "PULocationID", "DOLocationID"]
    pii_fields = [f for f in pii_fields if f in df.columns]

    # Simple field risk scores
    field_risk = {}
    for col in quasi_ids:
        if col.startswith("_"):
            continue
        nunique = df[col].nunique()
        total = len(df)
        field_risk[col] = round((nunique / max(total, 1)) * 100, 2)

    # Resolution comparison (compute uniqueness at different resolutions)
    res_data = []
    for res_label, freq in [("Hourly", "h"), ("Daily", "D"), ("Weekly", "W")]:
        if "tpep_pickup_datetime" in df.columns:
            temp_df = df.copy()
            temp_dt = pd.to_datetime(temp_df["tpep_pickup_datetime"], errors="coerce")
            temp_df["_temp_rounded"] = temp_dt.dt.floor(freq)
            temp_qids = [c for c in quasi_ids if not c.startswith("_")] + ["_temp_rounded"]
            temp_subset = temp_df[temp_qids].dropna()
            if len(temp_subset) > 0:
                temp_groups = temp_subset.groupby(temp_qids).size()
                temp_unique = (temp_groups == 1).sum()
                temp_pct = (temp_unique / len(temp_groups)) * 100
            else:
                temp_pct = 0
            res_data.append({"Resolution": res_label, "Uniqueness %": round(temp_pct, 2)})

    resolution_df = pd.DataFrame(res_data) if res_data else None

    return {
        "overall_risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "pii_fields": pii_fields,
        "uniqueness_percentage": round(unique_pct, 2),
        "k_anonymity_summary": k_values,
        "avg_entropy": 0,
        "linkage_rate": 0,
        "sensitivity_analysis": None,
        "field_risk_scores": field_risk,
        "resolution_comparison": resolution_df,
        "summary_text": (
            f"**Fallback assessment** (Module 2 not available): "
            f"Analysed {len(df):,} records with {len(quasi_ids)} quasi-identifiers. "
            f"Uniqueness: {unique_pct:.1f}%. Risk level: {risk_level}."
        ),
    }


# ─── Main page ──────────────────────────────────────────────────────────

def main():
    st.markdown("# Privacy Risk Assessment")
    st.markdown(
        "Evaluates re-identification risk through uniqueness analysis, "
        "k-anonymity, entropy measurement, and linkage attack simulation. "
        "Based on Sami Ullah's (B01750598) privacy assessor module. "
        "Uses **zone IDs** (1-265), not GPS coordinates."
    )
    st.markdown("---")

    results = st.session_state.get("privacy_results")

    if run:
        try:
            with st.spinner(f"Loading {year}-{month:02d} data..."):
                df = cached_load_month(year, month)
            st.success(f"Loaded {len(df):,} records.")
        except FileNotFoundError:
            st.error(f"Data file not found for {year}-{month:02d}.")
            return
        except Exception as exc:
            st.error(f"Error loading data: {exc}")
            return

        with st.spinner("Analysing privacy risks..."):
            try:
                if _module_available:
                    results = get_privacy_assessment(df, temporal_resolution=temporal_res)
                else:
                    st.warning(
                        "Module 2 (privacy_assessor) not yet available. "
                        "Showing fallback analysis."
                    )
                    results = _fallback_privacy(df, temporal_res)
            except Exception as exc:
                st.error(f"Privacy assessment failed: {exc}")
                st.warning("Falling back to basic privacy analysis.")
                results = _fallback_privacy(df, temporal_res)

        st.session_state["privacy_results"] = results

    if results is None:
        st.info(
            "No privacy assessment results yet. Select a year/month and "
            "click **Run Privacy Assessment** in the sidebar."
        )
        return

    # ── Risk level badge ────────────────────────────────────────────────
    risk_level = results.get("risk_level", "N/A")
    col_badge, col_space = st.columns([1, 2])
    with col_badge:
        _risk_badge(risk_level)

    st.markdown("")

    # ── Metric cards ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_score = results.get("overall_risk_score", "N/A")
        if isinstance(risk_score, (int, float)):
            st.metric("Risk Score", f"{risk_score:.1f}")
        else:
            st.metric("Risk Score", str(risk_score))
    with col2:
        uniq = results.get("uniqueness_percentage", "N/A")
        if isinstance(uniq, (int, float)):
            st.metric("Uniqueness", f"{uniq:.1f}%")
        else:
            st.metric("Uniqueness", str(uniq))
    with col3:
        k_summary = results.get("k_anonymity_summary", {}) or {}
        # Real module shape: {min_k, max_k, mean_k, median_k,
        #                     records_below_k5_pct, total_equivalence_classes,
        #                     k_distribution}
        # Fallback shape: {"k=1 (unique)": count, "k=2": count, ...}
        if isinstance(k_summary, dict) and "median_k" in k_summary:
            median_k = k_summary["median_k"]
            st.metric("K-Anonymity (median group)",
                      f"{int(median_k)}" if isinstance(median_k, (int, float)) else "N/A")
        elif isinstance(k_summary, dict) and k_summary:
            vals = [v for v in k_summary.values() if isinstance(v, (int, float))]
            median_k = int(np.median(vals)) if vals else "N/A"
            st.metric("K-Anonymity (median group)", median_k)
        else:
            st.metric("K-Anonymity", "N/A")

    st.markdown("---")

    # ── Charts ──────────────────────────────────────────────────────────
    tab_kanon, tab_uniq, tab_sens, tab_entropy = st.tabs([
        "K-Anonymity", "Uniqueness by Resolution", "Sensitivity", "Field Risk",
    ])

    with tab_kanon:
        fig_k = _k_anonymity_chart(results.get("k_anonymity_summary"))
        if fig_k:
            st.plotly_chart(fig_k, use_container_width=True, key="pr_kanon")
        else:
            st.info("K-anonymity data not available.")

    with tab_uniq:
        fig_res = _resolution_comparison_chart(results.get("resolution_comparison"))
        if fig_res:
            st.plotly_chart(fig_res, use_container_width=True, key="pr_res")
        else:
            st.info("Resolution comparison data not available.")

    with tab_sens:
        fig_sens = _sensitivity_chart(results.get("sensitivity_analysis"))
        if fig_sens:
            st.plotly_chart(fig_sens, use_container_width=True, key="pr_sens")
        else:
            st.info("Sensitivity analysis data not available.")

    with tab_entropy:
        fig_ent = _entropy_chart(results.get("field_risk_scores"))
        if fig_ent:
            st.plotly_chart(fig_ent, use_container_width=True, key="pr_entropy")
        else:
            st.info("Field risk/entropy data not available.")

    # ── PII field table ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### PII Field Classification")
    pii_fields = results.get("pii_fields", [])
    if pii_fields:
        pii_df = pd.DataFrame({
            "Field": pii_fields,
            "Classification": [
                "Quasi-identifier (temporal)" if "datetime" in f.lower()
                else "Quasi-identifier (spatial)" if "location" in f.lower()
                else "Quasi-identifier"
                for f in pii_fields
            ],
            "Risk": [
                "High" if "datetime" in f.lower() or "location" in f.lower()
                else "Medium"
                for f in pii_fields
            ],
        })
        st.dataframe(pii_df, use_container_width=True, hide_index=True)
    else:
        st.info("No PII fields identified.")

    # ── Linkage attack results ──────────────────────────────────────────
    st.markdown("### Linkage Attack Simulation")
    linkage = results.get("linkage_rate")
    if linkage is not None and isinstance(linkage, (int, float)):
        if linkage > 0:
            st.warning(f"Linkage attack success rate: {linkage:.2f}%")
        else:
            st.success("No successful linkage attacks detected in simulation.")
    else:
        st.info("Linkage attack data not available.")

    # ── Summary text ────────────────────────────────────────────────────
    if results.get("summary_text"):
        with st.expander("Assessment Summary", expanded=True):
            st.markdown(results["summary_text"])

    # ── Download ────────────────────────────────────────────────────────
    st.markdown("---")
    csv_bytes = export_privacy_csv(results)
    st.download_button(
        label="Download Privacy Report (CSV)",
        data=csv_bytes,
        file_name=f"privacy_report_{year}_{month:02d}.csv",
        mime="text/csv",
        use_container_width=True,
    )


main()
