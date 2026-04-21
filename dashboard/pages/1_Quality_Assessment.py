"""
Quality Assessment page — Module 1 integration.

Presents data quality profiling results across four dimensions:
completeness, accuracy, consistency, and timeliness.  Visualisations
follow Few's (2006) guidance on dashboard design: high data-ink ratio,
meaningful colour encoding, and progressive disclosure via expanders.

Author : Iqra Aziz (B01802319)
Module : Junaid Babar (B01802551) — Quality Profiler

References:
    Few, S. (2006) Information Dashboard Design. Analytics Press.
    Shneiderman, B. (1996) 'The eyes have it', Proc. IEEE Symp. Visual
        Languages, pp. 336-343.
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
    apply_custom_css, QUALITY_GREEN, QUALITY_GREEN_DARK, WARNING_ORANGE,
    SECURITY_RED, NEUTRAL_GREY, PLOTLY_TEMPLATE, CHART_FONT, CHART_MARGIN,
    score_colour,
)
from dashboard.utils.data_loader import cached_load_month
from dashboard.utils.export_reports import export_quality_csv
from config import AVAILABLE_YEARS, MONTHS, DATA_DIR

# ─── Page styling (no set_page_config — only Home.py sets it) ──────────
apply_custom_css()

# ─── Module import ──────────────────────────────────────────────────────
_module_available = False
try:
    from module1_quality.quality_profiler import get_quality_metrics
    _module_available = True
except ImportError:
    pass


# ─── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Quality Assessment")
    year = st.selectbox("Year", AVAILABLE_YEARS, key="qa_year")
    month = st.selectbox(
        "Month", list(MONTHS),
        format_func=lambda m: f"{m:02d}",
        key="qa_month",
    )
    run = st.button("Run Quality Assessment", type="primary", use_container_width=True)


# ─── Helper: radar chart for 4 quality dimensions ──────────────────────

def _quality_radar(metrics: dict) -> go.Figure:
    """Radar chart showing the four quality dimensions."""
    dims = ["Completeness", "Accuracy", "Consistency", "Timeliness"]
    keys = ["completeness", "accuracy", "consistency", "timeliness"]
    values = [metrics.get(k, 0) for k in keys]
    # Close the polygon
    values_closed = values + [values[0]]
    dims_closed = dims + [dims[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=dims_closed,
        fill="toself",
        fillcolor=f"rgba(46, 204, 113, 0.25)",
        line=dict(color=QUALITY_GREEN, width=2),
        marker=dict(size=8, color=QUALITY_GREEN),
        name="Quality Score",
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
        title="Quality Dimensions Radar",
    )
    return fig


# ─── Helper: completeness heatmap ──────────────────────────────────────

def _completeness_heatmap(detail_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of field-level completeness percentages.

    Parameters:
        detail_df: DataFrame with columns like 'field', 'completeness_pct'
                   or similar from the quality module.
    """
    if detail_df is None or detail_df.empty:
        return None

    # Try to identify field and value columns.  The real module returns
    # columns ``Field`` and ``Null_Percentage`` — Null_Percentage is the
    # INVERSE of completeness (0 null = 100% complete), so we invert it
    # before plotting.  The fallback dict uses ``completeness_pct``
    # directly (already in "higher is better" orientation).
    cols = detail_df.columns.tolist()
    field_col = None
    val_col = None
    invert = False
    for c in cols:
        cl = c.lower()
        if cl in ("field", "column", "column_name"):
            field_col = c
        elif cl == "null_percentage" or "null" in cl:
            val_col = c
            invert = True
        elif val_col is None and ("complete" in cl or cl.endswith("_pct")
                                   or "percent" in cl or "score" in cl):
            val_col = c

    if field_col is None:
        field_col = cols[0]
    if val_col is None:
        val_col = cols[1] if len(cols) > 1 else cols[0]

    fields = detail_df[field_col].astype(str).tolist()
    raw = pd.to_numeric(detail_df[val_col], errors="coerce").fillna(0)
    values = (100 - raw if invert else raw).clip(0, 100).tolist()

    # Reshape into a 2D matrix for heatmap (single row is fine)
    fig = go.Figure(go.Heatmap(
        z=[values],
        x=fields,
        y=["Completeness"],
        colorscale=[[0, SECURITY_RED], [0.5, WARNING_ORANGE], [1, QUALITY_GREEN]],
        zmin=0,
        zmax=100,
        text=[[f"{v:.1f}%" for v in values]],
        texttemplate="%{text}",
        textfont={"size": 11},
        hovertemplate="Field: %{x}<br>Completeness: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="%", ticksuffix="%"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=dict(l=80, r=40, t=40, b=100),
        height=200,
        title="Field Completeness Heatmap",
        xaxis=dict(tickangle=45),
    )
    return fig


# ─── Helper: outlier boxplots ──────────────────────────────────────────

def _outlier_boxplots(df: pd.DataFrame) -> go.Figure:
    """Box plots for fare_amount, trip_distance, passenger_count."""
    numeric_cols = ["fare_amount", "trip_distance", "passenger_count"]
    available = [c for c in numeric_cols if c in df.columns]

    if not available:
        return None

    fig = go.Figure()
    colours = [QUALITY_GREEN, WARNING_ORANGE, SECURITY_RED]

    for i, col in enumerate(available):
        series = df[col].dropna()
        # Clip extreme outliers for visualisation (show up to 99th percentile)
        upper = series.quantile(0.99)
        clipped = series[series <= upper]
        fig.add_trace(go.Box(
            y=clipped,
            name=col.replace("_", " ").title(),
            marker_color=colours[i % len(colours)],
            boxmean=True,
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=CHART_MARGIN,
        height=400,
        title="Outlier Analysis (clipped at 99th percentile)",
        yaxis_title="Value",
        showlegend=False,
    )
    return fig


# ─── Helper: distribution histograms ───────────────────────────────────

def _distribution_histograms(df: pd.DataFrame) -> go.Figure:
    """Side-by-side histograms for key numeric fields."""
    from plotly.subplots import make_subplots

    numeric_cols = ["fare_amount", "trip_distance", "passenger_count"]
    available = [c for c in numeric_cols if c in df.columns]

    if not available:
        return None

    fig = make_subplots(
        rows=1, cols=len(available),
        subplot_titles=[c.replace("_", " ").title() for c in available],
    )

    colours = [QUALITY_GREEN, WARNING_ORANGE, SECURITY_RED]
    for i, col in enumerate(available):
        series = df[col].dropna()
        upper = series.quantile(0.99)
        clipped = series[series <= upper]
        fig.add_trace(
            go.Histogram(
                x=clipped,
                marker_color=colours[i % len(colours)],
                opacity=0.75,
                name=col.replace("_", " ").title(),
                nbinsx=50,
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=dict(l=40, r=20, t=60, b=40),
        height=350,
        showlegend=False,
        title_text="Value Distributions (clipped at 99th percentile)",
    )
    return fig


# ─── Fallback: compute basic quality metrics from raw data ──────────────

def _fallback_quality_metrics(df: pd.DataFrame) -> dict:
    """
    Compute basic quality metrics directly when Module 1 is unavailable.

    This provides a functional dashboard even before the quality profiler
    module is complete, demonstrating the integration interface.
    """
    total_rows = len(df)
    total_cells = total_rows * len(df.columns)
    null_cells = df.isnull().sum().sum()

    # Completeness: percentage of non-null cells
    completeness = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0

    # Accuracy: percentage of rows with realistic values
    accuracy_checks = 0
    accuracy_pass = 0
    if "fare_amount" in df.columns:
        accuracy_checks += len(df)
        accuracy_pass += (df["fare_amount"].between(0, 500)).sum()
    if "trip_distance" in df.columns:
        accuracy_checks += len(df)
        accuracy_pass += (df["trip_distance"].between(0, 200)).sum()
    if "passenger_count" in df.columns:
        accuracy_checks += len(df)
        accuracy_pass += (df["passenger_count"].between(0, 6)).sum()
    accuracy = (accuracy_pass / accuracy_checks) * 100 if accuracy_checks > 0 else 0

    # Consistency: fare components sum check
    fare_cols = ["fare_amount", "extra", "mta_tax", "tip_amount",
                 "tolls_amount", "improvement_surcharge", "congestion_surcharge"]
    available_fare = [c for c in fare_cols if c in df.columns]
    if available_fare and "total_amount" in df.columns:
        computed = df[available_fare].fillna(0).sum(axis=1)
        diff = (computed - df["total_amount"].fillna(0)).abs()
        consistency = (diff <= 0.01).mean() * 100
    else:
        consistency = 0

    # Timeliness: check pickup datetime is within expected period
    if "tpep_pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        valid_dt = dt.notna()
        timeliness = valid_dt.mean() * 100
    else:
        timeliness = 0

    overall = (completeness + accuracy + consistency + timeliness) / 4

    # Completeness detail per field
    field_completeness = []
    for col in df.columns:
        pct = (1 - df[col].isnull().mean()) * 100
        field_completeness.append({"field": col, "completeness_pct": round(pct, 2)})
    completeness_detail = pd.DataFrame(field_completeness)

    # Field scores
    field_scores = completeness_detail.copy()

    return {
        "overall_score": round(overall, 2),
        "metrics": {
            "completeness": round(completeness, 2),
            "accuracy": round(accuracy, 2),
            "consistency": round(consistency, 2),
            "timeliness": round(timeliness, 2),
        },
        "completeness_detail": completeness_detail,
        "accuracy_detail": None,
        "consistency_detail": None,
        "timeliness_detail": None,
        "field_scores": field_scores,
        "summary_text": (
            f"**Fallback assessment** (Module 1 not available): "
            f"Analysed {total_rows:,} rows across {len(df.columns)} fields. "
            f"Overall quality score: {overall:.1f}%."
        ),
    }


# ─── Main page ──────────────────────────────────────────────────────────

def main():
    st.markdown("# Data Quality Assessment")
    st.markdown(
        "Comprehensive quality profiling across four dimensions: "
        "**completeness**, **accuracy**, **consistency**, and **timeliness**. "
        "Based on Junaid Babar's (B01802551) quality profiler module."
    )
    st.markdown("---")

    results = st.session_state.get("quality_results")

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

        with st.spinner("Analysing data quality..."):
            try:
                if _module_available:
                    results = get_quality_metrics(df, year=year, month=month)
                else:
                    st.warning(
                        "Module 1 (quality_profiler) is not yet available. "
                        "Showing fallback analysis computed directly from the data."
                    )
                    results = _fallback_quality_metrics(df)
            except Exception as exc:
                st.error(f"Quality assessment failed: {exc}")
                st.warning("Falling back to basic quality metrics.")
                results = _fallback_quality_metrics(df)

        st.session_state["quality_results"] = results
        st.session_state["quality_df"] = df

    if results is None:
        st.info(
            "No quality assessment results yet. Select a year/month and "
            "click **Run Quality Assessment** in the sidebar."
        )
        return

    df = st.session_state.get("quality_df")
    metrics = results.get("metrics", {})

    # ── Sub-metric cards ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    dim_data = [
        ("Completeness", "completeness", col1),
        ("Accuracy", "accuracy", col2),
        ("Consistency", "consistency", col3),
        ("Timeliness", "timeliness", col4),
    ]
    for label, key, col in dim_data:
        val = metrics.get(key)
        with col:
            if isinstance(val, (int, float)):
                st.metric(label, f"{val:.1f}%")
            else:
                st.metric(label, "N/A")

    st.markdown("---")

    # ── Radar chart + overall score ─────────────────────────────────────
    col_radar, col_detail = st.columns([1, 1])

    with col_radar:
        fig_radar = _quality_radar(metrics)
        st.plotly_chart(fig_radar, use_container_width=True, key="q_radar")

    with col_detail:
        overall = results.get("overall_score", 0)
        st.markdown("### Overall Quality Score")
        if isinstance(overall, (int, float)):
            colour = score_colour(overall)
            st.markdown(
                f'<p style="font-size:3rem; font-weight:700; color:{colour}; '
                f'text-align:center; margin:20px 0;">{overall:.1f}%</p>',
                unsafe_allow_html=True,
            )
        if results.get("summary_text"):
            st.markdown(results["summary_text"])

    st.markdown("---")

    # ── Completeness heatmap ────────────────────────────────────────────
    st.markdown("### Field Completeness")
    comp_detail = results.get("completeness_detail")
    if comp_detail is not None and isinstance(comp_detail, pd.DataFrame) and not comp_detail.empty:
        fig_hm = _completeness_heatmap(comp_detail)
        if fig_hm:
            st.plotly_chart(fig_hm, use_container_width=True, key="q_heatmap")
    else:
        st.info("Completeness detail not available.")

    # ── Outlier boxplots ────────────────────────────────────────────────
    if df is not None:
        st.markdown("### Outlier Analysis")
        tab_box, tab_hist = st.tabs(["Box Plots", "Distributions"])

        with tab_box:
            fig_box = _outlier_boxplots(df)
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True, key="q_box")
            else:
                st.info("No numeric columns available for box plots.")

        with tab_hist:
            fig_hist = _distribution_histograms(df)
            if fig_hist:
                st.plotly_chart(fig_hist, use_container_width=True, key="q_hist")
            else:
                st.info("No numeric columns available for histograms.")

    # ── Field scores table (progressive disclosure) ─────────────────────
    st.markdown("---")
    field_scores = results.get("field_scores")
    if field_scores is not None and isinstance(field_scores, pd.DataFrame) and not field_scores.empty:
        with st.expander("Detailed Field Scores", expanded=False):
            st.dataframe(field_scores, use_container_width=True, height=400)

    # ── Download button ─────────────────────────────────────────────────
    st.markdown("---")
    csv_bytes = export_quality_csv(results)
    st.download_button(
        label="Download Quality Report (CSV)",
        data=csv_bytes,
        file_name=f"quality_report_{year}_{month:02d}.csv",
        mime="text/csv",
        use_container_width=True,
    )


main()
