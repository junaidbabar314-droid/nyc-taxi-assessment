"""
Visualisation functions for the Data Quality Profiling module.

Generates publication-quality interactive charts using Plotly for
embedding in the governance dashboard (Streamlit) and HTML reports.

All chart functions return Plotly Figure objects, allowing callers to
render them interactively (Streamlit, Jupyter) or export as static
images for the dissertation document.

Author: Junaid Babar (B01802551)
Module: Data Quality Profiling
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Project imports ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Colour Palette (UWS-friendly, accessible) ─────────────────────
COLOUR_GOOD = "#2ecc71"
COLOUR_MINOR = "#f39c12"
COLOUR_CRITICAL = "#e74c3c"
COLOUR_PRIMARY = "#3498db"
COLOUR_SECONDARY = "#9b59b6"
COLOUR_BG = "#fafafa"
SEVERITY_COLOURS = {
    "Good": COLOUR_GOOD,
    "Minor": COLOUR_MINOR,
    "Critical": COLOUR_CRITICAL,
}


def create_completeness_heatmap(completeness_df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing null percentages across all fields.

    Fields are ordered by null percentage (descending), with colour
    intensity reflecting severity: green for Good, amber for Minor,
    red for Critical.

    Parameters:
        completeness_df: Output of assess_completeness().

    Returns:
        Plotly Figure object.
    """
    df = completeness_df.sort_values("Null_Percentage", ascending=True).copy()

    colours = df["Severity"].map(SEVERITY_COLOURS).tolist()

    fig = go.Figure(go.Bar(
        x=df["Null_Percentage"],
        y=df["Field"],
        orientation="h",
        marker_color=colours,
        text=df["Null_Percentage"].apply(lambda x: f"{x:.2f}%"),
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Null: %{x:.2f}%<br>"
            "Count: %{customdata[0]:,}<br>"
            "Severity: %{customdata[1]}"
            "<extra></extra>"
        ),
        customdata=df[["Null_Count", "Severity"]].values,
    ))

    fig.update_layout(
        title=dict(
            text="Data Completeness: Null Percentage by Field",
            font_size=16,
        ),
        xaxis_title="Null Percentage (%)",
        yaxis_title="Field",
        template="plotly_white",
        height=max(400, len(df) * 28),
        margin=dict(l=160, r=80, t=60, b=40),
    )

    # Add threshold lines
    fig.add_vline(x=5, line_dash="dash", line_color=COLOUR_CRITICAL,
                  annotation_text="Critical (5%)", annotation_position="top")
    fig.add_vline(x=1, line_dash="dot", line_color=COLOUR_MINOR,
                  annotation_text="Minor (1%)", annotation_position="top")

    return fig


def create_outlier_boxplots(
    df: pd.DataFrame,
    fields: Optional[List[str]] = None,
) -> go.Figure:
    """
    Create side-by-side box plots for selected numeric fields to
    visualise outlier distributions.

    Parameters:
        df: Trip data DataFrame.
        fields: List of column names to plot. Defaults to key fare
                and distance fields.

    Returns:
        Plotly Figure object.
    """
    if fields is None:
        fields = ["fare_amount", "trip_distance", "tip_amount", "total_amount"]

    available = [f for f in fields if f in df.columns]

    fig = make_subplots(
        rows=1, cols=len(available),
        subplot_titles=available,
        horizontal_spacing=0.05,
    )

    for i, field in enumerate(available, start=1):
        data = df[field].dropna()
        # Subsample for performance if large
        if len(data) > 50_000:
            data = data.sample(n=50_000, random_state=42)

        fig.add_trace(
            go.Box(
                y=data,
                name=field,
                marker_color=COLOUR_PRIMARY,
                boxmean=True,
                showlegend=False,
            ),
            row=1, col=i,
        )

    fig.update_layout(
        title=dict(
            text="Outlier Distribution: Box Plots (IQR Method)",
            font_size=16,
        ),
        template="plotly_white",
        height=500,
        margin=dict(t=80, b=40),
    )

    return fig


def create_distribution_histograms(
    df: pd.DataFrame,
    fields: Optional[List[str]] = None,
) -> go.Figure:
    """
    Create distribution histograms for selected numeric fields.

    Parameters:
        df: Trip data DataFrame.
        fields: List of column names to plot. Defaults to key fields.

    Returns:
        Plotly Figure object with subplot histograms.
    """
    if fields is None:
        fields = [
            "fare_amount", "trip_distance", "tip_amount",
            "passenger_count", "total_amount",
        ]

    available = [f for f in fields if f in df.columns]
    n_fields = len(available)
    n_cols = min(3, n_fields)
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=available,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, field in enumerate(available):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        data = df[field].dropna()

        # Clip extreme values for readability
        q99 = data.quantile(0.99) if len(data) > 0 else 0
        clipped = data[data <= q99]

        fig.add_trace(
            go.Histogram(
                x=clipped,
                name=field,
                marker_color=COLOUR_PRIMARY,
                opacity=0.8,
                showlegend=False,
                nbinsx=50,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title=dict(
            text="Value Distributions (clipped at 99th percentile)",
            font_size=16,
        ),
        template="plotly_white",
        height=300 * n_rows,
        margin=dict(t=80, b=40),
    )

    return fig


def create_quality_summary_radar(metrics: dict) -> go.Figure:
    """
    Create a radar (spider) chart summarising the four quality dimensions.

    Parameters:
        metrics: Dictionary with keys 'completeness', 'accuracy',
                 'consistency', 'timeliness' mapped to scores (0-100).

    Returns:
        Plotly Figure object.
    """
    categories = ["Completeness", "Accuracy", "Consistency", "Timeliness"]
    values = [
        metrics.get("completeness", 0),
        metrics.get("accuracy", 0),
        metrics.get("consistency", 0),
        metrics.get("timeliness", 0),
    ]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.25)",
        line=dict(color=COLOUR_PRIMARY, width=2),
        marker=dict(size=8, color=COLOUR_PRIMARY),
        name="Quality Score",
        hovertemplate="%{theta}: %{r:.1f}/100<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Data Quality Radar: Four Dimensions",
            font_size=16,
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
            ),
        ),
        template="plotly_white",
        height=500,
        width=550,
        showlegend=False,
    )

    return fig


def create_timeliness_chart(timeliness_results: dict) -> go.Figure:
    """
    Create a bar chart visualising timeliness freshness metrics.

    Displays the percentage of records within 30-day and 60-day
    freshness windows alongside the average and median lag.

    Parameters:
        timeliness_results: Output of assess_timeliness().

    Returns:
        Plotly Figure object.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Freshness Windows", "Lag Distribution"],
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.15,
    )

    # Freshness bars
    windows = ["30-day", "60-day"]
    pcts = [
        timeliness_results.get("freshness_30d_pct", 0),
        timeliness_results.get("freshness_60d_pct", 0),
    ]
    bar_colours = [
        COLOUR_GOOD if p >= 90 else COLOUR_MINOR if p >= 70 else COLOUR_CRITICAL
        for p in pcts
    ]

    fig.add_trace(
        go.Bar(
            x=windows,
            y=pcts,
            marker_color=bar_colours,
            text=[f"{p:.1f}%" for p in pcts],
            textposition="outside",
            name="Freshness",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Lag bars
    lag_labels = ["Average Lag", "Median Lag", "Max Lag"]
    lag_values = [
        timeliness_results.get("avg_lag_days", 0),
        timeliness_results.get("median_lag_days", 0),
        timeliness_results.get("max_lag_days", 0),
    ]

    fig.add_trace(
        go.Bar(
            x=lag_labels,
            y=lag_values,
            marker_color=[COLOUR_PRIMARY, COLOUR_SECONDARY, COLOUR_CRITICAL],
            text=[f"{v:.1f}d" for v in lag_values],
            textposition="outside",
            name="Lag",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Days", row=1, col=2)

    pub_date = timeliness_results.get("publication_date", "N/A")
    fig.update_layout(
        title=dict(
            text=f"Timeliness Assessment (Publication Date: {pub_date})",
            font_size=16,
        ),
        template="plotly_white",
        height=450,
        margin=dict(t=80, b=40),
    )

    return fig
