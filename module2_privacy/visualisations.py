"""
Privacy Visualisation Module for NYC Taxi Trip Records.

Generates publication-quality charts for all privacy metrics using
Plotly. Designed for embedding in the Streamlit governance dashboard
(Module 4) and for export as static images for the dissertation.

All charts follow a consistent academic colour scheme with proper
titles, axis labels, and annotations suitable for MSc-level
presentation.

Author: Sami Ullah (B01750598)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Colour palette (academic/professional) ─────────────────────────
COLOURS = {
    "critical": "#d62728",
    "high": "#ff7f0e",
    "medium": "#ffbb78",
    "low": "#2ca02c",
    "primary": "#1f77b4",
    "secondary": "#aec7e8",
    "accent": "#9467bd",
    "background": "#fafafa",
}

_LAYOUT_DEFAULTS = dict(
    font=dict(family="Arial, Helvetica, sans-serif", size=12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=30, t=60, b=50),
)


def _apply_layout(fig: go.Figure, title: str, **kwargs) -> go.Figure:
    """Apply consistent layout styling to a figure."""
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"),
                      **_LAYOUT_DEFAULTS, **kwargs)
    return fig


# ─── 1. Field-level privacy risk heatmap ────────────────────────────

def create_privacy_heatmap(field_scores: dict[str, float]) -> go.Figure:
    """
    Create a heatmap of field-level privacy risk scores.

    Parameters:
        field_scores: Dictionary mapping field_name -> risk score (0-100).

    Returns:
        Plotly Figure with a single-row heatmap showing colour-coded
        risk per field.
    """
    fields = list(field_scores.keys())
    scores = list(field_scores.values())

    # Sort by score descending for visual clarity
    sorted_pairs = sorted(zip(fields, scores), key=lambda x: -x[1])
    fields = [p[0] for p in sorted_pairs]
    scores = [p[1] for p in sorted_pairs]

    fig = go.Figure(data=go.Heatmap(
        z=[scores],
        x=fields,
        y=["Privacy Risk"],
        colorscale=[
            [0.0, "#2ca02c"],    # Low — green
            [0.25, "#ffdd57"],   # Medium — yellow
            [0.5, "#ff7f0e"],    # High — orange
            [0.75, "#d62728"],   # Critical — red
            [1.0, "#8b0000"],    # Extreme — dark red
        ],
        zmin=0,
        zmax=100,
        text=[[f"{s:.1f}" for s in scores]],
        texttemplate="%{text}",
        colorbar=dict(title="Risk Score"),
        hovertemplate="Field: %{x}<br>Score: %{z:.1f}<extra></extra>",
    ))

    fig = _apply_layout(
        fig,
        "Field-Level Privacy Risk Scores",
        height=250,
        xaxis=dict(tickangle=45),
    )
    return fig


# ─── 2. k-Anonymity distribution chart ─────────────────────────────

def create_k_distribution_chart(k_metrics: dict) -> go.Figure:
    """
    Create a bar chart of k-anonymity equivalence class distribution.

    Parameters:
        k_metrics: Dictionary with 'k_distribution' key mapping
                   bucket labels to counts.

    Returns:
        Plotly Figure with grouped bar chart.
    """
    k_dist = k_metrics.get("k_distribution", {})

    if not k_dist:
        fig = go.Figure()
        fig.add_annotation(text="No k-anonymity data available", showarrow=False)
        return _apply_layout(fig, "k-Anonymity Distribution")

    buckets = list(k_dist.keys())
    counts = list(k_dist.values())

    # Colour code: k=1 is critical, higher k is safer
    colours = [
        COLOURS["critical"],    # k=1
        COLOURS["high"],        # k=2-5
        COLOURS["medium"],      # k=6-10
        COLOURS["low"],         # k=11-50
        COLOURS["primary"],     # k>50
    ]
    # Pad colours if needed
    while len(colours) < len(buckets):
        colours.append(COLOURS["secondary"])

    fig = go.Figure(data=go.Bar(
        x=buckets,
        y=counts,
        marker_color=colours[:len(buckets)],
        text=counts,
        textposition="outside",
        hovertemplate="Bucket: %{x}<br>Classes: %{y:,}<extra></extra>",
    ))

    fig = _apply_layout(
        fig,
        "k-Anonymity: Equivalence Class Size Distribution",
        xaxis_title="Equivalence Class Size (k)",
        yaxis_title="Number of Equivalence Classes",
        height=400,
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


# ─── 3. Uniqueness by temporal resolution ───────────────────────────

def create_uniqueness_by_resolution(
    resolution_df: pd.DataFrame,
) -> go.Figure:
    """
    Create a bar chart showing uniqueness percentage at different
    temporal resolutions, demonstrating the privacy-utility trade-off.

    Parameters:
        resolution_df: DataFrame from compare_temporal_resolutions() with
                       columns: resolution, uniqueness_percentage.

    Returns:
        Plotly Figure with annotated bars.
    """
    fig = go.Figure()

    colours = [COLOURS["critical"], COLOURS["high"], COLOURS["low"]]

    fig.add_trace(go.Bar(
        x=resolution_df["resolution"],
        y=resolution_df["uniqueness_percentage"],
        marker_color=colours[:len(resolution_df)],
        text=[f"{v:.1f}%" for v in resolution_df["uniqueness_percentage"]],
        textposition="outside",
        hovertemplate=(
            "Resolution: %{x}<br>"
            "Uniqueness: %{y:.2f}%<extra></extra>"
        ),
    ))

    fig = _apply_layout(
        fig,
        "Impact of Temporal Resolution on Uniqueness",
        xaxis_title="Temporal Resolution",
        yaxis_title="Uniqueness (%)",
        height=400,
    )
    fig.update_yaxes(range=[0, 105], gridcolor="#eee")

    # Add annotation explaining the finding
    fig.add_annotation(
        text="Coarser resolution = lower uniqueness = better privacy",
        xref="paper", yref="paper",
        x=0.5, y=-0.18,
        showarrow=False,
        font=dict(size=10, color="grey"),
    )

    return fig


# ─── 4. Entropy distribution histogram ──────────────────────────────

def create_entropy_distribution(entropy_data: dict) -> go.Figure:
    """
    Create a histogram of per-zone entropy values.

    Parameters:
        entropy_data: Output of calculate_trajectory_entropy() or
                      calculate_temporal_entropy(), containing
                      'entropy_per_zone' DataFrame.

    Returns:
        Plotly Figure with histogram and summary statistics.
    """
    entropy_df = entropy_data.get("entropy_per_zone", pd.DataFrame())

    if entropy_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No entropy data available", showarrow=False)
        return _apply_layout(fig, "Entropy Distribution")

    fig = go.Figure(data=go.Histogram(
        x=entropy_df["entropy"],
        nbinsx=30,
        marker_color=COLOURS["primary"],
        opacity=0.8,
        hovertemplate="Entropy: %{x:.2f} bits<br>Count: %{y}<extra></extra>",
    ))

    # Add vertical line for average
    avg = entropy_data.get("avg_entropy", 0)
    fig.add_vline(
        x=avg, line_dash="dash", line_color=COLOURS["critical"],
        annotation_text=f"Mean: {avg:.2f}",
        annotation_position="top right",
    )

    fig = _apply_layout(
        fig,
        "Distribution of Per-Zone Trajectory Entropy",
        xaxis_title="Shannon Entropy (bits)",
        yaxis_title="Number of Zones",
        height=400,
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


# ─── 5. Sensitivity analysis chart ──────────────────────────────────

def create_sensitivity_chart(sensitivity_df: pd.DataFrame) -> go.Figure:
    """
    Create a grouped bar chart comparing composite scores across
    different weight configurations.

    Parameters:
        sensitivity_df: DataFrame from sensitivity_analysis() with
                        columns: weight_set, overall_score, risk_level.

    Returns:
        Plotly Figure with colour-coded bars by risk level.
    """
    risk_colours = {
        "Critical": COLOURS["critical"],
        "High": COLOURS["high"],
        "Medium": COLOURS["medium"],
        "Low": COLOURS["low"],
    }

    bar_colours = [
        risk_colours.get(rl, COLOURS["secondary"])
        for rl in sensitivity_df["risk_level"]
    ]

    fig = go.Figure(data=go.Bar(
        x=sensitivity_df["weight_set"],
        y=sensitivity_df["overall_score"],
        marker_color=bar_colours,
        text=[f"{s:.1f}" for s in sensitivity_df["overall_score"]],
        textposition="outside",
        hovertemplate=(
            "Weights: %{x}<br>"
            "Score: %{y:.2f}<br>"
            "<extra></extra>"
        ),
    ))

    # Add risk threshold lines
    fig.add_hline(y=75, line_dash="dot", line_color=COLOURS["critical"],
                  annotation_text="Critical (75)")
    fig.add_hline(y=50, line_dash="dot", line_color=COLOURS["high"],
                  annotation_text="High (50)")
    fig.add_hline(y=25, line_dash="dot", line_color=COLOURS["medium"],
                  annotation_text="Medium (25)")

    fig = _apply_layout(
        fig,
        "Sensitivity Analysis: Risk Score Across Weight Configurations",
        xaxis_title="Weight Configuration",
        yaxis_title="Composite Risk Score (0-100)",
        height=450,
    )
    fig.update_yaxes(range=[0, 105], gridcolor="#eee")
    fig.update_xaxes(tickangle=15)
    return fig


# ─── 6. Zone risk choropleth / bar chart ────────────────────────────

def create_zone_risk_choropleth(
    zone_scores: pd.DataFrame,
    taxi_zones: pd.DataFrame,
) -> go.Figure:
    """
    Create a bar chart of privacy risk by borough/zone.

    If geometry data is unavailable (common), falls back to a grouped
    bar chart showing average entropy by borough. This is the practical
    approach since taxi_zones.csv does not include geometry.

    Parameters:
        zone_scores:  DataFrame with zone_id, entropy columns.
        taxi_zones:   Taxi zone lookup with LocationID, Borough, Zone.

    Returns:
        Plotly Figure — bar chart grouped by borough.
    """
    if zone_scores.empty or taxi_zones.empty:
        fig = go.Figure()
        fig.add_annotation(text="No zone data available", showarrow=False)
        return _apply_layout(fig, "Zone-Level Privacy Risk")

    # Merge zone info
    merged = zone_scores.merge(
        taxi_zones[["LocationID", "Borough", "Zone"]],
        left_on="zone_id",
        right_on="LocationID",
        how="left",
    )

    if "Borough" not in merged.columns or merged["Borough"].isna().all():
        # Fallback: plot by zone_id
        top_zones = zone_scores.nsmallest(20, "entropy")
        fig = go.Figure(data=go.Bar(
            x=top_zones["zone_id"].astype(str),
            y=top_zones["entropy"],
            marker_color=COLOURS["critical"],
        ))
        return _apply_layout(
            fig,
            "Top 20 Lowest Entropy Zones (Highest Risk)",
            xaxis_title="Zone ID",
            yaxis_title="Entropy (bits)",
        )

    # Borough-level aggregation
    borough_stats = merged.groupby("Borough").agg(
        avg_entropy=("entropy", "mean"),
        min_entropy=("entropy", "min"),
        n_zones=("zone_id", "count"),
    ).reset_index().sort_values("avg_entropy")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=borough_stats["Borough"],
        y=borough_stats["avg_entropy"],
        name="Average Entropy",
        marker_color=COLOURS["primary"],
        text=[f"{v:.2f}" for v in borough_stats["avg_entropy"]],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        x=borough_stats["Borough"],
        y=borough_stats["min_entropy"],
        name="Minimum Entropy",
        marker_color=COLOURS["critical"],
        text=[f"{v:.2f}" for v in borough_stats["min_entropy"]],
        textposition="outside",
    ))

    fig = _apply_layout(
        fig,
        "Zone Entropy by Borough (Lower = Higher Privacy Risk)",
        xaxis_title="Borough",
        yaxis_title="Entropy (bits)",
        height=450,
        barmode="group",
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


# ─── 7. Risk components radar chart ─────────────────────────────────

def create_risk_components_radar(components: dict[str, float]) -> go.Figure:
    """
    Create a radar chart showing the four risk component scores.

    Parameters:
        components: Dictionary with keys: uniqueness, k_anonymity,
                    entropy, linkage (each 0-100).

    Returns:
        Plotly Figure with radar/spider chart.
    """
    categories = list(components.keys())
    values = list(components.values())

    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values_closed,
        theta=[c.replace("_", " ").title() for c in categories_closed],
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.2)",
        line=dict(color=COLOURS["primary"], width=2),
        marker=dict(size=8, color=COLOURS["primary"]),
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))

    fig = _apply_layout(
        fig,
        "Privacy Risk Components",
        height=450,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#ddd"),
            angularaxis=dict(gridcolor="#ddd"),
        ),
    )
    return fig


if __name__ == "__main__":
    print("Visualisations — Self Test (generating sample charts)")
    print("=" * 60)

    # Test with mock data
    field_scores = {
        "PULocationID": 72.5,
        "DOLocationID": 71.8,
        "tpep_pickup_datetime": 55.3,
        "tpep_dropoff_datetime": 48.1,
        "payment_type": 25.2,
        "fare_amount": 35.6,
        "VendorID": 5.0,
    }

    fig = create_privacy_heatmap(field_scores)
    print(f"  Heatmap: {len(fig.data)} traces")

    k_metrics = {
        "k_distribution": {
            "k=1 (unique)": 5000,
            "k=2-5": 3000,
            "k=6-10": 1500,
            "k=11-50": 800,
            "k>50": 200,
        }
    }
    fig = create_k_distribution_chart(k_metrics)
    print(f"  k-Distribution: {len(fig.data)} traces")

    res_df = pd.DataFrame({
        "resolution": ["15-minute", "Hourly", "Daily"],
        "uniqueness_percentage": [85.3, 62.1, 15.7],
    })
    fig = create_uniqueness_by_resolution(res_df)
    print(f"  Uniqueness chart: {len(fig.data)} traces")

    components = {
        "uniqueness": 65.0,
        "k_anonymity": 42.0,
        "entropy": 55.0,
        "linkage": 28.0,
    }
    fig = create_risk_components_radar(components)
    print(f"  Radar chart: {len(fig.data)} traces")

    print("\n  All visualisations generated successfully.")
