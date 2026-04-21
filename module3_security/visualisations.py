"""
Publication-quality security assessment visualisations.

Generates Plotly figures for the NIST CSF compliance heatmap, gap
analysis charts, radar chart, framework comparison, and remediation
priority display.  All charts use a consistent academic colour palette
and are sized for dissertation embedding and Streamlit dashboard display.

References:
    NIST (2024) Cybersecurity Framework (CSF) 2.0. National Institute of
        Standards and Technology, Gaithersburg, MD.
    Tufte, E.R. (2001) The Visual Display of Quantitative Information.
        2nd edn. Cheshire, CT: Graphics Press.

Author: Jannat Rafique (B01798960)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NIST_FUNCTIONS

# ── Colour palette ──────────────────────────────────────────────────────
_COLOURS = {
    "PASS": "#2ecc71",      # green
    "PARTIAL": "#f39c12",   # amber
    "FAIL": "#e74c3c",      # red
    "N/A": "#95a5a6",       # grey
    "NOT ASSESSED": "#bdc3c7",
}

_STATUS_NUM = {"PASS": 3, "PARTIAL": 2, "FAIL": 1, "N/A": 0, "NOT ASSESSED": 0}

_ACADEMIC_LAYOUT = dict(
    font=dict(family="Times New Roman, serif", size=12, color="#2c3e50"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=80, r=40, t=60, b=60),
)


def create_compliance_heatmap(checklist_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of NIST CSF controls coloured by assessment result.

    Rows represent NIST functions (Identify, Protect, Detect, Respond,
    Recover); columns represent individual controls within each function.
    Cell colour encodes PASS (green), PARTIAL (amber), FAIL (red), or
    N/A (grey).

    Parameters:
        checklist_df: Evaluated NIST checklist DataFrame.

    Returns:
        plotly.graph_objects.Figure
    """
    # Build a matrix: rows = NIST functions, columns = controls
    # Since control counts differ per function, we pad with N/A
    max_controls = checklist_df.groupby("NIST_Function").size().max()

    z_values = []
    text_values = []
    y_labels = []
    x_labels = [f"Control {i+1}" for i in range(max_controls)]

    for func in NIST_FUNCTIONS:
        subset = checklist_df.loc[checklist_df["NIST_Function"] == func].reset_index()
        row_z = []
        row_text = []
        for i in range(max_controls):
            if i < len(subset):
                result = subset.iloc[i]["Assessment_Result"]
                ctrl_id = subset.iloc[i]["Control_ID"]
                row_z.append(_STATUS_NUM.get(result, 0))
                row_text.append(f"{ctrl_id}<br>{result}")
            else:
                row_z.append(-1)
                row_text.append("")
        z_values.append(row_z)
        text_values.append(row_text)
        y_labels.append(func)

    # Custom colourscale: -1=white (pad), 0=grey, 1=red, 2=amber, 3=green
    colorscale = [
        [0.0, "white"],
        [0.25, "#95a5a6"],
        [0.5, "#e74c3c"],
        [0.75, "#f39c12"],
        [1.0, "#2ecc71"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=text_values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=colorscale,
        zmin=-1,
        zmax=3,
        showscale=False,
        hovertemplate="Function: %{y}<br>%{text}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="NIST CSF 2.0 Compliance Heatmap", x=0.5),
        xaxis_title="Controls",
        yaxis_title="NIST Function",
        yaxis=dict(autorange="reversed"),
        height=400,
        width=800,
        **_ACADEMIC_LAYOUT,
    )
    return fig


def create_gap_analysis_chart(checklist_df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart showing PASS/PARTIAL/FAIL counts per NIST function.

    Parameters:
        checklist_df: Evaluated NIST checklist DataFrame.

    Returns:
        plotly.graph_objects.Figure
    """
    counts = (
        checklist_df.groupby(["NIST_Function", "Assessment_Result"])
        .size()
        .unstack(fill_value=0)
        .reindex(NIST_FUNCTIONS)
        .fillna(0)
    )

    fig = go.Figure()

    for status in ["PASS", "PARTIAL", "FAIL", "N/A"]:
        if status in counts.columns:
            fig.add_trace(go.Bar(
                name=status,
                x=counts.index,
                y=counts[status],
                marker_color=_COLOURS.get(status, "#bdc3c7"),
                text=counts[status].astype(int),
                textposition="inside",
            ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Gap Analysis by NIST CSF Function", x=0.5),
        xaxis_title="NIST CSF Function",
        yaxis_title="Number of Controls",
        legend_title="Assessment Result",
        height=450,
        width=750,
        **_ACADEMIC_LAYOUT,
    )
    return fig


def create_nist_radar_chart(checklist_df: pd.DataFrame) -> go.Figure:
    """
    Radar (spider) chart with five axes for NIST CSF functions.

    Each axis shows the compliance percentage for that function, where
    PASS = 100%, PARTIAL = 50%, FAIL/N/A = 0%.

    Parameters:
        checklist_df: Evaluated NIST checklist DataFrame.

    Returns:
        plotly.graph_objects.Figure
    """
    scores = []
    for func in NIST_FUNCTIONS:
        subset = checklist_df.loc[
            (checklist_df["NIST_Function"] == func)
            & (checklist_df["Assessment_Result"] != "N/A")
        ]
        if subset.empty:
            scores.append(0)
            continue
        total = len(subset)
        pass_count = (subset["Assessment_Result"] == "PASS").sum()
        partial_count = (subset["Assessment_Result"] == "PARTIAL").sum()
        pct = ((pass_count + 0.5 * partial_count) / total) * 100
        scores.append(round(pct, 1))

    # Close the polygon
    categories = NIST_FUNCTIONS + [NIST_FUNCTIONS[0]]
    values = scores + [scores[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(46, 204, 113, 0.25)",
        line=dict(color="#2ecc71", width=2),
        marker=dict(size=8, color="#2ecc71"),
        name="Compliance %",
        text=[f"{v:.0f}%" for v in values],
        textposition="top center",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(size=10),
            ),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        title=dict(text="NIST CSF 2.0 Compliance Radar", x=0.5),
        showlegend=False,
        height=500,
        width=550,
        **_ACADEMIC_LAYOUT,
    )
    return fig


def create_framework_comparison(compliance_scores: dict) -> go.Figure:
    """
    Grouped bar chart comparing GDPR, ISO 27001, and NIST compliance.

    Parameters:
        compliance_scores: Dict from
            :func:`compliance_matrix.calculate_compliance_scores`.

    Returns:
        plotly.graph_objects.Figure
    """
    frameworks = ["GDPR", "ISO 27001", "NIST CSF 2.0"]
    scores = [
        compliance_scores.get("gdpr_compliance_pct", 0),
        compliance_scores.get("iso_compliance_pct", 0),
        compliance_scores.get("nist_compliance_pct", 0),
    ]

    colours = ["#3498db", "#9b59b6", "#e67e22"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=frameworks,
        y=scores,
        marker_color=colours,
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        width=0.5,
    ))

    # Add overall compliance line
    overall = compliance_scores.get("overall_compliance_pct", 0)
    fig.add_hline(
        y=overall,
        line_dash="dash",
        line_color="#2c3e50",
        annotation_text=f"Overall: {overall:.1f}%",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text="Compliance Score by Regulatory Framework", x=0.5),
        xaxis_title="Framework",
        yaxis_title="Compliance (%)",
        yaxis=dict(range=[0, 105]),
        showlegend=False,
        height=450,
        width=650,
        **_ACADEMIC_LAYOUT,
    )
    return fig


def create_remediation_priority_chart(gap_summary: dict) -> go.Figure:
    """
    Horizontal bar chart of remediation items ranked by priority.

    Parameters:
        gap_summary: Dict mapping priority ('High', 'Medium', 'Low')
            to lists of gap description strings.

    Returns:
        plotly.graph_objects.Figure
    """
    items = []
    colours_list = []
    priority_colour = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#3498db"}

    for priority in ["High", "Medium", "Low"]:
        for desc in gap_summary.get(priority, []):
            # Truncate for display
            label = desc[:70] + "..." if len(desc) > 70 else desc
            items.append(label)
            colours_list.append(priority_colour[priority])

    if not items:
        fig = go.Figure()
        fig.add_annotation(
            text="No remediation items identified.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16),
        )
        fig.update_layout(height=300, width=700, **_ACADEMIC_LAYOUT)
        return fig

    # Reverse so highest priority is at top
    items = items[::-1]
    colours_list = colours_list[::-1]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=items,
        x=[1] * len(items),  # uniform bar length — priority is shown by colour
        orientation="h",
        marker_color=colours_list,
        hovertemplate="%{y}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Remediation Priorities", x=0.5),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(tickfont=dict(size=9)),
        showlegend=False,
        height=max(350, len(items) * 35),
        width=850,
        **_ACADEMIC_LAYOUT,
    )

    # Add legend manually
    for priority, colour in priority_colour.items():
        fig.add_trace(go.Bar(
            y=[None], x=[None],
            marker_color=colour,
            name=priority,
            showlegend=True,
        ))
    fig.update_layout(showlegend=True, legend_title="Priority")

    return fig


# ── Self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from module3_security.nist_checklist import evaluate_checklist
    from module3_security.compliance_matrix import (
        create_full_compliance_matrix,
        calculate_compliance_scores,
    )

    print("Generating visualisations...")
    checklist = evaluate_checklist()
    matrix = create_full_compliance_matrix(checklist)
    scores = calculate_compliance_scores(matrix)
    gap_summary = {"High": ["Encryption missing", "No breach plan"],
                   "Medium": ["Partial permissions"], "Low": []}

    fig1 = create_compliance_heatmap(checklist)
    fig2 = create_gap_analysis_chart(checklist)
    fig3 = create_nist_radar_chart(checklist)
    fig4 = create_framework_comparison(scores)
    fig5 = create_remediation_priority_chart(gap_summary)

    print("All 5 visualisations generated successfully.")
    # Optionally write to HTML for inspection:
    # fig1.write_html("heatmap.html", auto_open=True)
