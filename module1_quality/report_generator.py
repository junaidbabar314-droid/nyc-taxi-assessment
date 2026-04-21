"""
Report generation for the Data Quality Profiling module.

Produces HTML and CSV reports from quality assessment results. The HTML
report embeds interactive Plotly charts and provides an executive summary
suitable for inclusion in the dissertation appendix or stakeholder review.

Author: Junaid Babar (B01802551)
Module: Data Quality Profiling
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd

# -- Project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module1_quality.visualisations import (
    create_completeness_heatmap,
    create_outlier_boxplots,
    create_quality_summary_radar,
    create_timeliness_chart,
)


# --- HTML Template ------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Quality Assessment Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --primary: #3498db;
            --success: #2ecc71;
            --warning: #f39c12;
            --danger: #e74c3c;
            --bg: #fafafa;
            --text: #2c3e50;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: var(--text);
            background: var(--bg);
            line-height: 1.6;
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }}
        h2 {{
            color: var(--text);
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid #ddd;
        }}
        .score-card {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            margin: 1.5rem 0;
        }}
        .score-box {{
            flex: 1;
            min-width: 180px;
            background: white;
            border-radius: 8px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .score-box .label {{ font-size: 0.85rem; color: #777; }}
        .score-box .value {{
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0.3rem 0;
        }}
        .score-good {{ color: var(--success); }}
        .score-fair {{ color: var(--warning); }}
        .score-poor {{ color: var(--danger); }}
        .summary {{ background: white; padding: 1.5rem; border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 1rem 0; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        th {{ background: var(--primary); color: white; padding: 0.7rem; text-align: left; }}
        td {{ padding: 0.6rem 0.7rem; border-bottom: 1px solid #eee; }}
        tr:hover td {{ background: #f0f7ff; }}
        .chart-container {{ margin: 1.5rem 0; }}
        .severity-Critical {{ color: var(--danger); font-weight: 600; }}
        .severity-Minor {{ color: var(--warning); font-weight: 600; }}
        .severity-Good {{ color: var(--success); font-weight: 600; }}
        footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #ddd;
            font-size: 0.8rem;
            color: #999;
        }}
    </style>
</head>
<body>
    <h1>Data Quality Assessment Report</h1>
    <p><strong>Generated:</strong> {generated_date}</p>
    <p><strong>Records assessed:</strong> {total_rows:,}</p>

    <h2>Executive Summary</h2>
    <div class="score-card">
        <div class="score-box">
            <div class="label">Overall Score</div>
            <div class="value {overall_class}">{overall_score:.1f}</div>
            <div class="label">out of 100</div>
        </div>
        <div class="score-box">
            <div class="label">Completeness</div>
            <div class="value {comp_class}">{completeness:.1f}</div>
        </div>
        <div class="score-box">
            <div class="label">Accuracy</div>
            <div class="value {acc_class}">{accuracy:.1f}</div>
        </div>
        <div class="score-box">
            <div class="label">Consistency</div>
            <div class="value {cons_class}">{consistency:.1f}</div>
        </div>
        <div class="score-box">
            <div class="label">Timeliness</div>
            <div class="value {time_class}">{timeliness:.1f}</div>
        </div>
    </div>

    <div class="summary">
        <p>{summary_text}</p>
    </div>

    <h2>Quality Radar</h2>
    <div class="chart-container">{radar_chart}</div>

    <h2>Completeness Analysis</h2>
    <div class="chart-container">{completeness_chart}</div>
    {completeness_table}

    <h2>Accuracy Analysis</h2>
    <h3>Location ID Validation</h3>
    {location_table}

    <h3>Impossible Values</h3>
    {impossible_table}

    <h3>Outlier Analysis (IQR Method)</h3>
    {outlier_table}
    <div class="chart-container">{outlier_chart}</div>

    <h2>Consistency Analysis</h2>
    <h3>Fare Consistency</h3>
    {fare_table}

    <h3>Timestamp Validation</h3>
    {timestamp_table}

    <h3>Trip Speed Validation</h3>
    {speed_table}

    <h2>Timeliness Analysis</h2>
    <div class="chart-container">{timeliness_chart}</div>

    <h2>Field-Level Quality Scores</h2>
    {field_scores_table}

    <footer>
        <p>MSc IT (Data Analysis) Dissertation &mdash; Data Security and Privacy
        Assessment Framework for Big Data Transportation Systems</p>
        <p>Junaid Babar (B01802551) &mdash; University of the West of Scotland</p>
    </footer>
</body>
</html>
"""


def _score_class(score: float) -> str:
    """Map a score to a CSS class for colour coding."""
    if score >= 75:
        return "score-good"
    elif score >= 60:
        return "score-fair"
    return "score-poor"


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a styled HTML table string."""
    html = df.to_html(index=False, classes="", border=0, escape=False)
    # Add severity CSS classes
    for sev in ("Critical", "Minor", "Good"):
        html = html.replace(
            f"<td>{sev}</td>",
            f'<td class="severity-{sev}">{sev}</td>',
        )
    return html


def _dict_to_html_table(data: dict, title: str = "") -> str:
    """Convert a flat dictionary to a two-column HTML table."""
    rows = ""
    for k, v in data.items():
        if isinstance(v, float):
            display_v = f"{v:,.4f}"
        elif isinstance(v, int):
            display_v = f"{v:,}"
        else:
            display_v = str(v)
        rows += f"<tr><td><strong>{k}</strong></td><td>{display_v}</td></tr>\n"

    return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"


def _plotly_to_html_div(fig) -> str:
    """Convert a Plotly figure to an HTML div string."""
    return fig.to_html(full_html=False, include_plotlyjs=False)


# --- Main Report Generators ---------------------------------------------------

def generate_html_report(
    quality_results: dict,
    output_path: str,
) -> str:
    """
    Generate a comprehensive interactive HTML report from quality results.

    The report includes embedded Plotly charts, executive summary,
    and detailed metric tables. Suitable for browser viewing and
    dissertation appendix.

    Parameters:
        quality_results: Output of get_quality_metrics().  Must include
            a 'dataframe' key with the assessed DataFrame for chart
            rendering.
        output_path: File path for the output HTML file.

    Returns:
        Absolute path to the generated HTML file.
    """
    metrics = quality_results["metrics"]
    completeness_df = quality_results["completeness_detail"]
    accuracy_detail = quality_results["accuracy_detail"]
    consistency_detail = quality_results["consistency_detail"]
    timeliness_detail = quality_results["timeliness_detail"]
    field_scores = quality_results["field_scores"]

    # Retrieve the assessed DataFrame for chart rendering.
    # Falls back to an empty DataFrame if not present (legacy callers).
    trip_df = quality_results.get("dataframe", pd.DataFrame())

    # Generate charts
    radar_fig = create_quality_summary_radar(metrics)
    comp_fig = create_completeness_heatmap(completeness_df)

    # Outlier boxplots use the actual trip data so the chart is meaningful
    outlier_fields = [
        entry["field_name"]
        for entry in accuracy_detail.get("outlier_analysis", [])
    ]
    if not trip_df.empty and outlier_fields:
        outlier_fig = create_outlier_boxplots(trip_df, fields=outlier_fields)
    else:
        outlier_fig = create_outlier_boxplots(pd.DataFrame(), fields=[])

    timeliness_fig = create_timeliness_chart(timeliness_detail)

    # Build detail tables
    loc_data = accuracy_detail.get("location_validation", {})
    imp_data = accuracy_detail.get("impossible_values", {})
    outlier_data = accuracy_detail.get("outlier_analysis", [])
    fare_data = consistency_detail.get("fare_consistency", {})
    ts_data = consistency_detail.get("timestamp_validation", {})
    speed_data = consistency_detail.get("speed_validation", {})

    outlier_df = pd.DataFrame(outlier_data) if outlier_data else pd.DataFrame()

    # Total rows from completeness
    total_rows = int(completeness_df["Total_Rows"].iloc[0]) if len(completeness_df) > 0 else 0

    html = _HTML_TEMPLATE.format(
        generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_rows=total_rows,
        overall_score=quality_results["overall_score"],
        overall_class=_score_class(quality_results["overall_score"]),
        completeness=metrics["completeness"],
        comp_class=_score_class(metrics["completeness"]),
        accuracy=metrics["accuracy"],
        acc_class=_score_class(metrics["accuracy"]),
        consistency=metrics["consistency"],
        cons_class=_score_class(metrics["consistency"]),
        timeliness=metrics["timeliness"],
        time_class=_score_class(metrics["timeliness"]),
        summary_text=quality_results["summary_text"],
        radar_chart=_plotly_to_html_div(radar_fig),
        completeness_chart=_plotly_to_html_div(comp_fig),
        completeness_table=_df_to_html_table(completeness_df),
        location_table=_dict_to_html_table(loc_data),
        impossible_table=_dict_to_html_table(imp_data),
        outlier_table=_df_to_html_table(outlier_df) if not outlier_df.empty else "<p>No outlier data.</p>",
        outlier_chart=_plotly_to_html_div(outlier_fig),
        fare_table=_dict_to_html_table(fare_data),
        timestamp_table=_dict_to_html_table(ts_data),
        speed_table=_dict_to_html_table(speed_data),
        timeliness_chart=_plotly_to_html_div(timeliness_fig),
        field_scores_table=_df_to_html_table(field_scores),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    return str(output.resolve())


def generate_csv_report(
    quality_results: dict,
    output_path: str,
) -> str:
    """
    Export tabular quality metrics as CSV files.

    Generates two CSV files:
        1. {output_path}_completeness.csv -- field-level completeness
        2. {output_path}_field_scores.csv -- per-field quality summary

    Parameters:
        quality_results: Output of get_quality_metrics().
        output_path: Base file path (without extension).

    Returns:
        Absolute path to the base output directory.
    """
    output_base = Path(output_path)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    # Completeness detail
    comp_path = str(output_base) + "_completeness.csv"
    quality_results["completeness_detail"].to_csv(comp_path, index=False)

    # Field scores
    field_path = str(output_base) + "_field_scores.csv"
    quality_results["field_scores"].to_csv(field_path, index=False)

    # Summary metrics as a single-row CSV
    summary_path = str(output_base) + "_summary.csv"
    summary_data = {
        "overall_score": quality_results["overall_score"],
        **{f"{k}_score": v for k, v in quality_results["metrics"].items()},
    }
    pd.DataFrame([summary_data]).to_csv(summary_path, index=False)

    return str(output_base.parent.resolve())
