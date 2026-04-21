# -*- coding: utf-8 -*-
"""
Comparative Analysis Module: 2019 vs 2024 NYC Taxi Data Governance.

Runs data quality profiling and privacy risk assessment on representative
months from both 2019 and 2024 NYC Yellow Taxi datasets, producing
structured comparative metrics and visualisations suitable for
Chapter 4 (Results) and Chapter 5 (Discussion) of the dissertation.

Key findings this module is designed to surface:
  - 2024 data uses zone-based location IDs exclusively (no GPS), reducing
    re-identification risk relative to any earlier GPS-era datasets.
  - Schema evolution between years (e.g., airport_fee introduced in 2024).
  - Quality dimension changes across the five-year gap.
  - Privacy posture improvement through NYC TLC's data governance reforms.

Methodology:
  A stratified temporal sampling approach is used: 3-4 months per year
  (Jan, Apr, Jul, Oct) provide seasonal coverage.  Each month is loaded
  with configurable sampling (default 1%) for tractable computation
  while maintaining statistical representativeness.

References:
    Wang, R.Y. and Strong, D.M. (1996) 'Beyond accuracy: what data quality
        means to data consumers', Journal of Management Information Systems,
        12(4), pp. 5-33.
    de Montjoye, Y.-A. et al. (2013) 'Unique in the Crowd: The privacy
        bounds of human mobility', Scientific Reports, 3, p. 1376.
    Sweeney, L. (2002) 'k-Anonymity: A model for protecting privacy',
        International Journal of Uncertainty, Fuzziness and Knowledge-Based
        Systems, 10(5), pp. 557-570.
    Hevner, A.R. et al. (2004) 'Design science in information systems
        research', MIS Quarterly, 28(1), pp. 75-105.
    NYC TLC (2024) TLC Trip Record Data. Available at:
        https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
        (Accessed: 15 March 2025).

Author: Junaid Babar (B01802551)
"""

from __future__ import annotations

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, OUTPUT_DIR, COLUMNS
from data_loader import load_month

from module1_quality.quality_profiler import get_quality_metrics
from module2_privacy.privacy_assessor import get_privacy_assessment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPRESENTATIVE_MONTHS = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct — seasonal coverage
DEFAULT_SAMPLE_FRAC = 0.01             # 1% sample for speed
FIGURE_DIR = OUTPUT_DIR / "figures"

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

# Greyscale-friendly colour palette
COLOUR_2019 = "#555555"
COLOUR_2024 = "#AAAAAA"
HATCH_2019 = "//"
HATCH_2024 = ".."


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Collection
# ═══════════════════════════════════════════════════════════════════════════

def _load_month_safe(
    year: int,
    month: int,
    sample_frac: float,
) -> Optional[pd.DataFrame]:
    """Load a single month, returning None on failure."""
    try:
        df = load_month(year, month, sample_frac=sample_frac)
        logger.info("Loaded %d-%02d: %d rows", year, month, len(df))
        return df
    except FileNotFoundError:
        logger.warning("File not found for %d-%02d — skipping", year, month)
        return None
    except Exception as exc:
        logger.error("Error loading %d-%02d: %s", year, month, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 2. Quality Comparison
# ═══════════════════════════════════════════════════════════════════════════

def _run_quality_for_year(
    year: int,
    months: list[int],
    sample_frac: float,
) -> dict[int, dict]:
    """
    Run quality profiling on each month for a given year.

    Returns:
        Dict mapping month number to quality metrics dict.
    """
    results = {}
    for m in months:
        df = _load_month_safe(year, m, sample_frac)
        if df is None:
            continue
        try:
            qm = get_quality_metrics(df, year=year, month=m)
            results[m] = qm
            logger.info(
                "Quality %d-%02d: overall=%.1f", year, m, qm["overall_score"]
            )
        except Exception as exc:
            logger.error("Quality profiling failed for %d-%02d: %s", year, m, exc)
    return results


def _compare_quality(
    q2019: dict[int, dict],
    q2024: dict[int, dict],
) -> dict[str, Any]:
    """
    Build comparative quality metrics from per-month results.

    Returns dict with:
        - avg_scores_2019 / avg_scores_2024: per-dimension averages
        - overall_2019 / overall_2024: overall score averages
        - monthly_scores: list of dicts for line-chart rendering
        - field_level_changes: per-field quality delta
        - schema_evolution: columns unique to each year
    """
    dimensions = ["completeness", "accuracy", "consistency", "timeliness"]

    def _avg_dim(results: dict[int, dict], dim: str) -> float:
        vals = [r["metrics"][dim] for r in results.values() if dim in r["metrics"]]
        return round(np.mean(vals), 2) if vals else 0.0

    def _avg_overall(results: dict[int, dict]) -> float:
        vals = [r["overall_score"] for r in results.values()]
        return round(np.mean(vals), 2) if vals else 0.0

    avg_2019 = {d: _avg_dim(q2019, d) for d in dimensions}
    avg_2024 = {d: _avg_dim(q2024, d) for d in dimensions}

    # Monthly scores for line charts
    monthly = []
    for year, results in [(2019, q2019), (2024, q2024)]:
        for m, r in sorted(results.items()):
            entry = {"year": year, "month": m, "month_name": MONTH_NAMES[m]}
            entry["overall"] = r["overall_score"]
            for d in dimensions:
                entry[d] = r["metrics"].get(d, None)
            monthly.append(entry)

    # Field-level comparison (averaged across months)
    field_scores_2019 = _aggregate_field_scores(q2019)
    field_scores_2024 = _aggregate_field_scores(q2024)
    field_changes = _compute_field_deltas(field_scores_2019, field_scores_2024)

    # Schema evolution
    cols_2019 = set()
    cols_2024 = set()
    for r in q2019.values():
        cols_2019.update(r["dataframe"].columns.tolist())
    for r in q2024.values():
        cols_2024.update(r["dataframe"].columns.tolist())

    schema_evolution = {
        "only_2019": sorted(cols_2019 - cols_2024),
        "only_2024": sorted(cols_2024 - cols_2019),
        "common": sorted(cols_2019 & cols_2024),
    }

    return {
        "avg_scores_2019": avg_2019,
        "avg_scores_2024": avg_2024,
        "overall_2019": _avg_overall(q2019),
        "overall_2024": _avg_overall(q2024),
        "monthly_scores": monthly,
        "field_level_changes": field_changes,
        "schema_evolution": schema_evolution,
        "n_months_2019": len(q2019),
        "n_months_2024": len(q2024),
    }


def _aggregate_field_scores(year_results: dict[int, dict]) -> dict[str, float]:
    """Average field scores across months for a single year."""
    field_totals: dict[str, list[float]] = {}
    for r in year_results.values():
        fs = r.get("field_scores")
        if fs is None or fs.empty:
            continue
        for _, row in fs.iterrows():
            fname = row["Field"]
            score = row["Field_Score"]
            if pd.notna(score):
                field_totals.setdefault(fname, []).append(score)
    return {f: round(np.mean(v), 2) for f, v in field_totals.items()}


def _compute_field_deltas(
    fs_2019: dict[str, float],
    fs_2024: dict[str, float],
) -> list[dict]:
    """Compute per-field quality deltas between 2019 and 2024."""
    all_fields = sorted(set(fs_2019) | set(fs_2024))
    deltas = []
    for f in all_fields:
        s19 = fs_2019.get(f)
        s24 = fs_2024.get(f)
        delta = None
        direction = "N/A"
        if s19 is not None and s24 is not None:
            delta = round(s24 - s19, 2)
            direction = "improved" if delta > 0 else "degraded" if delta < 0 else "unchanged"
        elif s24 is not None:
            direction = "new in 2024"
        elif s19 is not None:
            direction = "removed in 2024"
        deltas.append({
            "field": f,
            "score_2019": s19,
            "score_2024": s24,
            "delta": delta,
            "direction": direction,
        })
    return deltas


# ═══════════════════════════════════════════════════════════════════════════
# 3. Privacy Comparison
# ═══════════════════════════════════════════════════════════════════════════

def _run_privacy_for_year(
    year: int,
    months: list[int],
    sample_frac: float,
) -> dict[int, dict]:
    """
    Run privacy assessment on each month for a given year.

    Returns:
        Dict mapping month number to privacy assessment dict.
    """
    results = {}
    for m in months:
        df = _load_month_safe(year, m, sample_frac)
        if df is None:
            continue
        try:
            pa = get_privacy_assessment(df, temporal_resolution="H")
            results[m] = pa
            logger.info(
                "Privacy %d-%02d: risk=%.1f (%s)",
                year, m, pa["overall_risk_score"], pa["risk_level"],
            )
        except Exception as exc:
            logger.error("Privacy assessment failed for %d-%02d: %s", year, m, exc)
    return results


def _compare_privacy(
    p2019: dict[int, dict],
    p2024: dict[int, dict],
) -> dict[str, Any]:
    """
    Build comparative privacy metrics from per-month results.

    Returns dict with:
        - avg_risk_2019 / avg_risk_2024
        - avg_uniqueness_2019 / avg_uniqueness_2024
        - avg_k_min_2019 / avg_k_min_2024
        - avg_entropy_2019 / avg_entropy_2024
        - avg_linkage_2019 / avg_linkage_2024
        - monthly_privacy: list of dicts for charting
        - risk_reduction_pct: percentage improvement in risk score
        - key_finding: narrative string
    """

    def _safe_mean(results: dict[int, dict], key: str) -> float:
        vals = [r[key] for r in results.values() if key in r and r[key] is not None]
        return round(np.mean(vals), 2) if vals else 0.0

    def _safe_mean_nested(results: dict[int, dict], outer: str, inner: str) -> float:
        vals = []
        for r in results.values():
            nested = r.get(outer, {})
            if isinstance(nested, dict) and inner in nested:
                v = nested[inner]
                if v is not None:
                    vals.append(v)
        return round(np.mean(vals), 2) if vals else 0.0

    avg_risk_2019 = _safe_mean(p2019, "overall_risk_score")
    avg_risk_2024 = _safe_mean(p2024, "overall_risk_score")
    avg_uniq_2019 = _safe_mean(p2019, "uniqueness_percentage")
    avg_uniq_2024 = _safe_mean(p2024, "uniqueness_percentage")
    avg_k_min_2019 = _safe_mean_nested(p2019, "k_anonymity_summary", "min_k")
    avg_k_min_2024 = _safe_mean_nested(p2024, "k_anonymity_summary", "min_k")
    avg_entropy_2019 = _safe_mean(p2019, "avg_entropy")
    avg_entropy_2024 = _safe_mean(p2024, "avg_entropy")
    avg_linkage_2019 = _safe_mean(p2019, "linkage_rate")
    avg_linkage_2024 = _safe_mean(p2024, "linkage_rate")

    # Monthly privacy for charting
    monthly = []
    for year, results in [(2019, p2019), (2024, p2024)]:
        for m, r in sorted(results.items()):
            monthly.append({
                "year": year,
                "month": m,
                "month_name": MONTH_NAMES[m],
                "risk_score": r["overall_risk_score"],
                "uniqueness": r["uniqueness_percentage"],
                "linkage_rate": r["linkage_rate"],
                "avg_entropy": r["avg_entropy"],
            })

    # Risk reduction calculation
    risk_reduction = 0.0
    if avg_risk_2019 > 0:
        risk_reduction = round(
            ((avg_risk_2019 - avg_risk_2024) / avg_risk_2019) * 100, 2
        )

    # Key finding narrative
    if avg_risk_2024 < avg_risk_2019:
        key_finding = (
            f"The 2024 dataset demonstrates a {abs(risk_reduction):.1f}% reduction "
            f"in composite privacy risk score compared to 2019 "
            f"(from {avg_risk_2019:.1f} to {avg_risk_2024:.1f}). "
            f"Both years use zone-based location IDs rather than GPS coordinates, "
            f"but changes in data volume and trip patterns between years "
            f"affect the uniqueness and k-anonymity characteristics."
        )
    elif avg_risk_2024 > avg_risk_2019:
        key_finding = (
            f"The 2024 dataset shows a {abs(risk_reduction):.1f}% increase "
            f"in composite privacy risk relative to 2019 "
            f"(from {avg_risk_2019:.1f} to {avg_risk_2024:.1f}). "
            f"While both years use zone-based generalisation, differences "
            f"in trip volumes and patterns produce distinct privacy profiles."
        )
    else:
        key_finding = (
            f"The 2019 and 2024 datasets present comparable privacy risk "
            f"profiles (scores: {avg_risk_2019:.1f} vs {avg_risk_2024:.1f}), "
            f"both benefiting from the TLC's zone-based generalisation approach."
        )

    return {
        "avg_risk_2019": avg_risk_2019,
        "avg_risk_2024": avg_risk_2024,
        "avg_uniqueness_2019": avg_uniq_2019,
        "avg_uniqueness_2024": avg_uniq_2024,
        "avg_k_min_2019": avg_k_min_2019,
        "avg_k_min_2024": avg_k_min_2024,
        "avg_entropy_2019": avg_entropy_2019,
        "avg_entropy_2024": avg_entropy_2024,
        "avg_linkage_2019": avg_linkage_2019,
        "avg_linkage_2024": avg_linkage_2024,
        "monthly_privacy": monthly,
        "risk_reduction_pct": risk_reduction,
        "key_finding": key_finding,
        "n_months_2019": len(p2019),
        "n_months_2024": len(p2024),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_figure_dir() -> Path:
    """Create the figure output directory if needed."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR


def _setup_plot_style():
    """Configure matplotlib for academic, greyscale-friendly figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def generate_quality_bar_chart(quality_comparison: dict) -> Path:
    """
    Generate a grouped bar chart comparing quality dimension scores
    between 2019 and 2024.

    Parameters:
        quality_comparison: Output of _compare_quality().

    Returns:
        Path to saved figure.
    """
    _setup_plot_style()
    fig_dir = _ensure_figure_dir()

    dimensions = ["completeness", "accuracy", "consistency", "timeliness"]
    labels = [d.capitalize() for d in dimensions]
    scores_2019 = [quality_comparison["avg_scores_2019"][d] for d in dimensions]
    scores_2024 = [quality_comparison["avg_scores_2024"][d] for d in dimensions]

    x = np.arange(len(dimensions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_2019 = ax.bar(
        x - width / 2, scores_2019, width, label="2019",
        color=COLOUR_2019, hatch=HATCH_2019, edgecolor="black", linewidth=0.8,
    )
    bars_2024 = ax.bar(
        x + width / 2, scores_2024, width, label="2024",
        color=COLOUR_2024, hatch=HATCH_2024, edgecolor="black", linewidth=0.8,
    )

    # Value labels on bars
    for bars in [bars_2019, bars_2024]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Quality Dimension")
    ax.set_ylabel("Score (0-100)")
    ax.set_title("Data Quality Scores by Dimension: 2019 vs 2024")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    path = fig_dir / "comparative_quality_dimensions.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved quality bar chart: %s", path)
    return path


def generate_quality_line_chart(quality_comparison: dict) -> Path:
    """
    Generate a line chart showing overall quality scores across months
    for both 2019 and 2024.

    Parameters:
        quality_comparison: Output of _compare_quality().

    Returns:
        Path to saved figure.
    """
    _setup_plot_style()
    fig_dir = _ensure_figure_dir()

    monthly = quality_comparison["monthly_scores"]
    if not monthly:
        logger.warning("No monthly data for quality line chart")
        return fig_dir / "comparative_quality_monthly.png"

    df = pd.DataFrame(monthly)

    fig, ax = plt.subplots(figsize=(8, 5))

    for year, colour, marker in [(2019, COLOUR_2019, "s"), (2024, COLOUR_2024, "o")]:
        subset = df[df["year"] == year].sort_values("month")
        if subset.empty:
            continue
        ax.plot(
            subset["month_name"], subset["overall"],
            marker=marker, color=colour, linewidth=2,
            label=str(year), markersize=7,
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Overall Quality Score (0-100)")
    ax.set_title("Overall Quality Score Trend: 2019 vs 2024")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")

    path = fig_dir / "comparative_quality_monthly.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved quality line chart: %s", path)
    return path


def generate_privacy_comparison_chart(privacy_comparison: dict) -> Path:
    """
    Generate a grouped bar chart comparing key privacy metrics
    between 2019 and 2024.

    Parameters:
        privacy_comparison: Output of _compare_privacy().

    Returns:
        Path to saved figure.
    """
    _setup_plot_style()
    fig_dir = _ensure_figure_dir()

    metrics = ["Risk Score", "Uniqueness (%)", "Linkage Rate (%)", "Entropy (bits)"]
    vals_2019 = [
        privacy_comparison["avg_risk_2019"],
        privacy_comparison["avg_uniqueness_2019"],
        privacy_comparison["avg_linkage_2019"],
        privacy_comparison["avg_entropy_2019"],
    ]
    vals_2024 = [
        privacy_comparison["avg_risk_2024"],
        privacy_comparison["avg_uniqueness_2024"],
        privacy_comparison["avg_linkage_2024"],
        privacy_comparison["avg_entropy_2024"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_2019 = ax.bar(
        x - width / 2, vals_2019, width, label="2019",
        color=COLOUR_2019, hatch=HATCH_2019, edgecolor="black", linewidth=0.8,
    )
    bars_2024 = ax.bar(
        x + width / 2, vals_2024, width, label="2024",
        color=COLOUR_2024, hatch=HATCH_2024, edgecolor="black", linewidth=0.8,
    )

    for bars in [bars_2019, bars_2024]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Privacy Metric")
    ax.set_ylabel("Value")
    ax.set_title("Privacy Risk Metrics: 2019 vs 2024")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    path = fig_dir / "comparative_privacy_metrics.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved privacy comparison chart: %s", path)
    return path


def generate_privacy_monthly_chart(privacy_comparison: dict) -> Path:
    """
    Generate a line chart of privacy risk scores across months.

    Parameters:
        privacy_comparison: Output of _compare_privacy().

    Returns:
        Path to saved figure.
    """
    _setup_plot_style()
    fig_dir = _ensure_figure_dir()

    monthly = privacy_comparison["monthly_privacy"]
    if not monthly:
        logger.warning("No monthly data for privacy line chart")
        return fig_dir / "comparative_privacy_monthly.png"

    df = pd.DataFrame(monthly)

    fig, ax = plt.subplots(figsize=(8, 5))

    for year, colour, marker in [(2019, COLOUR_2019, "s"), (2024, COLOUR_2024, "o")]:
        subset = df[df["year"] == year].sort_values("month")
        if subset.empty:
            continue
        ax.plot(
            subset["month_name"], subset["risk_score"],
            marker=marker, color=colour, linewidth=2,
            label=str(year), markersize=7,
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Privacy Risk Score (0-100)")
    ax.set_title("Privacy Risk Score Trend: 2019 vs 2024")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")

    path = fig_dir / "comparative_privacy_monthly.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved privacy monthly chart: %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 5. Report Narrative
# ═══════════════════════════════════════════════════════════════════════════

def generate_comparative_report_text(results: dict) -> str:
    """
    Generate a multi-paragraph academic narrative summarising the
    comparative analysis results. Suitable for insertion into
    Chapter 4 (Results) and Chapter 5 (Discussion) of the dissertation.

    Parameters:
        results: Output of run_comparative().

    Returns:
        Multi-paragraph string with Harvard-style in-text references.
    """
    qc = results["quality_comparison"]
    pc = results["privacy_comparison"]

    paragraphs = []

    # --- Paragraph 1: Overview ---
    paragraphs.append(
        f"The comparative analysis examined {qc['n_months_2019']} months of 2019 "
        f"data and {qc['n_months_2024']} months of 2024 data from the NYC Yellow "
        f"Taxi Trip Records, assessing both data quality and privacy risk "
        f"characteristics. This longitudinal comparison spans a five-year period "
        f"during which the NYC Taxi and Limousine Commission (TLC) continued to "
        f"refine its data governance practices (NYC TLC, 2024)."
    )

    # --- Paragraph 2: Quality overview ---
    overall_delta = round(qc["overall_2024"] - qc["overall_2019"], 2)
    direction = "improvement" if overall_delta > 0 else "decline" if overall_delta < 0 else "stability"
    paragraphs.append(
        f"Overall data quality scores averaged {qc['overall_2019']:.1f}/100 for "
        f"2019 and {qc['overall_2024']:.1f}/100 for 2024, representing a "
        f"{abs(overall_delta):.1f}-point {direction}. Applying the Wang and Strong "
        f"(1996) multi-dimensional framework, the four quality dimensions exhibited "
        f"distinct patterns: completeness ({qc['avg_scores_2019']['completeness']:.1f} "
        f"vs {qc['avg_scores_2024']['completeness']:.1f}), accuracy "
        f"({qc['avg_scores_2019']['accuracy']:.1f} vs "
        f"{qc['avg_scores_2024']['accuracy']:.1f}), consistency "
        f"({qc['avg_scores_2019']['consistency']:.1f} vs "
        f"{qc['avg_scores_2024']['consistency']:.1f}), and timeliness "
        f"({qc['avg_scores_2019']['timeliness']:.1f} vs "
        f"{qc['avg_scores_2024']['timeliness']:.1f})."
    )

    # --- Paragraph 3: Schema evolution ---
    only_2024 = qc["schema_evolution"]["only_2024"]
    only_2019 = qc["schema_evolution"]["only_2019"]
    schema_notes = []
    if only_2024:
        schema_notes.append(
            f"columns introduced in 2024 ({', '.join(only_2024)})"
        )
    if only_2019:
        schema_notes.append(
            f"columns present in 2019 but absent in 2024 ({', '.join(only_2019)})"
        )
    if schema_notes:
        paragraphs.append(
            f"Schema evolution between the two periods revealed "
            f"{' and '.join(schema_notes)}. This structural change reflects the "
            f"TLC's ongoing adjustments to its data collection framework, including "
            f"the addition of airport-related surcharge fields to support evolving "
            f"fare structures."
        )
    else:
        paragraphs.append(
            "The schema remained consistent between 2019 and 2024, with both "
            "years sharing an identical set of columns."
        )

    # --- Paragraph 4: Field-level quality ---
    field_changes = qc["field_level_changes"]
    improved = [f for f in field_changes if f["direction"] == "improved"]
    degraded = [f for f in field_changes if f["direction"] == "degraded"]
    if improved or degraded:
        parts = []
        if improved:
            top_improved = sorted(improved, key=lambda x: x["delta"] or 0, reverse=True)[:3]
            names = ", ".join(f["field"] for f in top_improved)
            parts.append(f"fields showing quality improvement include {names}")
        if degraded:
            top_degraded = sorted(degraded, key=lambda x: x["delta"] or 0)[:3]
            names = ", ".join(f["field"] for f in top_degraded)
            parts.append(f"fields with quality degradation include {names}")
        paragraphs.append(
            f"At the field level, {'; '.join(parts)}. These field-level changes "
            f"provide actionable insights for data stewardship, consistent with "
            f"the continuous improvement philosophy advocated by Batini et al. (2009)."
        )

    # --- Paragraph 5: Privacy comparison ---
    paragraphs.append(
        f"Privacy risk assessment using the composite scoring methodology "
        f"(incorporating uniqueness, k-anonymity, entropy, and linkage attack "
        f"simulation) yielded average risk scores of {pc['avg_risk_2019']:.1f}/100 "
        f"for 2019 and {pc['avg_risk_2024']:.1f}/100 for 2024. "
        f"{pc['key_finding']}"
    )

    # --- Paragraph 6: Privacy details ---
    paragraphs.append(
        f"Uniqueness analysis, following de Montjoye et al. (2013), showed "
        f"that {pc['avg_uniqueness_2019']:.1f}% of 2019 trips and "
        f"{pc['avg_uniqueness_2024']:.1f}% of 2024 trips had unique "
        f"quasi-identifier combinations at hourly temporal resolution. "
        f"Minimum k-anonymity values averaged {pc['avg_k_min_2019']:.0f} (2019) "
        f"and {pc['avg_k_min_2024']:.0f} (2024), while trajectory entropy "
        f"averaged {pc['avg_entropy_2019']:.2f} bits (2019) and "
        f"{pc['avg_entropy_2024']:.2f} bits (2024). Linkage attack rates "
        f"were {pc['avg_linkage_2019']:.1f}% (2019) and "
        f"{pc['avg_linkage_2024']:.1f}% (2024), indicating the proportion "
        f"of trips involving zones associated with known NYC landmarks "
        f"(Sweeney, 2002)."
    )

    # --- Paragraph 7: Security note ---
    paragraphs.append(
        "The security assessment (Module 3) evaluates file-level and "
        "compliance-level controls that are independent of the data year. "
        "The same Parquet distribution mechanism, encryption status, and "
        "NIST CSF 2.0 compliance profile apply uniformly to both the 2019 "
        "and 2024 datasets. Consequently, security findings are reported "
        "as a single cross-year assessment rather than a year-on-year comparison."
    )

    # --- Paragraph 8: Governance narrative ---
    paragraphs.append(
        "Taken together, these findings illustrate the evolution of NYC TLC's "
        "data governance over a five-year period. The consistent use of "
        "zone-based generalisation across both years demonstrates an "
        "established commitment to privacy protection, while changes in data "
        "quality characteristics reflect the dynamic nature of large-scale "
        "transportation data collection systems. This comparative analysis "
        "provides empirical evidence supporting the Design Science Research "
        "artefact's ability to assess governance posture longitudinally "
        "(Hevner et al., 2004)."
    )

    return "\n\n".join(paragraphs)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_comparative(
    months: Optional[list[int]] = None,
    sample_frac: float = DEFAULT_SAMPLE_FRAC,
    generate_figures: bool = True,
) -> dict[str, Any]:
    """
    Run the full 2019 vs 2024 comparative analysis.

    This is the primary public interface for the module. It orchestrates
    data loading, quality profiling, privacy assessment, comparative
    metric computation, visualisation, and narrative generation.

    Parameters:
        months:           Months to assess (default: [1, 4, 7, 10]).
        sample_frac:      Fraction of each month to sample (default: 0.01).
        generate_figures: Whether to generate and save matplotlib figures.

    Returns:
        Structured dictionary with all comparative findings:
            - quality_comparison: dict of comparative quality metrics
            - privacy_comparison: dict of comparative privacy metrics
            - quality_raw_2019: per-month quality results for 2019
            - quality_raw_2024: per-month quality results for 2024
            - privacy_raw_2019: per-month privacy results for 2019
            - privacy_raw_2024: per-month privacy results for 2024
            - report_text: multi-paragraph narrative for dissertation
            - figures: list of Path objects to generated figures
            - metadata: run configuration and timestamps
    """
    if months is None:
        months = REPRESENTATIVE_MONTHS

    logger.info("=" * 65)
    logger.info("COMPARATIVE ANALYSIS: 2019 vs 2024")
    logger.info("Months: %s | Sample: %.1f%%", months, sample_frac * 100)
    logger.info("=" * 65)

    start_time = datetime.now()

    # --- Quality profiling ---
    logger.info("\n--- Phase 1/3: Quality Profiling ---")
    q2019 = _run_quality_for_year(2019, months, sample_frac)
    q2024 = _run_quality_for_year(2024, months, sample_frac)
    quality_comparison = _compare_quality(q2019, q2024)

    # --- Privacy assessment ---
    logger.info("\n--- Phase 2/3: Privacy Assessment ---")
    p2019 = _run_privacy_for_year(2019, months, sample_frac)
    p2024 = _run_privacy_for_year(2024, months, sample_frac)
    privacy_comparison = _compare_privacy(p2019, p2024)

    # --- Build results dict (needed for report text and figures) ---
    results = {
        "quality_comparison": quality_comparison,
        "privacy_comparison": privacy_comparison,
        "quality_raw_2019": q2019,
        "quality_raw_2024": q2024,
        "privacy_raw_2019": p2019,
        "privacy_raw_2024": p2024,
        "figures": [],
        "metadata": {
            "months_assessed": months,
            "sample_frac": sample_frac,
            "start_time": start_time.isoformat(),
            "years": [2019, 2024],
        },
    }

    # --- Visualisations ---
    if generate_figures:
        logger.info("\n--- Phase 3/3: Generating Figures ---")
        figures = []
        try:
            figures.append(generate_quality_bar_chart(quality_comparison))
        except Exception as exc:
            logger.error("Failed to generate quality bar chart: %s", exc)
        try:
            figures.append(generate_quality_line_chart(quality_comparison))
        except Exception as exc:
            logger.error("Failed to generate quality line chart: %s", exc)
        try:
            figures.append(generate_privacy_comparison_chart(privacy_comparison))
        except Exception as exc:
            logger.error("Failed to generate privacy comparison chart: %s", exc)
        try:
            figures.append(generate_privacy_monthly_chart(privacy_comparison))
        except Exception as exc:
            logger.error("Failed to generate privacy monthly chart: %s", exc)
        results["figures"] = figures

    # --- Report narrative ---
    results["report_text"] = generate_comparative_report_text(results)

    # --- Finalise ---
    end_time = datetime.now()
    results["metadata"]["end_time"] = end_time.isoformat()
    results["metadata"]["duration_seconds"] = (end_time - start_time).total_seconds()

    logger.info(
        "Comparative analysis complete in %.1f seconds",
        results["metadata"]["duration_seconds"],
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 7. CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def _print_results(results: dict) -> None:
    """Pretty-print key results to the console."""
    qc = results["quality_comparison"]
    pc = results["privacy_comparison"]

    print("\n" + "=" * 65)
    print("  COMPARATIVE ANALYSIS RESULTS: 2019 vs 2024")
    print("=" * 65)

    # Quality summary
    print("\n--- DATA QUALITY ---")
    print(f"  Overall Score:  2019 = {qc['overall_2019']:.1f}  |  2024 = {qc['overall_2024']:.1f}")
    print(f"  {'Dimension':<15s}  {'2019':>8s}  {'2024':>8s}  {'Delta':>8s}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}")
    for dim in ["completeness", "accuracy", "consistency", "timeliness"]:
        s19 = qc["avg_scores_2019"][dim]
        s24 = qc["avg_scores_2024"][dim]
        delta = s24 - s19
        sign = "+" if delta > 0 else ""
        print(f"  {dim.capitalize():<15s}  {s19:>8.1f}  {s24:>8.1f}  {sign}{delta:>7.1f}")

    # Schema evolution
    schema = qc["schema_evolution"]
    if schema["only_2024"]:
        print(f"\n  New columns in 2024: {', '.join(schema['only_2024'])}")
    if schema["only_2019"]:
        print(f"  Removed in 2024:     {', '.join(schema['only_2019'])}")

    # Field-level changes
    field_changes = qc["field_level_changes"]
    improved = [f for f in field_changes if f["direction"] == "improved"]
    degraded = [f for f in field_changes if f["direction"] == "degraded"]
    if improved:
        top = sorted(improved, key=lambda x: x["delta"] or 0, reverse=True)[:5]
        print(f"\n  Top improved fields:")
        for f in top:
            print(f"    {f['field']:<30s}  {f['score_2019']:.1f} -> {f['score_2024']:.1f}  (+{f['delta']:.1f})")
    if degraded:
        top = sorted(degraded, key=lambda x: x["delta"] or 0)[:5]
        print(f"\n  Top degraded fields:")
        for f in top:
            print(f"    {f['field']:<30s}  {f['score_2019']:.1f} -> {f['score_2024']:.1f}  ({f['delta']:.1f})")

    # Privacy summary
    print("\n--- PRIVACY RISK ---")
    print(f"  Risk Score:     2019 = {pc['avg_risk_2019']:.1f}  |  2024 = {pc['avg_risk_2024']:.1f}")
    print(f"  Uniqueness:     2019 = {pc['avg_uniqueness_2019']:.1f}%  |  2024 = {pc['avg_uniqueness_2024']:.1f}%")
    print(f"  Min k-anon:     2019 = {pc['avg_k_min_2019']:.0f}    |  2024 = {pc['avg_k_min_2024']:.0f}")
    print(f"  Entropy:        2019 = {pc['avg_entropy_2019']:.2f}  |  2024 = {pc['avg_entropy_2024']:.2f}")
    print(f"  Linkage Rate:   2019 = {pc['avg_linkage_2019']:.1f}%  |  2024 = {pc['avg_linkage_2024']:.1f}%")
    print(f"  Risk Reduction: {pc['risk_reduction_pct']:.1f}%")
    print(f"\n  Key Finding: {pc['key_finding']}")

    # Security note
    print("\n--- SECURITY ---")
    print("  Security assessment is file-level and year-independent.")
    print("  Both 2019 and 2024 datasets share the same compliance profile.")

    # Figures
    if results["figures"]:
        print(f"\n--- FIGURES ({len(results['figures'])}) ---")
        for fig_path in results["figures"]:
            print(f"  {fig_path}")

    # Timing
    meta = results["metadata"]
    print(f"\n--- METADATA ---")
    print(f"  Months assessed: {meta['months_assessed']}")
    print(f"  Sample fraction: {meta['sample_frac']*100:.0f}%")
    print(f"  Duration:        {meta['duration_seconds']:.1f}s")

    # Report text preview
    print(f"\n--- REPORT TEXT (first 500 chars) ---")
    print(results["report_text"][:500] + "...")
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress verbose pandas/numpy warnings during batch processing
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

    print("=" * 65)
    print("  Comparative Analysis Module")
    print("  2019 vs 2024 NYC Taxi Data Governance Assessment")
    print("  Junaid Babar (B01802551)")
    print("=" * 65)

    results = run_comparative(
        months=REPRESENTATIVE_MONTHS,
        sample_frac=DEFAULT_SAMPLE_FRAC,
        generate_figures=True,
    )

    _print_results(results)
    print("Comparative analysis complete.")
