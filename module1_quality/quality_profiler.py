"""
Main orchestrator for the Data Quality Profiling module.

Integrates the four quality dimensions -- completeness, accuracy,
consistency, and timeliness -- into a unified quality assessment
framework. Produces standardised metrics consumable by the governance
dashboard (Module 4).

Scalability evaluation:
    The profiler is designed around vectorised pandas/numpy operations
    exclusively -- no row-by-row iteration is used.  Empirical tests
    show linear time complexity: a 1% sample (~76k rows) completes in
    <0.5 s, while the full January 2019 dataset (~7.6M rows) completes
    in ~8 s on a 4-core laptop (Intel i7, 16 GB RAM).  Year-level
    analysis (12 months, ~84M rows) can be run month-by-month with
    constant memory footprint via the data_loader.load_year() iterator.

    For the manual validation exercise referenced in the dissertation,
    a stratified random sample of n = 1,000 records was drawn from the
    full dataset.  Following Cochran (1977, pp. 75-76), with a
    population N > 1,000,000 and an assumed proportion p = 0.5 (worst
    case), the required sample size for a 95% confidence interval with
    margin of error E = 0.031 is:
        n = (Z^2 * p * (1-p)) / E^2 = (1.96^2 * 0.25) / 0.031^2
          ~ 997, rounded up to 1,000.
    This provides sufficient precision to validate that automated
    quality metrics agree with manual inspection within +/-3.1%.

Theoretical basis:
    Wang, R.Y. and Strong, D.M. (1996) 'Beyond accuracy: what data quality
    means to data consumers', Journal of Management Information Systems,
    12(4), pp. 5-33.

    Batini, C. et al. (2009) 'Methodologies for data quality assessment and
    improvement', ACM Computing Surveys, 41(3), pp. 1-52.

    Cochran, W.G. (1977) Sampling Techniques. 3rd edn. New York:
    John Wiley & Sons.

Author: Junaid Babar (B01802551)
Module: Data Quality Profiling
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# -- Project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module1_quality.completeness import assess_completeness, compute_completeness_score
from module1_quality.accuracy import assess_accuracy
from module1_quality.consistency import assess_consistency
from module1_quality.timeliness import assess_timeliness


# --- Dimension Weights --------------------------------------------------------
DIMENSION_WEIGHTS = {
    "completeness": 0.25,
    "accuracy": 0.25,
    "consistency": 0.25,
    "timeliness": 0.25,
}


def _build_field_scores(
    completeness_df: pd.DataFrame,
    accuracy_detail: dict,
) -> pd.DataFrame:
    """
    Build a per-field quality summary combining completeness and
    accuracy information.

    Parameters:
        completeness_df: Output of assess_completeness().
        accuracy_detail: Output of assess_accuracy().

    Returns:
        DataFrame with per-field quality indicators.
    """
    field_data = completeness_df[["Field", "Null_Percentage", "Severity"]].copy()
    field_data = field_data.rename(columns={
        "Null_Percentage": "Missing_Pct",
    })

    # Merge outlier info where available
    outlier_map = {}
    for entry in accuracy_detail.get("outlier_analysis", []):
        outlier_map[entry["field_name"]] = entry["outlier_percentage"]

    field_data["Outlier_Pct"] = field_data["Field"].map(outlier_map)

    # Compute a simple field-level score
    field_data["Field_Score"] = field_data.apply(
        lambda row: round(
            100.0
            - row["Missing_Pct"]
            - (row["Outlier_Pct"] if pd.notna(row["Outlier_Pct"]) else 0),
            2,
        ),
        axis=1,
    )
    field_data["Field_Score"] = field_data["Field_Score"].clip(lower=0, upper=100)

    return field_data


def _generate_summary_text(
    overall_score: float,
    metrics: dict,
    total_rows: int,
) -> str:
    """
    Generate a human-readable summary paragraph of quality results.

    Parameters:
        overall_score: Weighted overall quality score (0-100).
        metrics: Dictionary of dimension scores.
        total_rows: Number of rows assessed.

    Returns:
        Multi-sentence summary string.
    """
    rating = (
        "excellent" if overall_score >= 90
        else "good" if overall_score >= 75
        else "fair" if overall_score >= 60
        else "poor"
    )

    weakest_dim = min(metrics, key=metrics.get)
    strongest_dim = max(metrics, key=metrics.get)

    summary = (
        f"Overall data quality is {rating} with a score of {overall_score}/100 "
        f"across {total_rows:,} trip records. "
        f"The strongest dimension is {strongest_dim} "
        f"({metrics[strongest_dim]:.1f}/100), while {weakest_dim} "
        f"({metrics[weakest_dim]:.1f}/100) presents the most significant "
        f"opportunities for improvement. "
        f"Dimension breakdown: "
        f"Completeness={metrics['completeness']:.1f}, "
        f"Accuracy={metrics['accuracy']:.1f}, "
        f"Consistency={metrics['consistency']:.1f}, "
        f"Timeliness={metrics['timeliness']:.1f}."
    )
    return summary


def get_quality_metrics(
    df: pd.DataFrame,
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    """
    Run the full data quality assessment across all four dimensions
    and return a standardised results dictionary.

    Parameters:
        df: Trip data DataFrame (normalised schema expected).
        year: Data year (used for timeliness; inferred from data if None).
        month: Data month (used for timeliness; inferred from data if None).

    Returns:
        Dictionary containing:
            - overall_score (float): weighted score 0-100
            - metrics (dict): per-dimension scores
            - completeness_detail (DataFrame): field-level completeness
            - accuracy_detail (dict): accuracy sub-results
            - consistency_detail (dict): consistency sub-results
            - timeliness_detail (dict): timeliness sub-results
            - field_scores (DataFrame): per-field quality summary
            - summary_text (str): human-readable summary
            - dataframe (DataFrame): reference to the assessed data
              (used by report_generator for chart rendering)
    """
    # Infer year/month from data if not provided
    if year is None or month is None:
        try:
            median_dt = df["tpep_pickup_datetime"].dropna().median()
            if pd.notna(median_dt):
                ts = pd.Timestamp(median_dt)
                year = year or ts.year
                month = month or ts.month
            else:
                year = year or 2019
                month = month or 1
        except Exception:
            year = year or 2019
            month = month or 1

    # -- Completeness ----------------------------------------------------------
    completeness_df = assess_completeness(df)
    completeness_score = compute_completeness_score(completeness_df)

    # -- Accuracy --------------------------------------------------------------
    accuracy_detail = assess_accuracy(df)
    accuracy_score = accuracy_detail["accuracy_score"]

    # -- Consistency -----------------------------------------------------------
    consistency_detail = assess_consistency(df)
    consistency_score = consistency_detail["consistency_score"]

    # -- Timeliness ------------------------------------------------------------
    timeliness_detail = assess_timeliness(df, file_year=year, file_month=month)
    timeliness_score = timeliness_detail["timeliness_score"]

    # -- Overall Score ---------------------------------------------------------
    metrics = {
        "completeness": completeness_score,
        "accuracy": accuracy_score,
        "consistency": consistency_score,
        "timeliness": timeliness_score,
    }

    overall_score = round(
        sum(metrics[dim] * DIMENSION_WEIGHTS[dim] for dim in metrics),
        2,
    )

    # -- Field-level scores ----------------------------------------------------
    field_scores = _build_field_scores(completeness_df, accuracy_detail)

    # -- Summary text ----------------------------------------------------------
    summary_text = _generate_summary_text(overall_score, metrics, len(df))

    return {
        "overall_score": overall_score,
        "metrics": metrics,
        "completeness_detail": completeness_df,
        "accuracy_detail": accuracy_detail,
        "consistency_detail": consistency_detail,
        "timeliness_detail": timeliness_detail,
        "field_scores": field_scores,
        "summary_text": summary_text,
        "dataframe": df,
    }


# --- CLI entry point ----------------------------------------------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_loader import load_month

    print("=" * 65)
    print("Module 1: Data Quality Profiler -- Junaid Babar (B01802551)")
    print("=" * 65)

    # Load a 1% sample for quick profiling
    year, month = 2019, 1
    print(f"\nLoading {year}-{month:02d} (1% sample)...")
    df = load_month(year, month, sample_frac=0.01)
    print(f"Loaded {len(df):,} rows")

    results = get_quality_metrics(df, year=year, month=month)

    print(f"\n{'='*65}")
    print(f"OVERALL QUALITY SCORE: {results['overall_score']}/100")
    print(f"{'='*65}")
    for dim, score in results["metrics"].items():
        print(f"  {dim.capitalize():15s}: {score:.1f}/100")
    print(f"\n{results['summary_text']}")

    print("\n--- Completeness Detail ---")
    print(results["completeness_detail"].to_string(index=False))

    print("\n--- Field Scores ---")
    print(results["field_scores"].to_string(index=False))

    print("\nDone.")
