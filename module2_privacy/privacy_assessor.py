"""
Privacy Assessment Orchestrator for NYC Taxi Trip Records.

Main entry point for Module 2 (Privacy Risk Detection). Coordinates
all sub-modules — PII classification, uniqueness analysis, k-anonymity,
entropy, linkage attack simulation, and risk scoring — into a single
comprehensive privacy assessment.

This module produces a standardised output dictionary consumed by
Module 4 (Streamlit governance dashboard) and used for the dissertation
results chapter.

Academic context:
  The NYC Taxi & Limousine Commission (TLC) applied zone-level
  generalisation as a privacy protection mechanism, replacing GPS
  coordinates with 265 taxi zone IDs. This assessment evaluates the
  effectiveness of that generalisation using multiple privacy metrics.

References:
    de Montjoye, Y.-A. et al. (2013) 'Unique in the Crowd: The privacy
        bounds of human mobility', Scientific Reports, 3, p. 1376.
    Sweeney, L. (2002) 'k-Anonymity: A model for protecting privacy',
        International Journal of Uncertainty, Fuzziness and Knowledge-Based
        Systems, 10(5), pp. 557-570.
    Dwork, C. and Roth, A. (2014) 'The Algorithmic Foundations of
        Differential Privacy', Foundations and Trends in Theoretical
        Computer Science, 9(3-4), pp. 211-407.

Author: Sami Ullah (B01750598)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from module2_privacy.pii_classifier import (
    classify_pii_fields,
    get_all_field_scores,
)
from module2_privacy.uniqueness import (
    calculate_uniqueness,
    compare_temporal_resolutions,
)
from module2_privacy.k_anonymity import assess_k_anonymity
from module2_privacy.entropy import (
    calculate_trajectory_entropy,
    calculate_temporal_entropy,
)
from module2_privacy.linkage_attack import simulate_linkage_attack
from module2_privacy.risk_scorer import (
    calculate_privacy_risk_score,
    sensitivity_analysis,
)


def _generate_summary_text(
    overall_score: float,
    risk_level: str,
    uniqueness_pct: float,
    k_metrics: dict,
    avg_entropy: float,
    linkage_rate: float,
    resolution: str,
) -> str:
    """
    Generate a human-readable summary paragraph for the assessment.

    Parameters:
        overall_score:  Composite risk score (0-100).
        risk_level:     Risk classification string.
        uniqueness_pct: Uniqueness percentage.
        k_metrics:      k-anonymity metrics dictionary.
        avg_entropy:    Average trajectory entropy.
        linkage_rate:   Linkage attack rate.
        resolution:     Temporal resolution used.

    Returns:
        Multi-sentence summary string suitable for dashboard display
        and dissertation results section.
    """
    summary_parts = [
        f"The overall privacy risk score is {overall_score:.1f}/100 "
        f"({risk_level} risk) at {resolution} temporal resolution.",
    ]

    # Uniqueness interpretation
    if uniqueness_pct > 70:
        summary_parts.append(
            f"Uniqueness is critically high at {uniqueness_pct:.1f}%, "
            f"indicating that the majority of quasi-identifier combinations "
            f"appear only once in the dataset."
        )
    elif uniqueness_pct > 40:
        summary_parts.append(
            f"Uniqueness is moderately high at {uniqueness_pct:.1f}%, "
            f"suggesting that a significant proportion of trips can be "
            f"distinguished by their zone-time combination."
        )
    else:
        summary_parts.append(
            f"Uniqueness is {uniqueness_pct:.1f}%, indicating that zone-level "
            f"generalisation provides reasonable protection at this resolution."
        )

    # k-anonymity interpretation
    below_k5 = k_metrics.get("records_below_k5_pct", 0)
    min_k = k_metrics.get("min_k", 0)
    summary_parts.append(
        f"The minimum k-anonymity value is {min_k}, with "
        f"{below_k5:.1f}% of records in equivalence classes smaller than 5."
    )

    # Entropy interpretation
    summary_parts.append(
        f"Average trajectory entropy is {avg_entropy:.2f} bits "
        f"({'low — trips are highly predictable' if avg_entropy < 3 else 'moderate — some diversity in destinations' if avg_entropy < 5 else 'high — destinations are well-distributed'})."
    )

    # Linkage interpretation
    summary_parts.append(
        f"Linkage attack simulation found that {linkage_rate:.1f}% of trips "
        f"involve zones associated with known NYC landmarks, making them "
        f"potentially identifiable through background knowledge."
    )

    return " ".join(summary_parts)


def get_privacy_assessment(
    df: pd.DataFrame,
    temporal_resolution: str = "H",
) -> dict[str, Any]:
    """
    Run the complete privacy risk assessment pipeline.

    Orchestrates all sub-modules and returns a standardised dictionary
    that serves as the interface between Module 2 and the governance
    dashboard (Module 4).

    Pipeline:
      1. PII field classification and scoring
      2. Quasi-identifier uniqueness analysis
      3. Multi-resolution uniqueness comparison
      4. k-anonymity assessment
      5. Trajectory and temporal entropy
      6. Linkage attack simulation
      7. Weighted composite risk scoring
      8. Sensitivity analysis across weight configurations

    Parameters:
        df:                   NYC Taxi trip DataFrame (raw, not cleaned).
        temporal_resolution:  Temporal rounding for QI analysis ('15min', 'H', 'D').

    Returns:
        Standardised dictionary with all assessment results:
            - overall_risk_score: float (0-100)
            - risk_level: str (Low/Medium/High/Critical)
            - pii_fields: list of fields classified as PII/QI
            - uniqueness_percentage: float
            - k_anonymity_summary: dict
            - avg_entropy: float
            - linkage_rate: float
            - sensitivity_analysis: pd.DataFrame
            - field_risk_scores: dict
            - resolution_comparison: pd.DataFrame
            - trajectory_entropy: dict (full results)
            - temporal_entropy: dict (full results)
            - linkage_details: dict (full results)
            - risk_components: dict (normalised sub-scores)
            - summary_text: str
    """
    # 1. PII classification
    classification = classify_pii_fields(df)
    pii_fields = [
        field for field, info in classification.items()
        if info["risk_level"] not in ("None", "Low")
    ]
    field_risk_scores = get_all_field_scores(df)

    # 2. Uniqueness analysis
    uniqueness_result = calculate_uniqueness(df, temporal_resolution=temporal_resolution)
    uniqueness_pct = uniqueness_result["uniqueness_percentage"]

    # 3. Multi-resolution comparison
    resolution_comparison = compare_temporal_resolutions(df)

    # 4. k-anonymity
    k_metrics = assess_k_anonymity(df, temporal_resolution=temporal_resolution)
    k_summary = {
        "min_k": k_metrics["min_k"],
        "max_k": k_metrics["max_k"],
        "mean_k": k_metrics["mean_k"],
        "median_k": k_metrics["median_k"],
        "records_below_k5_pct": k_metrics["records_below_k5_pct"],
        "total_equivalence_classes": k_metrics["total_equivalence_classes"],
        "k_distribution": k_metrics["k_distribution"],
    }

    # 5. Entropy analysis
    traj_entropy = calculate_trajectory_entropy(df)
    temp_entropy = calculate_temporal_entropy(df)
    avg_entropy = traj_entropy["avg_entropy"]

    # 6. Linkage attack
    linkage_result = simulate_linkage_attack(df)
    linkage_rate = linkage_result["linkage_rate"]

    # 7. Composite risk score
    risk_result = calculate_privacy_risk_score(
        uniqueness_pct, k_summary, avg_entropy, linkage_rate
    )

    # 8. Sensitivity analysis
    sa_df = sensitivity_analysis(
        uniqueness_pct, k_summary, avg_entropy, linkage_rate
    )

    # 9. Summary text
    summary_text = _generate_summary_text(
        risk_result["overall_score"],
        risk_result["risk_level"],
        uniqueness_pct,
        k_summary,
        avg_entropy,
        linkage_rate,
        uniqueness_result["resolution"],
    )

    return {
        "overall_risk_score": risk_result["overall_score"],
        "risk_level": risk_result["risk_level"],
        "pii_fields": pii_fields,
        "uniqueness_percentage": uniqueness_pct,
        "k_anonymity_summary": k_summary,
        "avg_entropy": avg_entropy,
        "linkage_rate": linkage_rate,
        "sensitivity_analysis": sa_df,
        "field_risk_scores": field_risk_scores,
        "resolution_comparison": resolution_comparison,
        "trajectory_entropy": traj_entropy,
        "temporal_entropy": temp_entropy,
        "linkage_details": linkage_result,
        "risk_components": risk_result["components"],
        "summary_text": summary_text,
    }


if __name__ == "__main__":
    import logging

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_loader import load_month

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Privacy Assessor — Full Pipeline Test")
    print("=" * 60)
    df = load_month(2019, 1, sample_frac=0.01)

    result = get_privacy_assessment(df, temporal_resolution="H")

    print(f"\nOverall Risk Score: {result['overall_risk_score']:.2f}/100")
    print(f"Risk Level:         {result['risk_level']}")
    print(f"Uniqueness:         {result['uniqueness_percentage']:.2f}%")
    print(f"Avg Entropy:        {result['avg_entropy']:.4f} bits")
    print(f"Linkage Rate:       {result['linkage_rate']:.2f}%")
    print(f"PII Fields:         {result['pii_fields']}")

    print(f"\nk-Anonymity Summary:")
    for k, v in result["k_anonymity_summary"].items():
        if k != "k_distribution":
            print(f"  {k}: {v}")

    print(f"\nRisk Components:")
    for k, v in result["risk_components"].items():
        print(f"  {k}: {v:.2f}")

    print(f"\nSensitivity Analysis:")
    print(result["sensitivity_analysis"].to_string(index=False))

    print(f"\nResolution Comparison:")
    print(result["resolution_comparison"].to_string(index=False))

    print(f"\nSummary:")
    print(result["summary_text"])
