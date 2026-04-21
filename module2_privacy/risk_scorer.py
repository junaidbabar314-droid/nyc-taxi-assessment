"""
Privacy Risk Scoring Module for NYC Taxi Trip Records.

Computes a weighted composite privacy risk score from four sub-metrics:
uniqueness, k-anonymity, entropy, and linkage vulnerability. The scoring
model follows a configurable weighted-sum approach with sensitivity
analysis across alternative weight configurations to demonstrate
robustness — a critical requirement for MSc-level research.

Risk score interpretation (0-100):
  - Critical (>=75): Zone generalisation is ineffective; records are
    highly identifiable.
  - High (>=50): Significant re-identification risk; additional
    anonymisation measures required.
  - Medium (>=25): Moderate risk; zone generalisation provides partial
    protection.
  - Low (<25): Zone generalisation is effective for this data subset.

References:
    El Emam, K. and Dankar, F.K. (2008) 'Protecting privacy using
        k-anonymity', Journal of the American Medical Informatics
        Association, 15(5), pp. 627-637.
    Dwork, C. and Roth, A. (2014) 'The Algorithmic Foundations of
        Differential Privacy', Foundations and Trends in Theoretical
        Computer Science, 9(3-4), pp. 211-407.
    NIST (2020) 'De-Identification of Personal Information', NIST
        Special Publication 800-188.

Author: Sami Ullah (B01750598)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PRIVACY_WEIGHT_UNIQUENESS,
    PRIVACY_WEIGHT_K_ANONYMITY,
    PRIVACY_WEIGHT_ENTROPY,
    PRIVACY_WEIGHT_LINKAGE,
    PRIVACY_RISK_CRITICAL,
    PRIVACY_RISK_HIGH,
    PRIVACY_RISK_MEDIUM,
)


def _normalise_uniqueness(uniqueness_pct: float) -> float:
    """
    Convert uniqueness percentage to a 0-100 risk score.

    Higher uniqueness = higher risk. Direct mapping since
    uniqueness_pct is already 0-100.
    """
    return min(100.0, max(0.0, uniqueness_pct))


def _normalise_k_anonymity(k_metrics: dict) -> float:
    """
    Convert k-anonymity metrics to a 0-100 risk score.

    Uses records_below_k5_pct as the primary indicator: a higher
    percentage of records in small equivalence classes means higher risk.
    """
    pct_below_k5 = k_metrics.get("records_below_k5_pct", 0.0)
    return min(100.0, max(0.0, pct_below_k5))


def _normalise_entropy(avg_entropy: float, max_possible: float = 8.0) -> float:
    """
    Convert average entropy to a 0-100 risk score.

    Lower entropy = more predictable = higher risk. We invert the
    scale so that low entropy produces a high risk score.

    max_possible: theoretical maximum entropy for 265 zones is
    log2(265) ~ 8.05 bits.
    """
    if max_possible <= 0:
        return 50.0
    normalised = (1.0 - min(avg_entropy, max_possible) / max_possible) * 100.0
    return min(100.0, max(0.0, normalised))


def _normalise_linkage(linkage_rate: float) -> float:
    """
    Convert linkage rate to a 0-100 risk score.

    Higher linkage rate = more trips touchable by landmark-based
    attack = higher risk. Direct mapping.
    """
    return min(100.0, max(0.0, linkage_rate))


def _classify_risk(score: float) -> str:
    """Map a numeric score to a risk level label."""
    if score >= PRIVACY_RISK_CRITICAL:
        return "Critical"
    elif score >= PRIVACY_RISK_HIGH:
        return "High"
    elif score >= PRIVACY_RISK_MEDIUM:
        return "Medium"
    else:
        return "Low"


def calculate_privacy_risk_score(
    uniqueness_pct: float,
    k_metrics: dict,
    avg_entropy: float,
    linkage_rate: float,
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Calculate a weighted composite privacy risk score.

    Combines four normalised sub-scores using configurable weights
    (default from config.py). Each sub-metric is normalised to 0-100
    before weighting.

    Parameters:
        uniqueness_pct: Percentage of unique QI combinations (0-100).
        k_metrics:      Output of assess_k_anonymity() — must contain
                        'records_below_k5_pct'.
        avg_entropy:    Average trajectory entropy in bits.
        linkage_rate:   Percentage of trips touching landmark zones (0-100).
        weights:        Optional custom weights dict with keys:
                        uniqueness, k_anonymity, entropy, linkage.

    Returns:
        Dictionary with keys:
            - overall_score: float (0-100)
            - risk_level: str ('Low'/'Medium'/'High'/'Critical')
            - components: dict with individual normalised sub-scores
            - weights_used: dict of applied weights
    """
    if weights is None:
        weights = {
            "uniqueness": PRIVACY_WEIGHT_UNIQUENESS,
            "k_anonymity": PRIVACY_WEIGHT_K_ANONYMITY,
            "entropy": PRIVACY_WEIGHT_ENTROPY,
            "linkage": PRIVACY_WEIGHT_LINKAGE,
        }

    # Normalise each component to 0-100
    comp_uniqueness = _normalise_uniqueness(uniqueness_pct)
    comp_k_anon = _normalise_k_anonymity(k_metrics)
    comp_entropy = _normalise_entropy(avg_entropy)
    comp_linkage = _normalise_linkage(linkage_rate)

    # Weighted sum
    overall = (
        weights["uniqueness"] * comp_uniqueness
        + weights["k_anonymity"] * comp_k_anon
        + weights["entropy"] * comp_entropy
        + weights["linkage"] * comp_linkage
    )
    overall = min(100.0, max(0.0, overall))

    return {
        "overall_score": round(overall, 2),
        "risk_level": _classify_risk(overall),
        "components": {
            "uniqueness": round(comp_uniqueness, 2),
            "k_anonymity": round(comp_k_anon, 2),
            "entropy": round(comp_entropy, 2),
            "linkage": round(comp_linkage, 2),
        },
        "weights_used": weights,
    }


def sensitivity_analysis(
    uniqueness_pct: float,
    k_metrics: dict,
    avg_entropy: float,
    linkage_rate: float,
) -> pd.DataFrame:
    """
    Run privacy risk scoring with multiple weight configurations to
    demonstrate the robustness (or sensitivity) of the composite score.

    This is critical for MSc-level research: showing that the overall
    risk assessment is not an artefact of a single weight choice.

    Weight sets:
      1. Default (from config): 35/30/20/15
      2. Equal: 25/25/25/25
      3. Uniqueness-heavy: 50/20/15/15
      4. k-Anonymity-heavy: 20/50/15/15

    Parameters:
        uniqueness_pct: Percentage of unique QI combinations.
        k_metrics:      Output of assess_k_anonymity().
        avg_entropy:    Average trajectory entropy in bits.
        linkage_rate:   Percentage of trips in landmark zones.

    Returns:
        DataFrame with columns: weight_set, uniqueness_w, k_anonymity_w,
        entropy_w, linkage_w, overall_score, risk_level.
    """
    weight_sets = {
        "Default (35/30/20/15)": {
            "uniqueness": PRIVACY_WEIGHT_UNIQUENESS,
            "k_anonymity": PRIVACY_WEIGHT_K_ANONYMITY,
            "entropy": PRIVACY_WEIGHT_ENTROPY,
            "linkage": PRIVACY_WEIGHT_LINKAGE,
        },
        "Equal (25/25/25/25)": {
            "uniqueness": 0.25,
            "k_anonymity": 0.25,
            "entropy": 0.25,
            "linkage": 0.25,
        },
        "Uniqueness-heavy (50/20/15/15)": {
            "uniqueness": 0.50,
            "k_anonymity": 0.20,
            "entropy": 0.15,
            "linkage": 0.15,
        },
        "k-Anonymity-heavy (20/50/15/15)": {
            "uniqueness": 0.20,
            "k_anonymity": 0.50,
            "entropy": 0.15,
            "linkage": 0.15,
        },
    }

    rows = []
    for name, w in weight_sets.items():
        result = calculate_privacy_risk_score(
            uniqueness_pct, k_metrics, avg_entropy, linkage_rate, weights=w
        )
        rows.append({
            "weight_set": name,
            "uniqueness_w": w["uniqueness"],
            "k_anonymity_w": w["k_anonymity"],
            "entropy_w": w["entropy"],
            "linkage_w": w["linkage"],
            "overall_score": result["overall_score"],
            "risk_level": result["risk_level"],
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Risk Scorer — Self Test")
    print("=" * 60)

    # Simulated inputs
    test_uniqueness = 65.0
    test_k_metrics = {"records_below_k5_pct": 42.0}
    test_entropy = 4.5
    test_linkage = 28.0

    result = calculate_privacy_risk_score(
        test_uniqueness, test_k_metrics, test_entropy, test_linkage
    )
    print(f"\n  Overall score: {result['overall_score']:.2f}")
    print(f"  Risk level:    {result['risk_level']}")
    print(f"  Components:")
    for k, v in result["components"].items():
        print(f"    {k}: {v:.2f}")

    print("\nSensitivity Analysis:")
    sa = sensitivity_analysis(
        test_uniqueness, test_k_metrics, test_entropy, test_linkage
    )
    print(sa.to_string(index=False))
