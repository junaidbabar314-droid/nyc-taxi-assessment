"""
PII Classification Module for NYC Taxi Trip Records.

Classifies dataset fields into privacy-relevant categories and assigns
risk levels based on their potential for re-identification. The taxonomy
follows the quasi-identifier (QI) framework established by Sweeney (2002),
adapted for transportation data where direct identifiers have already been
removed through zone-level generalisation by NYC TLC.

Field categories:
  - Location QI (High risk): Zone IDs that encode spatial information.
  - Temporal QI (Medium risk): Timestamps enabling temporal correlation.
  - Behavioural QI (Low-Medium): Trip attributes that characterise behaviour.
  - Non-PII (None): Administrative or fixed-value fields.

References:
    Sweeney, L. (2002) 'k-Anonymity: A model for protecting privacy',
        International Journal of Uncertainty, Fuzziness and Knowledge-Based
        Systems, 10(5), pp. 557-570.
    Article 29 Data Protection Working Party (2014) 'Opinion 05/2014 on
        Anonymisation Techniques', WP216.

Author: Sami Ullah (B01750598)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PICKUP_LOCATION, DROPOFF_LOCATION, PICKUP_DATETIME, DROPOFF_DATETIME

# ─── Field classification taxonomy ─────────────────────────────────

PII_TAXONOMY: dict[str, dict[str, str]] = {
    # Location quasi-identifiers — high risk
    PICKUP_LOCATION: {
        "category": "Location QI",
        "risk_level": "High",
        "justification": (
            "Pickup zone encodes spatial origin of trip. Even after "
            "generalisation to zone level, trips from low-traffic zones "
            "or landmark locations can be re-identified (Sweeney, 2002)."
        ),
    },
    DROPOFF_LOCATION: {
        "category": "Location QI",
        "risk_level": "High",
        "justification": (
            "Dropoff zone encodes spatial destination. Combined with "
            "pickup zone, creates a trajectory that may be unique, "
            "especially for rare origin-destination pairs."
        ),
    },
    # Temporal quasi-identifiers — medium risk
    PICKUP_DATETIME: {
        "category": "Temporal QI",
        "risk_level": "Medium",
        "justification": (
            "Precise pickup timestamp enables temporal correlation. "
            "At second-level precision, timestamps significantly "
            "increase uniqueness of trip records (de Montjoye et al., 2013)."
        ),
    },
    DROPOFF_DATETIME: {
        "category": "Temporal QI",
        "risk_level": "Medium",
        "justification": (
            "Dropoff timestamp combined with pickup time reveals trip "
            "duration, which serves as an additional quasi-identifier."
        ),
    },
    # Behavioural quasi-identifiers — low to medium risk
    "payment_type": {
        "category": "Behavioural QI",
        "risk_level": "Low-Medium",
        "justification": (
            "Payment method reveals behavioural preference. Habitual "
            "patterns (e.g., always cash) can aid linkage attacks."
        ),
    },
    "fare_amount": {
        "category": "Behavioural QI",
        "risk_level": "Low-Medium",
        "justification": (
            "Fare is largely determined by distance and time but exact "
            "amounts can narrow down specific trips."
        ),
    },
    "tip_amount": {
        "category": "Behavioural QI",
        "risk_level": "Low-Medium",
        "justification": (
            "Tip amount is a personal choice that varies by individual. "
            "Unusual tip patterns could serve as a fingerprint."
        ),
    },
    "trip_distance": {
        "category": "Behavioural QI",
        "risk_level": "Low-Medium",
        "justification": (
            "Trip distance combined with zone IDs could help confirm "
            "a specific route, reducing the anonymity set."
        ),
    },
    "passenger_count": {
        "category": "Behavioural QI",
        "risk_level": "Low-Medium",
        "justification": (
            "Passenger count has low cardinality (1-6) but can serve "
            "as a supplementary quasi-identifier in combination attacks."
        ),
    },
    "total_amount": {
        "category": "Behavioural QI",
        "risk_level": "Low-Medium",
        "justification": (
            "Total fare encapsulates all charges. Precise totals can "
            "narrow the anonymity set when combined with other QIs."
        ),
    },
    # Non-PII — negligible privacy risk
    "VendorID": {
        "category": "Non-PII",
        "risk_level": "None",
        "justification": "Vendor identifier (1 or 2) has negligible cardinality and no personal link.",
    },
    "RatecodeID": {
        "category": "Non-PII",
        "risk_level": "None",
        "justification": "Rate code is a categorical fare structure indicator with few values.",
    },
    "store_and_fwd_flag": {
        "category": "Non-PII",
        "risk_level": "None",
        "justification": "Binary flag for store-and-forward; no privacy relevance.",
    },
    "improvement_surcharge": {
        "category": "Non-PII",
        "risk_level": "None",
        "justification": "Fixed surcharge ($0.30) applied uniformly; no discriminatory value.",
    },
    "mta_tax": {
        "category": "Non-PII",
        "risk_level": "None",
        "justification": "Fixed tax amount applied uniformly to all trips.",
    },
    "extra": {
        "category": "Non-PII",
        "risk_level": "None",
        "justification": "Miscellaneous surcharges (rush hour, overnight) with few values.",
    },
    "congestion_surcharge": {
        "category": "Non-PII",
        "risk_level": "None",
        "justification": "Fixed congestion surcharge; not available in 2019 data.",
    },
    "airport_fee": {
        "category": "Non-PII",
        "risk_level": "None",
        "justification": "Fixed airport fee; not available in 2019 data.",
    },
    "tolls_amount": {
        "category": "Non-PII",
        "risk_level": "None",
        "justification": "Toll amount is route-dependent but has low cardinality in practice.",
    },
}

# Risk level numeric mapping for scoring
_RISK_BASE_SCORES: dict[str, float] = {
    "High": 70.0,
    "Medium": 45.0,
    "Low-Medium": 25.0,
    "None": 5.0,
}


def classify_pii_fields(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """
    Classify each column of a taxi trip DataFrame into privacy categories.

    Applies the predefined PII taxonomy to every column present in the
    DataFrame. Columns not in the taxonomy are labelled as 'Unknown' with
    a Low risk level for conservative assessment.

    Parameters:
        df: NYC Taxi trip DataFrame with standard column names.

    Returns:
        Dictionary mapping field_name -> {category, risk_level, justification}.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'PULocationID': [1], 'VendorID': [2]})
        >>> result = classify_pii_fields(df)
        >>> result['PULocationID']['category']
        'Location QI'
    """
    classification: dict[str, dict[str, str]] = {}

    for col in df.columns:
        if col in PII_TAXONOMY:
            classification[col] = PII_TAXONOMY[col].copy()
        else:
            classification[col] = {
                "category": "Unknown",
                "risk_level": "Low",
                "justification": (
                    f"Column '{col}' is not in the predefined taxonomy. "
                    "Assigned conservative Low risk pending manual review."
                ),
            }

    return classification


def score_field_privacy_risk(
    field_name: str,
    df: pd.DataFrame,
    classification: dict[str, dict[str, str]],
) -> float:
    """
    Calculate a 0-100 privacy risk score for a single field.

    The score combines a base risk from the field's classification category
    with an adjustment based on the actual uniqueness (cardinality ratio)
    of values in the dataset. High-cardinality fields within high-risk
    categories receive the largest scores.

    Score formula:
        score = base_risk + (uniqueness_ratio * 30)
        where uniqueness_ratio = n_unique / n_total

    Parameters:
        field_name:     Name of the column to score.
        df:             DataFrame containing the column data.
        classification: Output of classify_pii_fields().

    Returns:
        Float between 0 and 100 representing the privacy risk score.

    Raises:
        KeyError: If field_name is not in the classification dict.
    """
    if field_name not in classification:
        raise KeyError(f"Field '{field_name}' not found in classification.")

    risk_level = classification[field_name]["risk_level"]
    base_score = _RISK_BASE_SCORES.get(risk_level, 10.0)

    # Compute uniqueness adjustment if column exists in the DataFrame
    if field_name in df.columns:
        n_total = len(df)
        if n_total > 0:
            n_unique = df[field_name].nunique()
            uniqueness_ratio = n_unique / n_total
        else:
            uniqueness_ratio = 0.0
    else:
        uniqueness_ratio = 0.0

    # Scale uniqueness contribution (max +30 points)
    uniqueness_adjustment = uniqueness_ratio * 30.0

    score = min(100.0, base_score + uniqueness_adjustment)
    return round(score, 2)


def get_all_field_scores(df: pd.DataFrame) -> dict[str, float]:
    """
    Score all fields in the DataFrame for privacy risk.

    Convenience function that classifies and scores every column.

    Parameters:
        df: NYC Taxi trip DataFrame.

    Returns:
        Dictionary mapping field_name -> risk score (0-100).
    """
    classification = classify_pii_fields(df)
    scores: dict[str, float] = {}
    for field_name in classification:
        scores[field_name] = score_field_privacy_risk(field_name, df, classification)
    return scores


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_loader import load_month

    print("PII Classifier — Self Test")
    print("=" * 60)
    df = load_month(2019, 1, sample_frac=0.01)
    classification = classify_pii_fields(df)

    print(f"\n{'Field':<30} {'Category':<20} {'Risk Level':<12}")
    print("-" * 62)
    for field, info in classification.items():
        print(f"{field:<30} {info['category']:<20} {info['risk_level']:<12}")

    print("\nField Risk Scores:")
    scores = get_all_field_scores(df)
    for field, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {field:<30} {score:>6.2f}")
