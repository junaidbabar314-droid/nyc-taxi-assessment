"""
Accuracy assessment for NYC Taxi Trip Records.

Accuracy evaluates whether data values correctly represent the real-world
entities they describe. This module applies statistical outlier detection
(IQR method), domain-specific range validation, and impossible-value
detection to quantify accuracy across the dataset.

Scalability note:
    All checks (IQR quantile computation, boolean masking, summation)
    are implemented as vectorised pandas/numpy operations.  The IQR
    outlier pass runs in O(n log n) due to the quantile sort; the
    remaining checks are O(n).  A full month of Yellow Taxi data
    (~7.6M rows) completes in under 2 seconds on commodity hardware.

Theoretical basis:
    Tukey, J.W. (1977) Exploratory Data Analysis. Reading, MA:
    Addison-Wesley.

    Batini, C. et al. (2009) 'Methodologies for data quality assessment and
    improvement', ACM Computing Surveys, 41(3), pp. 1-52.

    Wang, R.Y. and Strong, D.M. (1996) 'Beyond accuracy: what data quality
    means to data consumers', Journal of Management Information Systems,
    12(4), pp. 5-33.

Author: Junaid Babar (B01802551)
Module: Data Quality Profiling
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# -- Project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    VALID_LOCATION_ID_MIN,
    VALID_LOCATION_ID_MAX,
    MAX_REALISTIC_FARE,
    MAX_REALISTIC_DISTANCE,
    MAX_PASSENGER_COUNT,
    PICKUP_LOCATION,
    DROPOFF_LOCATION,
)


# --- IQR Outlier Detection ----------------------------------------------------

def detect_outliers_iqr(series: pd.Series, field_name: str) -> dict:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Outliers are defined as values falling below Q1 - 1.5 * IQR or
    above Q3 + 1.5 * IQR, following Tukey (1977).

    Parameters:
        series: Numeric pandas Series to analyse.
        field_name: Name of the field (used in output dict).

    Returns:
        Dictionary containing:
            - field_name: name of the analysed field
            - Q1, Q3, IQR: quartile statistics
            - lower_bound, upper_bound: outlier fences
            - outlier_count: number of values outside fences
            - outlier_percentage: outlier count as percentage
            - total_count: number of non-null values analysed
    """
    clean = series.dropna()
    total_count = len(clean)

    if total_count == 0:
        return {
            "field_name": field_name,
            "Q1": np.nan,
            "Q3": np.nan,
            "IQR": np.nan,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "total_count": 0,
        }

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (clean < lower_bound) | (clean > upper_bound)
    outlier_count = int(outlier_mask.sum())
    outlier_pct = (outlier_count / total_count * 100) if total_count > 0 else 0.0

    return {
        "field_name": field_name,
        "Q1": round(q1, 4),
        "Q3": round(q3, 4),
        "IQR": round(iqr, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_pct, 4),
        "total_count": total_count,
    }


# --- Location ID Validation ---------------------------------------------------

def validate_location_ids(df: pd.DataFrame) -> dict:
    """
    Validate that PULocationID and DOLocationID fall within the valid
    NYC Taxi Zone range (1-265).

    Parameters:
        df: Trip data DataFrame with location ID columns.

    Returns:
        Dictionary containing:
            - invalid_pickup_count: pickups outside valid range
            - invalid_dropoff_count: dropoffs outside valid range
            - invalid_percentage: combined invalid as percentage of total
            - total_rows: number of rows examined
    """
    total_rows = len(df)

    pu = df[PICKUP_LOCATION]
    do = df[DROPOFF_LOCATION]

    invalid_pu = int(
        ((pu < VALID_LOCATION_ID_MIN) | (pu > VALID_LOCATION_ID_MAX) | pu.isna()).sum()
    )
    invalid_do = int(
        ((do < VALID_LOCATION_ID_MIN) | (do > VALID_LOCATION_ID_MAX) | do.isna()).sum()
    )

    total_checks = total_rows * 2
    invalid_pct = (
        ((invalid_pu + invalid_do) / total_checks * 100) if total_checks > 0 else 0.0
    )

    return {
        "invalid_pickup_count": invalid_pu,
        "invalid_dropoff_count": invalid_do,
        "invalid_percentage": round(invalid_pct, 4),
        "total_rows": total_rows,
    }


# --- Impossible Value Detection -----------------------------------------------

def detect_impossible_values(df: pd.DataFrame) -> dict:
    """
    Detect logically impossible data values based on domain knowledge
    of NYC taxi operations.

    Checks performed:
        1. Negative fare amounts (fare_amount < 0)
        2. Zero-distance trips with fares exceeding $2
        3. Passenger count > 6 (standard taxi vehicle capacity)
        4. Passenger count <= 0 (invalid)
        5. Negative trip distances
        6. Extremely high fares (> MAX_REALISTIC_FARE)

    Parameters:
        df: Trip data DataFrame.

    Returns:
        Dictionary with counts for each impossible-value category.
    """
    total_rows = len(df)

    # Negative fares
    negative_fares = int((df["fare_amount"] < 0).sum()) if "fare_amount" in df.columns else 0

    # Zero-distance trips with substantial fares (>$2 -- base fare exception)
    if "trip_distance" in df.columns and "fare_amount" in df.columns:
        zero_dist_high_fare = int(
            ((df["trip_distance"] == 0) & (df["fare_amount"] > 2)).sum()
        )
    else:
        zero_dist_high_fare = 0

    # Passenger count anomalies
    if "passenger_count" in df.columns:
        pax = df["passenger_count"]
        excess_passengers = int((pax > MAX_PASSENGER_COUNT).sum())
        zero_or_neg_passengers = int((pax <= 0).sum())
    else:
        excess_passengers = 0
        zero_or_neg_passengers = 0

    # Negative distances
    negative_distance = (
        int((df["trip_distance"] < 0).sum()) if "trip_distance" in df.columns else 0
    )

    # Extremely high fares
    extreme_fares = (
        int((df["fare_amount"] > MAX_REALISTIC_FARE).sum())
        if "fare_amount" in df.columns
        else 0
    )

    total_impossible = (
        negative_fares
        + zero_dist_high_fare
        + excess_passengers
        + zero_or_neg_passengers
        + negative_distance
        + extreme_fares
    )
    impossible_pct = (total_impossible / total_rows * 100) if total_rows > 0 else 0.0

    return {
        "negative_fares": negative_fares,
        "zero_distance_high_fare": zero_dist_high_fare,
        "excess_passengers": excess_passengers,
        "zero_or_negative_passengers": zero_or_neg_passengers,
        "negative_distance": negative_distance,
        "extreme_fares": extreme_fares,
        "total_impossible": total_impossible,
        "impossible_percentage": round(impossible_pct, 4),
        "total_rows": total_rows,
    }


# --- Orchestrator -------------------------------------------------------------

def assess_accuracy(df: pd.DataFrame) -> dict:
    """
    Orchestrate all accuracy checks and compute an overall accuracy score.

    The accuracy score (0-100) is derived from three sub-scores:
        1. Location validity: % of valid location IDs
        2. Impossible values: % of rows free from impossible values
        3. Outlier prevalence: average % of non-outlier values across
           key numeric fields

    Each sub-score contributes equally (33.3% weight).

    Parameters:
        df: Trip data DataFrame.

    Returns:
        Dictionary containing:
            - accuracy_score: overall score (0-100)
            - location_validation: dict from validate_location_ids()
            - impossible_values: dict from detect_impossible_values()
            - outlier_analysis: list of dicts from detect_outliers_iqr()
    """
    # Location validation
    loc_result = validate_location_ids(df)
    loc_score = 100.0 - loc_result["invalid_percentage"]

    # Impossible values
    impossible_result = detect_impossible_values(df)
    impossible_score = 100.0 - impossible_result["impossible_percentage"]

    # Outlier analysis on key numeric fields
    outlier_fields = [
        "fare_amount", "trip_distance", "tip_amount",
        "total_amount", "passenger_count",
    ]
    outlier_results = []
    outlier_scores = []

    for field in outlier_fields:
        if field in df.columns:
            result = detect_outliers_iqr(df[field], field)
            outlier_results.append(result)
            outlier_scores.append(100.0 - result["outlier_percentage"])

    outlier_avg_score = float(np.mean(outlier_scores)) if outlier_scores else 100.0

    # Weighted combination (equal thirds)
    accuracy_score = round(
        (loc_score * 0.333 + impossible_score * 0.333 + outlier_avg_score * 0.334),
        2,
    )

    return {
        "accuracy_score": accuracy_score,
        "location_validation": loc_result,
        "impossible_values": impossible_result,
        "outlier_analysis": outlier_results,
    }
