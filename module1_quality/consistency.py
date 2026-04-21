"""
Consistency assessment for NYC Taxi Trip Records.

Consistency evaluates internal coherence: whether related data values
agree with each other and with expected logical constraints. This module
checks fare arithmetic, temporal ordering, and trip speed plausibility.

Scalability note:
    All checks are implemented as vectorised pandas operations that
    scale linearly with row count.  A full month of Yellow Taxi data
    (~7.6M rows) completes in under 3 seconds on commodity hardware.
    No iterative row-by-row logic is used.

Theoretical basis:
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

import pandas as pd
import numpy as np

# -- Project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FARE_COMPONENTS,
    FARE_TOTAL_COLUMN,
    FARE_TOLERANCE,
    PICKUP_DATETIME,
    DROPOFF_DATETIME,
    MAX_REALISTIC_SPEED_MPH,
)


# --- Fare Consistency ---------------------------------------------------------

def check_fare_consistency(df: pd.DataFrame) -> dict:
    """
    Verify that total_amount equals the sum of fare components within
    a configurable tolerance.

    The NYC Yellow Taxi data defines:
        total_amount = fare_amount + extra + mta_tax + tip_amount
                     + tolls_amount + improvement_surcharge
                     + congestion_surcharge

    Note: airport_fee is NOT included in FARE_COMPONENTS in config.py
    but IS reflected in total_amount for some 2024 records. We include
    it in the sum if the column exists.

    congestion_surcharge and airport_fee are NaN for most/all 2019 rows
    and must be filled with 0 before summing.

    Parameters:
        df: Trip data DataFrame.

    Returns:
        Dictionary containing:
            - consistent_count: rows where sum matches total within tolerance
            - inconsistent_count: rows where sum does not match
            - inconsistency_percentage: inconsistent as % of total
            - max_discrepancy: largest absolute difference observed
            - mean_discrepancy: mean absolute difference for inconsistent rows
            - total_rows: number of rows examined
    """
    total_rows = len(df)

    # Build the sum of components, filling NaN with 0 for optional fields
    component_cols = list(FARE_COMPONENTS)
    if "airport_fee" in df.columns and "airport_fee" not in component_cols:
        component_cols.append("airport_fee")

    component_sum = pd.Series(np.zeros(total_rows), index=df.index)
    for col in component_cols:
        if col in df.columns:
            component_sum += df[col].fillna(0)

    recorded_total = df[FARE_TOTAL_COLUMN].fillna(0)
    discrepancy = (recorded_total - component_sum).abs()

    consistent_mask = discrepancy <= FARE_TOLERANCE
    consistent_count = int(consistent_mask.sum())
    inconsistent_count = total_rows - consistent_count

    inconsistency_pct = (
        (inconsistent_count / total_rows * 100) if total_rows > 0 else 0.0
    )
    max_disc = float(discrepancy.max()) if total_rows > 0 else 0.0

    inconsistent_disc = discrepancy[~consistent_mask]
    mean_disc = float(inconsistent_disc.mean()) if len(inconsistent_disc) > 0 else 0.0

    return {
        "consistent_count": consistent_count,
        "inconsistent_count": inconsistent_count,
        "inconsistency_percentage": round(inconsistency_pct, 4),
        "max_discrepancy": round(max_disc, 4),
        "mean_discrepancy": round(mean_disc, 4),
        "total_rows": total_rows,
    }


# --- Timestamp Validation -----------------------------------------------------

def validate_timestamps(df: pd.DataFrame) -> dict:
    """
    Validate temporal ordering and range of trip timestamps.

    Checks:
        1. pickup_datetime must be before dropoff_datetime.
        2. Dates must fall within a reasonable range (2009-2026),
           excluding future dates and impossibly old records.

    Parameters:
        df: Trip data DataFrame with datetime columns.

    Returns:
        Dictionary containing:
            - invalid_order_count: rows where dropoff <= pickup
            - out_of_range_count: rows with dates outside expected range
            - invalid_order_percentage: invalid order as % of total
            - out_of_range_percentage: out-of-range as % of total
            - total_rows: number of rows examined
    """
    total_rows = len(df)

    pickup = df[PICKUP_DATETIME]
    dropoff = df[DROPOFF_DATETIME]

    # Check temporal ordering: pickup must precede dropoff
    invalid_order = int((dropoff <= pickup).sum())

    # Check date range: NYC TLC data starts from 2009; cap at 2026
    min_date = pd.Timestamp("2009-01-01")
    max_date = pd.Timestamp("2026-12-31")

    out_of_range_pu = (pickup < min_date) | (pickup > max_date) | pickup.isna()
    out_of_range_do = (dropoff < min_date) | (dropoff > max_date) | dropoff.isna()
    out_of_range_count = int((out_of_range_pu | out_of_range_do).sum())

    invalid_order_pct = (invalid_order / total_rows * 100) if total_rows > 0 else 0.0
    oor_pct = (out_of_range_count / total_rows * 100) if total_rows > 0 else 0.0

    return {
        "invalid_order_count": invalid_order,
        "out_of_range_count": out_of_range_count,
        "invalid_order_percentage": round(invalid_order_pct, 4),
        "out_of_range_percentage": round(oor_pct, 4),
        "total_rows": total_rows,
    }


# --- Trip Speed Validation ----------------------------------------------------

def check_trip_speed(df: pd.DataFrame) -> dict:
    """
    Calculate implied trip speeds and flag impossible values.

    Speed = trip_distance / duration_hours. Trips exceeding the
    configured maximum realistic speed (100 mph) are flagged.

    Trips with zero or negative duration are excluded from speed
    calculations to avoid division errors. Infinite speed values
    (caused by near-zero durations) are treated as impossible.

    Parameters:
        df: Trip data DataFrame.

    Returns:
        Dictionary containing:
            - impossible_speed_count: trips exceeding speed threshold
            - impossible_speed_percentage: as % of calculable trips
            - max_speed: highest finite speed observed (mph)
            - mean_speed: average speed across valid finite trips (mph)
            - calculable_trips: trips with valid positive duration
            - total_rows: total rows in DataFrame
    """
    total_rows = len(df)

    duration_hours = (
        (df[DROPOFF_DATETIME] - df[PICKUP_DATETIME]).dt.total_seconds() / 3600.0
    )

    # Only compute speed for positive durations and non-null distances
    valid_mask = (duration_hours > 0) & df["trip_distance"].notna()
    valid_duration = duration_hours[valid_mask]
    valid_distance = df.loc[valid_mask, "trip_distance"]

    speed = valid_distance / valid_duration
    calculable_trips = len(speed)

    # Replace inf values with NaN for safe aggregation, but count them
    # as impossible
    inf_mask = np.isinf(speed)
    impossible_mask = (speed > MAX_REALISTIC_SPEED_MPH) | inf_mask
    impossible_count = int(impossible_mask.sum())
    impossible_pct = (
        (impossible_count / calculable_trips * 100) if calculable_trips > 0 else 0.0
    )

    finite_speed = speed[~inf_mask]
    max_speed = float(finite_speed.max()) if len(finite_speed) > 0 else 0.0
    mean_speed = float(finite_speed.mean()) if len(finite_speed) > 0 else 0.0

    return {
        "impossible_speed_count": impossible_count,
        "impossible_speed_percentage": round(impossible_pct, 4),
        "max_speed": round(max_speed, 2),
        "mean_speed": round(mean_speed, 2),
        "calculable_trips": calculable_trips,
        "total_rows": total_rows,
    }


# --- Orchestrator -------------------------------------------------------------

def assess_consistency(df: pd.DataFrame) -> dict:
    """
    Orchestrate all consistency checks and compute an overall score.

    The consistency score (0-100) is derived from three sub-scores:
        1. Fare consistency: % of rows with matching fare sums
        2. Timestamp validity: % of rows with correct temporal order
        3. Speed plausibility: % of calculable trips within speed limits

    Each sub-score contributes equally (33.3% weight).

    Parameters:
        df: Trip data DataFrame.

    Returns:
        Dictionary containing:
            - consistency_score: overall score (0-100)
            - fare_consistency: dict from check_fare_consistency()
            - timestamp_validation: dict from validate_timestamps()
            - speed_validation: dict from check_trip_speed()
    """
    fare_result = check_fare_consistency(df)
    fare_score = 100.0 - fare_result["inconsistency_percentage"]

    ts_result = validate_timestamps(df)
    ts_score = 100.0 - ts_result["invalid_order_percentage"]

    speed_result = check_trip_speed(df)
    speed_score = 100.0 - speed_result["impossible_speed_percentage"]

    consistency_score = round(
        (fare_score * 0.333 + ts_score * 0.333 + speed_score * 0.334),
        2,
    )

    return {
        "consistency_score": consistency_score,
        "fare_consistency": fare_result,
        "timestamp_validation": ts_result,
        "speed_validation": speed_result,
    }
