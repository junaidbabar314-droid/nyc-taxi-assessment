"""
Unit tests for Module 1: Data Quality Profiling.

Tests use controlled synthetic DataFrames with known characteristics
to verify correctness of each quality dimension assessment.

Author: Junaid Babar (B01802551)
Module: Data Quality Profiling
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# -- Ensure project root is on the path ----------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from completeness import assess_completeness, compute_completeness_score
from accuracy import (
    detect_outliers_iqr,
    validate_location_ids,
    detect_impossible_values,
    assess_accuracy,
)
from consistency import (
    check_fare_consistency,
    validate_timestamps,
    check_trip_speed,
    assess_consistency,
)
from timeliness import assess_timeliness


# --- Fixtures -----------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal valid taxi trip DataFrame for testing."""
    n = 100
    np.random.seed(42)
    return pd.DataFrame({
        "VendorID": np.random.choice([1, 2], size=n),
        "tpep_pickup_datetime": pd.date_range("2019-01-01", periods=n, freq="h"),
        "tpep_dropoff_datetime": pd.date_range("2019-01-01 00:15:00", periods=n, freq="h"),
        "passenger_count": np.random.randint(1, 5, size=n).astype(float),
        "trip_distance": np.random.exponential(3.0, size=n).round(2),
        "RatecodeID": np.ones(n, dtype=int),
        "store_and_fwd_flag": ["N"] * n,
        "PULocationID": np.random.randint(1, 266, size=n),
        "DOLocationID": np.random.randint(1, 266, size=n),
        "payment_type": np.random.choice([1, 2], size=n),
        "fare_amount": np.random.uniform(5, 50, size=n).round(2),
        "extra": np.zeros(n),
        "mta_tax": np.full(n, 0.5),
        "tip_amount": np.random.uniform(0, 10, size=n).round(2),
        "tolls_amount": np.zeros(n),
        "improvement_surcharge": np.full(n, 0.3),
        "total_amount": np.zeros(n),  # Will be computed below
        "congestion_surcharge": np.full(n, 2.5),
        "airport_fee": np.zeros(n),
    })


@pytest.fixture
def df_with_consistent_fares(sample_df) -> pd.DataFrame:
    """Create a DataFrame where total_amount = sum of components."""
    df = sample_df.copy()
    df["total_amount"] = (
        df["fare_amount"]
        + df["extra"]
        + df["mta_tax"]
        + df["tip_amount"]
        + df["tolls_amount"]
        + df["improvement_surcharge"]
        + df["congestion_surcharge"]
        + df["airport_fee"]
    )
    return df


# --- Test 1: Completeness with controlled nulls ------------------------------

def test_completeness_known_nulls():
    """Verify null counts match exactly for a controlled DataFrame."""
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4, 5],
        "B": [np.nan, np.nan, 3, 4, 5],
        "C": [1, 2, 3, 4, 5],
    })
    result = assess_completeness(df)

    a_row = result[result["Field"] == "A"].iloc[0]
    b_row = result[result["Field"] == "B"].iloc[0]
    c_row = result[result["Field"] == "C"].iloc[0]

    assert a_row["Null_Count"] == 1
    assert a_row["Null_Percentage"] == 20.0
    assert a_row["Severity"] == "Critical"  # 20% > 5%

    assert b_row["Null_Count"] == 2
    assert b_row["Null_Percentage"] == 40.0

    assert c_row["Null_Count"] == 0
    assert c_row["Null_Percentage"] == 0.0
    assert c_row["Severity"] == "Good"


def test_completeness_score():
    """Verify the completeness score calculation."""
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4, 5],       # 20% null
        "B": [1, 2, 3, 4, 5],             # 0% null
    })
    comp_df = assess_completeness(df)
    score = compute_completeness_score(comp_df)
    # Average: (80 + 100) / 2 = 90
    assert score == 90.0


# --- Test 2: IQR outlier detection -------------------------------------------

def test_iqr_known_outliers():
    """Verify IQR detects known outliers in a controlled series."""
    # 20 normal values + 2 extreme outliers
    values = list(range(1, 21)) + [100, -50]
    series = pd.Series(values, dtype=float)
    result = detect_outliers_iqr(series, "test_field")

    assert result["field_name"] == "test_field"
    assert result["total_count"] == 22
    assert result["outlier_count"] >= 2  # At least the two extremes
    assert result["Q1"] < result["Q3"]
    assert result["lower_bound"] < result["upper_bound"]


def test_iqr_empty_series():
    """Verify IQR handles empty series gracefully."""
    series = pd.Series([], dtype=float)
    result = detect_outliers_iqr(series, "empty")
    assert result["total_count"] == 0
    assert result["outlier_count"] == 0


# --- Test 3: Location ID validation ------------------------------------------

def test_location_id_out_of_range():
    """Verify detection of out-of-range location IDs."""
    df = pd.DataFrame({
        "PULocationID": [1, 265, 0, 300, np.nan],
        "DOLocationID": [100, 265, -1, 266, 150],
    })
    result = validate_location_ids(df)

    # Invalid PU: 0 (below min), 300 (above max), NaN = 3
    assert result["invalid_pickup_count"] == 3
    # Invalid DO: -1 (below min), 266 (above max) = 2
    assert result["invalid_dropoff_count"] == 2
    assert result["total_rows"] == 5


# --- Test 4: Impossible values ------------------------------------------------

def test_impossible_values():
    """Verify detection of negative fares and excess passengers."""
    df = pd.DataFrame({
        "fare_amount": [10, -5, 15, 20, 3],
        "trip_distance": [2, 3, 0, 5, 0],
        "passenger_count": [1, 2, 7, 0, 3],
    })
    result = detect_impossible_values(df)

    assert result["negative_fares"] == 1           # -5
    assert result["zero_distance_high_fare"] == 2  # rows 3 (fare=15) and 5 (fare=3) both > $2
    assert result["excess_passengers"] == 1        # 7 > 6
    assert result["zero_or_negative_passengers"] == 1  # 0


# --- Test 5: Fare consistency -------------------------------------------------

def test_fare_consistency_exact_match(df_with_consistent_fares):
    """Verify 100% consistency when totals match exactly."""
    result = check_fare_consistency(df_with_consistent_fares)
    assert result["inconsistent_count"] == 0
    assert result["inconsistency_percentage"] == 0.0


def test_fare_consistency_with_discrepancy():
    """Verify detection of deliberate fare discrepancies."""
    df = pd.DataFrame({
        "fare_amount": [10.0, 20.0, 15.0],
        "extra": [0.0, 0.0, 0.0],
        "mta_tax": [0.5, 0.5, 0.5],
        "tip_amount": [2.0, 3.0, 1.0],
        "tolls_amount": [0.0, 0.0, 0.0],
        "improvement_surcharge": [0.3, 0.3, 0.3],
        "congestion_surcharge": [2.5, 2.5, 2.5],
        "airport_fee": [0.0, 0.0, 0.0],
        "total_amount": [15.3, 99.99, 19.3],  # Row 1 correct, row 2 wrong
    })
    result = check_fare_consistency(df)
    assert result["inconsistent_count"] >= 1
    assert result["max_discrepancy"] > 0


# --- Test 6: Timestamp validation ---------------------------------------------

def test_timestamp_invalid_order():
    """Verify detection of dropoff before pickup."""
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2019-01-01 10:00", "2019-01-01 12:00", "2019-01-01 14:00"
        ]),
        "tpep_dropoff_datetime": pd.to_datetime([
            "2019-01-01 10:30", "2019-01-01 11:00", "2019-01-01 14:30"  # Row 2: dropoff < pickup
        ]),
    })
    result = validate_timestamps(df)
    assert result["invalid_order_count"] == 1  # Row 2


def test_timestamp_out_of_range():
    """Verify detection of dates outside expected range."""
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2019-01-01 10:00", "1990-01-01 10:00",
        ]),
        "tpep_dropoff_datetime": pd.to_datetime([
            "2019-01-01 10:30", "1990-01-01 10:30",
        ]),
    })
    result = validate_timestamps(df)
    assert result["out_of_range_count"] == 1  # 1990 row


# --- Test 7: Trip speed validation --------------------------------------------

def test_impossible_speed():
    """Verify detection of physically impossible trip speeds."""
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2019-01-01 10:00", "2019-01-01 12:00",
        ]),
        "tpep_dropoff_datetime": pd.to_datetime([
            "2019-01-01 10:01", "2019-01-01 13:00",  # 1 minute trip
        ]),
        "trip_distance": [200.0, 30.0],  # 200 miles in 1 min = 12000 mph
    })
    result = check_trip_speed(df)
    assert result["impossible_speed_count"] >= 1
    assert result["max_speed"] > 100


# --- Test 8: Timeliness assessment --------------------------------------------

def test_timeliness_january_2019():
    """Verify timeliness for a known month with in-range data."""
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.date_range("2019-01-01", periods=50, freq="12h"),
    })
    result = assess_timeliness(df, file_year=2019, file_month=1)

    assert result["total_records"] == 50
    assert result["timeliness_score"] > 0
    assert "publication_date" in result
    assert result["avg_lag_days"] > 0


def test_timeliness_all_in_month():
    """Verify score is 100 when all records fall within labelled month."""
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.date_range("2019-01-01", periods=31, freq="D"),
    })
    result = assess_timeliness(df, file_year=2019, file_month=1)
    assert result["timeliness_score"] == 100.0
    assert result["freshness_30d_pct"] == 100.0


def test_timeliness_all_outside_month():
    """Verify score is 0 when no records fall within labelled month."""
    # All records are from March 2019, but file claims to be January
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.date_range("2019-03-01", periods=10, freq="D"),
    })
    result = assess_timeliness(df, file_year=2019, file_month=1)
    assert result["timeliness_score"] == 0.0
    assert result["freshness_30d_pct"] == 0.0


# --- Test 9: Accuracy orchestrator -------------------------------------------

def test_accuracy_orchestrator(sample_df):
    """Verify the accuracy orchestrator returns all expected keys."""
    result = assess_accuracy(sample_df)

    assert "accuracy_score" in result
    assert 0 <= result["accuracy_score"] <= 100
    assert "location_validation" in result
    assert "impossible_values" in result
    assert "outlier_analysis" in result
    assert isinstance(result["outlier_analysis"], list)


# --- Test 10: Consistency orchestrator ----------------------------------------

def test_consistency_orchestrator(df_with_consistent_fares):
    """Verify the consistency orchestrator returns all expected keys."""
    result = assess_consistency(df_with_consistent_fares)

    assert "consistency_score" in result
    assert 0 <= result["consistency_score"] <= 100
    assert "fare_consistency" in result
    assert "timestamp_validation" in result
    assert "speed_validation" in result


# --- Test 11: Fare consistency with NaN surcharges ----------------------------

def test_fare_consistency_nan_surcharges():
    """Verify that NaN congestion_surcharge/airport_fee are handled (filled with 0)."""
    df = pd.DataFrame({
        "fare_amount": [10.0],
        "extra": [0.0],
        "mta_tax": [0.5],
        "tip_amount": [2.0],
        "tolls_amount": [0.0],
        "improvement_surcharge": [0.3],
        "congestion_surcharge": [np.nan],  # 2019-style NaN
        "airport_fee": [np.nan],           # 2019-style NaN
        "total_amount": [12.8],            # 10 + 0 + 0.5 + 2 + 0 + 0.3 = 12.8
    })
    result = check_fare_consistency(df)
    assert result["inconsistent_count"] == 0


# --- Test 12: Full quality profiler integration -------------------------------

def test_quality_profiler_integration(df_with_consistent_fares):
    """Verify the main orchestrator returns the complete results structure."""
    from quality_profiler import get_quality_metrics

    result = get_quality_metrics(df_with_consistent_fares, year=2019, month=1)

    assert "overall_score" in result
    assert "metrics" in result
    assert "completeness_detail" in result
    assert "accuracy_detail" in result
    assert "consistency_detail" in result
    assert "timeliness_detail" in result
    assert "field_scores" in result
    assert "summary_text" in result
    assert "dataframe" in result

    assert 0 <= result["overall_score"] <= 100
    assert isinstance(result["completeness_detail"], pd.DataFrame)
    assert isinstance(result["field_scores"], pd.DataFrame)
    assert isinstance(result["summary_text"], str)
    assert len(result["summary_text"]) > 50

    # Verify the dataframe reference is the same object
    assert result["dataframe"] is df_with_consistent_fares


# --- Test 13: Speed check handles inf values ----------------------------------

def test_speed_check_no_inf():
    """Verify speed check returns finite max_speed even with near-zero durations."""
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2019-01-01 10:00:00", "2019-01-01 12:00:00",
        ]),
        "tpep_dropoff_datetime": pd.to_datetime([
            "2019-01-01 10:00:01", "2019-01-01 13:00:00",  # 1-second trip
        ]),
        "trip_distance": [50.0, 30.0],  # 50 miles in 1 second
    })
    result = check_trip_speed(df)
    assert np.isfinite(result["max_speed"])
    assert np.isfinite(result["mean_speed"])
    assert result["impossible_speed_count"] >= 1
