"""
Unit Tests for Module 2: Privacy Risk Detection.

Tests cover all sub-modules with controlled data to verify correctness
of PII classification, uniqueness analysis, k-anonymity, entropy
calculations, linkage attack simulation, and risk scoring.

Run with: pytest module2_privacy/tests/test_privacy.py -v

Author: Sami Ullah (B01750598)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from module2_privacy.pii_classifier import (
    classify_pii_fields,
    score_field_privacy_risk,
    get_all_field_scores,
    PII_TAXONOMY,
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
from module2_privacy.linkage_attack import (
    get_landmark_zones,
    simulate_linkage_attack,
)
from module2_privacy.risk_scorer import (
    calculate_privacy_risk_score,
    sensitivity_analysis,
)


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal taxi trip DataFrame for testing."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "VendorID": np.random.choice([1, 2], n),
        "tpep_pickup_datetime": pd.date_range("2019-01-01", periods=n, freq="10min"),
        "tpep_dropoff_datetime": pd.date_range("2019-01-01 00:15:00", periods=n, freq="10min"),
        "PULocationID": np.random.choice([132, 138, 161, 186, 230, 43], n),
        "DOLocationID": np.random.choice([132, 138, 161, 186, 230, 43, 261], n),
        "passenger_count": np.random.choice([1, 2, 3, 4], n),
        "trip_distance": np.random.uniform(0.5, 20.0, n).round(2),
        "fare_amount": np.random.uniform(5.0, 100.0, n).round(2),
        "tip_amount": np.random.uniform(0.0, 20.0, n).round(2),
        "total_amount": np.random.uniform(5.0, 120.0, n).round(2),
        "payment_type": np.random.choice([1, 2, 3], n),
        "RatecodeID": np.random.choice([1, 2], n),
        "store_and_fwd_flag": np.random.choice(["Y", "N"], n),
        "improvement_surcharge": 0.3,
        "mta_tax": 0.5,
        "extra": np.random.choice([0.0, 0.5, 1.0], n),
        "congestion_surcharge": np.random.choice([0.0, 2.5], n),
        "airport_fee": np.nan,
        "tolls_amount": np.random.choice([0.0, 5.76], n),
    })


@pytest.fixture
def controlled_k_df() -> pd.DataFrame:
    """
    Create a DataFrame with known k-anonymity properties.
    Each (PU, DO, hour) combination appears exactly 3 times.
    """
    rows = []
    base_time = pd.Timestamp("2019-01-01 10:00:00")
    for pu in [100, 200]:
        for do in [150, 250]:
            for i in range(3):
                rows.append({
                    "PULocationID": pu,
                    "DOLocationID": do,
                    "tpep_pickup_datetime": base_time + pd.Timedelta(minutes=i * 5),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def uniform_entropy_df() -> pd.DataFrame:
    """
    Create a DataFrame where one zone has perfectly uniform distribution
    over 4 destinations (entropy = log2(4) = 2.0 bits).
    """
    rows = []
    base_time = pd.Timestamp("2019-01-01 08:00:00")
    # Zone 100 sends equal trips to 4 destinations
    for dest in [201, 202, 203, 204]:
        for i in range(25):  # 25 trips each = 100 total
            rows.append({
                "PULocationID": 100,
                "DOLocationID": dest,
                "tpep_pickup_datetime": base_time + pd.Timedelta(hours=i % 24),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Test 1: PII Classification
# ═══════════════════════════════════════════════════════════════════

class TestPIIClassification:
    """Tests for the PII classifier module."""

    def test_classify_returns_all_columns(self, sample_df: pd.DataFrame):
        """Every DataFrame column should appear in the classification."""
        result = classify_pii_fields(sample_df)
        for col in sample_df.columns:
            assert col in result, f"Column '{col}' missing from classification"

    def test_location_fields_classified_as_high(self, sample_df: pd.DataFrame):
        """PULocationID and DOLocationID should be classified as High risk."""
        result = classify_pii_fields(sample_df)
        assert result["PULocationID"]["category"] == "Location QI"
        assert result["PULocationID"]["risk_level"] == "High"
        assert result["DOLocationID"]["category"] == "Location QI"
        assert result["DOLocationID"]["risk_level"] == "High"

    def test_temporal_fields_classified_as_medium(self, sample_df: pd.DataFrame):
        """Datetime columns should be classified as Medium risk."""
        result = classify_pii_fields(sample_df)
        assert result["tpep_pickup_datetime"]["risk_level"] == "Medium"
        assert result["tpep_dropoff_datetime"]["risk_level"] == "Medium"

    def test_vendor_classified_as_non_pii(self, sample_df: pd.DataFrame):
        """VendorID should be Non-PII with None risk."""
        result = classify_pii_fields(sample_df)
        assert result["VendorID"]["category"] == "Non-PII"
        assert result["VendorID"]["risk_level"] == "None"

    def test_score_high_risk_field(self, sample_df: pd.DataFrame):
        """High-risk fields should score > 60."""
        classification = classify_pii_fields(sample_df)
        score = score_field_privacy_risk("PULocationID", sample_df, classification)
        assert score >= 60.0, f"Expected >= 60, got {score}"

    def test_score_non_pii_field(self, sample_df: pd.DataFrame):
        """Non-PII fields should score low."""
        classification = classify_pii_fields(sample_df)
        score = score_field_privacy_risk("VendorID", sample_df, classification)
        assert score <= 20.0, f"Expected <= 20, got {score}"

    def test_score_range(self, sample_df: pd.DataFrame):
        """All scores should be between 0 and 100."""
        scores = get_all_field_scores(sample_df)
        for field, score in scores.items():
            assert 0 <= score <= 100, f"{field}: score {score} out of range"


# ═══════════════════════════════════════════════════════════════════
# Test 2: Uniqueness Analysis
# ═══════════════════════════════════════════════════════════════════

class TestUniqueness:
    """Tests for the uniqueness analysis module."""

    def test_uniqueness_returns_expected_keys(self, sample_df: pd.DataFrame):
        """Result dictionary should contain all expected keys."""
        result = calculate_uniqueness(sample_df, temporal_resolution="H")
        expected_keys = {
            "uniqueness_percentage", "unique_count", "total_records",
            "total_combinations", "resolution", "value_counts",
        }
        assert expected_keys.issubset(result.keys())

    def test_uniqueness_percentage_range(self, sample_df: pd.DataFrame):
        """Uniqueness percentage should be between 0 and 100."""
        result = calculate_uniqueness(sample_df, temporal_resolution="H")
        assert 0 <= result["uniqueness_percentage"] <= 100

    def test_daily_less_unique_than_15min(self, sample_df: pd.DataFrame):
        """Daily resolution should produce lower uniqueness than 15-minute."""
        res_15 = calculate_uniqueness(sample_df, temporal_resolution="15min")
        res_d = calculate_uniqueness(sample_df, temporal_resolution="D")
        assert res_d["uniqueness_percentage"] <= res_15["uniqueness_percentage"]

    def test_compare_resolutions_returns_3_rows(self, sample_df: pd.DataFrame):
        """Resolution comparison should return exactly 3 rows."""
        comp = compare_temporal_resolutions(sample_df)
        assert len(comp) == 3
        assert "resolution" in comp.columns
        assert "uniqueness_percentage" in comp.columns


# ═══════════════════════════════════════════════════════════════════
# Test 3: k-Anonymity
# ═══════════════════════════════════════════════════════════════════

class TestKAnonymity:
    """Tests for the k-anonymity assessment module."""

    def test_controlled_k_equals_3(self, controlled_k_df: pd.DataFrame):
        """With 3 records per group, min k should be 3."""
        # All records are within the same hour, so at hourly resolution
        # each (PU, DO) pair has 3 records
        result = assess_k_anonymity(controlled_k_df, temporal_resolution="H")
        assert result["min_k"] == 3
        assert result["max_k"] == 3

    def test_k_distribution_keys(self, sample_df: pd.DataFrame):
        """k-distribution should contain the expected bucket labels."""
        result = assess_k_anonymity(sample_df, temporal_resolution="H")
        expected_buckets = {"k=1 (unique)", "k=2-5", "k=6-10", "k=11-50", "k>50"}
        assert expected_buckets == set(result["k_distribution"].keys())

    def test_k_metrics_types(self, sample_df: pd.DataFrame):
        """All metric values should have correct types."""
        result = assess_k_anonymity(sample_df, temporal_resolution="H")
        assert isinstance(result["min_k"], int)
        assert isinstance(result["mean_k"], float)
        assert isinstance(result["median_k"], float)
        assert isinstance(result["records_below_k5_pct"], float)


# ═══════════════════════════════════════════════════════════════════
# Test 4: Entropy
# ═══════════════════════════════════════════════════════════════════

class TestEntropy:
    """Tests for the entropy analysis module."""

    def test_uniform_distribution_entropy(self, uniform_entropy_df: pd.DataFrame):
        """
        A zone with perfectly uniform distribution over 4 destinations
        should have entropy of exactly 2.0 bits (log2(4)).
        """
        result = calculate_trajectory_entropy(uniform_entropy_df, min_trips=10)
        zone_data = result["entropy_per_zone"]
        assert len(zone_data) == 1  # Only zone 100
        assert abs(zone_data.iloc[0]["entropy"] - 2.0) < 0.01

    def test_entropy_returns_expected_keys(self, sample_df: pd.DataFrame):
        """Trajectory entropy result should contain all expected keys."""
        result = calculate_trajectory_entropy(sample_df)
        expected = {"avg_entropy", "median_entropy", "min_entropy",
                    "max_entropy", "std_entropy", "n_zones_analysed",
                    "entropy_per_zone"}
        assert expected.issubset(result.keys())

    def test_temporal_entropy_runs(self, sample_df: pd.DataFrame):
        """Temporal entropy should execute without errors."""
        result = calculate_temporal_entropy(sample_df)
        assert result["n_zones_analysed"] > 0
        assert result["avg_entropy"] >= 0

    def test_entropy_non_negative(self, sample_df: pd.DataFrame):
        """All entropy values should be non-negative."""
        result = calculate_trajectory_entropy(sample_df)
        assert result["min_entropy"] >= 0


# ═══════════════════════════════════════════════════════════════════
# Test 5: Linkage Attack
# ═══════════════════════════════════════════════════════════════════

class TestLinkageAttack:
    """Tests for the linkage attack simulation module."""

    def test_landmark_zones_not_empty(self):
        """Landmark zones dictionary should contain entries."""
        landmarks = get_landmark_zones()
        assert len(landmarks) > 0
        assert "JFK Airport Terminal 1" in landmarks

    def test_linkage_rate_range(self, sample_df: pd.DataFrame):
        """Linkage rate should be between 0 and 100."""
        result = simulate_linkage_attack(sample_df)
        assert 0 <= result["linkage_rate"] <= 100

    def test_linkage_with_landmark_zones(self, sample_df: pd.DataFrame):
        """
        Since sample_df uses zones 132, 138, 161, 186, 230, 43
        which are all landmark zones, linkage rate should be high.
        """
        result = simulate_linkage_attack(sample_df)
        assert result["linkage_rate"] > 50  # Most zones are landmarks

    def test_linkage_breakdown_has_columns(self, sample_df: pd.DataFrame):
        """Breakdown DataFrame should have expected columns."""
        result = simulate_linkage_attack(sample_df)
        breakdown = result["landmark_breakdown"]
        assert "landmark" in breakdown.columns
        assert "zone_id" in breakdown.columns
        assert "total_trips" in breakdown.columns


# ═══════════════════════════════════════════════════════════════════
# Test 6: Risk Scoring
# ═══════════════════════════════════════════════════════════════════

class TestRiskScoring:
    """Tests for the risk scoring module."""

    def test_high_risk_inputs(self):
        """High uniqueness + low k + low entropy + high linkage = Critical."""
        result = calculate_privacy_risk_score(
            uniqueness_pct=90.0,
            k_metrics={"records_below_k5_pct": 80.0},
            avg_entropy=1.0,
            linkage_rate=70.0,
        )
        assert result["overall_score"] >= 75
        assert result["risk_level"] == "Critical"

    def test_low_risk_inputs(self):
        """Low uniqueness + high k + high entropy + low linkage = Low."""
        result = calculate_privacy_risk_score(
            uniqueness_pct=5.0,
            k_metrics={"records_below_k5_pct": 2.0},
            avg_entropy=7.0,
            linkage_rate=5.0,
        )
        assert result["overall_score"] < 25
        assert result["risk_level"] == "Low"

    def test_score_range(self):
        """Score should always be between 0 and 100."""
        result = calculate_privacy_risk_score(
            uniqueness_pct=50.0,
            k_metrics={"records_below_k5_pct": 50.0},
            avg_entropy=4.0,
            linkage_rate=30.0,
        )
        assert 0 <= result["overall_score"] <= 100

    def test_components_present(self):
        """Result should contain all four component scores."""
        result = calculate_privacy_risk_score(
            uniqueness_pct=50.0,
            k_metrics={"records_below_k5_pct": 50.0},
            avg_entropy=4.0,
            linkage_rate=30.0,
        )
        assert set(result["components"].keys()) == {
            "uniqueness", "k_anonymity", "entropy", "linkage"
        }

    def test_sensitivity_returns_4_rows(self):
        """Sensitivity analysis should return 4 weight configurations."""
        sa = sensitivity_analysis(
            uniqueness_pct=50.0,
            k_metrics={"records_below_k5_pct": 40.0},
            avg_entropy=4.0,
            linkage_rate=25.0,
        )
        assert len(sa) >= 3  # At least 3 alternative weight sets
        assert "weight_set" in sa.columns
        assert "overall_score" in sa.columns
        assert "risk_level" in sa.columns

    def test_custom_weights(self):
        """Custom weights should be applied correctly."""
        w = {"uniqueness": 1.0, "k_anonymity": 0.0, "entropy": 0.0, "linkage": 0.0}
        result = calculate_privacy_risk_score(
            uniqueness_pct=80.0,
            k_metrics={"records_below_k5_pct": 0.0},
            avg_entropy=8.0,
            linkage_rate=0.0,
            weights=w,
        )
        # With 100% weight on uniqueness (80.0) and 0% on everything else
        assert abs(result["overall_score"] - 80.0) < 0.1


# ═══════════════════════════════════════════════════════════════════
# Test 7: Integration — empty and edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe_uniqueness(self):
        """Uniqueness on empty DataFrame should return 0."""
        df = pd.DataFrame({
            "PULocationID": pd.Series(dtype=int),
            "DOLocationID": pd.Series(dtype=int),
            "tpep_pickup_datetime": pd.Series(dtype="datetime64[ns]"),
        })
        result = calculate_uniqueness(df)
        assert result["uniqueness_percentage"] == 0.0

    def test_empty_dataframe_k_anonymity(self):
        """k-anonymity on empty DataFrame should handle gracefully."""
        df = pd.DataFrame({
            "PULocationID": pd.Series(dtype=int),
            "DOLocationID": pd.Series(dtype=int),
            "tpep_pickup_datetime": pd.Series(dtype="datetime64[ns]"),
        })
        result = assess_k_anonymity(df)
        assert result["min_k"] == 0

    def test_single_record(self):
        """A single record should have uniqueness of 100%."""
        df = pd.DataFrame({
            "PULocationID": [100],
            "DOLocationID": [200],
            "tpep_pickup_datetime": [pd.Timestamp("2019-01-01 12:00:00")],
        })
        result = calculate_uniqueness(df)
        assert result["uniqueness_percentage"] == 100.0

    def test_linkage_empty_df(self):
        """Linkage on empty DataFrame should return 0."""
        df = pd.DataFrame({
            "PULocationID": pd.Series(dtype=int),
            "DOLocationID": pd.Series(dtype=int),
        })
        result = simulate_linkage_attack(df)
        assert result["linkage_rate"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
