"""
Entropy Analysis Module for NYC Taxi Trip Records.

Applies Shannon entropy to measure the predictability of trip patterns
at the zone level. Lower entropy for a zone means trips from that zone
go to fewer distinct destinations, making individual trips more
predictable and thus posing higher privacy risk.

Two entropy dimensions are analysed:
  1. **Trajectory entropy**: For each pickup zone, the distribution of
     dropoff zones. Measures spatial predictability.
  2. **Temporal entropy**: For each pickup zone, the distribution of
     pickup hours. Measures temporal predictability.

These metrics draw on de Montjoye et al.'s (2013) finding that human
mobility is highly predictable, and that even coarse-grained location
data can reveal individual patterns.

References:
    de Montjoye, Y.-A. et al. (2013) 'Unique in the Crowd: The privacy
        bounds of human mobility', Scientific Reports, 3, p. 1376.
    Shannon, C.E. (1948) 'A Mathematical Theory of Communication',
        The Bell System Technical Journal, 27(3), pp. 379-423.
    Song, C. et al. (2010) 'Limits of Predictability in Human Mobility',
        Science, 327(5968), pp. 1018-1021.

Author: Sami Ullah (B01750598)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PICKUP_LOCATION, DROPOFF_LOCATION, PICKUP_DATETIME


def _compute_zone_entropy(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    min_trips: int = 10,
) -> pd.DataFrame:
    """
    Compute Shannon entropy for each group based on the distribution
    of values within that group.

    Parameters:
        df:         DataFrame with the relevant columns.
        group_col:  Column to group by (e.g., PULocationID).
        value_col:  Column whose distribution is measured (e.g., DOLocationID).
        min_trips:  Minimum trips per zone to include (avoids noisy estimates).

    Returns:
        DataFrame with columns: zone_id, entropy, trip_count, n_distinct.
    """
    results = []

    for zone_id, group in df.groupby(group_col):
        n_trips = len(group)
        if n_trips < min_trips:
            continue

        # Value distribution as probabilities
        value_counts = group[value_col].value_counts(normalize=True)
        probabilities = value_counts.values

        # Shannon entropy (natural log base; scipy uses ln by default)
        # Convert to log base 2 for information-theoretic interpretation (bits)
        h = float(scipy_entropy(probabilities, base=2))

        results.append({
            "zone_id": int(zone_id) if not pd.isna(zone_id) else zone_id,
            "entropy": round(h, 4),
            "trip_count": n_trips,
            "n_distinct": len(value_counts),
        })

    return pd.DataFrame(results)


def calculate_trajectory_entropy(
    df: pd.DataFrame,
    min_trips: int = 10,
) -> dict:
    """
    Calculate trajectory entropy: for each pickup zone, measure the
    Shannon entropy of the dropoff zone distribution.

    Higher entropy means more diverse destinations (better privacy).
    Lower entropy means trips from that zone are more predictable
    (e.g., a residential zone where most trips go to the same workplace).

    Parameters:
        df:         NYC Taxi trip DataFrame.
        min_trips:  Minimum trips per zone to include in analysis.

    Returns:
        Dictionary with keys:
            - avg_entropy: float (bits)
            - median_entropy: float
            - min_entropy: float
            - max_entropy: float
            - std_entropy: float
            - n_zones_analysed: int
            - entropy_per_zone: pd.DataFrame (zone_id, entropy, trip_count, n_distinct)
    """
    required = [PICKUP_LOCATION, DROPOFF_LOCATION]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    entropy_df = _compute_zone_entropy(
        df, group_col=PICKUP_LOCATION, value_col=DROPOFF_LOCATION,
        min_trips=min_trips,
    )

    if entropy_df.empty:
        return {
            "avg_entropy": 0.0,
            "median_entropy": 0.0,
            "min_entropy": 0.0,
            "max_entropy": 0.0,
            "std_entropy": 0.0,
            "n_zones_analysed": 0,
            "entropy_per_zone": entropy_df,
        }

    return {
        "avg_entropy": round(float(entropy_df["entropy"].mean()), 4),
        "median_entropy": round(float(entropy_df["entropy"].median()), 4),
        "min_entropy": round(float(entropy_df["entropy"].min()), 4),
        "max_entropy": round(float(entropy_df["entropy"].max()), 4),
        "std_entropy": round(float(entropy_df["entropy"].std()), 4),
        "n_zones_analysed": len(entropy_df),
        "entropy_per_zone": entropy_df,
    }


def calculate_temporal_entropy(
    df: pd.DataFrame,
    min_trips: int = 10,
) -> dict:
    """
    Calculate temporal entropy: for each pickup zone, measure the
    Shannon entropy of the pickup hour distribution.

    Low temporal entropy indicates that a zone has concentrated activity
    at specific hours (e.g., a business district with morning peaks),
    making trips more temporally predictable.

    Parameters:
        df:         NYC Taxi trip DataFrame.
        min_trips:  Minimum trips per zone to include in analysis.

    Returns:
        Dictionary with same structure as calculate_trajectory_entropy().
    """
    if PICKUP_DATETIME not in df.columns:
        raise ValueError(f"Missing required column: {PICKUP_DATETIME}")
    if PICKUP_LOCATION not in df.columns:
        raise ValueError(f"Missing required column: {PICKUP_LOCATION}")

    work = df.copy()
    work["_pickup_hour"] = work[PICKUP_DATETIME].dt.hour

    entropy_df = _compute_zone_entropy(
        work, group_col=PICKUP_LOCATION, value_col="_pickup_hour",
        min_trips=min_trips,
    )

    if entropy_df.empty:
        return {
            "avg_entropy": 0.0,
            "median_entropy": 0.0,
            "min_entropy": 0.0,
            "max_entropy": 0.0,
            "std_entropy": 0.0,
            "n_zones_analysed": 0,
            "entropy_per_zone": entropy_df,
        }

    return {
        "avg_entropy": round(float(entropy_df["entropy"].mean()), 4),
        "median_entropy": round(float(entropy_df["entropy"].median()), 4),
        "min_entropy": round(float(entropy_df["entropy"].min()), 4),
        "max_entropy": round(float(entropy_df["entropy"].max()), 4),
        "std_entropy": round(float(entropy_df["entropy"].std()), 4),
        "n_zones_analysed": len(entropy_df),
        "entropy_per_zone": entropy_df,
    }


def identify_high_risk_zones(
    trajectory_entropy: dict,
    temporal_entropy: dict,
    entropy_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Identify zones with low entropy in both trajectory and temporal
    dimensions — these are the highest privacy risk zones.

    Parameters:
        trajectory_entropy: Output of calculate_trajectory_entropy().
        temporal_entropy:   Output of calculate_temporal_entropy().
        entropy_threshold:  Zones below this entropy (bits) are flagged.

    Returns:
        DataFrame of high-risk zones with both entropy values.
    """
    traj = trajectory_entropy["entropy_per_zone"].rename(
        columns={"entropy": "trajectory_entropy"}
    )
    temp = temporal_entropy["entropy_per_zone"].rename(
        columns={"entropy": "temporal_entropy"}
    )

    merged = traj[["zone_id", "trajectory_entropy", "trip_count"]].merge(
        temp[["zone_id", "temporal_entropy"]],
        on="zone_id",
        how="inner",
    )

    high_risk = merged[
        (merged["trajectory_entropy"] < entropy_threshold)
        | (merged["temporal_entropy"] < entropy_threshold)
    ].sort_values("trajectory_entropy")

    return high_risk


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_loader import load_month

    print("Entropy Analysis — Self Test")
    print("=" * 60)
    df = load_month(2019, 1, sample_frac=0.01)

    traj = calculate_trajectory_entropy(df)
    print(f"\nTrajectory Entropy (spatial predictability):")
    print(f"  Average: {traj['avg_entropy']:.4f} bits")
    print(f"  Median:  {traj['median_entropy']:.4f} bits")
    print(f"  Min:     {traj['min_entropy']:.4f} bits")
    print(f"  Max:     {traj['max_entropy']:.4f} bits")
    print(f"  Zones:   {traj['n_zones_analysed']}")

    temp = calculate_temporal_entropy(df)
    print(f"\nTemporal Entropy (temporal predictability):")
    print(f"  Average: {temp['avg_entropy']:.4f} bits")
    print(f"  Median:  {temp['median_entropy']:.4f} bits")
    print(f"  Min:     {temp['min_entropy']:.4f} bits")
    print(f"  Max:     {temp['max_entropy']:.4f} bits")
    print(f"  Zones:   {temp['n_zones_analysed']}")
