"""
Uniqueness Analysis Module for NYC Taxi Trip Records.

Measures how unique individual trip records are when quasi-identifiers
(location zones and temporal attributes) are combined. High uniqueness
means individual trips can be singled out despite zone-level
generalisation — a critical finding for transportation privacy research.

The module evaluates uniqueness at multiple temporal resolutions (15-min,
hourly, daily) to quantify the privacy impact of temporal precision,
demonstrating that NYC TLC's zone-based anonymisation is more effective
when temporal resolution is coarser.

References:
    de Montjoye, Y.-A. et al. (2013) 'Unique in the Crowd: The privacy
        bounds of human mobility', Scientific Reports, 3, p. 1376.
    Xu, F. et al. (2017) 'Trajectory recovery from ash: User privacy is
        NOT preserved in aggregated mobility data', Proceedings of the
        26th International Conference on World Wide Web, pp. 1241-1250.

Author: Sami Ullah (B01750598)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PICKUP_LOCATION, DROPOFF_LOCATION, PICKUP_DATETIME

# Type alias for supported temporal resolutions
TemporalResolution = Literal["15min", "H", "D"]

# Human-readable labels for resolutions
_RESOLUTION_LABELS: dict[str, str] = {
    "15min": "15-minute",
    "H": "Hourly",
    "D": "Daily",
}


def _round_temporal(
    dt_series: pd.Series, resolution: TemporalResolution
) -> pd.Series:
    """
    Round a datetime Series to the specified temporal resolution.

    Parameters:
        dt_series:  Series of datetime values.
        resolution: One of '15min', 'H', 'D'.

    Returns:
        Series with datetimes floored to the given resolution.
    """
    if resolution == "15min":
        return dt_series.dt.floor("15min")
    elif resolution == "H":
        return dt_series.dt.floor("h")
    elif resolution == "D":
        return dt_series.dt.floor("D")
    else:
        raise ValueError(f"Unsupported resolution: {resolution}. Use '15min', 'H', or 'D'.")


def calculate_uniqueness(
    df: pd.DataFrame,
    temporal_resolution: TemporalResolution = "H",
) -> dict:
    """
    Calculate the uniqueness of trip records based on quasi-identifier
    combinations of (PULocationID, DOLocationID, rounded_pickup_time).

    A record is 'unique' if its quasi-identifier combination appears
    exactly once in the dataset. Following de Montjoye et al. (2013),
    high uniqueness indicates that zone-level generalisation is
    insufficient to prevent re-identification at the given temporal
    precision.

    Parameters:
        df:                   NYC Taxi trip DataFrame.
        temporal_resolution:  Temporal rounding: '15min', 'H', or 'D'.

    Returns:
        Dictionary with keys:
            - uniqueness_percentage: float (0-100)
            - unique_count: int (records with count == 1)
            - total_records: int
            - total_combinations: int (distinct QI tuples)
            - resolution: str
            - value_counts: pd.Series (group sizes, indexed by size)
    """
    required = [PICKUP_LOCATION, DROPOFF_LOCATION, PICKUP_DATETIME]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[[PICKUP_LOCATION, DROPOFF_LOCATION, PICKUP_DATETIME]].copy()
    work["_pickup_rounded"] = _round_temporal(work[PICKUP_DATETIME], temporal_resolution)

    qi_cols = [PICKUP_LOCATION, DROPOFF_LOCATION, "_pickup_rounded"]

    # Drop rows with NaN in any QI column
    work = work.dropna(subset=qi_cols)

    group_sizes = work.groupby(qi_cols).size()
    total_records = len(work)
    total_combinations = len(group_sizes)
    unique_count = int((group_sizes == 1).sum())

    # Distribution of group sizes (how many groups have size 1, 2, 3, ...)
    value_counts = group_sizes.value_counts().sort_index()

    uniqueness_pct = (unique_count / total_records * 100) if total_records > 0 else 0.0

    return {
        "uniqueness_percentage": round(uniqueness_pct, 2),
        "unique_count": unique_count,
        "total_records": total_records,
        "total_combinations": total_combinations,
        "resolution": _RESOLUTION_LABELS.get(temporal_resolution, temporal_resolution),
        "value_counts": value_counts,
    }


def compare_temporal_resolutions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare uniqueness across multiple temporal resolutions.

    Demonstrates how temporal precision affects the privacy guarantees
    of zone-level generalisation — a key academic finding showing the
    trade-off between data utility and privacy.

    Parameters:
        df: NYC Taxi trip DataFrame.

    Returns:
        DataFrame with columns:
            - resolution: human-readable label
            - uniqueness_percentage: float
            - unique_count: int
            - total_combinations: int
            - total_records: int
    """
    resolutions: list[TemporalResolution] = ["15min", "H", "D"]
    rows = []

    for res in resolutions:
        result = calculate_uniqueness(df, temporal_resolution=res)
        rows.append({
            "resolution": result["resolution"],
            "resolution_code": res,
            "uniqueness_percentage": result["uniqueness_percentage"],
            "unique_count": result["unique_count"],
            "total_combinations": result["total_combinations"],
            "total_records": result["total_records"],
        })

    comparison_df = pd.DataFrame(rows)
    return comparison_df


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_loader import load_month

    print("Uniqueness Analysis — Self Test")
    print("=" * 60)
    df = load_month(2019, 1, sample_frac=0.01)

    for res in ["15min", "H", "D"]:
        result = calculate_uniqueness(df, temporal_resolution=res)
        print(f"\n{result['resolution']} resolution:")
        print(f"  Uniqueness: {result['uniqueness_percentage']:.2f}%")
        print(f"  Unique combinations: {result['unique_count']:,}")
        print(f"  Total combinations:  {result['total_combinations']:,}")

    print("\nResolution comparison:")
    comp = compare_temporal_resolutions(df)
    print(comp.to_string(index=False))
