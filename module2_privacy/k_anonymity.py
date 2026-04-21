"""
k-Anonymity Assessment Module for NYC Taxi Trip Records.

Implements k-anonymity analysis following Sweeney's (2002) model to
evaluate the effectiveness of NYC TLC's zone-based generalisation as
a privacy protection mechanism. A record satisfies k-anonymity if it
is indistinguishable from at least (k-1) other records with respect
to a set of quasi-identifiers.

For transportation data, the quasi-identifiers are:
  - PULocationID (pickup zone)
  - DOLocationID (dropoff zone)
  - Pickup time (rounded to configurable resolution)

Low k-values indicate that some trips can be singled out despite
zone-level generalisation — a significant finding for privacy risk
assessment in big data transportation systems.

References:
    Sweeney, L. (2002) 'k-Anonymity: A model for protecting privacy',
        International Journal of Uncertainty, Fuzziness and Knowledge-Based
        Systems, 10(5), pp. 557-570.
    Machanavajjhala, A. et al. (2007) 'l-Diversity: Privacy beyond
        k-anonymity', ACM Transactions on Knowledge Discovery from Data,
        1(1), Article 3.

Author: Sami Ullah (B01750598)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PICKUP_LOCATION, DROPOFF_LOCATION, PICKUP_DATETIME


def _round_temporal(dt_series: pd.Series, resolution: str) -> pd.Series:
    """Round datetime to the specified resolution."""
    if resolution == "15min":
        return dt_series.dt.floor("15min")
    elif resolution == "H":
        return dt_series.dt.floor("h")
    elif resolution == "D":
        return dt_series.dt.floor("D")
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")


def assess_k_anonymity(
    df: pd.DataFrame,
    qi_columns: Optional[list[str]] = None,
    temporal_resolution: str = "H",
) -> dict:
    """
    Assess k-anonymity of the dataset with respect to quasi-identifiers.

    Groups records by their quasi-identifier values and analyses the
    distribution of equivalence class sizes. Following Sweeney (2002),
    a dataset achieves k-anonymity when every equivalence class contains
    at least k records.

    Parameters:
        df:                   NYC Taxi trip DataFrame.
        qi_columns:           List of quasi-identifier columns. If None,
                              defaults to [PULocationID, DOLocationID, pickup_hour].
        temporal_resolution:  Temporal rounding for the pickup time QI.

    Returns:
        Dictionary with keys:
            - min_k: int — minimum equivalence class size (worst case)
            - max_k: int — largest equivalence class size
            - mean_k: float — average equivalence class size
            - median_k: float — median equivalence class size
            - total_equivalence_classes: int
            - records_below_k5_pct: float — % of records in classes with k < 5
            - k_distribution: dict — {bucket_label: count_of_classes}
            - equivalence_class_sizes: pd.Series — raw group sizes
    """
    work = df.copy()

    # Build quasi-identifier set
    if qi_columns is None:
        # Default: location pair + rounded pickup time
        if PICKUP_DATETIME in work.columns:
            work["_pickup_rounded"] = _round_temporal(
                work[PICKUP_DATETIME], temporal_resolution
            )
            qi_columns = [PICKUP_LOCATION, DROPOFF_LOCATION, "_pickup_rounded"]
        else:
            qi_columns = [PICKUP_LOCATION, DROPOFF_LOCATION]

    # Validate columns exist
    missing = [c for c in qi_columns if c not in work.columns]
    if missing:
        raise ValueError(f"Missing QI columns: {missing}")

    # Drop NaN rows in QI columns
    work = work.dropna(subset=qi_columns)

    if len(work) == 0:
        return {
            "min_k": 0,
            "max_k": 0,
            "mean_k": 0.0,
            "median_k": 0.0,
            "total_equivalence_classes": 0,
            "records_below_k5_pct": 0.0,
            "k_distribution": {},
            "equivalence_class_sizes": pd.Series(dtype=int),
        }

    # Group by QI columns → equivalence classes
    group_sizes = work.groupby(qi_columns).size()

    min_k = int(group_sizes.min())
    max_k = int(group_sizes.max())
    mean_k = float(group_sizes.mean())
    median_k = float(group_sizes.median())
    total_classes = len(group_sizes)

    # Records in equivalence classes with k < 5
    small_classes = group_sizes[group_sizes < 5]
    records_in_small = int(small_classes.sum())
    total_records = int(group_sizes.sum())
    records_below_k5_pct = (
        (records_in_small / total_records * 100) if total_records > 0 else 0.0
    )

    # Distribution buckets
    k_distribution = {
        "k=1 (unique)": int((group_sizes == 1).sum()),
        "k=2-5": int(((group_sizes >= 2) & (group_sizes <= 5)).sum()),
        "k=6-10": int(((group_sizes >= 6) & (group_sizes <= 10)).sum()),
        "k=11-50": int(((group_sizes >= 11) & (group_sizes <= 50)).sum()),
        "k>50": int((group_sizes > 50).sum()),
    }

    return {
        "min_k": min_k,
        "max_k": max_k,
        "mean_k": round(mean_k, 2),
        "median_k": round(median_k, 2),
        "total_equivalence_classes": total_classes,
        "records_below_k5_pct": round(records_below_k5_pct, 2),
        "k_distribution": k_distribution,
        "equivalence_class_sizes": group_sizes,
    }


def k_anonymity_by_borough(
    df: pd.DataFrame,
    taxi_zones: pd.DataFrame,
    temporal_resolution: str = "H",
) -> pd.DataFrame:
    """
    Assess k-anonymity broken down by pickup borough.

    Merges trip data with taxi zone metadata and computes k-anonymity
    metrics per borough, revealing geographic disparities in privacy
    protection (e.g., outer boroughs may have lower k due to fewer trips).

    Parameters:
        df:                   NYC Taxi trip DataFrame.
        taxi_zones:           Taxi zone lookup DataFrame with LocationID, Borough.
        temporal_resolution:  Temporal rounding for pickup time QI.

    Returns:
        DataFrame with columns: Borough, min_k, mean_k, median_k,
        records_below_k5_pct, total_equivalence_classes.
    """
    # Merge borough info
    if "Borough" not in df.columns:
        zone_lookup = taxi_zones[["LocationID", "Borough"]].copy()
        merged = df.merge(
            zone_lookup,
            left_on=PICKUP_LOCATION,
            right_on="LocationID",
            how="left",
        )
    else:
        merged = df.copy()

    results = []
    for borough, group in merged.groupby("Borough"):
        if len(group) < 10:
            continue
        metrics = assess_k_anonymity(group, temporal_resolution=temporal_resolution)
        results.append({
            "Borough": borough,
            "min_k": metrics["min_k"],
            "mean_k": metrics["mean_k"],
            "median_k": metrics["median_k"],
            "records_below_k5_pct": metrics["records_below_k5_pct"],
            "total_equivalence_classes": metrics["total_equivalence_classes"],
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_loader import load_month

    print("k-Anonymity Assessment — Self Test")
    print("=" * 60)
    df = load_month(2019, 1, sample_frac=0.01)
    metrics = assess_k_anonymity(df, temporal_resolution="H")

    print(f"  Min k:               {metrics['min_k']}")
    print(f"  Max k:               {metrics['max_k']}")
    print(f"  Mean k:              {metrics['mean_k']:.2f}")
    print(f"  Median k:            {metrics['median_k']:.2f}")
    print(f"  Equivalence classes: {metrics['total_equivalence_classes']:,}")
    print(f"  Records below k=5:   {metrics['records_below_k5_pct']:.2f}%")
    print(f"\n  k-Distribution:")
    for bucket, count in metrics["k_distribution"].items():
        print(f"    {bucket:<20} {count:>8,}")
