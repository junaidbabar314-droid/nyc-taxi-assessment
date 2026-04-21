"""
Linkage Attack Simulation Module for NYC Taxi Trip Records.

Simulates a background-knowledge linkage attack where an adversary
knows that a target visited a specific NYC landmark and attempts to
identify their trip record. Even though NYC TLC generalised GPS
coordinates to zone-level IDs, many landmarks map to a small number
of zones — meaning trips to/from these landmarks can be linked with
high confidence.

This module demonstrates that zone-level generalisation provides
uneven protection: trips involving well-known, zone-identifiable
landmarks (e.g., JFK Airport maps to zone 132) are more vulnerable
than trips in zones containing many diverse locations.

Attack model:
  1. Adversary knows the target visited landmark L at time T.
  2. Adversary looks up the zone ID for L.
  3. Adversary queries the dataset for trips with matching zone and
     approximate time.
  4. If the resulting set is small, the target can be re-identified.

References:
    Narayanan, A. and Shmatikov, V. (2008) 'Robust de-anonymization of
        large sparse datasets', Proceedings of the 2008 IEEE Symposium
        on Security and Privacy, pp. 111-125.
    Zang, H. and Bolot, J. (2011) 'Anonymization of location data does
        not work: A large-scale measurement study', Proceedings of the
        17th Annual International Conference on Mobile Computing and
        Networking (MobiCom), pp. 145-156.

Author: Sami Ullah (B01750598)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    NYC_LANDMARKS,
    PICKUP_LOCATION,
    DROPOFF_LOCATION,
    PICKUP_DATETIME,
)


def get_landmark_zones() -> dict[str, int]:
    """
    Return the NYC landmark-to-zone mapping from shared configuration.

    These 30 landmarks are well-known locations whose zone IDs are
    publicly available, enabling an adversary to link trip records
    to specific real-world locations.

    Returns:
        Dictionary mapping landmark_name -> zone_id.
    """
    return NYC_LANDMARKS.copy()


def simulate_linkage_attack(
    df: pd.DataFrame,
    landmark_zones: Optional[dict[str, int]] = None,
) -> dict:
    """
    Simulate a linkage attack using known landmark-to-zone mappings.

    For each landmark, counts how many trips have their pickup or dropoff
    in the landmark's zone. Trips involving landmark zones are considered
    'linkable' because an adversary with background knowledge (e.g.,
    "the target was at JFK Airport") can narrow down candidate records.

    The linkage rate represents the proportion of all trips that touch
    at least one landmark zone — a measure of the dataset's vulnerability
    to background-knowledge attacks.

    Parameters:
        df:              NYC Taxi trip DataFrame.
        landmark_zones:  Landmark-to-zone mapping. Defaults to NYC_LANDMARKS.

    Returns:
        Dictionary with keys:
            - linkage_rate: float (0-100, % of trips touching landmark zones)
            - total_linkable_trips: int
            - total_trips: int
            - n_landmarks: int
            - n_unique_zones: int (distinct zone IDs across all landmarks)
            - landmark_breakdown: pd.DataFrame with per-landmark stats
    """
    if landmark_zones is None:
        landmark_zones = get_landmark_zones()

    total_trips = len(df)
    if total_trips == 0:
        return {
            "linkage_rate": 0.0,
            "total_linkable_trips": 0,
            "total_trips": 0,
            "n_landmarks": len(landmark_zones),
            "n_unique_zones": 0,
            "landmark_breakdown": pd.DataFrame(),
        }

    # Collect all unique landmark zone IDs
    unique_zones = set(landmark_zones.values())

    # Identify trips touching any landmark zone (pickup OR dropoff)
    pu_in_landmark = df[PICKUP_LOCATION].isin(unique_zones)
    do_in_landmark = df[DROPOFF_LOCATION].isin(unique_zones)
    linkable_mask = pu_in_landmark | do_in_landmark
    total_linkable = int(linkable_mask.sum())

    linkage_rate = (total_linkable / total_trips * 100) if total_trips > 0 else 0.0

    # Per-landmark breakdown
    breakdown_rows = []
    for landmark_name, zone_id in landmark_zones.items():
        pu_count = int((df[PICKUP_LOCATION] == zone_id).sum())
        do_count = int((df[DROPOFF_LOCATION] == zone_id).sum())
        trip_count = pu_count + do_count  # may double-count if PU==DO==zone
        pct = (trip_count / total_trips * 100) if total_trips > 0 else 0.0

        breakdown_rows.append({
            "landmark": landmark_name,
            "zone_id": zone_id,
            "pickup_trips": pu_count,
            "dropoff_trips": do_count,
            "total_trips": trip_count,
            "pct_of_dataset": round(pct, 4),
        })

    breakdown_df = pd.DataFrame(breakdown_rows).sort_values(
        "total_trips", ascending=False
    ).reset_index(drop=True)

    return {
        "linkage_rate": round(linkage_rate, 2),
        "total_linkable_trips": total_linkable,
        "total_trips": total_trips,
        "n_landmarks": len(landmark_zones),
        "n_unique_zones": len(unique_zones),
        "landmark_breakdown": breakdown_df,
    }


def simulate_temporal_linkage(
    df: pd.DataFrame,
    landmark_zones: Optional[dict[str, int]] = None,
    temporal_resolution: str = "H",
) -> dict:
    """
    Enhanced linkage attack combining landmark zone knowledge with
    temporal constraints.

    An adversary who knows both the landmark AND approximate time
    can narrow down candidates further. This function measures
    the average anonymity set size for trips to landmark zones
    at each time slot.

    Parameters:
        df:                   NYC Taxi trip DataFrame.
        landmark_zones:       Landmark-to-zone mapping.
        temporal_resolution:  Time rounding ('15min', 'H', 'D').

    Returns:
        Dictionary with keys:
            - avg_anonymity_set: float (average group size for landmark+time)
            - min_anonymity_set: int
            - max_anonymity_set: int
            - pct_singleton: float (% of landmark+time groups with size 1)
    """
    from module2_privacy.uniqueness import _round_temporal

    if landmark_zones is None:
        landmark_zones = get_landmark_zones()

    unique_zones = set(landmark_zones.values())
    work = df[df[PICKUP_LOCATION].isin(unique_zones)].copy()

    if len(work) == 0:
        return {
            "avg_anonymity_set": 0.0,
            "min_anonymity_set": 0,
            "max_anonymity_set": 0,
            "pct_singleton": 0.0,
        }

    work["_time_slot"] = _round_temporal(
        work[PICKUP_DATETIME], temporal_resolution
    )

    group_sizes = work.groupby([PICKUP_LOCATION, "_time_slot"]).size()

    return {
        "avg_anonymity_set": round(float(group_sizes.mean()), 2),
        "min_anonymity_set": int(group_sizes.min()),
        "max_anonymity_set": int(group_sizes.max()),
        "pct_singleton": round(
            float((group_sizes == 1).sum() / len(group_sizes) * 100), 2
        ),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_loader import load_month

    print("Linkage Attack Simulation — Self Test")
    print("=" * 60)
    df = load_month(2019, 1, sample_frac=0.01)
    result = simulate_linkage_attack(df)

    print(f"\n  Linkage rate:     {result['linkage_rate']:.2f}%")
    print(f"  Linkable trips:   {result['total_linkable_trips']:,}")
    print(f"  Total trips:      {result['total_trips']:,}")
    print(f"  Landmarks:        {result['n_landmarks']}")
    print(f"  Unique zones:     {result['n_unique_zones']}")

    print("\n  Top 10 landmarks by trip count:")
    top = result["landmark_breakdown"].head(10)
    print(top.to_string(index=False))
