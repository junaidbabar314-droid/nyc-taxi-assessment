"""
Shared data loading layer for the NYC Taxi Assessment Framework.

Provides consistent Parquet loading with schema normalisation across
2019 and 2024 datasets. All four modules (quality, privacy, security,
dashboard) use this as their single entry point for data access.

Key design decisions:
  - Parquet files live in `data/` at the project root (no duplication
    of the ~1.85 GB corpus).
  - Schema normalisation handles 2019 vs 2024 dtype differences
    (int64→int32 VendorID, string→float airport_fee).
  - Raw data is loaded without cleaning — the quality module
    assesses data as-is to identify issues authentically.
  - Optional sampling via sample_frac for development speed.

Group project — MSc IT (Data Analysis), University of the West of Scotland:
    Junaid Babar   (B01802551) — Module 1 · Data Quality Profiling
    Sami Ullah     (B01750598) — Module 2 · Privacy Risk Detection
    Jannat Rafique (B01798960) — Module 3 · Security Compliance
    Iqra Aziz      (B01802319) — Module 4 · Governance Dashboard

This loader is a shared dependency consumed by every module above.

References:
  McKinney, W. (2012) Python for Data Analysis. O'Reilly Media.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    PARQUET_PATTERN,
    PICKUP_DATETIME,
    DROPOFF_DATETIME,
    TAXI_ZONE_CSV,
)

logger = logging.getLogger(__name__)


# ─── Schema normalisation ───────────────────────────────────────────

def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent dtypes across 2019 and 2024 data.

    Handles known differences:
      - VendorID: int64 (2019) vs int32 (2024) → coerce to Int64
      - airport_fee: object/None (2019) vs float64 (2024) → coerce to float64
      - congestion_surcharge: may be object in some months → coerce to float64
      - Datetime columns: ensure proper datetime64 dtype

    Parameters:
        df: Raw DataFrame loaded from Parquet.

    Returns:
        DataFrame with normalised dtypes.
    """
    df = df.copy()

    # Numeric columns that may contain mixed types
    numeric_coerce = [
        "VendorID", "passenger_count", "RatecodeID",
        "fare_amount", "extra", "mta_tax", "tip_amount",
        "tolls_amount", "improvement_surcharge", "total_amount",
        "congestion_surcharge", "airport_fee", "trip_distance",
    ]

    for col in numeric_coerce:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure datetime columns
    for dt_col in [PICKUP_DATETIME, DROPOFF_DATETIME]:
        if dt_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    # Nullable integer types for ID columns
    for id_col in ["VendorID", "PULocationID", "DOLocationID", "RatecodeID"]:
        if id_col in df.columns:
            df[id_col] = df[id_col].astype("Int64")

    return df


# ─── Single month loader ────────────────────────────────────────────

def load_month(
    year: int,
    month: int,
    sample_frac: Optional[float] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load a single month of NYC Yellow Taxi trip data from Parquet.

    Parameters:
        year:        Data year (2019 or 2024).
        month:       Data month (1-12).
        sample_frac: If provided, randomly sample this fraction (0.0-1.0).
                     Uses random_state=42 for reproducibility.
        data_dir:    Override default data directory.

    Returns:
        DataFrame with normalised schema.

    Raises:
        FileNotFoundError: If the Parquet file does not exist.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    filename = PARQUET_PATTERN.format(year=year, month=month)
    filepath = data_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    logger.info("Loading %s (%s)", filename, f"{filepath.stat().st_size / 1e6:.1f} MB")
    df = pd.read_parquet(filepath)

    df = normalise_schema(df)

    if sample_frac is not None and 0 < sample_frac < 1:
        n_sample = max(1, int(len(df) * sample_frac))
        df = df.sample(n=n_sample, random_state=42)
        logger.info("Sampled %d rows (%.1f%%)", n_sample, sample_frac * 100)

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


# ─── Multi-month loader ─────────────────────────────────────────────

def load_year(
    year: int,
    months: range = range(1, 13),
    sample_frac: Optional[float] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load multiple months of data for a given year, concatenated.

    Parameters:
        year:        Data year (2019 or 2024).
        months:      Range of months to load (default: all 12).
        sample_frac: If provided, sample each month before concatenation.
        data_dir:    Override default data directory.

    Returns:
        Concatenated DataFrame with normalised schema.
    """
    frames = []
    for m in months:
        try:
            df = load_month(year, m, sample_frac=sample_frac, data_dir=data_dir)
            df["source_month"] = m
            frames.append(df)
        except FileNotFoundError:
            logger.warning("Skipping missing file: year=%d, month=%d", year, m)

    if not frames:
        raise ValueError(f"No data files found for year {year}")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined %d months: %d total rows", len(frames), len(combined))
    return combined


# ─── Taxi zone lookup ────────────────────────────────────────────────

def load_taxi_zones(zone_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    Load NYC Taxi Zone lookup table mapping LocationID to zone names.

    Parameters:
        zone_csv: Path to taxi_zones.csv. Defaults to config path.

    Returns:
        DataFrame with columns: LocationID, Borough, Zone, service_zone.
    """
    zone_csv = Path(zone_csv) if zone_csv else TAXI_ZONE_CSV

    if not zone_csv.exists():
        logger.warning(
            "Taxi zone CSV not found at %s. "
            "Download from: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip",
            zone_csv,
        )
        return pd.DataFrame(columns=["LocationID", "Borough", "Zone", "service_zone"])

    zones = pd.read_csv(zone_csv)

    # Normalise column names (different sources use different headers)
    rename_map = {}
    for col in zones.columns:
        lower = col.lower().strip()
        if lower in ("locationid", "location_id", "objectid"):
            rename_map[col] = "LocationID"
        elif lower == "borough":
            rename_map[col] = "Borough"
        elif lower == "zone":
            rename_map[col] = "Zone"
        elif lower in ("service_zone", "service zone"):
            rename_map[col] = "service_zone"

    zones = zones.rename(columns=rename_map)
    logger.info("Loaded %d taxi zones", len(zones))
    return zones


# ─── Convenience: merge zone names ──────────────────────────────────

def load_with_zones(
    year: int,
    month: int,
    sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load trip data with pickup and dropoff zone names joined.

    Parameters:
        year:        Data year.
        month:       Data month.
        sample_frac: Optional sampling fraction.

    Returns:
        DataFrame with added columns: PU_Borough, PU_Zone, DO_Borough, DO_Zone.
    """
    df = load_month(year, month, sample_frac=sample_frac)
    zones = load_taxi_zones()

    if zones.empty:
        return df

    # Merge pickup zone
    pu_zones = zones[["LocationID", "Borough", "Zone"]].rename(
        columns={"Borough": "PU_Borough", "Zone": "PU_Zone"}
    )
    df = df.merge(pu_zones, left_on="PULocationID", right_on="LocationID", how="left")
    df = df.drop(columns=["LocationID"], errors="ignore")

    # Merge dropoff zone
    do_zones = zones[["LocationID", "Borough", "Zone"]].rename(
        columns={"Borough": "DO_Borough", "Zone": "DO_Zone"}
    )
    df = df.merge(do_zones, left_on="DOLocationID", right_on="LocationID", how="left")
    df = df.drop(columns=["LocationID"], errors="ignore")

    return df


# ─── File listing utility ───────────────────────────────────────────

def list_parquet_files(data_dir: Optional[Path] = None) -> list[Path]:
    """
    List all Parquet files in the data directory, sorted by name.

    Returns:
        List of Path objects for each .parquet file.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    files = sorted(data_dir.glob("*.parquet"))
    return files


# ─── Self-test when run directly ─────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("NYC Taxi Data Loader — Self Test")
    print("=" * 60)

    # List available files
    files = list_parquet_files()
    print(f"\nFound {len(files)} Parquet files in {DATA_DIR}")
    for f in files[:3]:
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    if len(files) > 3:
        print(f"  ... and {len(files) - 3} more")

    # Load a sample
    print("\nLoading 2019-01 (1% sample)...")
    df = load_month(2019, 1, sample_frac=0.01)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes.to_string()}")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())

    # Check taxi zones
    print("\nLoading taxi zones...")
    zones = load_taxi_zones()
    if not zones.empty:
        print(f"  {len(zones)} zones loaded")
        print(zones.head())
    else:
        print("  ⚠ Taxi zones CSV not found — download it for zone-based analysis")

    print("\n✓ Data loader self-test complete")
