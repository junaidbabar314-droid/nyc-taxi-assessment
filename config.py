"""
Shared configuration constants for the NYC Taxi Assessment Framework.

All modules reference this file for dataset paths, valid ranges, and
field mappings to ensure consistency across quality, privacy, and
security assessments.

Group project — MSc IT (Data Analysis), University of the West of Scotland:
    Junaid Babar   (B01802551) — Module 1 · Data Quality Profiling
    Sami Ullah     (B01750598) — Module 2 · Privacy Risk Detection
    Jannat Rafique (B01798960) — Module 3 · Security Compliance
    Iqra Aziz      (B01802319) — Module 4 · Governance Dashboard

This file is a shared dependency for every module above. Any change to
thresholds, weights, or paths must be coordinated across the team so
that all four modules continue to produce consistent results.
"""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TAXI_ZONE_CSV = DATA_DIR / "taxi_zones.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ─── Parquet file pattern ────────────────────────────────────────────
PARQUET_PATTERN = "yellow_tripdata_{year}-{month:02d}.parquet"
AVAILABLE_YEARS = [2019, 2024]
MONTHS = range(1, 13)

# ─── Schema: column names present in the actual Parquet files ────────
COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
]

# ─── Valid ranges ────────────────────────────────────────────────────
# NYC Taxi Zone IDs (1-263 standard zones, 264-265 unknown/other)
VALID_LOCATION_ID_MIN = 1
VALID_LOCATION_ID_MAX = 265

# Vendor IDs observed in the dataset
VALID_VENDOR_IDS = {1, 2, 4, 5, 6}

# Maximum realistic values for outlier context
MAX_REALISTIC_FARE = 500.0        # dollars
MAX_REALISTIC_DISTANCE = 200.0    # miles (JFK round-trips can be long)
MAX_PASSENGER_COUNT = 6           # standard taxi vehicle capacity
MAX_REALISTIC_SPEED_MPH = 100.0   # flag trips exceeding this

# ─── Fare component columns (for consistency checks) ────────────────
FARE_COMPONENTS = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
]
FARE_TOTAL_COLUMN = "total_amount"
FARE_TOLERANCE = 0.01  # dollars — tolerance for fare sum verification

# ─── Payment type mapping ────────────────────────────────────────────
PAYMENT_TYPE_MAP = {
    0: "Unknown",
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Voided trip",
    6: "Unknown",
}

# ─── Rate code mapping ──────────────────────────────────────────────
RATECODE_MAP = {
    1: "Standard rate",
    2: "JFK",
    3: "Newark",
    4: "Nassau or Westchester",
    5: "Negotiated fare",
    6: "Group ride",
    99: "Unknown",
}

# ─── Datetime columns ───────────────────────────────────────────────
PICKUP_DATETIME = "tpep_pickup_datetime"
DROPOFF_DATETIME = "tpep_dropoff_datetime"

# ─── Location columns ───────────────────────────────────────────────
PICKUP_LOCATION = "PULocationID"
DROPOFF_LOCATION = "DOLocationID"

# ─── Key NYC landmarks mapped to Taxi Zone IDs ──────────────────────
# Used by privacy module for linkage attack simulation
NYC_LANDMARKS = {
    "JFK Airport Terminal 1": 132,
    "JFK Airport Terminal 4": 132,
    "JFK Airport (other)": 138,
    "LaGuardia Airport": 138,
    "Newark Airport": 1,       # EWR is zone 1 in some mappings
    "Times Square": 186,
    "Penn Station": 186,
    "Grand Central Terminal": 161,
    "Empire State Building": 161,
    "World Trade Center": 261,
    "Wall Street": 261,
    "Central Park South": 43,
    "Columbus Circle": 43,
    "Brooklyn Bridge": 26,
    "Yankee Stadium": 69,
    "Citi Field (Mets)": 201,
    "Madison Square Garden": 186,
    "Statue of Liberty Ferry": 103,
    "Broadway Theater District": 230,
    "SoHo": 211,
    "Chelsea Market": 68,
    "Williamsburg": 257,
    "Barclays Center": 25,
    "Coney Island": 29,
    "Harlem - 125th St": 116,
    "Columbia University": 166,
    "NYU": 113,
    "MetLife Stadium": 1,
    "Rockefeller Center": 161,
    "Museum Mile (Met Museum)": 236,
}

# ─── Quality severity thresholds ────────────────────────────────────
COMPLETENESS_CRITICAL_THRESHOLD = 5.0   # >5% null → Critical
COMPLETENESS_MINOR_THRESHOLD = 1.0      # >1% null → Minor, else Good

# ─── Privacy risk scoring weights ────────────────────────────────────
# Based on Sami Ullah's spec: weighted composite risk score
PRIVACY_WEIGHT_UNIQUENESS = 0.35
PRIVACY_WEIGHT_K_ANONYMITY = 0.30
PRIVACY_WEIGHT_ENTROPY = 0.20
PRIVACY_WEIGHT_LINKAGE = 0.15

# Risk level thresholds
PRIVACY_RISK_CRITICAL = 75
PRIVACY_RISK_HIGH = 50
PRIVACY_RISK_MEDIUM = 25

# ─── NIST CSF Functions ─────────────────────────────────────────────
NIST_FUNCTIONS = ["Govern", "Identify", "Protect", "Detect", "Respond", "Recover"]
