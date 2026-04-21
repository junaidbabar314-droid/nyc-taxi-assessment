"""
Cached data-loading wrappers for the Streamlit dashboard.

Wraps the shared Phase2/data_loader.py functions with Streamlit's
@st.cache_data decorator to avoid redundant I/O across reruns and pages.
Auto-samples datasets exceeding 5 million rows to maintain interactive
response times, following Shneiderman's (1996) mantra of "overview first,
zoom and filter, then details-on-demand."

References:
    Shneiderman, B. (1996) 'The eyes have it: a task by data type taxonomy
        for information visualizations', in Proceedings IEEE Symposium on
        Visual Languages, pp. 336-343.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Ensure Phase2 root is importable
_PHASE2_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE2_ROOT))

from data_loader import (            # noqa: E402 — path-dependent import
    load_month as _raw_load_month,
    load_year as _raw_load_year,
    load_taxi_zones as _raw_load_zones,
    list_parquet_files as _raw_list_files,
    load_with_zones as _raw_load_with_zones,
)
from config import DATA_DIR, PARQUET_PATTERN  # noqa: E402

# ─── Constants ──────────────────────────────────────────────────────────
AUTO_SAMPLE_THRESHOLD = 5_000_000   # rows — auto-sample above this
AUTO_SAMPLE_FRAC = 0.10             # 10% sample when auto-triggered

# On Streamlit Community Cloud the repo ships without the raw Parquet
# files (they are release assets).  Auto-fetch the requested month on
# first load so the dashboard works out of the box when deployed.
_RELEASE_BASE = (
    "https://github.com/junaidbabar314-droid/nyc-taxi-assessment/"
    "releases/download/v1.0-data"
)


def _ensure_month_present(year: int, month: int) -> None:
    """Download a single month Parquet from the GitHub release if missing.

    Silent no-op when the file already exists locally.  Intended for
    Streamlit Cloud deployments where ``data/`` starts empty.
    """
    filename = PARQUET_PATTERN.format(year=year, month=month)
    target = DATA_DIR / filename
    if target.exists() and target.stat().st_size > 0:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_RELEASE_BASE}/{filename}"
    import urllib.request
    with st.spinner(f"Downloading {filename} from release (first run)..."):
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)


@st.cache_data(show_spinner="Loading trip data...", ttl=3600)
def cached_load_month(
    year: int,
    month: int,
    sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load a single month of taxi data with Streamlit caching.

    If the loaded DataFrame exceeds AUTO_SAMPLE_THRESHOLD rows and no
    explicit sample_frac was requested, automatically samples to keep
    the dashboard responsive.

    Parameters:
        year:        Data year (2019 or 2024).
        month:       Data month (1-12).
        sample_frac: Explicit sampling fraction (0.0-1.0). None = auto.

    Returns:
        Cached DataFrame.
    """
    _ensure_month_present(year, month)
    df = _raw_load_month(year, month, sample_frac=sample_frac)

    # Auto-sample large datasets when no explicit fraction was given
    if sample_frac is None and len(df) > AUTO_SAMPLE_THRESHOLD:
        n = int(len(df) * AUTO_SAMPLE_FRAC)
        df = df.sample(n=n, random_state=42)
        # Note: cannot call st.toast inside cached function — caller should
        # check len(df) and display a message if needed.

    return df


@st.cache_data(show_spinner="Loading full year...", ttl=3600)
def cached_load_year(
    year: int,
    sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    """Load all 12 months for a year with caching."""
    return _raw_load_year(year, sample_frac=sample_frac or AUTO_SAMPLE_FRAC)


@st.cache_data(show_spinner="Loading taxi zones...", ttl=86400)
def cached_load_zones() -> pd.DataFrame:
    """Load taxi zone lookup table with long TTL caching (static data)."""
    return _raw_load_zones()


@st.cache_data(show_spinner="Loading trip data with zones...", ttl=3600)
def cached_load_with_zones(
    year: int,
    month: int,
    sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    """Load trip data merged with zone names, with caching."""
    _ensure_month_present(year, month)
    df = _raw_load_with_zones(year, month, sample_frac=sample_frac)

    if sample_frac is None and len(df) > AUTO_SAMPLE_THRESHOLD:
        n = int(len(df) * AUTO_SAMPLE_FRAC)
        df = df.sample(n=n, random_state=42)

    return df


def cached_list_files() -> list:
    """List available Parquet files (lightweight, no caching needed)."""
    return _raw_list_files()
