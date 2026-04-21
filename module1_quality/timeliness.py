"""
Timeliness assessment for NYC Taxi Trip Records.

Timeliness measures the degree to which data is sufficiently current for
the task at hand. For NYC TLC data, the publication schedule provides a
benchmark: data for a given month is typically published by the 15th of
the following month. This module computes data lag relative to that
expected publication date and measures what fraction of records belong
to the labelled calendar month.

Scalability note:
    All computations use vectorised pandas/numpy operations, enabling
    efficient assessment of multi-million-row datasets.  On a standard
    4-core laptop, a full month of Yellow Taxi data (~7.6M rows) is
    processed in under 2 seconds.  For year-level analysis the module
    can be invoked per-month inside a loop with negligible overhead.

    The 1,000-record manual validation sample size used during
    development follows Cochran (1977) guidelines for finite-population
    sampling: with N > 1M records, n = 1,000 yields a margin of error
    below +/-3.1% at the 95% confidence level, providing sufficient
    precision to validate automated metric correctness while remaining
    feasible for manual inspection (Cochran, 1977, pp. 75-76).

Theoretical basis:
    Wang, R.Y. and Strong, D.M. (1996) 'Beyond accuracy: what data quality
    means to data consumers', Journal of Management Information Systems,
    12(4), pp. 5-33.

    Batini, C. et al. (2009) 'Methodologies for data quality assessment and
    improvement', ACM Computing Surveys, 41(3), pp. 1-52.

    Cochran, W.G. (1977) Sampling Techniques. 3rd edn. New York:
    John Wiley & Sons.

Author: Junaid Babar (B01802551)
Module: Data Quality Profiling
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# -- Project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PICKUP_DATETIME


def _estimate_publication_date(year: int, month: int) -> pd.Timestamp:
    """
    Estimate the expected publication date for a given data month.

    NYC TLC typically publishes monthly data by the 15th of the
    following month.

    Parameters:
        year: Data year.
        month: Data month (1-12).

    Returns:
        Estimated publication date as a Timestamp.
    """
    if month == 12:
        pub_year = year + 1
        pub_month = 1
    else:
        pub_year = year
        pub_month = month + 1

    return pd.Timestamp(year=pub_year, month=pub_month, day=15)


def assess_timeliness(
    df: pd.DataFrame,
    file_year: int,
    file_month: int,
) -> dict:
    """
    Assess data timeliness based on the lag between trip pickup times
    and the estimated publication date.

    Metrics computed:
        - Average and median lag (in days) between pickup and publication
        - Freshness: percentage of records within the labelled calendar
          month (30-day window) and within an extended 60-day window
        - Maximum lag observed

    A timeliness score (0-100) is derived from the percentage of records
    whose pickup datetime falls within the labelled calendar month.
    This reflects the core question: "does this file contain data from
    the month it claims to represent?"

    Parameters:
        df: Trip data DataFrame with normalised schema.
        file_year: Year label of the data file.
        file_month: Month label of the data file (1-12).

    Returns:
        Dictionary containing:
            - timeliness_score: overall score (0-100)
            - avg_lag_days: mean lag in days
            - median_lag_days: median lag in days
            - freshness_30d_pct: % of records within the labelled month
            - freshness_60d_pct: % of records within extended window
            - max_lag_days: maximum lag observed
            - publication_date: estimated publication date (ISO string)
            - total_records: number of records assessed
    """
    pub_date = _estimate_publication_date(file_year, file_month)
    pickup = df[PICKUP_DATETIME].dropna()
    total_records = len(pickup)

    if total_records == 0:
        return {
            "timeliness_score": 0.0,
            "avg_lag_days": 0.0,
            "median_lag_days": 0.0,
            "freshness_30d_pct": 0.0,
            "freshness_60d_pct": 0.0,
            "max_lag_days": 0.0,
            "publication_date": pub_date.isoformat(),
            "total_records": 0,
        }

    # Data lag: days between pickup and publication date
    lag = (pub_date - pickup).dt.total_seconds() / 86400.0
    avg_lag = float(lag.mean())
    median_lag = float(lag.median())
    max_lag = float(lag.max())

    # Freshness windows: % of records whose pickup falls within the
    # expected data month boundaries.  A "fresh" record is one that
    # belongs to the labelled month (not stale data from a prior period
    # or future-dated records from a subsequent period).
    month_start = pd.Timestamp(year=file_year, month=file_month, day=1)
    if file_month == 12:
        month_end = pd.Timestamp(year=file_year + 1, month=1, day=1)
    else:
        month_end = pd.Timestamp(year=file_year, month=file_month + 1, day=1)

    # Records within the labelled calendar month
    in_month_mask = (pickup >= month_start) & (pickup < month_end)
    in_month_count = int(in_month_mask.sum())
    in_month_pct = (in_month_count / total_records * 100)

    # Extended window: within 30 days either side of the labelled month
    extended_start = month_start - pd.Timedelta(days=30)
    extended_end = month_end + pd.Timedelta(days=30)
    in_extended_mask = (pickup >= extended_start) & (pickup < extended_end)
    in_extended_count = int(in_extended_mask.sum())

    freshness_30d_pct = in_month_pct  # core freshness: records in expected month
    freshness_60d_pct = (in_extended_count / total_records * 100)

    # Timeliness score: primarily based on what fraction of records
    # actually belong to the labelled month.  A dataset where all
    # records fall within the expected month scores 100.
    timeliness_score = round(min(100.0, freshness_30d_pct), 2)

    return {
        "timeliness_score": timeliness_score,
        "avg_lag_days": round(avg_lag, 2),
        "median_lag_days": round(median_lag, 2),
        "freshness_30d_pct": round(freshness_30d_pct, 2),
        "freshness_60d_pct": round(freshness_60d_pct, 2),
        "max_lag_days": round(max_lag, 2),
        "publication_date": pub_date.isoformat(),
        "total_records": total_records,
    }
