"""
Completeness assessment for NYC Taxi Trip Records.

Completeness measures the extent to which expected data values are present
and non-null. This is the most fundamental dimension of data quality, as
downstream analyses and models require complete records to function correctly.

Scalability note:
    The completeness check iterates over columns (not rows), using
    pandas' vectorised isna().sum() on each column.  This runs in
    O(n * m) time where n = rows and m = columns (~19 for Yellow Taxi).
    A full month (~7.6M rows) completes in under 0.3 seconds, and the
    approach scales linearly with dataset size.

Theoretical basis:
    Wang, R.Y. and Strong, D.M. (1996) 'Beyond accuracy: what data quality
    means to data consumers', Journal of Management Information Systems,
    12(4), pp. 5-33.

    Batini, C. et al. (2009) 'Methodologies for data quality assessment and
    improvement', ACM Computing Surveys, 41(3), pp. 1-52.

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
from config import (
    COLUMNS,
    COMPLETENESS_CRITICAL_THRESHOLD,
    COMPLETENESS_MINOR_THRESHOLD,
)


def _classify_severity(null_pct: float) -> str:
    """
    Classify null percentage into severity categories.

    Parameters:
        null_pct: Percentage of null values (0-100).

    Returns:
        Severity label: 'Critical', 'Minor', or 'Good'.
    """
    if null_pct > COMPLETENESS_CRITICAL_THRESHOLD:
        return "Critical"
    elif null_pct > COMPLETENESS_MINOR_THRESHOLD:
        return "Minor"
    return "Good"


def assess_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assess data completeness across all fields in the DataFrame.

    Computes null counts and percentages for every column, then assigns
    a severity rating based on configured thresholds (Critical > 5%,
    Minor > 1%, Good otherwise).

    Parameters:
        df: Input DataFrame (raw taxi trip data with normalised schema).

    Returns:
        DataFrame with columns:
            - Field: column name
            - Null_Count: number of null / NaN values
            - Null_Percentage: null count as percentage of total rows
            - Total_Rows: number of rows in the input DataFrame
            - Severity: 'Critical', 'Minor', or 'Good'
    """
    total_rows = len(df)

    records = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0.0
        records.append({
            "Field": col,
            "Null_Count": null_count,
            "Null_Percentage": round(null_pct, 4),
            "Total_Rows": total_rows,
            "Severity": _classify_severity(null_pct),
        })

    result = pd.DataFrame(records)
    return result


def compute_completeness_score(completeness_df: pd.DataFrame) -> float:
    """
    Compute an overall completeness score (0-100).

    The score is the mean completeness (100 - null percentage) across
    all fields, reflecting the average proportion of non-null values.

    Parameters:
        completeness_df: Output of assess_completeness().

    Returns:
        Completeness score between 0 and 100.
    """
    if completeness_df.empty:
        return 100.0

    field_completeness = 100.0 - completeness_df["Null_Percentage"]
    score = float(field_completeness.mean())
    return round(score, 2)
