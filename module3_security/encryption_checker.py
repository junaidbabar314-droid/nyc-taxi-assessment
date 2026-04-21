"""
Encryption assessment for Parquet data files.

This module inspects NYC Yellow Taxi Parquet files to determine whether
data-at-rest encryption is applied.  Apache Parquet supports column-level
and footer encryption (Parquet Modular Encryption, PME) since format
version 2.6, but the publicly distributed NYC TLC files do not use it.

The checker reads Parquet metadata via PyArrow without loading row data,
making it efficient even for large files.

References:
    NIST (2024) Cybersecurity Framework (CSF) 2.0. National Institute of
        Standards and Technology, Gaithersburg, MD.
    Sharma, S. and Garg, V.K. (2024) 'Big data security and privacy
        issues in transportation systems', Journal of Big Data, 11(1),
        pp. 1-25.
    Apache Software Foundation (2023) Parquet Format Specification,
        available at: https://parquet.apache.org/documentation/latest/.

Author: Jannat Rafique (B01798960)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

logger = logging.getLogger(__name__)


def check_parquet_encryption(file_path: Union[str, Path]) -> dict:
    """
    Inspect a single Parquet file for encryption metadata.

    Uses PyArrow to open the Parquet file and examine its metadata,
    including whether Parquet Modular Encryption (PME) is active and
    which compression codec is applied to row groups.

    Parameters:
        file_path: Absolute or relative path to a .parquet file.

    Returns:
        dict with keys:
            - file_name (str): Base name of the file.
            - file_size_mb (float): File size in megabytes.
            - encrypted (bool): Whether the file uses Parquet encryption.
            - encryption_algorithm (str | None): Encryption algorithm if
              encrypted, else None.
            - num_row_groups (int): Number of row groups in the file.
            - num_columns (int): Number of columns in the schema.
            - compression_codec (str): Compression codec of the first
              row group's first column (e.g. 'SNAPPY', 'GZIP', 'NONE').
            - status (str): 'PASS' if encrypted, 'FAIL' otherwise.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        pyarrow.ArrowInvalidError: If the file is not valid Parquet.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)

    pf = pq.ParquetFile(file_path)
    metadata = pf.metadata

    # --- Encryption detection -------------------------------------------
    # PyArrow exposes encryption information through the file metadata.
    # If the file is encrypted with PME, the footer will indicate it.
    encrypted = False
    encryption_algorithm: Optional[str] = None

    # Check for encryption via the file-level metadata key-value pairs
    file_meta = metadata.metadata or {}
    for key in file_meta:
        decoded_key = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        if "encrypt" in decoded_key.lower():
            encrypted = True
            val = file_meta[key]
            encryption_algorithm = (
                val.decode("utf-8") if isinstance(val, bytes) else str(val)
            )
            break

    # Additional heuristic: PyArrow ParquetFile will raise or set flags
    # for encrypted footers.  We also inspect the created_by string.
    created_by = metadata.created_by or ""
    if "encryption" in created_by.lower():
        encrypted = True

    # --- Compression detection ------------------------------------------
    compression_codec = "UNKNOWN"
    if metadata.num_row_groups > 0:
        row_group_0 = metadata.row_group(0)
        if row_group_0.num_columns > 0:
            col_0 = row_group_0.column(0)
            compression_codec = col_0.compression

    num_columns = metadata.num_columns
    num_row_groups = metadata.num_row_groups

    status = "PASS" if encrypted else "FAIL"

    result = {
        "file_name": file_path.name,
        "file_size_mb": file_size_mb,
        "encrypted": encrypted,
        "encryption_algorithm": encryption_algorithm,
        "num_row_groups": num_row_groups,
        "num_columns": num_columns,
        "compression_codec": str(compression_codec),
        "status": status,
    }

    logger.info(
        "Encryption check: %s | encrypted=%s | compression=%s",
        file_path.name,
        encrypted,
        compression_codec,
    )
    return result


def scan_all_files(data_dir: Union[str, Path, None] = None) -> pd.DataFrame:
    """
    Scan every Parquet file in *data_dir* for encryption status.

    Parameters:
        data_dir: Directory containing .parquet files.
                  Defaults to ``config.DATA_DIR``.

    Returns:
        pandas.DataFrame with one row per file and columns matching
        the keys returned by :func:`check_parquet_encryption`.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    files = sorted(data_dir.glob("*.parquet"))

    if not files:
        logger.warning("No Parquet files found in %s", data_dir)
        return pd.DataFrame()

    results = []
    for fp in files:
        try:
            results.append(check_parquet_encryption(fp))
        except Exception as exc:
            logger.error("Failed to check %s: %s", fp.name, exc)
            results.append({
                "file_name": fp.name,
                "file_size_mb": None,
                "encrypted": None,
                "encryption_algorithm": None,
                "num_row_groups": None,
                "num_columns": None,
                "compression_codec": None,
                "status": "ERROR",
            })

    df = pd.DataFrame(results)
    logger.info(
        "Scanned %d files: %d encrypted, %d unencrypted",
        len(df),
        df["encrypted"].sum() if df["encrypted"].notna().any() else 0,
        (~df["encrypted"].fillna(False)).sum(),
    )
    return df


# ── Self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 60)
    print("Encryption Checker — Self-Test")
    print("=" * 60)

    df = scan_all_files()
    if not df.empty:
        print(df.to_string(index=False))
        n_enc = df["encrypted"].sum()
        print(f"\nSummary: {n_enc}/{len(df)} files encrypted")
    else:
        print("No files found.")
