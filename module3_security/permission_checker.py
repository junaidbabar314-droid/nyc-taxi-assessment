"""
File permission assessment for Parquet data files.

This module evaluates operating-system-level access controls on data
files.  On POSIX systems it reads the full permission mask (owner, group,
other); on Windows it inspects the read-only attribute and attempts to
query NTFS Access Control Lists (ACLs) via the ``win32security`` API
when available.

Limitations:
    - Windows ``os.stat()`` only exposes the read-only bit; full ACL
      inspection requires the optional ``pywin32`` package.
    - Running inside WSL or Cygwin may report POSIX-style permissions
      that do not reflect actual NTFS ACLs.

References:
    NIST (2024) Cybersecurity Framework (CSF) 2.0. National Institute of
        Standards and Technology, Gaithersburg, MD.
    ISO (2013) ISO/IEC 27001:2013 — Information security management
        systems.  International Organization for Standardization.
    European Union (2018) General Data Protection Regulation (GDPR),
        Regulation (EU) 2016/679.

Author: Jannat Rafique (B01798960)
"""

from __future__ import annotations

import logging
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"

# Try to import pywin32 for detailed Windows ACL inspection
_HAS_WIN32 = False
if IS_WINDOWS:
    try:
        import win32security  # type: ignore[import-untyped]
        import ntsecuritycon as con  # type: ignore[import-untyped]
        _HAS_WIN32 = True
    except ImportError:
        pass


def _posix_permissions(file_path: Path) -> dict:
    """Extract POSIX permission details from *file_path*."""
    st = file_path.stat()
    mode = st.st_mode

    perms_str = stat.filemode(mode)
    is_world_readable = bool(mode & stat.S_IROTH)
    is_world_writable = bool(mode & stat.S_IWOTH)

    # Determine owner — best-effort
    try:
        import pwd
        owner = pwd.getpwuid(st.st_uid).pw_name
    except (ImportError, KeyError):
        owner = str(st.st_uid)

    # Risk assessment
    if is_world_writable:
        risk_level = "High"
        status = "FAIL"
        recommendation = (
            "Remove world-writable permission (chmod o-w) to prevent "
            "unauthorised modification of data files."
        )
    elif is_world_readable:
        risk_level = "Medium"
        status = "PARTIAL"
        recommendation = (
            "Restrict read access to the data-owner group only "
            "(chmod o-r) to reduce exposure of potentially sensitive "
            "transportation records."
        )
    else:
        risk_level = "Low"
        status = "PASS"
        recommendation = "File permissions are appropriately restrictive."

    return {
        "file_name": file_path.name,
        "permissions_str": perms_str,
        "is_world_readable": is_world_readable,
        "is_world_writable": is_world_writable,
        "owner": owner,
        "risk_level": risk_level,
        "status": status,
        "recommendation": recommendation,
    }


def _windows_permissions(file_path: Path) -> dict:
    """Extract Windows permission details from *file_path*."""
    st = file_path.stat()
    is_read_only = not (st.st_mode & stat.S_IWRITE)

    owner = "UNKNOWN"
    acl_info = "N/A (pywin32 not installed)"
    is_world_readable = True   # conservative default on Windows
    is_world_writable = not is_read_only

    if _HAS_WIN32:
        try:
            sd = win32security.GetFileSecurity(
                str(file_path),
                win32security.OWNER_SECURITY_INFORMATION
                | win32security.DACL_SECURITY_INFORMATION,
            )
            owner_sid = sd.GetSecurityDescriptorOwner()
            name, domain, _ = win32security.LookupAccountSid(None, owner_sid)
            owner = f"{domain}\\{name}"

            dacl = sd.GetSecurityDescriptorDacl()
            if dacl is not None:
                everyone_sid = win32security.ConvertStringSidToSid("S-1-1-0")
                for i in range(dacl.GetAceCount()):
                    ace = dacl.GetAce(i)
                    ace_sid = ace[2]
                    if ace_sid == everyone_sid:
                        mask = ace[1]
                        if mask & con.FILE_GENERIC_WRITE:
                            is_world_writable = True
                        if mask & con.FILE_GENERIC_READ:
                            is_world_readable = True
                acl_info = f"{dacl.GetAceCount()} ACEs inspected"
            else:
                acl_info = "NULL DACL (full access)"
                is_world_readable = True
                is_world_writable = True
        except Exception as exc:
            logger.debug("win32security inspection failed: %s", exc)
            acl_info = f"ACL query error: {exc}"
    else:
        acl_info = "pywin32 not available — limited to read-only attribute"

    # Build permissions string for Windows
    perms_str = f"{'R' if True else '-'}{'W' if not is_read_only else '-'} (Windows; {acl_info})"

    # Risk assessment
    if is_world_writable:
        risk_level = "High"
        status = "FAIL"
        recommendation = (
            "Set file to read-only and restrict NTFS ACLs to authorised "
            "users/groups only.  Use icacls or Properties > Security tab."
        )
    elif is_world_readable:
        risk_level = "Medium"
        status = "PARTIAL"
        recommendation = (
            "Restrict NTFS read permissions to the data-owner group. "
            "Remove the 'Everyone' or 'Users' ACE if present."
        )
    else:
        risk_level = "Low"
        status = "PASS"
        recommendation = "File permissions are appropriately restrictive."

    return {
        "file_name": file_path.name,
        "permissions_str": perms_str,
        "is_world_readable": is_world_readable,
        "is_world_writable": is_world_writable,
        "owner": owner,
        "risk_level": risk_level,
        "status": status,
        "recommendation": recommendation,
    }


def check_file_permissions(file_path: Union[str, Path]) -> dict:
    """
    Assess operating-system file permissions for a data file.

    On POSIX (Linux/macOS) the full permission mask is inspected.
    On Windows the read-only attribute is checked and, if ``pywin32``
    is installed, NTFS ACLs are queried for the 'Everyone' SID.

    Parameters:
        file_path: Path to the file to inspect.

    Returns:
        dict with keys:
            - file_name (str)
            - permissions_str (str): Human-readable permission string.
            - is_world_readable (bool)
            - is_world_writable (bool)
            - owner (str): File owner username or UID.
            - risk_level (str): 'Low', 'Medium', or 'High'.
            - status (str): 'PASS', 'PARTIAL', or 'FAIL'.
            - recommendation (str): Actionable remediation guidance.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if IS_WINDOWS:
        return _windows_permissions(file_path)
    else:
        return _posix_permissions(file_path)


def scan_all_permissions(data_dir: Union[str, Path, None] = None) -> pd.DataFrame:
    """
    Check file permissions on every Parquet file in *data_dir*.

    Parameters:
        data_dir: Directory containing .parquet files.
                  Defaults to ``config.DATA_DIR``.

    Returns:
        pandas.DataFrame with one row per file.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    files = sorted(data_dir.glob("*.parquet"))

    if not files:
        logger.warning("No Parquet files found in %s", data_dir)
        return pd.DataFrame()

    results = []
    for fp in files:
        try:
            results.append(check_file_permissions(fp))
        except Exception as exc:
            logger.error("Permission check failed for %s: %s", fp.name, exc)
            results.append({
                "file_name": fp.name,
                "permissions_str": "ERROR",
                "is_world_readable": None,
                "is_world_writable": None,
                "owner": "UNKNOWN",
                "risk_level": "High",
                "status": "FAIL",
                "recommendation": f"Could not inspect permissions: {exc}",
            })

    df = pd.DataFrame(results)
    risk_counts = df["risk_level"].value_counts().to_dict()
    logger.info("Permission scan complete: %s", risk_counts)
    return df


# ── Self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 60)
    print("Permission Checker — Self-Test")
    print(f"Platform: {platform.system()} | pywin32: {_HAS_WIN32}")
    print("=" * 60)

    df = scan_all_permissions()
    if not df.empty:
        print(df[["file_name", "permissions_str", "risk_level", "status"]].to_string(index=False))
    else:
        print("No files found.")
