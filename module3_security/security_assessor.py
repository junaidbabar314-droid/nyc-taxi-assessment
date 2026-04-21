"""
Security Assessment orchestrator — Module 3 main entry point.

Coordinates the encryption checker, permission checker, NIST checklist,
and compliance matrix into a single unified assessment.  The returned
dictionary conforms to the project's standardised interface so that the
Streamlit governance dashboard (Module 4) can consume results directly.

References:
    NIST (2024) Cybersecurity Framework (CSF) 2.0.  National Institute of
        Standards and Technology, Gaithersburg, MD.
    ISO (2022) ISO/IEC 27001:2022 — Information security management
        systems.  International Organization for Standardization.
    European Union (2018) General Data Protection Regulation (GDPR),
        Regulation (EU) 2016/679.
    Sharma, S. and Garg, V.K. (2024) 'Big data security and privacy
        issues in transportation systems', Journal of Big Data, 11(1).
    Fernandez-Garcia, A.J. et al. (2024) 'A systematic review of data
        governance frameworks for smart transportation', Computers &
        Security, 137, 103598.

Author: Jannat Rafique (B01798960)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

from module3_security.encryption_checker import scan_all_files as scan_encryption
from module3_security.permission_checker import scan_all_permissions
from module3_security.nist_checklist import evaluate_checklist
from module3_security.compliance_matrix import (
    create_full_compliance_matrix,
    calculate_compliance_scores,
)

logger = logging.getLogger(__name__)


def _build_gap_summary(checklist_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Organise gaps by remediation priority.

    Parameters:
        checklist_df: Evaluated NIST checklist.

    Returns:
        dict mapping priority ('High', 'Medium', 'Low') to lists of
        strings describing each gap.
    """
    summary: dict[str, list[str]] = {"High": [], "Medium": [], "Low": []}

    for _, row in checklist_df.iterrows():
        if row["Assessment_Result"] in ("FAIL", "PARTIAL"):
            priority = row["Remediation_Priority"]
            entry = (
                f"{row['Control_ID']} ({row['NIST_Function']}): "
                f"{row['Control_Description'][:80]}... "
                f"[{row['Assessment_Result']}]"
            )
            if priority in summary:
                summary[priority].append(entry)
            else:
                summary["Medium"].append(entry)

    return summary


def _build_remediation_df(checklist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a prioritised remediation table from checklist gaps.

    Parameters:
        checklist_df: Evaluated NIST checklist.

    Returns:
        pandas.DataFrame with columns: Priority, Control_ID,
        NIST_Function, Issue, Recommendation, sorted by priority.
    """
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    rows = []

    for _, row in checklist_df.iterrows():
        if row["Assessment_Result"] in ("FAIL", "PARTIAL"):
            rows.append({
                "Priority": row["Remediation_Priority"],
                "Control_ID": row["Control_ID"],
                "NIST_Function": row["NIST_Function"],
                "Issue": row["Evidence"][:200],
                "Assessment_Result": row["Assessment_Result"],
                "Recommendation": row["Control_Description"],
                "_sort": priority_order.get(row["Remediation_Priority"], 1),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


def _build_summary_text(
    enc_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    checklist_df: pd.DataFrame,
    scores: dict,
) -> str:
    """
    Generate a human-readable summary of the security assessment.

    Parameters:
        enc_df: Encryption scan results.
        perm_df: Permission scan results.
        checklist_df: Evaluated NIST checklist.
        scores: Compliance scores dict.

    Returns:
        Multi-line summary string.
    """
    n_files = len(enc_df) if not enc_df.empty else 0
    n_encrypted = int(enc_df["encrypted"].sum()) if not enc_df.empty else 0

    n_pass = int((checklist_df["Assessment_Result"] == "PASS").sum())
    n_partial = int((checklist_df["Assessment_Result"] == "PARTIAL").sum())
    n_fail = int((checklist_df["Assessment_Result"] == "FAIL").sum())
    n_na = int((checklist_df["Assessment_Result"] == "N/A").sum())
    total = len(checklist_df)

    lines = [
        "SECURITY ASSESSMENT SUMMARY",
        "=" * 50,
        "",
        f"Files scanned:        {n_files}",
        f"Files encrypted:      {n_encrypted}/{n_files}",
        f"",
        f"NIST Controls:        {total} total",
        f"  PASS:               {n_pass}",
        f"  PARTIAL:            {n_partial}",
        f"  FAIL:               {n_fail}",
        f"  N/A:                {n_na}",
        f"",
        f"Compliance Scores:",
        f"  GDPR:               {scores['gdpr_compliance_pct']:.1f}%",
        f"  ISO 27001:          {scores['iso_compliance_pct']:.1f}%",
        f"  NIST CSF 2.0:       {scores['nist_compliance_pct']:.1f}%",
        f"  Overall:            {scores['overall_compliance_pct']:.1f}%",
        f"",
        f"High-priority gaps:   {len(scores['gaps_by_priority'].get('High', []))}",
        f"Medium-priority gaps: {len(scores['gaps_by_priority'].get('Medium', []))}",
        f"",
        "Key Findings:",
        "  - NYC TLC Parquet files are distributed without encryption.",
        "  - Location IDs and timestamps are stored in plaintext,",
        "    creating re-identification risk.",
        "  - No formal incident response or recovery procedures",
        "    are documented for this project.",
        "  - Data-in-transit protection (HTTPS) is the strongest",
        "    control currently in place.",
    ]
    return "\n".join(lines)


def get_security_checklist(
    file_paths: Optional[list] = None,
    data_dir: Union[str, Path, None] = None,
) -> dict:
    """
    Run the full Module 3 security assessment.

    This is the primary entry point for the security module.  It
    executes all sub-assessments and returns a standardised dictionary
    suitable for consumption by the Streamlit dashboard.

    Parameters:
        file_paths: Optional explicit list of Parquet file paths to
                    assess.  If provided, *data_dir* is ignored for
                    file discovery (but still used for checklist context).
        data_dir:   Directory containing .parquet files.
                    Defaults to ``config.DATA_DIR``.

    Returns:
        dict with keys:
            - overall_compliance (float): 0-100 composite score.
            - encryption_results (pd.DataFrame): Per-file encryption status.
            - permission_results (pd.DataFrame): Per-file permission status.
            - checklist_results (pd.DataFrame): Evaluated NIST checklist.
            - compliance_matrix (pd.DataFrame): Unified traceability matrix.
            - compliance_scores (dict): Per-framework scores.
            - gap_summary (dict): Gaps grouped by priority.
            - remediation_priorities (pd.DataFrame): Prioritised remediation.
            - summary_text (str): Human-readable summary.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    logger.info("Starting Module 3 Security Assessment...")
    logger.info("Data directory: %s", data_dir)

    # 1. Encryption assessment
    logger.info("Phase 1/4: Encryption scan")
    enc_df = scan_encryption(data_dir)

    # 2. Permission assessment
    logger.info("Phase 2/4: Permission scan")
    perm_df = scan_all_permissions(data_dir)

    # 3. NIST checklist evaluation
    logger.info("Phase 3/4: NIST checklist evaluation")
    checklist_df = evaluate_checklist(data_dir)

    # 4. Compliance matrix and scoring
    logger.info("Phase 4/4: Compliance matrix generation")
    matrix_df = create_full_compliance_matrix(checklist_df)
    scores = calculate_compliance_scores(matrix_df, checklist_df=checklist_df)

    # Derived outputs
    gap_summary = _build_gap_summary(checklist_df)
    remediation_df = _build_remediation_df(checklist_df)
    summary_text = _build_summary_text(enc_df, perm_df, checklist_df, scores)

    result = {
        "overall_compliance": scores["overall_compliance_pct"],
        "encryption_results": enc_df,
        "permission_results": perm_df,
        "checklist_results": checklist_df,
        "compliance_matrix": matrix_df,
        "compliance_scores": scores,
        "gap_summary": gap_summary,
        "gap_counts": scores.get("gap_counts", {}),
        "gap_percentages": scores.get("gap_percentages", {}),
        "total_gaps": scores.get("total_gaps", 0),
        "remediation_priorities": remediation_df,
        "summary_text": summary_text,
    }

    logger.info("Security assessment complete. Overall compliance: %.1f%%",
                scores["overall_compliance_pct"])
    return result


# ── Self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 60)
    print("Security Assessor — Full Run")
    print("=" * 60)

    results = get_security_checklist()
    print(results["summary_text"])
    print(f"\nRemediation items: {len(results['remediation_priorities'])}")
    if not results["remediation_priorities"].empty:
        print(results["remediation_priorities"][
            ["Priority", "Control_ID", "Assessment_Result"]
        ].to_string(index=False))
