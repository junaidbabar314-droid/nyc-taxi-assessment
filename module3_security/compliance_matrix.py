"""
Regulatory compliance traceability matrix.

Maps NIST CSF 2.0 controls to GDPR articles and ISO 27001 Annex A
controls, producing a unified compliance matrix that enables gap analysis
across all three frameworks simultaneously.

The matrix supports the dissertation's argument that transportation
big-data systems must satisfy overlapping regulatory obligations, and
that a unified view simplifies governance (Fernandez-Garcia et al., 2024).

References:
    European Union (2018) General Data Protection Regulation (GDPR),
        Regulation (EU) 2016/679.
    ISO (2013) ISO/IEC 27001:2013 — Information security management
        systems — Requirements.
    NIST (2024) Cybersecurity Framework (CSF) 2.0. National Institute of
        Standards and Technology, Gaithersburg, MD.
    Fernandez-Garcia, A.J. et al. (2024) 'A systematic review of data
        governance frameworks for smart transportation', Computers &
        Security, 137, 103598.

Author: Jannat Rafique (B01798960)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ── GDPR mapping table ─────────────────────────────────────────────────

_GDPR_MAPPINGS: list[dict] = [
    {
        "Regulation": "GDPR",
        "Article": "Art. 25 — Data Protection by Design and Default",
        "Requirement": (
            "Implement appropriate technical and organisational measures "
            "to ensure data protection principles are embedded from "
            "the design stage."
        ),
        "Mapped_Controls": "ID.AM-5, PR.DS-5",
    },
    {
        "Regulation": "GDPR",
        "Article": "Art. 30 — Records of Processing Activities",
        "Requirement": (
            "Maintain records of processing activities including "
            "purposes, data categories, and recipients."
        ),
        "Mapped_Controls": "ID.AM-1, ID.AM-2",
    },
    {
        "Regulation": "GDPR",
        "Article": "Art. 32 — Security of Processing",
        "Requirement": (
            "Implement appropriate technical measures including "
            "encryption, pseudonymisation, and access controls to "
            "ensure a level of security appropriate to the risk."
        ),
        "Mapped_Controls": "PR.AC-1, PR.AC-4, PR.DS-1, PR.DS-2, ID.RA-1",
    },
    {
        "Regulation": "GDPR",
        "Article": "Art. 33 — Notification of Breach to Supervisory Authority",
        "Requirement": (
            "Notify the supervisory authority within 72 hours of "
            "becoming aware of a personal data breach."
        ),
        "Mapped_Controls": "DE.DP-4, RS.CO-2, DE.AE-3",
    },
    {
        "Regulation": "GDPR",
        "Article": "Art. 34 — Communication of Breach to Data Subject",
        "Requirement": (
            "Communicate a personal data breach to the data subject "
            "when the breach is likely to result in a high risk to "
            "their rights and freedoms."
        ),
        "Mapped_Controls": "RC.CO-3, DE.DP-4",
    },
    {
        "Regulation": "GDPR",
        "Article": "Art. 9 — Processing of Special Categories of Data",
        "Requirement": (
            "Special categories of personal data require additional "
            "safeguards.  Location data can constitute special-category "
            "data when it reveals sensitive patterns."
        ),
        "Mapped_Controls": "ID.AM-5, PR.DS-5",
    },
]


# ── ISO 27001 mapping table ────────────────────────────────────────────

_ISO_MAPPINGS: list[dict] = [
    {
        "Regulation": "ISO 27001",
        "Article": "A.8.1 — Responsibility for Assets / Asset Inventory",
        "Requirement": (
            "Assets associated with information and information-"
            "processing facilities shall be identified and an "
            "inventory maintained."
        ),
        "Mapped_Controls": "ID.AM-1, ID.AM-2",
    },
    {
        "Regulation": "ISO 27001",
        "Article": "A.8.2 — Classification of Information",
        "Requirement": (
            "Information shall be classified in terms of legal "
            "requirements, value, criticality, and sensitivity."
        ),
        "Mapped_Controls": "ID.AM-5",
    },
    {
        "Regulation": "ISO 27001",
        "Article": "A.9.2 — User Access Management",
        "Requirement": (
            "A formal user registration and de-registration process "
            "shall be implemented to enable assignment of access rights."
        ),
        "Mapped_Controls": "PR.AC-1",
    },
    {
        "Regulation": "ISO 27001",
        "Article": "A.9.4 — System and Application Access Control",
        "Requirement": (
            "Access to systems and applications shall be controlled "
            "by a secure log-on procedure."
        ),
        "Mapped_Controls": "PR.AC-4",
    },
    {
        "Regulation": "ISO 27001",
        "Article": "A.10.1 — Cryptographic Controls",
        "Requirement": (
            "A policy on the use of cryptographic controls for "
            "protection of information shall be developed."
        ),
        "Mapped_Controls": "PR.DS-1, PR.DS-2",
    },
    {
        "Regulation": "ISO 27001",
        "Article": "A.12.3 — Information Backup",
        "Requirement": (
            "Backup copies of information shall be taken and tested "
            "regularly in accordance with an agreed backup policy."
        ),
        "Mapped_Controls": "RC.RP-1",
    },
    {
        "Regulation": "ISO 27001",
        "Article": "A.12.4 — Logging and Monitoring",
        "Requirement": (
            "Event logs recording user activities, exceptions, and "
            "information security events shall be produced and kept."
        ),
        "Mapped_Controls": "DE.AE-3, DE.CM-1",
    },
    {
        "Regulation": "ISO 27001",
        "Article": "A.12.6 — Technical Vulnerability Management",
        "Requirement": (
            "Information about technical vulnerabilities shall be "
            "obtained in a timely fashion and remediated."
        ),
        "Mapped_Controls": "ID.RA-1",
    },
    {
        "Regulation": "ISO 27001",
        "Article": "A.16.1 — Management of Information Security Incidents",
        "Requirement": (
            "Responsibilities and procedures shall be established to "
            "ensure a quick, effective, and orderly response to "
            "information security incidents."
        ),
        "Mapped_Controls": "RS.CO-2, RS.AN-1, RS.MI-1, RC.IM-1, RC.CO-3",
    },
    {
        "Regulation": "ISO 27001",
        "Article": "A.18.1 — Compliance with Legal and Contractual Requirements",
        "Requirement": (
            "All relevant legislative, regulatory, and contractual "
            "requirements shall be explicitly identified and documented."
        ),
        "Mapped_Controls": "PR.DS-5, DE.DP-4",
    },
]


def _resolve_status(
    mapped_controls_str: str, checklist_df: pd.DataFrame
) -> tuple[str, str]:
    """
    Derive composite status and gap description from mapped NIST controls.

    Parameters:
        mapped_controls_str: Comma-separated Control_ID values.
        checklist_df: Evaluated NIST checklist DataFrame.

    Returns:
        Tuple of (status, gap_description).
    """
    ctrl_ids = [c.strip() for c in mapped_controls_str.split(",")]
    statuses = []
    gaps = []

    for cid in ctrl_ids:
        match = checklist_df.loc[checklist_df["Control_ID"] == cid]
        if match.empty:
            statuses.append("N/A")
        else:
            result = match.iloc[0]["Assessment_Result"]
            statuses.append(result)
            if result in ("FAIL", "PARTIAL"):
                evidence = match.iloc[0]["Evidence"]
                gaps.append(f"{cid}: {evidence[:120]}")

    if all(s == "PASS" for s in statuses):
        composite = "PASS"
    elif any(s == "FAIL" for s in statuses):
        composite = "FAIL"
    elif any(s == "PARTIAL" for s in statuses):
        composite = "PARTIAL"
    else:
        composite = "N/A"

    gap_desc = "; ".join(gaps) if gaps else "No gaps identified."
    return composite, gap_desc


def create_gdpr_mapping(checklist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map NIST CSF controls to GDPR articles and assess compliance.

    Parameters:
        checklist_df: Evaluated NIST checklist (from
                      :func:`nist_checklist.evaluate_checklist`).

    Returns:
        pandas.DataFrame with columns: Regulation, Article, Requirement,
        Mapped_Controls, Status, Gap_Description.
    """
    rows = []
    for mapping in _GDPR_MAPPINGS:
        status, gap = _resolve_status(mapping["Mapped_Controls"], checklist_df)
        rows.append({
            "Regulation": mapping["Regulation"],
            "Article": mapping["Article"],
            "Requirement": mapping["Requirement"],
            "Mapped_Controls": mapping["Mapped_Controls"],
            "Status": status,
            "Gap_Description": gap,
        })
    return pd.DataFrame(rows)


def create_iso27001_mapping(checklist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map NIST CSF controls to ISO 27001 Annex A controls.

    Parameters:
        checklist_df: Evaluated NIST checklist.

    Returns:
        pandas.DataFrame with the same structure as
        :func:`create_gdpr_mapping`.
    """
    rows = []
    for mapping in _ISO_MAPPINGS:
        status, gap = _resolve_status(mapping["Mapped_Controls"], checklist_df)
        rows.append({
            "Regulation": mapping["Regulation"],
            "Article": mapping["Article"],
            "Requirement": mapping["Requirement"],
            "Mapped_Controls": mapping["Mapped_Controls"],
            "Status": status,
            "Gap_Description": gap,
        })
    return pd.DataFrame(rows)


def create_full_compliance_matrix(checklist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a unified traceability matrix across GDPR, ISO 27001, and NIST.

    Combines GDPR and ISO mappings with a direct NIST listing into a
    single DataFrame for cross-framework gap analysis.

    Parameters:
        checklist_df: Evaluated NIST checklist.

    Returns:
        pandas.DataFrame with columns: Framework, Requirement,
        Control_ID, Status, Gap, Remediation.
    """
    gdpr_df = create_gdpr_mapping(checklist_df)
    iso_df = create_iso27001_mapping(checklist_df)

    rows = []

    # GDPR rows
    for _, r in gdpr_df.iterrows():
        for ctrl_id in r["Mapped_Controls"].split(","):
            ctrl_id = ctrl_id.strip()
            match = checklist_df.loc[checklist_df["Control_ID"] == ctrl_id]
            remediation = ""
            if not match.empty and match.iloc[0]["Assessment_Result"] in ("FAIL", "PARTIAL"):
                remediation = (
                    f"Address {ctrl_id} to achieve {r['Article']} compliance."
                )
            rows.append({
                "Framework": "GDPR",
                "Requirement": r["Article"],
                "Control_ID": ctrl_id,
                "Status": r["Status"],
                "Gap": r["Gap_Description"],
                "Remediation": remediation,
            })

    # ISO 27001 rows
    for _, r in iso_df.iterrows():
        for ctrl_id in r["Mapped_Controls"].split(","):
            ctrl_id = ctrl_id.strip()
            match = checklist_df.loc[checklist_df["Control_ID"] == ctrl_id]
            remediation = ""
            if not match.empty and match.iloc[0]["Assessment_Result"] in ("FAIL", "PARTIAL"):
                remediation = (
                    f"Address {ctrl_id} to achieve {r['Article']} compliance."
                )
            rows.append({
                "Framework": "ISO 27001",
                "Requirement": r["Article"],
                "Control_ID": ctrl_id,
                "Status": r["Status"],
                "Gap": r["Gap_Description"],
                "Remediation": remediation,
            })

    # NIST direct listing
    for _, r in checklist_df.iterrows():
        remediation = ""
        if r["Assessment_Result"] in ("FAIL", "PARTIAL"):
            remediation = (
                f"Remediate {r['Control_ID']} ({r['NIST_Function']}): "
                f"{r['Evidence'][:100]}"
            )
        rows.append({
            "Framework": "NIST CSF 2.0",
            "Requirement": f"{r['Control_ID']} — {r['NIST_Function']}",
            "Control_ID": r["Control_ID"],
            "Status": r["Assessment_Result"],
            "Gap": r["Evidence"] if r["Assessment_Result"] in ("FAIL", "PARTIAL") else "",
            "Remediation": remediation,
        })

    return pd.DataFrame(rows)


def calculate_compliance_scores(
    compliance_matrix: pd.DataFrame,
    checklist_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Calculate compliance percentage per framework.

    A control is considered compliant if its status is 'PASS'.  'PARTIAL'
    controls contribute 50% to the compliance score.  'N/A' controls are
    excluded from the denominator.

    Parameters:
        compliance_matrix: Output of :func:`create_full_compliance_matrix`.
        checklist_df: Optional evaluated NIST checklist with actual
                      Remediation_Priority values.  When provided, gaps
                      are grouped using the real priority rather than
                      inferring from status alone.

    Returns:
        dict with keys:
            - gdpr_compliance_pct (float)
            - iso_compliance_pct (float)
            - nist_compliance_pct (float)
            - overall_compliance_pct (float)
            - gaps_by_priority (dict): Mapping of priority level to
              list of Control_IDs with gaps.
    """
    scores = {}

    for framework in ["GDPR", "ISO 27001", "NIST CSF 2.0"]:
        subset = compliance_matrix.loc[
            (compliance_matrix["Framework"] == framework)
            & (compliance_matrix["Status"] != "N/A")
        ].drop_duplicates(subset=["Control_ID"])

        if subset.empty:
            scores[framework] = 0.0
            continue

        total = len(subset)
        pass_count = (subset["Status"] == "PASS").sum()
        partial_count = (subset["Status"] == "PARTIAL").sum()
        score = ((pass_count + 0.5 * partial_count) / total) * 100
        scores[framework] = round(score, 1)

    # Overall = average of the three
    framework_scores = list(scores.values())
    overall = round(sum(framework_scores) / len(framework_scores), 1) if framework_scores else 0.0

    # Gaps by priority — use actual Remediation_Priority from checklist if available
    gaps_by_priority: dict[str, list[str]] = {"High": [], "Medium": [], "Low": []}

    if checklist_df is not None and "Remediation_Priority" in checklist_df.columns:
        for _, r in checklist_df.iterrows():
            if r["Assessment_Result"] in ("FAIL", "PARTIAL"):
                priority = r["Remediation_Priority"]
                if priority in gaps_by_priority:
                    gaps_by_priority[priority].append(r["Control_ID"])
    else:
        # Fallback: infer from compliance matrix NIST rows
        nist_rows = compliance_matrix.loc[
            compliance_matrix["Framework"] == "NIST CSF 2.0"
        ]
        for _, r in nist_rows.iterrows():
            if r["Status"] == "FAIL":
                gaps_by_priority["High"].append(r["Control_ID"])
            elif r["Status"] == "PARTIAL":
                gaps_by_priority["Medium"].append(r["Control_ID"])

    # Deduplicate
    for k in gaps_by_priority:
        gaps_by_priority[k] = sorted(set(gaps_by_priority[k]))

    total_gaps = sum(len(v) for v in gaps_by_priority.values())
    gap_percentages = {
        p: round(100 * len(ids) / total_gaps, 1) if total_gaps else 0.0
        for p, ids in gaps_by_priority.items()
    }
    gap_counts = {p: len(ids) for p, ids in gaps_by_priority.items()}

    return {
        "gdpr_compliance_pct": scores.get("GDPR", 0.0),
        "iso_compliance_pct": scores.get("ISO 27001", 0.0),
        "nist_compliance_pct": scores.get("NIST CSF 2.0", 0.0),
        "overall_compliance_pct": overall,
        "gaps_by_priority": gaps_by_priority,
        "gap_counts": gap_counts,
        "gap_percentages": gap_percentages,
        "total_gaps": total_gaps,
    }


# ── Self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from module3_security.nist_checklist import evaluate_checklist

    print("=" * 70)
    print("Compliance Matrix — Self-Test")
    print("=" * 70)

    checklist = evaluate_checklist()
    matrix = create_full_compliance_matrix(checklist)
    scores = calculate_compliance_scores(matrix)

    print(f"\nGDPR compliance:     {scores['gdpr_compliance_pct']:.1f}%")
    print(f"ISO 27001 compliance: {scores['iso_compliance_pct']:.1f}%")
    print(f"NIST CSF compliance:  {scores['nist_compliance_pct']:.1f}%")
    print(f"Overall compliance:   {scores['overall_compliance_pct']:.1f}%")
    print(f"\nGaps by priority: {scores['gaps_by_priority']}")
    print(f"\nMatrix shape: {matrix.shape}")
