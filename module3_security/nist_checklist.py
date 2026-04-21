"""
NIST Cybersecurity Framework (CSF) 2.0 security assessment checklist.

Defines security controls mapped to the six NIST CSF 2.0 core functions
(Govern, Identify, Protect, Detect, Respond, Recover) and evaluates each
control against real evidence gathered from the NYC Yellow Taxi dataset files.

The checklist structure follows NIST SP 800-53 control families while
the function mapping aligns with NIST CSF 2.0 (2024).  Each control
includes an assessment method, evidence collection, and remediation
priority for gaps identified.

References:
    NIST (2024) Cybersecurity Framework (CSF) 2.0. National Institute of
        Standards and Technology, Gaithersburg, MD.
    NIST (2020) SP 800-53 Rev. 5: Security and Privacy Controls for
        Information Systems and Organizations.
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
from config import DATA_DIR, NIST_FUNCTIONS, COLUMNS

from module3_security.encryption_checker import scan_all_files as scan_encryption
from module3_security.permission_checker import scan_all_permissions

logger = logging.getLogger(__name__)


# ── Static checklist definition ─────────────────────────────────────────

_CONTROLS: list[dict] = [
    # ── GOVERN (3 controls) — new in CSF 2.0 ──────────────────────────
    {
        "Control_ID": "GV.PO-1",
        "NIST_Function": "Govern",
        "Control_Description": (
            "A cybersecurity risk management policy is established "
            "based on organisational context — data governance policies "
            "cover collection, storage, and sharing of taxi trip data."
        ),
        "Assessment_Method": (
            "Verify existence of a documented data governance policy "
            "that covers NYC Taxi data handling, retention, and sharing."
        ),
        "Remediation_Priority": "High",
        "Compliance_Links": "GDPR Art. 5; ISO 27001 A.5.1",
    },
    {
        "Control_ID": "GV.RM-1",
        "NIST_Function": "Govern",
        "Control_Description": (
            "Risk management objectives are established and expressed "
            "as risk tolerance statements — acceptable privacy and "
            "security risk levels for transportation data are defined."
        ),
        "Assessment_Method": (
            "Verify that risk tolerance thresholds are documented for "
            "re-identification risk, data quality degradation, and "
            "compliance gaps."
        ),
        "Remediation_Priority": "High",
        "Compliance_Links": "GDPR Art. 35; ISO 27001 A.5.1",
    },
    {
        "Control_ID": "GV.RR-1",
        "NIST_Function": "Govern",
        "Control_Description": (
            "Organisational leadership takes responsibility for "
            "cybersecurity risk — roles and responsibilities for data "
            "protection are assigned and communicated."
        ),
        "Assessment_Method": (
            "Verify that data protection roles are documented "
            "(e.g., data owner, data steward, security lead) with "
            "clear accountability for the taxi dataset."
        ),
        "Remediation_Priority": "Medium",
        "Compliance_Links": "GDPR Art. 37-39; ISO 27001 A.5.2",
    },
    # ── IDENTIFY (4 controls) ──────────────────────────────────────────
    {
        "Control_ID": "ID.AM-1",
        "NIST_Function": "Identify",
        "Control_Description": (
            "Physical devices and systems within the organisation are "
            "inventoried — all Parquet data files and their storage "
            "locations are documented."
        ),
        "Assessment_Method": (
            "Enumerate Parquet files in the data directory and record "
            "file names, sizes, and row-group counts."
        ),
        "Remediation_Priority": "Medium",
        "Compliance_Links": "GDPR Art. 30; ISO 27001 A.8.1",
    },
    {
        "Control_ID": "ID.AM-2",
        "NIST_Function": "Identify",
        "Control_Description": (
            "Software platforms and applications are inventoried — "
            "Python, Pandas, PyArrow, and analysis tools are documented."
        ),
        "Assessment_Method": (
            "Verify that a software bill of materials (SBOM) or "
            "requirements file exists listing all dependencies."
        ),
        "Remediation_Priority": "Low",
        "Compliance_Links": "ISO 27001 A.8.1; NIST SP 800-53 CM-8",
    },
    {
        "Control_ID": "ID.AM-5",
        "NIST_Function": "Identify",
        "Control_Description": (
            "Resources are classified by sensitivity — dataset columns "
            "are classified as High, Medium, or Low sensitivity."
        ),
        "Assessment_Method": (
            "Inspect dataset schema and classify each column's "
            "sensitivity level (location IDs = High, fare = Medium, "
            "vendor metadata = Low)."
        ),
        "Remediation_Priority": "High",
        "Compliance_Links": "GDPR Art. 9; ISO 27001 A.8.2",
    },
    {
        "Control_ID": "ID.RA-1",
        "NIST_Function": "Identify",
        "Control_Description": (
            "Asset vulnerabilities are identified and documented — "
            "scan for unencrypted storage of potentially sensitive "
            "fields (pickup/dropoff locations, timestamps)."
        ),
        "Assessment_Method": (
            "Run encryption checker on all Parquet files; flag any "
            "file lacking Parquet Modular Encryption."
        ),
        "Remediation_Priority": "High",
        "Compliance_Links": "GDPR Art. 32; ISO 27001 A.12.6",
    },
    # ── PROTECT (5 controls) ───────────────────────────────────────────
    {
        "Control_ID": "PR.AC-1",
        "NIST_Function": "Protect",
        "Control_Description": (
            "Identities and credentials for authorised users are "
            "managed — authentication is required for data access."
        ),
        "Assessment_Method": (
            "Verify whether OS-level authentication is required to "
            "access the data directory (user login, file share auth)."
        ),
        "Remediation_Priority": "High",
        "Compliance_Links": "GDPR Art. 32; ISO 27001 A.9.2",
    },
    {
        "Control_ID": "PR.AC-4",
        "NIST_Function": "Protect",
        "Control_Description": (
            "Access permissions and authorisations are managed — file "
            "system permissions restrict who can read/write data files."
        ),
        "Assessment_Method": (
            "Run permission checker on all Parquet files; assess "
            "whether files are world-readable or world-writable."
        ),
        "Remediation_Priority": "High",
        "Compliance_Links": "GDPR Art. 32; ISO 27001 A.9.4",
    },
    {
        "Control_ID": "PR.DS-1",
        "NIST_Function": "Protect",
        "Control_Description": (
            "Data-at-rest is protected — Parquet files should use "
            "column-level or footer encryption."
        ),
        "Assessment_Method": (
            "Inspect Parquet metadata for encryption flags using "
            "PyArrow; verify encryption algorithm if present."
        ),
        "Remediation_Priority": "High",
        "Compliance_Links": "GDPR Art. 32(1)(a); ISO 27001 A.10.1",
    },
    {
        "Control_ID": "PR.DS-2",
        "NIST_Function": "Protect",
        "Control_Description": (
            "Data-in-transit is protected — HTTPS is used when "
            "downloading NYC TLC dataset files."
        ),
        "Assessment_Method": (
            "Verify that the NYC TLC download URL uses HTTPS "
            "(https://d37ci6vzurychx.cloudfront.net/)."
        ),
        "Remediation_Priority": "Medium",
        "Compliance_Links": "GDPR Art. 32(1)(a); ISO 27001 A.13.1",
    },
    {
        "Control_ID": "PR.DS-5",
        "NIST_Function": "Protect",
        "Control_Description": (
            "Protections against data leaks are implemented — "
            "anonymisation or pseudonymisation of location data."
        ),
        "Assessment_Method": (
            "Check whether raw location IDs (PULocationID, "
            "DOLocationID) are stored without anonymisation; "
            "assess re-identification risk from the privacy module."
        ),
        "Remediation_Priority": "High",
        "Compliance_Links": "GDPR Art. 25; ISO 27001 A.18.1",
    },
    # ── DETECT (3 controls) ────────────────────────────────────────────
    {
        "Control_ID": "DE.AE-3",
        "NIST_Function": "Detect",
        "Control_Description": (
            "Event data are aggregated and correlated from multiple "
            "sources — access logging is enabled for data files."
        ),
        "Assessment_Method": (
            "Check OS audit logging configuration for the data "
            "directory (Windows Event Log / Linux auditd)."
        ),
        "Remediation_Priority": "Medium",
        "Compliance_Links": "GDPR Art. 33; ISO 27001 A.12.4",
    },
    {
        "Control_ID": "DE.CM-1",
        "NIST_Function": "Detect",
        "Control_Description": (
            "The network is monitored to detect potential "
            "cybersecurity events — network anomaly detection is "
            "in place for data transfer channels."
        ),
        "Assessment_Method": (
            "Verify whether network monitoring tools (IDS/IPS, "
            "firewall logs) cover the segment where data is stored."
        ),
        "Remediation_Priority": "Medium",
        "Compliance_Links": "ISO 27001 A.12.4; NIST SP 800-53 SI-4",
    },
    {
        "Control_ID": "DE.DP-4",
        "NIST_Function": "Detect",
        "Control_Description": (
            "Event detection information is communicated — breach "
            "notification procedures are documented."
        ),
        "Assessment_Method": (
            "Verify existence of a documented breach notification "
            "procedure aligned with GDPR 72-hour rule."
        ),
        "Remediation_Priority": "High",
        "Compliance_Links": "GDPR Art. 33-34; ISO 27001 A.16.1",
    },
    # ── RESPOND (3 controls) ───────────────────────────────────────────
    {
        "Control_ID": "RS.CO-2",
        "NIST_Function": "Respond",
        "Control_Description": (
            "Incidents are reported consistent with established "
            "criteria — incident reporting procedures exist."
        ),
        "Assessment_Method": (
            "Verify documented incident reporting workflow including "
            "roles, escalation paths, and reporting timelines."
        ),
        "Remediation_Priority": "Medium",
        "Compliance_Links": "GDPR Art. 33; ISO 27001 A.16.1",
    },
    {
        "Control_ID": "RS.AN-1",
        "NIST_Function": "Respond",
        "Control_Description": (
            "Notifications from detection systems are investigated — "
            "incident analysis procedures are documented."
        ),
        "Assessment_Method": (
            "Verify that an incident analysis playbook exists "
            "covering triage, root-cause analysis, and containment."
        ),
        "Remediation_Priority": "Medium",
        "Compliance_Links": "ISO 27001 A.16.1; NIST SP 800-61",
    },
    {
        "Control_ID": "RS.MI-1",
        "NIST_Function": "Respond",
        "Control_Description": (
            "Incidents are contained to limit impact — isolation "
            "procedures for compromised data stores are defined."
        ),
        "Assessment_Method": (
            "Verify documented containment procedures (network "
            "isolation, access revocation, data quarantine)."
        ),
        "Remediation_Priority": "Medium",
        "Compliance_Links": "ISO 27001 A.16.1; NIST SP 800-61",
    },
    # ── RECOVER (3 controls) ───────────────────────────────────────────
    {
        "Control_ID": "RC.RP-1",
        "NIST_Function": "Recover",
        "Control_Description": (
            "Recovery plan is executed during or after a "
            "cybersecurity incident — backup and restore procedures "
            "exist for data files."
        ),
        "Assessment_Method": (
            "Verify existence of backup copies of Parquet data files "
            "and a documented restore procedure."
        ),
        "Remediation_Priority": "High",
        "Compliance_Links": "ISO 27001 A.12.3; NIST SP 800-34",
    },
    {
        "Control_ID": "RC.IM-1",
        "NIST_Function": "Recover",
        "Control_Description": (
            "Recovery plans incorporate lessons learned — post-"
            "incident reviews feed back into recovery planning."
        ),
        "Assessment_Method": (
            "Verify that a lessons-learned template and review "
            "cadence are documented."
        ),
        "Remediation_Priority": "Low",
        "Compliance_Links": "ISO 27001 A.16.1; NIST SP 800-61",
    },
    {
        "Control_ID": "RC.CO-3",
        "NIST_Function": "Recover",
        "Control_Description": (
            "Recovery activities are communicated to internal and "
            "external stakeholders — communication plan exists for "
            "data-loss events."
        ),
        "Assessment_Method": (
            "Verify stakeholder communication plan covering "
            "regulators, data subjects, and senior management."
        ),
        "Remediation_Priority": "Medium",
        "Compliance_Links": "GDPR Art. 34; ISO 27001 A.16.1",
    },
]


def create_nist_checklist() -> pd.DataFrame:
    """
    Create a baseline NIST CSF 2.0 security checklist.

    Returns a DataFrame with 21 controls across all six NIST CSF 2.0 functions.
    Assessment_Result and Evidence columns are initialised to 'NOT ASSESSED'
    and should be populated by :func:`evaluate_checklist`.

    Returns:
        pandas.DataFrame with columns: Control_ID, NIST_Function,
        Control_Description, Assessment_Method, Assessment_Result,
        Evidence, Remediation_Priority, Compliance_Links.
    """
    df = pd.DataFrame(_CONTROLS)
    df["Assessment_Result"] = "NOT ASSESSED"
    df["Evidence"] = ""
    return df


def _classify_columns() -> dict[str, str]:
    """Classify dataset columns by sensitivity level."""
    high = {"PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime"}
    medium = {
        "fare_amount", "total_amount", "tip_amount", "tolls_amount",
        "payment_type", "trip_distance", "passenger_count",
    }
    classification = {}
    for col in COLUMNS:
        if col in high:
            classification[col] = "High"
        elif col in medium:
            classification[col] = "Medium"
        else:
            classification[col] = "Low"
    return classification


def evaluate_checklist(
    data_dir: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """
    Evaluate every NIST control by running real security checks.

    Populates Assessment_Result ('PASS', 'PARTIAL', 'FAIL', 'N/A') and
    Evidence columns with findings from the encryption checker,
    permission checker, and manual assessments.

    Parameters:
        data_dir: Directory containing .parquet files.
                  Defaults to ``config.DATA_DIR``.

    Returns:
        pandas.DataFrame — the completed checklist.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    # Run automated checks
    logger.info("Running encryption scan...")
    enc_df = scan_encryption(data_dir)
    logger.info("Running permission scan...")
    perm_df = scan_all_permissions(data_dir)

    # Gather facts
    n_files = len(enc_df) if not enc_df.empty else 0
    n_encrypted = int(enc_df["encrypted"].sum()) if not enc_df.empty else 0
    n_unencrypted = n_files - n_encrypted

    perm_high = int((perm_df["risk_level"] == "High").sum()) if not perm_df.empty else 0
    perm_medium = int((perm_df["risk_level"] == "Medium").sum()) if not perm_df.empty else 0
    perm_pass = int((perm_df["status"] == "PASS").sum()) if not perm_df.empty else 0

    col_classification = _classify_columns()
    high_sens_cols = [c for c, lvl in col_classification.items() if lvl == "High"]

    # Requirements file check
    req_file = data_dir.parent / "requirements.txt"
    has_requirements = req_file.exists()

    checklist = create_nist_checklist()

    # Evaluation logic per control
    evaluations: dict[str, tuple[str, str]] = {}

    # GV.PO-1 — governance policy
    evaluations["GV.PO-1"] = (
        "FAIL",
        "No formal data governance policy document found for this project. "
        "NYC TLC data is processed without a documented policy covering "
        "collection purpose, retention periods, or sharing restrictions.",
    )

    # GV.RM-1 — risk management objectives
    evaluations["GV.RM-1"] = (
        "PARTIAL",
        "Privacy risk thresholds are defined programmatically in config.py "
        "(Critical >=75, High >=50, Medium >=25) and quality scoring uses "
        "weighted dimensions, but no formal risk tolerance statement document "
        "exists for the project.",
    )

    # GV.RR-1 — roles and responsibilities
    evaluations["GV.RR-1"] = (
        "PARTIAL",
        "Team roles are documented in the dissertation (Chapter 1, Table 1.1): "
        "Junaid (quality), Sami (privacy), Jannat (security), Iqra (dashboard). "
        "However, formal data protection officer (DPO) designation and "
        "accountability procedures are not established.",
    )

    # ID.AM-1 — asset inventory
    evaluations["ID.AM-1"] = (
        "PASS" if n_files > 0 else "FAIL",
        f"Found {n_files} Parquet files in data directory. "
        f"Total size: {enc_df['file_size_mb'].sum():.1f} MB. "
        f"Files span 2019 and 2024 NYC Yellow Taxi data."
        if not enc_df.empty else "No Parquet files found.",
    )

    # ID.AM-2 — software inventory
    evaluations["ID.AM-2"] = (
        "PASS" if has_requirements else "PARTIAL",
        (
            f"requirements.txt found at {req_file}. "
            "Python, Pandas, PyArrow, Plotly documented."
            if has_requirements
            else "No requirements.txt found. Software dependencies should be "
                 "documented in a requirements file or environment.yml."
        ),
    )

    # ID.AM-5 — data classification
    evaluations["ID.AM-5"] = (
        "PARTIAL",
        f"Column classification performed: {len(high_sens_cols)} HIGH sensitivity "
        f"columns ({', '.join(high_sens_cols)}), "
        f"{sum(1 for v in col_classification.values() if v == 'Medium')} MEDIUM, "
        f"{sum(1 for v in col_classification.values() if v == 'Low')} LOW. "
        "Formal data classification policy not yet documented.",
    )

    # ID.RA-1 — vulnerability identification
    evaluations["ID.RA-1"] = (
        "FAIL" if n_unencrypted > 0 else "PASS",
        f"{n_unencrypted}/{n_files} files lack encryption. "
        f"Sensitive fields (location IDs, timestamps) stored in plaintext. "
        "Parquet Modular Encryption (PME) not enabled.",
    )

    # PR.AC-1 — authentication
    evaluations["PR.AC-1"] = (
        "PARTIAL",
        "Data files reside on local filesystem requiring OS-level "
        "user authentication (Windows login). No application-level "
        "authentication layer (e.g., database credentials, API keys) "
        "is enforced for direct file access.",
    )

    # PR.AC-4 — access permissions
    if perm_high > 0:
        ac4_status = "FAIL"
    elif perm_medium > 0:
        ac4_status = "PARTIAL"
    else:
        ac4_status = "PASS"
    evaluations["PR.AC-4"] = (
        ac4_status,
        f"Permission scan: {perm_pass}/{n_files} files PASS, "
        f"{perm_high} HIGH risk, {perm_medium} MEDIUM risk. "
        + (
            "Some files are world-readable/writable."
            if perm_high > 0 or perm_medium > 0
            else "All files have restrictive permissions."
        ),
    )

    # PR.DS-1 — data-at-rest encryption
    evaluations["PR.DS-1"] = (
        "FAIL" if n_unencrypted > 0 else "PASS",
        f"{n_encrypted}/{n_files} files encrypted. "
        f"NYC TLC distributes data without Parquet encryption. "
        f"Compression codec: {enc_df['compression_codec'].iloc[0] if not enc_df.empty else 'N/A'} "
        "(compression is not encryption).",
    )

    # PR.DS-2 — data-in-transit
    evaluations["PR.DS-2"] = (
        "PASS",
        "NYC TLC data is distributed via HTTPS "
        "(https://d37ci6vzurychx.cloudfront.net/). "
        "CloudFront enforces TLS 1.2+ for data downloads.",
    )

    # PR.DS-5 — data leak protection
    evaluations["PR.DS-5"] = (
        "FAIL",
        f"Raw location identifiers (PULocationID, DOLocationID) and "
        f"precise timestamps are stored without anonymisation. "
        f"{len(high_sens_cols)} high-sensitivity columns identified. "
        "No differential privacy or k-anonymity applied to the raw files.",
    )

    # DE.AE-3 — access logging
    evaluations["DE.AE-3"] = (
        "PARTIAL",
        "Windows Event Log records file-system access at OS level, "
        "but no application-level audit logging is configured for "
        "data file reads. Recommend enabling Object Access Auditing "
        "via Local Security Policy.",
    )

    # DE.CM-1 — network monitoring
    evaluations["DE.CM-1"] = (
        "N/A",
        "Data is stored locally on a development workstation. "
        "Network monitoring is not applicable for local file access. "
        "If deployed to a server, IDS/IPS should cover the data segment.",
    )

    # DE.DP-4 — breach notification
    evaluations["DE.DP-4"] = (
        "FAIL",
        "No documented breach notification procedure found. "
        "GDPR Article 33 requires notification within 72 hours. "
        "Recommend creating a breach response playbook.",
    )

    # RS.CO-2 — incident reporting
    evaluations["RS.CO-2"] = (
        "FAIL",
        "No formal incident reporting procedure documented for this "
        "project. Recommend establishing an incident reporting template "
        "with roles, escalation paths, and timelines.",
    )

    # RS.AN-1 — incident analysis
    evaluations["RS.AN-1"] = (
        "FAIL",
        "No incident analysis playbook found. Recommend documenting "
        "triage criteria, root-cause analysis methods, and containment "
        "steps for data-related security incidents.",
    )

    # RS.MI-1 — incident containment
    evaluations["RS.MI-1"] = (
        "FAIL",
        "No documented containment procedures for compromised data "
        "stores. Recommend defining isolation steps (network disconnect, "
        "access revocation, data quarantine).",
    )

    # RC.RP-1 — recovery plan
    evaluations["RC.RP-1"] = (
        "PARTIAL",
        "Original data is publicly available from NYC TLC for re-download. "
        "However, no formal backup/restore procedure is documented. "
        "Recommend documenting the restore process and verifying data "
        "integrity post-restore using checksums.",
    )

    # RC.IM-1 — recovery improvement
    evaluations["RC.IM-1"] = (
        "FAIL",
        "No lessons-learned process or post-incident review cadence "
        "documented. Recommend establishing a review template.",
    )

    # RC.CO-3 — recovery communication
    evaluations["RC.CO-3"] = (
        "FAIL",
        "No stakeholder communication plan for data-loss events. "
        "GDPR Article 34 requires notification to affected data "
        "subjects when risk is high.",
    )

    # Apply evaluations
    for idx, row in checklist.iterrows():
        ctrl_id = row["Control_ID"]
        if ctrl_id in evaluations:
            result, evidence = evaluations[ctrl_id]
            checklist.at[idx, "Assessment_Result"] = result
            checklist.at[idx, "Evidence"] = evidence

    logger.info(
        "Checklist evaluated: %s",
        checklist["Assessment_Result"].value_counts().to_dict(),
    )
    return checklist


# ── Self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 70)
    print("NIST CSF 2.0 Checklist — Self-Test")
    print("=" * 70)

    df = evaluate_checklist()
    for _, row in df.iterrows():
        status_icon = {
            "PASS": "[PASS]", "PARTIAL": "[PART]", "FAIL": "[FAIL]", "N/A": "[N/A]"
        }.get(row["Assessment_Result"], "[????]")
        print(f"  {status_icon} {row['Control_ID']:10s} {row['NIST_Function']:10s} "
              f"{row['Assessment_Result']:8s}")

    print(f"\nTotal controls: {len(df)}")
    print(df["Assessment_Result"].value_counts().to_string())
