"""
Test suite for Module 3: Security Assessment.

Validates encryption checking, permission inspection, NIST checklist
completeness, compliance matrix mappings, score calculation, and
end-to-end orchestration against real NYC Yellow Taxi Parquet files.

Uses the 2019-01 file as a concrete test fixture to ensure assertions
are grounded in actual file metadata rather than mocked data.

Author: Jannat Rafique (B01798960)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_DIR, NIST_FUNCTIONS

from module3_security.encryption_checker import (
    check_parquet_encryption,
    scan_all_files,
)
from module3_security.permission_checker import (
    check_file_permissions,
    scan_all_permissions,
)
from module3_security.nist_checklist import (
    create_nist_checklist,
    evaluate_checklist,
)
from module3_security.compliance_matrix import (
    create_gdpr_mapping,
    create_iso27001_mapping,
    create_full_compliance_matrix,
    calculate_compliance_scores,
)
from module3_security.security_assessor import get_security_checklist


# ── Fixtures ────────────────────────────────────────────────────────────

SAMPLE_FILE = DATA_DIR / "yellow_tripdata_2019-01.parquet"


@pytest.fixture(scope="module")
def sample_file() -> Path:
    """Path to the January 2019 Parquet file."""
    if not SAMPLE_FILE.exists():
        pytest.skip(f"Test data not found: {SAMPLE_FILE}")
    return SAMPLE_FILE


@pytest.fixture(scope="module")
def evaluated_checklist() -> pd.DataFrame:
    """Pre-evaluated NIST checklist (expensive, run once per module)."""
    return evaluate_checklist(DATA_DIR)


@pytest.fixture(scope="module")
def full_results() -> dict:
    """Full security assessment results."""
    return get_security_checklist(data_dir=DATA_DIR)


# ── Test: encryption_checker ────────────────────────────────────────────

class TestEncryptionChecker:
    """Tests for encryption_checker module."""

    def test_check_single_file_structure(self, sample_file: Path) -> None:
        """check_parquet_encryption returns all expected keys."""
        result = check_parquet_encryption(sample_file)
        expected_keys = {
            "file_name", "file_size_mb", "encrypted",
            "encryption_algorithm", "num_row_groups", "num_columns",
            "compression_codec", "status",
        }
        assert set(result.keys()) == expected_keys

    def test_nyc_data_is_unencrypted(self, sample_file: Path) -> None:
        """NYC TLC data files are known to be unencrypted."""
        result = check_parquet_encryption(sample_file)
        assert result["encrypted"] is False
        assert result["encryption_algorithm"] is None
        assert result["status"] == "FAIL"

    def test_file_metadata_realistic(self, sample_file: Path) -> None:
        """Metadata values are within realistic bounds."""
        result = check_parquet_encryption(sample_file)
        assert result["file_size_mb"] > 0
        assert result["num_row_groups"] >= 1
        assert result["num_columns"] >= 17  # NYC TLC schema has 17-19 cols
        assert result["compression_codec"] != "UNKNOWN"

    def test_scan_all_returns_dataframe(self) -> None:
        """scan_all_files returns a DataFrame with expected shape."""
        df = scan_all_files(DATA_DIR)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "encrypted" in df.columns
        assert "file_name" in df.columns

    def test_file_not_found_raises(self) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            check_parquet_encryption("/nonexistent/file.parquet")


# ── Test: permission_checker ────────────────────────────────────────────

class TestPermissionChecker:
    """Tests for permission_checker module."""

    def test_check_permissions_structure(self, sample_file: Path) -> None:
        """check_file_permissions returns all expected keys."""
        result = check_file_permissions(sample_file)
        expected_keys = {
            "file_name", "permissions_str", "is_world_readable",
            "is_world_writable", "owner", "risk_level", "status",
            "recommendation",
        }
        assert set(result.keys()) == expected_keys

    def test_risk_level_valid(self, sample_file: Path) -> None:
        """Risk level is one of the defined categories."""
        result = check_file_permissions(sample_file)
        assert result["risk_level"] in ("Low", "Medium", "High")

    def test_status_valid(self, sample_file: Path) -> None:
        """Status is one of PASS, PARTIAL, FAIL."""
        result = check_file_permissions(sample_file)
        assert result["status"] in ("PASS", "PARTIAL", "FAIL")

    def test_scan_all_permissions_returns_dataframe(self) -> None:
        """scan_all_permissions returns DataFrame with correct length."""
        df = scan_all_permissions(DATA_DIR)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "risk_level" in df.columns


# ── Test: nist_checklist ────────────────────────────────────────────────

class TestNISTChecklist:
    """Tests for nist_checklist module."""

    def test_all_six_functions_covered(self) -> None:
        """Baseline checklist covers all 6 NIST CSF 2.0 functions."""
        df = create_nist_checklist()
        functions_present = set(df["NIST_Function"].unique())
        for func in NIST_FUNCTIONS:
            assert func in functions_present, f"Missing NIST function: {func}"

    def test_minimum_control_count(self) -> None:
        """Checklist has at least 21 controls (18 original + 3 Govern)."""
        df = create_nist_checklist()
        assert len(df) >= 21

    def test_evaluate_produces_real_evidence(
        self, evaluated_checklist: pd.DataFrame
    ) -> None:
        """evaluate_checklist fills in Evidence for all controls."""
        for _, row in evaluated_checklist.iterrows():
            assert row["Assessment_Result"] in (
                "PASS", "PARTIAL", "FAIL", "N/A"
            ), f"Unexpected result for {row['Control_ID']}: {row['Assessment_Result']}"
            assert len(row["Evidence"]) > 0, (
                f"No evidence for {row['Control_ID']}"
            )

    def test_encryption_control_fails(
        self, evaluated_checklist: pd.DataFrame
    ) -> None:
        """PR.DS-1 (data-at-rest encryption) should FAIL for NYC TLC data."""
        pr_ds1 = evaluated_checklist.loc[
            evaluated_checklist["Control_ID"] == "PR.DS-1"
        ]
        assert len(pr_ds1) == 1
        assert pr_ds1.iloc[0]["Assessment_Result"] == "FAIL"


# ── Test: compliance_matrix ─────────────────────────────────────────────

class TestComplianceMatrix:
    """Tests for compliance_matrix module."""

    def test_gdpr_mapping_has_articles(
        self, evaluated_checklist: pd.DataFrame
    ) -> None:
        """GDPR mapping covers key articles (25, 32, 33)."""
        gdpr = create_gdpr_mapping(evaluated_checklist)
        articles_text = " ".join(gdpr["Article"].tolist())
        assert "Art. 25" in articles_text
        assert "Art. 32" in articles_text
        assert "Art. 33" in articles_text

    def test_iso_mapping_has_annexes(
        self, evaluated_checklist: pd.DataFrame
    ) -> None:
        """ISO 27001 mapping covers Annex A sections (A.9, A.10, A.12)."""
        iso = create_iso27001_mapping(evaluated_checklist)
        articles_text = " ".join(iso["Article"].tolist())
        assert "A.9" in articles_text
        assert "A.10" in articles_text
        assert "A.12" in articles_text

    def test_full_matrix_has_all_frameworks(
        self, evaluated_checklist: pd.DataFrame
    ) -> None:
        """Unified matrix contains GDPR, ISO 27001, and NIST entries."""
        matrix = create_full_compliance_matrix(evaluated_checklist)
        frameworks = set(matrix["Framework"].unique())
        assert "GDPR" in frameworks
        assert "ISO 27001" in frameworks
        assert "NIST CSF 2.0" in frameworks

    def test_compliance_scores_structure(
        self, evaluated_checklist: pd.DataFrame
    ) -> None:
        """Compliance scores dict has all expected keys."""
        matrix = create_full_compliance_matrix(evaluated_checklist)
        scores = calculate_compliance_scores(matrix)

        assert "gdpr_compliance_pct" in scores
        assert "iso_compliance_pct" in scores
        assert "nist_compliance_pct" in scores
        assert "overall_compliance_pct" in scores
        assert "gaps_by_priority" in scores

        # Scores should be between 0 and 100
        for key in ["gdpr_compliance_pct", "iso_compliance_pct",
                     "nist_compliance_pct", "overall_compliance_pct"]:
            assert 0 <= scores[key] <= 100, f"{key} out of range: {scores[key]}"


# ── Test: security_assessor (end-to-end) ────────────────────────────────

class TestSecurityAssessor:
    """Tests for the main orchestrator."""

    def test_result_keys(self, full_results: dict) -> None:
        """get_security_checklist returns all expected keys."""
        expected = {
            "overall_compliance", "encryption_results", "permission_results",
            "checklist_results", "compliance_matrix", "compliance_scores",
            "gap_summary", "gap_counts", "gap_percentages", "total_gaps",
            "remediation_priorities", "summary_text",
        }
        assert set(full_results.keys()) == expected

    def test_overall_compliance_is_numeric(self, full_results: dict) -> None:
        """Overall compliance is a float between 0 and 100."""
        score = full_results["overall_compliance"]
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_summary_text_nonempty(self, full_results: dict) -> None:
        """Summary text is generated and contains key phrases."""
        text = full_results["summary_text"]
        assert len(text) > 100
        assert "SECURITY ASSESSMENT" in text
        assert "GDPR" in text

    def test_remediation_priorities_populated(self, full_results: dict) -> None:
        """Remediation table is non-empty (NYC TLC data has known gaps)."""
        df = full_results["remediation_priorities"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Priority" in df.columns
        assert "Control_ID" in df.columns
