"""
NYC Taxi Data Governance Dashboard — Home / Executive Overview.

Module 4 of the MSc dissertation: "Data Security and Privacy Assessment
Framework for Big Data Transportation Systems."  This entry-point page
presents a consolidated governance scorecard integrating the three
analysis modules (Quality, Privacy, Security).

Design rationale follows Shneiderman's (1996) visual information-seeking
mantra — overview first, then details on demand via sub-pages — and
Few's (2006) principles for effective dashboard layout.

Author : Iqra Aziz (B01802319) — Module 4 (Dashboard)
Group  : Junaid Babar (B01802551) — Module 1 Quality Profiling
         Sami Ullah   (B01750598) — Module 2 Privacy Assessment
         Jannat Rafique (B01798960) — Module 3 Security Compliance
         Iqra Aziz    (B01802319) — Module 4 Dashboard Integration
Programme: MSc IT (Data Analysis), University of the West of Scotland

References:
    Few, S. (2006) Information Dashboard Design. Analytics Press.
    Nielsen, J. (1994) Usability Engineering. Morgan Kaufmann.
    Shneiderman, B. (1996) 'The eyes have it', Proc. IEEE Symp. Visual
        Languages, pp. 336-343.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# ─── Path setup ─────────────────────────────────────────────────────────
_DASHBOARD_DIR = Path(__file__).resolve().parent
_PHASE2_ROOT = _DASHBOARD_DIR.parent
if str(_PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE2_ROOT))

from dashboard.utils.styling import (   # noqa: E402
    apply_custom_css,
    QUALITY_GREEN,
    PRIVACY_BLUE,
    SECURITY_RED,
    WARNING_ORANGE,
    NEUTRAL_GREY,
    GAUGE_STEPS,
    PLOTLY_TEMPLATE,
    CHART_FONT,
    CHART_MARGIN,
    score_colour,
    risk_colour,
)
from dashboard.utils.data_loader import cached_load_month   # noqa: E402
from dashboard.utils.export_reports import generate_summary_pdf  # noqa: E402

from config import AVAILABLE_YEARS, MONTHS, DATA_DIR   # noqa: E402

# ─── Page configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Governance Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_css()

# ─── Module imports with graceful fallback ──────────────────────────────
_quality_available = False
_privacy_available = False
_security_available = False

try:
    from module1_quality.quality_profiler import get_quality_metrics
    _quality_available = True
except ImportError:
    pass

try:
    from module2_privacy.privacy_assessor import get_privacy_assessment
    _privacy_available = True
except ImportError:
    pass

try:
    from module3_security.security_assessor import get_security_checklist
    _security_available = True
except ImportError:
    pass


# ─── Sidebar: global filters ───────────────────────────────────────────

def _render_sidebar():
    """Build sidebar controls shared across all pages."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/"
            "University_of_the_West_of_Scotland_logo.svg/220px-"
            "University_of_the_West_of_Scotland_logo.svg.png",
            width=180,
        )
        st.markdown("### Data Selection")

        year = st.selectbox(
            "Year",
            options=AVAILABLE_YEARS,
            index=0,
            key="global_year",
            help="Select data year (2019 or 2024).",
        )
        month = st.selectbox(
            "Month",
            options=list(MONTHS),
            format_func=lambda m: f"{m:02d} — {_month_name(m)}",
            index=0,
            key="global_month",
            help="Select data month (1-12).",
        )

        st.markdown("---")
        run = st.button(
            "Run Full Assessment",
            type="primary",
            use_container_width=True,
            help="Run all three assessment modules on the selected data.",
        )

        # Module availability indicators
        st.markdown("### Module Status")
        _status_indicator("Quality Profiler", _quality_available)
        _status_indicator("Privacy Assessor", _privacy_available)
        _status_indicator("Security Assessor", _security_available)

        st.markdown("---")
        st.markdown("### Group Members")
        st.caption(
            "**MSc IT (Data Analysis)**  \n"
            "University of the West of Scotland\n\n"
            "**Junaid Babar** — B01802551  \n"
            "_Module 1 · Data Quality Profiling_\n\n"
            "**Sami Ullah** — B01750598  \n"
            "_Module 2 · Privacy Risk Detection_\n\n"
            "**Jannat Rafique** — B01798960  \n"
            "_Module 3 · Security Compliance_\n\n"
            "**Iqra Aziz** — B01802319  \n"
            "_Module 4 · Governance Dashboard_"
        )

    return year, month, run


def _status_indicator(label: str, available: bool):
    """Show a green/red status dot for a module."""
    dot = "🟢" if available else "🔴"
    st.markdown(f"{dot} {label}")


def _month_name(m: int) -> str:
    """Return abbreviated month name."""
    import calendar
    return calendar.month_abbr[m]


# ─── Helper: compute governance score ──────────────────────────────────

def _compute_governance_score(
    quality_res: dict | None,
    privacy_res: dict | None,
    security_res: dict | None,
) -> float:
    """
    Weighted composite governance score (0-100).

    Weights: Quality 40%, Security 35%, Privacy 25%.
    Privacy risk is inverted (high risk = low governance score).
    """
    scores = []
    weights = []

    if quality_res and isinstance(quality_res.get("overall_score"), (int, float)):
        scores.append(quality_res["overall_score"])
        weights.append(0.40)

    if security_res and isinstance(security_res.get("overall_compliance"), (int, float)):
        scores.append(security_res["overall_compliance"])
        weights.append(0.35)

    if privacy_res and isinstance(privacy_res.get("overall_risk_score"), (int, float)):
        # Invert: 0 risk = 100 governance, 100 risk = 0 governance
        scores.append(100 - privacy_res["overall_risk_score"])
        weights.append(0.25)

    if not scores:
        return 0.0

    # Normalise weights to sum to 1.0
    total_w = sum(weights)
    return sum(s * w / total_w for s, w in zip(scores, weights))


# ─── Helper: governance gauge ──────────────────────────────────────────

def _governance_gauge(score: float) -> go.Figure:
    """Create a Plotly gauge indicator for the overall governance score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": "%", "font": {"size": 48}},
        title={"text": "Overall Governance Score", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": score_colour(score)},
            "steps": GAUGE_STEPS,
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        margin=dict(l=30, r=30, t=60, b=20),
        height=300,
    )
    return fig


# ─── Main page content ─────────────────────────────────────────────────

def main():
    year, month, run_assessment = _render_sidebar()

    # Title
    st.markdown(
        "# NYC Taxi Data Governance Dashboard\n"
        "**Data Security and Privacy Assessment Framework for Big Data "
        "Transportation Systems**"
    )
    st.markdown(
        "This dashboard integrates data quality profiling, privacy risk "
        "detection, and security compliance assessment for NYC Yellow Taxi "
        "Trip Records. Select a year and month in the sidebar, then click "
        "'Run Full Assessment' to analyse the data."
    )

    with st.expander("Group Members & Contributions", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                "**Junaid Babar**  \n"
                "`B01802551`  \n"
                "Module 1 · Data Quality Profiling"
            )
        with m2:
            st.markdown(
                "**Sami Ullah**  \n"
                "`B01750598`  \n"
                "Module 2 · Privacy Risk Detection"
            )
        with m3:
            st.markdown(
                "**Jannat Rafique**  \n"
                "`B01798960`  \n"
                "Module 3 · Security Compliance"
            )
        with m4:
            st.markdown(
                "**Iqra Aziz**  \n"
                "`B01802319`  \n"
                "Module 4 · Governance Dashboard"
            )
    st.markdown("---")

    # ── Run assessment if requested ─────────────────────────────────────
    quality_res = st.session_state.get("quality_results")
    privacy_res = st.session_state.get("privacy_results")
    security_res = st.session_state.get("security_results")

    if run_assessment:
        # Load data
        try:
            with st.spinner(f"Loading {year}-{month:02d} trip data..."):
                df = cached_load_month(year, month)
            st.success(f"Loaded {len(df):,} trip records for {year}-{month:02d}.")
        except FileNotFoundError:
            st.error(
                f"Data file for {year}-{month:02d} not found. "
                f"Ensure Parquet files are in `{DATA_DIR}`."
            )
            return
        except Exception as exc:
            st.error(f"Error loading data: {exc}")
            return

        # Module 1: Quality
        if _quality_available:
            with st.spinner("Running data quality assessment..."):
                try:
                    quality_res = get_quality_metrics(df, year=year, month=month)
                    st.session_state["quality_results"] = quality_res
                except Exception as exc:
                    st.warning(f"Quality module error: {exc}")
                    quality_res = None
        else:
            st.info("Quality module not yet available — skipping.")

        # Module 2: Privacy
        if _privacy_available:
            with st.spinner("Running privacy risk assessment..."):
                try:
                    privacy_res = get_privacy_assessment(df, temporal_resolution="H")
                    st.session_state["privacy_results"] = privacy_res
                except Exception as exc:
                    st.warning(f"Privacy module error: {exc}")
                    privacy_res = None
        else:
            st.info("Privacy module not yet available — skipping.")

        # Module 3: Security
        if _security_available:
            with st.spinner("Running security compliance assessment..."):
                try:
                    security_res = get_security_checklist(data_dir=DATA_DIR)
                    st.session_state["security_results"] = security_res
                except Exception as exc:
                    st.warning(f"Security module error: {exc}")
                    security_res = None
        else:
            st.info("Security module not yet available — skipping.")

        st.session_state["assessment_year"] = year
        st.session_state["assessment_month"] = month

    # ── Display results or placeholder ──────────────────────────────────
    has_results = any(r is not None for r in [quality_res, privacy_res, security_res])

    if not has_results:
        st.markdown("---")
        st.markdown("### Getting Started")
        st.info(
            "No assessment results available yet. Use the sidebar to select a "
            "year and month, then click **Run Full Assessment** to begin.\n\n"
            "You can also explore individual module pages from the sidebar navigation."
        )

        # Show module availability summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quality Module", "Available" if _quality_available else "Pending")
        with col2:
            st.metric("Privacy Module", "Available" if _privacy_available else "Pending")
        with col3:
            st.metric("Security Module", "Available" if _security_available else "Pending")
        return

    # ── Metric summary cards ────────────────────────────────────────────
    assessed = st.session_state.get("assessment_year", "—")
    assessed_m = st.session_state.get("assessment_month", "—")
    if isinstance(assessed_m, int):
        st.markdown(f"**Assessment period:** {assessed}-{assessed_m:02d}")
    else:
        st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        if quality_res and isinstance(quality_res.get("overall_score"), (int, float)):
            q_score = quality_res["overall_score"]
            st.metric(
                "Data Quality Score",
                f"{q_score:.1f}%",
                delta=None,
                help="Composite score across completeness, accuracy, consistency, timeliness.",
            )
        else:
            st.metric("Data Quality Score", "N/A")

    with col2:
        if privacy_res:
            risk_level = privacy_res.get("risk_level", "N/A")
            risk_score = privacy_res.get("overall_risk_score", "N/A")
            label = f"{risk_level} ({risk_score})" if isinstance(risk_score, (int, float)) else risk_level
            st.metric(
                "Privacy Risk Level",
                label,
                help="Composite privacy risk from uniqueness, k-anonymity, entropy, linkage.",
            )
        else:
            st.metric("Privacy Risk Level", "N/A")

    with col3:
        if security_res and isinstance(security_res.get("overall_compliance"), (int, float)):
            s_score = security_res["overall_compliance"]
            st.metric(
                "Security Compliance",
                f"{s_score:.1f}%",
                help="Weighted compliance across NIST, GDPR, and ISO 27001 controls.",
            )
        else:
            st.metric("Security Compliance", "N/A")

    st.markdown("---")

    # ── Governance gauge ────────────────────────────────────────────────
    gov_score = _compute_governance_score(quality_res, privacy_res, security_res)

    col_gauge, col_issues = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(
            _governance_gauge(gov_score),
            use_container_width=True,
            key="gov_gauge",
        )

    # ── Recent issues / findings ────────────────────────────────────────
    with col_issues:
        st.markdown("### Key Findings")
        _display_findings(quality_res, privacy_res, security_res)

    # ── Module summary tabs ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Module Summaries")

    tab_q, tab_p, tab_s = st.tabs([
        "Quality Summary", "Privacy Summary", "Security Summary",
    ])

    with tab_q:
        if quality_res:
            _quality_summary(quality_res)
        else:
            st.info("Quality assessment not yet run.")

    with tab_p:
        if privacy_res:
            _privacy_summary(privacy_res)
        else:
            st.info("Privacy assessment not yet run.")

    with tab_s:
        if security_res:
            _security_summary(security_res)
        else:
            st.info("Security assessment not yet run.")

    # ── Export section ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Export Report")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        pdf_bytes = generate_summary_pdf(quality_res, privacy_res, security_res)
        pdf_filename = (
            f"governance_report_{assessed}_{assessed_m:02d}.pdf"
            if isinstance(assessed_m, int)
            else "governance_report.pdf"
        )
        st.download_button(
            label="Download Summary Report (PDF)",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
        )


# ─── Findings display ──────────────────────────────────────────────────

def _display_findings(quality_res, privacy_res, security_res):
    """Show top findings as Streamlit alert boxes."""
    findings_shown = 0

    if quality_res:
        q_score = quality_res.get("overall_score", 0)
        if isinstance(q_score, (int, float)):
            if q_score < 50:
                st.error(f"Data quality is critically low ({q_score:.1f}%). Immediate attention required.")
                findings_shown += 1
            elif q_score < 70:
                st.warning(f"Data quality score ({q_score:.1f}%) is below acceptable threshold.")
                findings_shown += 1
            else:
                st.success(f"Data quality score is healthy ({q_score:.1f}%).")
                findings_shown += 1

        # Check individual dimensions
        metrics = quality_res.get("metrics", {})
        for dim in ("completeness", "accuracy", "consistency", "timeliness"):
            val = metrics.get(dim)
            if isinstance(val, (int, float)) and val < 50:
                st.warning(f"{dim.title()} score is low: {val:.1f}%")
                findings_shown += 1

    if privacy_res:
        risk_level = privacy_res.get("risk_level", "")
        if risk_level == "Critical":
            st.error("Privacy risk is CRITICAL — high re-identification potential.")
            findings_shown += 1
        elif risk_level == "High":
            st.warning("Privacy risk is HIGH — significant uniqueness detected.")
            findings_shown += 1
        elif risk_level in ("Medium", "Low"):
            st.info(f"Privacy risk level: {risk_level}.")
            findings_shown += 1

    if security_res:
        compliance = security_res.get("overall_compliance", 0)
        if isinstance(compliance, (int, float)):
            if compliance < 50:
                st.error(f"Security compliance is critically low ({compliance:.1f}%).")
                findings_shown += 1
            elif compliance < 70:
                st.warning(f"Security compliance ({compliance:.1f}%) needs improvement.")
                findings_shown += 1

        gaps = security_res.get("gap_summary", [])
        if isinstance(gaps, dict):
            # Real module: {"High": [ids], "Medium": [ids], "Low": [ids]}
            high_ids = gaps.get("High", [])
            if high_ids:
                preview = ", ".join(high_ids[:3])
                extra = f" (+{len(high_ids) - 3} more)" if len(high_ids) > 3 else ""
                st.warning(f"High-priority security gaps: {preview}{extra}")
                findings_shown += 1
        elif isinstance(gaps, list) and gaps:
            for gap in gaps[:2]:
                st.warning(f"Gap: {gap}")
                findings_shown += 1

    if findings_shown == 0:
        st.info("No critical findings. Run a full assessment for detailed analysis.")


# ─── Module summaries ──────────────────────────────────────────────────

def _quality_summary(res: dict):
    """Render quality summary in a tab."""
    cols = st.columns(4)
    metrics = res.get("metrics", {})
    labels = ["Completeness", "Accuracy", "Consistency", "Timeliness"]
    keys = ["completeness", "accuracy", "consistency", "timeliness"]

    for col, label, key in zip(cols, labels, keys):
        val = metrics.get(key)
        with col:
            if isinstance(val, (int, float)):
                st.metric(label, f"{val:.1f}%")
            else:
                st.metric(label, "N/A")

    if res.get("summary_text"):
        st.markdown(res["summary_text"])


def _privacy_summary(res: dict):
    """Render privacy summary in a tab."""
    cols = st.columns(3)
    with cols[0]:
        st.metric("Risk Score", res.get("overall_risk_score", "N/A"))
    with cols[1]:
        st.metric("Uniqueness", f"{res.get('uniqueness_percentage', 'N/A')}%")
    with cols[2]:
        st.metric("Linkage Rate", res.get("linkage_rate", "N/A"))

    if res.get("summary_text"):
        st.markdown(res["summary_text"])


def _security_summary(res: dict):
    """Render security summary in a tab."""
    compliance = res.get("overall_compliance")
    if isinstance(compliance, (int, float)):
        st.metric("Overall Compliance", f"{compliance:.1f}%")

    scores = res.get("compliance_scores", {}) or {}
    # Whitelist the three framework scores; accept both the real-module
    # "_compliance_pct" keys and the fallback-dict human-readable names.
    framework_pairs = [
        ("GDPR",      scores.get("gdpr_compliance_pct", scores.get("GDPR"))),
        ("ISO 27001", scores.get("iso_compliance_pct", scores.get("ISO 27001"))),
        ("NIST CSF",  scores.get("nist_compliance_pct", scores.get("NIST CSF"))),
    ]
    framework_pairs = [(k, v) for k, v in framework_pairs if isinstance(v, (int, float))]
    if framework_pairs:
        cols = st.columns(len(framework_pairs))
        for col, (framework, score) in zip(cols, framework_pairs):
            with col:
                st.metric(framework, f"{score:.1f}%")

    if res.get("summary_text"):
        st.markdown(res["summary_text"])


# ─── Entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
else:
    main()
