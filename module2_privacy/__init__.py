# Module 2: Privacy Risk Detection — Sami Ullah (B01750598)
#
# Sub-modules:
#   pii_classifier   — Field-level PII classification and scoring
#   uniqueness        — Quasi-identifier uniqueness analysis
#   k_anonymity       — k-anonymity equivalence class assessment
#   entropy           — Shannon entropy for trajectory/temporal predictability
#   linkage_attack    — Landmark-based linkage attack simulation
#   risk_scorer       — Weighted composite risk scoring with sensitivity analysis
#   privacy_assessor  — Main orchestrator combining all sub-modules
#   visualisations    — Publication-quality Plotly charts

from module2_privacy.privacy_assessor import get_privacy_assessment

__all__ = ["get_privacy_assessment"]
