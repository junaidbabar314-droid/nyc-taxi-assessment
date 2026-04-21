"""
run_all.py — CLI entry point for running all assessment modules.

Usage:
    python run_all.py                          # Full assessment, 2019-01 (1% sample)
    python run_all.py --year 2024 --month 6    # Specific month
    python run_all.py --full                   # No sampling (slow for full months)
    python run_all.py --year 2019 --all-months # All 12 months of a year

Group project — MSc IT (Data Analysis), University of the West of Scotland:
    Junaid Babar   (B01802551) — Module 1 · Data Quality Profiling
    Sami Ullah     (B01750598) — Module 2 · Privacy Risk Detection
    Jannat Rafique (B01798960) — Module 3 · Security Compliance
    Iqra Aziz      (B01802319) — Module 4 · Governance Dashboard

This shared CLI harness integrates all four modules into a single
reproducible command-line workflow.
"""

import sys
import argparse
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_month, load_year, list_parquet_files
from config import DATA_DIR, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_quality(df, year, month):
    """Run Module 1: Data Quality Profiling."""
    from module1_quality.quality_profiler import get_quality_metrics

    print("\n" + "=" * 60)
    print("MODULE 1: DATA QUALITY PROFILING (Junaid Babar)")
    print("=" * 60)

    t0 = time.time()
    result = get_quality_metrics(df, year=year, month=month)
    elapsed = time.time() - t0

    print(f"\n  Overall Quality Score: {result['overall_score']:.1f}/100")
    print(f"  Completeness:  {result['metrics']['completeness']:.1f}/100")
    print(f"  Accuracy:      {result['metrics']['accuracy']:.1f}/100")
    print(f"  Consistency:   {result['metrics']['consistency']:.1f}/100")
    print(f"  Timeliness:    {result['metrics']['timeliness']:.1f}/100")
    print(f"\n  {result['summary_text']}")
    print(f"  [Completed in {elapsed:.1f}s]")

    return result


def run_privacy(df):
    """Run Module 2: Privacy Risk Detection."""
    from module2_privacy.privacy_assessor import get_privacy_assessment

    print("\n" + "=" * 60)
    print("MODULE 2: PRIVACY RISK DETECTION (Sami Ullah)")
    print("=" * 60)

    t0 = time.time()
    result = get_privacy_assessment(df)
    elapsed = time.time() - t0

    print(f"\n  Overall Risk Score: {result['overall_risk_score']:.1f}/100")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Uniqueness: {result['uniqueness_percentage']:.1f}%")
    print(f"  Linkage Rate: {result['linkage_rate']:.1f}%")
    print(f"  PII Fields: {len(result['pii_fields'])}")
    print(f"\n  {result['summary_text']}")
    print(f"  [Completed in {elapsed:.1f}s]")

    return result


def run_security():
    """Run Module 3: Security Assessment."""
    from module3_security.security_assessor import get_security_checklist

    print("\n" + "=" * 60)
    print("MODULE 3: SECURITY ASSESSMENT (Jannat Rafique)")
    print("=" * 60)

    t0 = time.time()
    result = get_security_checklist()
    elapsed = time.time() - t0

    print(f"\n  Overall Compliance: {result['overall_compliance']:.1f}%")
    scores = result['compliance_scores']
    print(f"  NIST CSF:  {scores.get('nist_compliance_pct', scores.get('NIST CSF 2.0', 0)):.1f}%")
    print(f"  GDPR:      {scores.get('gdpr_compliance_pct', scores.get('GDPR', 0)):.1f}%")
    print(f"  ISO 27001: {scores.get('iso_compliance_pct', scores.get('ISO 27001', 0)):.1f}%")

    gap_summary = result['gap_summary']
    for priority in ['High', 'Medium', 'Low']:
        items = gap_summary.get(priority, [])
        if items:
            print(f"  {priority}-priority gaps: {len(items)}")

    print(f"\n  {result['summary_text']}")
    print(f"  [Completed in {elapsed:.1f}s]")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="NYC Taxi Data Security & Privacy Assessment Framework"
    )
    parser.add_argument("--year", type=int, default=2019, help="Data year (default: 2019)")
    parser.add_argument("--month", type=int, default=1, help="Data month (default: 1)")
    parser.add_argument("--sample", type=float, default=0.01, help="Sample fraction (default: 0.01)")
    parser.add_argument("--full", action="store_true", help="Use full data (no sampling)")
    parser.add_argument("--quality-only", action="store_true", help="Run only quality module")
    parser.add_argument("--privacy-only", action="store_true", help="Run only privacy module")
    parser.add_argument("--security-only", action="store_true", help="Run only security module")
    args = parser.parse_args()

    sample_frac = None if args.full else args.sample

    print("=" * 60)
    print("NYC TAXI DATA GOVERNANCE ASSESSMENT FRAMEWORK")
    print("=" * 60)
    print(f"  Year: {args.year}  |  Month: {args.month}")
    print(f"  Sampling: {'Full data' if args.full else f'{args.sample*100:.0f}%'}")
    print(f"  Data dir: {DATA_DIR}")

    # List available files
    files = list_parquet_files()
    print(f"  Available files: {len(files)}")

    # Load data
    print(f"\nLoading data for {args.year}-{args.month:02d}...")
    t_start = time.time()
    df = load_month(args.year, args.month, sample_frac=sample_frac)
    print(f"  Loaded {len(df):,} rows in {time.time() - t_start:.1f}s")

    run_all = not (args.quality_only or args.privacy_only or args.security_only)

    results = {}

    # Module 1: Quality
    if run_all or args.quality_only:
        results['quality'] = run_quality(df, args.year, args.month)

    # Module 2: Privacy
    if run_all or args.privacy_only:
        results['privacy'] = run_privacy(df)

    # Module 3: Security
    if run_all or args.security_only:
        results['security'] = run_security()

    # Summary
    print("\n" + "=" * 60)
    print("ASSESSMENT COMPLETE")
    print("=" * 60)

    if 'quality' in results:
        print(f"  Quality Score:       {results['quality']['overall_score']:.1f}/100")
    if 'privacy' in results:
        print(f"  Privacy Risk:        {results['privacy']['overall_risk_score']:.1f}/100 ({results['privacy']['risk_level']})")
    if 'security' in results:
        print(f"  Security Compliance: {results['security']['overall_compliance']:.1f}%")

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"\n  Dashboard: streamlit run dashboard/Home.py")

    return results


if __name__ == "__main__":
    main()
