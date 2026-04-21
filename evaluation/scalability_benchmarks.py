"""
Scalability Benchmarking Suite for the Data Governance Framework.

Evaluates processing throughput, memory consumption, and scaling behaviour
of the quality profiling (Module 1) and privacy assessment (Module 2)
pipelines across a range of dataset sizes from 1,000 to the maximum
available rows (~24 million across 2024).

The results feed directly into the dissertation's evaluation chapter,
providing empirical evidence for the framework's scalability claims.
All charts are rendered in grayscale for print-friendly dissertation
figures.

Methodology:
    Each pipeline is timed using time.perf_counter (wall-clock) and
    profiled with tracemalloc (peak RSS delta).  Three iterations per
    sample size yield mean and standard-deviation estimates.  Scaling
    behaviour is assessed by fitting a power-law model (t = a * N^b)
    via least-squares on log-transformed data; b ~ 1.0 indicates O(n)
    linear scaling.

References:
    McConnell, S. (2004) Code Complete. 2nd edn. Redmond, WA:
        Microsoft Press, pp. 587-606.
    Knuth, D.E. (1997) The Art of Computer Programming, Volume 1:
        Fundamental Algorithms. 3rd edn. Boston, MA: Addison-Wesley.

Author: Junaid Babar (B01802551)
Module: Evaluation / Scalability Benchmarking
"""

from __future__ import annotations

import gc
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from data_loader import load_month, load_year  # noqa: E402
from module1_quality.quality_profiler import get_quality_metrics  # noqa: E402
from module1_quality.completeness import assess_completeness, compute_completeness_score  # noqa: E402
from module1_quality.accuracy import assess_accuracy  # noqa: E402
from module1_quality.consistency import assess_consistency  # noqa: E402
from module1_quality.timeliness import assess_timeliness  # noqa: E402
from module2_privacy.privacy_assessor import get_privacy_assessment  # noqa: E402
from config import OUTPUT_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_SIZES = [1_000, 10_000, 100_000, 500_000, 1_000_000]
ITERATIONS = 3
BASE_YEAR = 2024
BASE_MONTH = 1
FIGURES_DIR = OUTPUT_DIR / "figures"

QUALITY_DIMENSIONS = {
    "completeness": lambda df, y, m: _run_completeness(df),
    "accuracy": lambda df, y, m: assess_accuracy(df),
    "consistency": lambda df, y, m: assess_consistency(df),
    "timeliness": lambda df, y, m: assess_timeliness(df, file_year=y, file_month=m),
}


def _run_completeness(df: pd.DataFrame) -> float:
    """Run completeness and return the score (wrapper for consistent API)."""
    cdf = assess_completeness(df)
    return compute_completeness_score(cdf)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_base_data(max_rows: int | None = None) -> pd.DataFrame:
    """
    Load enough data to satisfy the largest requested sample size.

    Starts with 2024-01 (~2.9M rows).  If more rows are needed,
    concatenates additional months from 2024.

    Parameters:
        max_rows: Maximum rows required.  If None, load a single month.

    Returns:
        DataFrame with at least max_rows rows (or all available data).
    """
    df = load_month(BASE_YEAR, BASE_MONTH)
    logger.info("Base month loaded: %d rows", len(df))

    if max_rows is not None and len(df) < max_rows:
        # Load additional months until we have enough
        for extra_month in range(2, 13):
            if len(df) >= max_rows:
                break
            try:
                extra = load_month(BASE_YEAR, extra_month)
                df = pd.concat([df, extra], ignore_index=True)
                logger.info(
                    "Added month %d: now %d rows", extra_month, len(df)
                )
            except FileNotFoundError:
                logger.warning("Month %d not available", extra_month)

    return df


def _subsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Draw exactly n rows from df with a fixed random seed.

    Parameters:
        df: Source DataFrame.
        n:  Number of rows to sample.

    Returns:
        DataFrame with exactly n rows (or all rows if n >= len(df)).
    """
    if n >= len(df):
        return df.copy()
    return df.sample(n=n, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Timing / memory utilities
# ---------------------------------------------------------------------------

def _timed_run(func, *args, **kwargs) -> dict:
    """
    Execute *func* while measuring wall-clock time and peak memory.

    Parameters:
        func:   Callable to benchmark.
        *args:  Positional arguments forwarded to func.
        **kwargs: Keyword arguments forwarded to func.

    Returns:
        Dictionary with keys: elapsed_s, peak_memory_mb, result.
    """
    gc.collect()
    tracemalloc.start()

    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "elapsed_s": elapsed,
        "peak_memory_mb": peak / (1024 * 1024),
        "result": result,
    }


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def benchmark_quality_scaling(
    sample_sizes: list[int] | None = None,
    iterations: int = ITERATIONS,
) -> list[dict]:
    """
    Benchmark quality profiling at multiple data scales.

    For each sample size, runs get_quality_metrics() *iterations* times
    and records elapsed time, peak memory, and throughput.

    Parameters:
        sample_sizes: List of row counts to test.  Defaults to SAMPLE_SIZES.
        iterations:   Number of repetitions per size for statistics.

    Returns:
        List of dicts, one per sample size, with keys:
            n_rows, mean_time_s, std_time_s, mean_memory_mb,
            mean_throughput_rows_per_s, timings (raw list).
    """
    sample_sizes = sample_sizes or SAMPLE_SIZES
    max_needed = max(sample_sizes)
    print(f"Loading base data (need up to {max_needed:,} rows)...")
    base_df = _load_base_data(max_needed)
    print(f"Base data ready: {len(base_df):,} rows")

    results = []
    for n in sample_sizes:
        if n > len(base_df):
            print(f"  Skipping {n:,} rows (only {len(base_df):,} available)")
            continue

        df_sample = _subsample(base_df, n)
        timings = []
        memories = []

        print(f"  Benchmarking quality @ {n:,} rows ({iterations} iterations)...")
        for i in range(iterations):
            run = _timed_run(get_quality_metrics, df_sample, year=BASE_YEAR, month=BASE_MONTH)
            timings.append(run["elapsed_s"])
            memories.append(run["peak_memory_mb"])
            print(f"    Iter {i+1}: {run['elapsed_s']:.3f}s, {run['peak_memory_mb']:.1f} MB")

        mean_t = np.mean(timings)
        results.append({
            "n_rows": n,
            "mean_time_s": mean_t,
            "std_time_s": np.std(timings),
            "mean_memory_mb": np.mean(memories),
            "std_memory_mb": np.std(memories),
            "mean_throughput_rows_per_s": n / mean_t if mean_t > 0 else 0,
            "timings": timings,
        })

    return results


def benchmark_privacy_scaling(
    sample_sizes: list[int] | None = None,
    iterations: int = ITERATIONS,
) -> list[dict]:
    """
    Benchmark privacy assessment at multiple data scales.

    Parameters:
        sample_sizes: List of row counts to test.
        iterations:   Number of repetitions per size.

    Returns:
        List of dicts with the same structure as benchmark_quality_scaling().
    """
    sample_sizes = sample_sizes or SAMPLE_SIZES
    max_needed = max(sample_sizes)
    print(f"Loading base data for privacy benchmarks ({max_needed:,} rows)...")
    base_df = _load_base_data(max_needed)
    print(f"Base data ready: {len(base_df):,} rows")

    results = []
    for n in sample_sizes:
        if n > len(base_df):
            print(f"  Skipping {n:,} rows (only {len(base_df):,} available)")
            continue

        df_sample = _subsample(base_df, n)
        timings = []
        memories = []

        print(f"  Benchmarking privacy @ {n:,} rows ({iterations} iterations)...")
        for i in range(iterations):
            run = _timed_run(get_privacy_assessment, df_sample, temporal_resolution="H")
            timings.append(run["elapsed_s"])
            memories.append(run["peak_memory_mb"])
            print(f"    Iter {i+1}: {run['elapsed_s']:.3f}s, {run['peak_memory_mb']:.1f} MB")

        mean_t = np.mean(timings)
        results.append({
            "n_rows": n,
            "mean_time_s": mean_t,
            "std_time_s": np.std(timings),
            "mean_memory_mb": np.mean(memories),
            "std_memory_mb": np.std(memories),
            "mean_throughput_rows_per_s": n / mean_t if mean_t > 0 else 0,
            "timings": timings,
        })

    return results


def benchmark_dimension_breakdown(
    sample_sizes: list[int] | None = None,
    iterations: int = ITERATIONS,
) -> dict[str, list[dict]]:
    """
    Benchmark each quality dimension independently to identify bottlenecks.

    Parameters:
        sample_sizes: List of row counts to test.
        iterations:   Number of repetitions per size.

    Returns:
        Dictionary keyed by dimension name, each mapping to a list of
        per-size result dicts (same schema as benchmark_quality_scaling).
    """
    sample_sizes = sample_sizes or SAMPLE_SIZES
    max_needed = max(sample_sizes)
    print(f"Loading base data for dimension breakdown ({max_needed:,} rows)...")
    base_df = _load_base_data(max_needed)
    print(f"Base data ready: {len(base_df):,} rows")

    all_results: dict[str, list[dict]] = {}

    for dim_name, dim_func in QUALITY_DIMENSIONS.items():
        dim_results = []
        print(f"\n  Dimension: {dim_name}")

        for n in sample_sizes:
            if n > len(base_df):
                continue

            df_sample = _subsample(base_df, n)
            timings = []
            memories = []

            print(f"    {n:,} rows ({iterations} iters)...", end=" ")
            for _ in range(iterations):
                run = _timed_run(dim_func, df_sample, BASE_YEAR, BASE_MONTH)
                timings.append(run["elapsed_s"])
                memories.append(run["peak_memory_mb"])

            mean_t = np.mean(timings)
            dim_results.append({
                "n_rows": n,
                "mean_time_s": mean_t,
                "std_time_s": np.std(timings),
                "mean_memory_mb": np.mean(memories),
                "mean_throughput_rows_per_s": n / mean_t if mean_t > 0 else 0,
            })
            print(f"{mean_t:.4f}s")

        all_results[dim_name] = dim_results

    return all_results


# ---------------------------------------------------------------------------
# Scaling analysis
# ---------------------------------------------------------------------------

def _fit_power_law(sizes: list[int], times: list[float]) -> dict:
    """
    Fit t = a * N^b via least-squares on log-transformed data.

    Parameters:
        sizes: Sample sizes (N).
        times: Corresponding mean elapsed times.

    Returns:
        Dictionary with a, b (exponent), r_squared, complexity_class.
    """
    log_n = np.log10(np.array(sizes, dtype=float))
    log_t = np.log10(np.array(times, dtype=float))

    # Linear regression on log-log
    coeffs = np.polyfit(log_n, log_t, 1)
    b, log_a = coeffs
    a = 10 ** log_a

    # R-squared
    predicted = np.polyval(coeffs, log_n)
    ss_res = np.sum((log_t - predicted) ** 2)
    ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Classify complexity
    if b < 1.15:
        complexity = "O(n) — linear"
    elif b < 1.5:
        complexity = "O(n log n) — linearithmic"
    elif b < 2.1:
        complexity = "O(n^2) — quadratic"
    else:
        complexity = f"O(n^{b:.2f}) — super-quadratic"

    return {
        "a": a,
        "b": b,
        "r_squared": r_squared,
        "complexity_class": complexity,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_benchmark_report(
    quality_results: list[dict],
    privacy_results: list[dict],
    dimension_results: dict[str, list[dict]],
) -> str:
    """
    Produce a formatted text report suitable for the dissertation
    evaluation chapter.

    Parameters:
        quality_results:   Output of benchmark_quality_scaling().
        privacy_results:   Output of benchmark_privacy_scaling().
        dimension_results: Output of benchmark_dimension_breakdown().

    Returns:
        Multi-paragraph report string with tables and analysis.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("SCALABILITY BENCHMARK REPORT")
    lines.append("Data Governance Framework — NYC Taxi Trip Records")
    lines.append("=" * 72)

    # --- Quality profiling table ---
    lines.append("\n1. QUALITY PROFILING PIPELINE — SCALING RESULTS")
    lines.append("-" * 72)
    lines.append(
        f"{'Rows':>12s}  {'Mean Time (s)':>14s}  {'Std (s)':>9s}  "
        f"{'Throughput':>14s}  {'Memory (MB)':>12s}"
    )
    lines.append("-" * 72)

    q_sizes, q_times = [], []
    for r in quality_results:
        q_sizes.append(r["n_rows"])
        q_times.append(r["mean_time_s"])
        lines.append(
            f"{r['n_rows']:>12,d}  {r['mean_time_s']:>14.4f}  "
            f"{r['std_time_s']:>9.4f}  "
            f"{r['mean_throughput_rows_per_s']:>12,.0f}/s  "
            f"{r['mean_memory_mb']:>12.1f}"
        )

    if len(q_sizes) >= 2:
        fit = _fit_power_law(q_sizes, q_times)
        lines.append(f"\nScaling exponent (b): {fit['b']:.3f}  "
                      f"(R² = {fit['r_squared']:.4f})")
        lines.append(f"Complexity class: {fit['complexity_class']}")
        lines.append(f"Model: t = {fit['a']:.6e} * N^{fit['b']:.3f}")

    # --- Privacy assessment table ---
    lines.append("\n\n2. PRIVACY ASSESSMENT PIPELINE — SCALING RESULTS")
    lines.append("-" * 72)
    lines.append(
        f"{'Rows':>12s}  {'Mean Time (s)':>14s}  {'Std (s)':>9s}  "
        f"{'Throughput':>14s}  {'Memory (MB)':>12s}"
    )
    lines.append("-" * 72)

    p_sizes, p_times = [], []
    for r in privacy_results:
        p_sizes.append(r["n_rows"])
        p_times.append(r["mean_time_s"])
        lines.append(
            f"{r['n_rows']:>12,d}  {r['mean_time_s']:>14.4f}  "
            f"{r['std_time_s']:>9.4f}  "
            f"{r['mean_throughput_rows_per_s']:>12,.0f}/s  "
            f"{r['mean_memory_mb']:>12.1f}"
        )

    if len(p_sizes) >= 2:
        fit = _fit_power_law(p_sizes, p_times)
        lines.append(f"\nScaling exponent (b): {fit['b']:.3f}  "
                      f"(R² = {fit['r_squared']:.4f})")
        lines.append(f"Complexity class: {fit['complexity_class']}")

    # --- Dimension breakdown ---
    lines.append("\n\n3. QUALITY DIMENSION BREAKDOWN (time in seconds)")
    lines.append("-" * 72)

    # Build a table: rows = sizes, columns = dimensions
    dim_names = list(dimension_results.keys())
    header = f"{'Rows':>12s}  " + "  ".join(f"{d:>14s}" for d in dim_names)
    lines.append(header)
    lines.append("-" * 72)

    # Align by row count
    all_sizes = sorted({
        r["n_rows"]
        for dim_list in dimension_results.values()
        for r in dim_list
    })

    for n in all_sizes:
        row_parts = [f"{n:>12,d}"]
        for dim in dim_names:
            matching = [r for r in dimension_results[dim] if r["n_rows"] == n]
            if matching:
                row_parts.append(f"{matching[0]['mean_time_s']:>14.4f}")
            else:
                row_parts.append(f"{'N/A':>14s}")
        lines.append("  ".join(row_parts))

    # Identify bottleneck at largest size
    largest_size = max(all_sizes) if all_sizes else 0
    bottleneck_dim = None
    bottleneck_time = 0
    for dim in dim_names:
        matching = [r for r in dimension_results[dim] if r["n_rows"] == largest_size]
        if matching and matching[0]["mean_time_s"] > bottleneck_time:
            bottleneck_time = matching[0]["mean_time_s"]
            bottleneck_dim = dim

    if bottleneck_dim:
        lines.append(
            f"\nBottleneck at {largest_size:,} rows: {bottleneck_dim} "
            f"({bottleneck_time:.4f}s)"
        )

    # --- Summary ---
    lines.append("\n\n4. SCALABILITY ASSESSMENT")
    lines.append("-" * 72)

    if quality_results:
        max_throughput = max(r["mean_throughput_rows_per_s"] for r in quality_results)
        max_rows_tested = max(r["n_rows"] for r in quality_results)
        lines.append(
            f"Peak quality throughput: {max_throughput:,.0f} rows/second"
        )
        lines.append(
            f"Maximum dataset size tested: {max_rows_tested:,} rows"
        )

    if privacy_results:
        max_throughput_p = max(r["mean_throughput_rows_per_s"] for r in privacy_results)
        lines.append(
            f"Peak privacy throughput: {max_throughput_p:,.0f} rows/second"
        )

    lines.append("")
    lines.append("Note: The dissertation specification targets benchmarks at 1M, 10M,")
    lines.append("and 100M rows.  The locally available dataset contains approximately")
    if quality_results:
        lines.append(
            f"{max_rows_tested:,} rows for the largest feasible test.  Extrapolation"
        )
    lines.append("based on the observed scaling exponent suggests that processing")
    lines.append("100M rows would require approximately:")

    if len(q_sizes) >= 2:
        fit = _fit_power_law(q_sizes, q_times)
        est_100m = fit["a"] * (100_000_000 ** fit["b"])
        lines.append(f"  Quality profiling: ~{est_100m:.0f} seconds ({est_100m/60:.1f} min)")

    if len(p_sizes) >= 2:
        fit_p = _fit_power_law(p_sizes, p_times)
        est_100m_p = fit_p["a"] * (100_000_000 ** fit_p["b"])
        lines.append(f"  Privacy assessment: ~{est_100m_p:.0f} seconds ({est_100m_p/60:.1f} min)")

    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chart generation (grayscale, dissertation-ready)
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    """Configure matplotlib for grayscale, print-friendly charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def generate_scaling_chart(
    quality_results: list[dict],
    privacy_results: list[dict],
    output_path: Path | None = None,
) -> Path:
    """
    Log-log plot of data size vs processing time for both pipelines.

    Parameters:
        quality_results: Output of benchmark_quality_scaling().
        privacy_results: Output of benchmark_privacy_scaling().
        output_path:     Save path.  Defaults to figures directory.

    Returns:
        Path to the saved figure.
    """
    plt = _setup_matplotlib()
    output_path = output_path or FIGURES_DIR / "benchmark_quality_scaling.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    # Quality line
    q_sizes = [r["n_rows"] for r in quality_results]
    q_times = [r["mean_time_s"] for r in quality_results]
    q_stds = [r["std_time_s"] for r in quality_results]
    ax.errorbar(
        q_sizes, q_times, yerr=q_stds,
        marker="o", color="black", linestyle="-", linewidth=1.5,
        capsize=4, label="Quality profiling",
    )

    # Privacy line
    if privacy_results:
        p_sizes = [r["n_rows"] for r in privacy_results]
        p_times = [r["mean_time_s"] for r in privacy_results]
        p_stds = [r["std_time_s"] for r in privacy_results]
        ax.errorbar(
            p_sizes, p_times, yerr=p_stds,
            marker="s", color="gray", linestyle="--", linewidth=1.5,
            capsize=4, label="Privacy assessment",
        )

    # Reference lines for O(n) and O(n^2)
    if len(q_sizes) >= 2:
        ref_sizes = np.array(q_sizes, dtype=float)
        # Normalise O(n) reference to pass through the first quality data point
        on_ref = q_times[0] * (ref_sizes / ref_sizes[0])
        ax.plot(
            ref_sizes, on_ref,
            color="lightgray", linestyle=":", linewidth=1,
            label="O(n) reference",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size (rows)")
    ax.set_ylabel("Processing Time (seconds)")
    ax.set_title("Framework Scalability: Processing Time vs Dataset Size")
    ax.legend(frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


def generate_dimension_breakdown_chart(
    dimension_results: dict[str, list[dict]],
    output_path: Path | None = None,
) -> Path:
    """
    Stacked bar chart showing time per quality dimension at each scale.

    Parameters:
        dimension_results: Output of benchmark_dimension_breakdown().
        output_path:       Save path.

    Returns:
        Path to the saved figure.
    """
    plt = _setup_matplotlib()
    output_path = output_path or FIGURES_DIR / "benchmark_dimension_breakdown.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dim_names = list(dimension_results.keys())
    all_sizes = sorted({
        r["n_rows"]
        for dim_list in dimension_results.values()
        for r in dim_list
    })

    # Build arrays for stacking
    data = {}
    for dim in dim_names:
        dim_times = []
        for n in all_sizes:
            matching = [r for r in dimension_results[dim] if r["n_rows"] == n]
            dim_times.append(matching[0]["mean_time_s"] if matching else 0)
        data[dim] = dim_times

    # Grayscale hatching patterns for each dimension
    hatches = ["///", "\\\\\\", "...", "xxx"]
    grays = ["0.2", "0.4", "0.6", "0.8"]

    fig, ax = plt.subplots()
    x = np.arange(len(all_sizes))
    bar_width = 0.6
    bottom = np.zeros(len(all_sizes))

    for i, dim in enumerate(dim_names):
        bars = ax.bar(
            x, data[dim], bar_width,
            bottom=bottom,
            color=grays[i % len(grays)],
            hatch=hatches[i % len(hatches)],
            edgecolor="black",
            linewidth=0.5,
            label=dim.capitalize(),
        )
        bottom += np.array(data[dim])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in all_sizes], rotation=30, ha="right")
    ax.set_xlabel("Dataset Size (rows)")
    ax.set_ylabel("Processing Time (seconds)")
    ax.set_title("Quality Profiling: Time Breakdown by Dimension")
    ax.legend(frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


def generate_throughput_chart(
    quality_results: list[dict],
    privacy_results: list[dict],
    output_path: Path | None = None,
) -> Path:
    """
    Throughput (rows/sec) vs dataset size for both pipelines.

    Parameters:
        quality_results: Output of benchmark_quality_scaling().
        privacy_results: Output of benchmark_privacy_scaling().
        output_path:     Save path.

    Returns:
        Path to the saved figure.
    """
    plt = _setup_matplotlib()
    output_path = output_path or FIGURES_DIR / "benchmark_throughput.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    q_sizes = [r["n_rows"] for r in quality_results]
    q_throughput = [r["mean_throughput_rows_per_s"] for r in quality_results]
    ax.plot(
        q_sizes, q_throughput,
        marker="o", color="black", linestyle="-", linewidth=1.5,
        label="Quality profiling",
    )

    if privacy_results:
        p_sizes = [r["n_rows"] for r in privacy_results]
        p_throughput = [r["mean_throughput_rows_per_s"] for r in privacy_results]
        ax.plot(
            p_sizes, p_throughput,
            marker="s", color="gray", linestyle="--", linewidth=1.5,
            label="Privacy assessment",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Dataset Size (rows)")
    ax.set_ylabel("Throughput (rows/second)")
    ax.set_title("Framework Throughput at Different Data Scales")
    ax.legend(frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


def generate_memory_chart(
    quality_results: list[dict],
    privacy_results: list[dict],
    output_path: Path | None = None,
) -> Path:
    """
    Memory usage (peak MB) vs dataset size for both pipelines.

    Parameters:
        quality_results: Output of benchmark_quality_scaling().
        privacy_results: Output of benchmark_privacy_scaling().
        output_path:     Save path.

    Returns:
        Path to the saved figure.
    """
    plt = _setup_matplotlib()
    output_path = output_path or FIGURES_DIR / "benchmark_memory.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    q_sizes = [r["n_rows"] for r in quality_results]
    q_mem = [r["mean_memory_mb"] for r in quality_results]
    ax.plot(
        q_sizes, q_mem,
        marker="o", color="black", linestyle="-", linewidth=1.5,
        label="Quality profiling",
    )

    if privacy_results:
        p_sizes = [r["n_rows"] for r in privacy_results]
        p_mem = [r["mean_memory_mb"] for r in privacy_results]
        ax.plot(
            p_sizes, p_mem,
            marker="s", color="gray", linestyle="--", linewidth=1.5,
            label="Privacy assessment",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Dataset Size (rows)")
    ax.set_ylabel("Peak Memory Usage (MB)")
    ax.set_title("Framework Memory Consumption at Different Data Scales")
    ax.legend(frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# All-in-one runner
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    sample_sizes: list[int] | None = None,
    iterations: int = ITERATIONS,
    generate_charts: bool = True,
    save_report: bool = True,
) -> dict[str, Any]:
    """
    Execute the complete benchmarking suite and optionally generate
    charts and a text report.

    Parameters:
        sample_sizes:    Row counts to benchmark.
        iterations:      Repetitions per size.
        generate_charts: Whether to save matplotlib figures.
        save_report:     Whether to write the report to a text file.

    Returns:
        Dictionary with keys: quality, privacy, dimensions, report,
        chart_paths.
    """
    sample_sizes = sample_sizes or SAMPLE_SIZES

    print("\n" + "=" * 72)
    print("SCALABILITY BENCHMARKING SUITE")
    print("=" * 72)

    # 1. Quality scaling
    print("\n--- Phase 1: Quality Profiling Scaling ---")
    quality_results = benchmark_quality_scaling(sample_sizes, iterations)

    # 2. Privacy scaling
    print("\n--- Phase 2: Privacy Assessment Scaling ---")
    privacy_results = benchmark_privacy_scaling(sample_sizes, iterations)

    # 3. Dimension breakdown
    print("\n--- Phase 3: Quality Dimension Breakdown ---")
    dimension_results = benchmark_dimension_breakdown(sample_sizes, iterations)

    # 4. Report
    print("\n--- Generating Report ---")
    report = generate_benchmark_report(quality_results, privacy_results, dimension_results)
    print(report)

    chart_paths = []
    if generate_charts:
        print("\n--- Generating Charts ---")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        chart_paths.append(generate_scaling_chart(quality_results, privacy_results))
        chart_paths.append(generate_dimension_breakdown_chart(dimension_results))
        chart_paths.append(generate_throughput_chart(quality_results, privacy_results))
        chart_paths.append(generate_memory_chart(quality_results, privacy_results))

    if save_report:
        report_path = OUTPUT_DIR / "scalability_benchmark_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"\n  Report saved: {report_path}")

    return {
        "quality": quality_results,
        "privacy": privacy_results,
        "dimensions": dimension_results,
        "report": report,
        "chart_paths": [str(p) for p in chart_paths],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 72)
    print("Scalability Benchmarking Suite")
    print("Junaid Babar (B01802551) — MSc Data Governance Framework")
    print("=" * 72)

    # Default run: all standard sample sizes, 3 iterations
    # For a quick test, uncomment the line below instead:
    # results = run_all_benchmarks(sample_sizes=[1_000, 10_000, 100_000], iterations=2)
    results = run_all_benchmarks()

    print("\n\nBenchmarking complete.")
    if results["chart_paths"]:
        print("Charts saved to:")
        for p in results["chart_paths"]:
            print(f"  {p}")
