# Data Security and Privacy Assessment Framework for Big Data Transportation Systems

**MSc IT (Data Analysis) — University of the West of Scotland**

A governance framework that assesses data quality, privacy risk, and security compliance for NYC Yellow Taxi Trip Records (2019-2024). Built as a four-person group dissertation with an interactive Streamlit dashboard.

---

## Team

| Member | Banner ID | Contribution |
|--------|-----------|--------------|
| Junaid Babar | B01802551 | Module 1 — Data Quality Profiling (`module1_quality/`) |
| Sami Ullah | B01750598 | Module 2 — Privacy Risk Detection (`module2_privacy/`) |
| Jannat Rafique | B01798960 | Module 3 — Security Compliance (`module3_security/`) |
| Iqra Aziz | B01802319 | Module 4 — Governance Dashboard (`dashboard/`) |

---

## Requirements

- **Python 3.10** (managed by Conda — do **not** install Python separately)
- **Miniconda** or **Anaconda** (Miniconda recommended; ~400 MB)
- **~3 GB free disk space** for the NYC Taxi Parquet data
- **Windows, macOS, or Linux** — instructions below cover all three

---

## Step-by-Step Setup

### Step 1 — Install Miniconda

**Windows (PowerShell or Command Prompt):**

1. Download the installer:
   ```
   https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
   ```
2. Run the `.exe` and accept the defaults. At the final screen, tick **"Add Miniconda to my PATH environment variable"** (even though the installer warns against it — we need it for the terminal commands below).
3. Close and reopen your terminal (or open **"Anaconda Prompt"** from the Start menu).
4. Verify the install:
   ```bash
   conda --version
   ```
   You should see something like `conda 24.x.x`.

**macOS (Terminal):**

```bash
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
# Restart your terminal, then:
conda --version
```

**Linux (Terminal):**

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Restart your terminal, then:
conda --version
```

---

### Step 2 — Clone / download the project and open a terminal in it

```bash
cd path/to/nyc-taxi-assessment
```

Confirm you are in the right folder:

```bash
ls    # You should see: environment.yml, run_all.py, dashboard/, module1_quality/, ...
```

---

### Step 3 — Create the Conda environment (installs all dependencies)

The file **`environment.yml`** lists every package the project needs (pandas, pyarrow, streamlit, plotly, scikit-learn, reportlab, etc.). Run this **once**:

```bash
conda env create -f environment.yml
```

This takes 5-10 minutes — it downloads Python 3.10 and ~25 packages into an isolated environment named `nyc-taxi-assessment`.

If you ever need to rebuild from scratch:

```bash
conda env remove -n nyc-taxi-assessment
conda env create -f environment.yml
```

---

### Step 4 — Activate the environment

**Every time you open a new terminal to work on this project, run:**

```bash
conda activate nyc-taxi-assessment
```

Your prompt should now show `(nyc-taxi-assessment)` at the front. Verify:

```bash
python --version     # → Python 3.10.x
```

---

### Step 5 — Add the NYC Taxi data

The 24 Yellow Taxi Parquet files (about 1.8 GB total) are too large to
keep inside the git repository, so they are published as assets of this
repo's `v1.0-data` GitHub Release.  Fetch them with the bundled
downloader:

```bash
python scripts/download_data.py                    # all 24 months
python scripts/download_data.py --year 2024        # only 2024 (12 files)
python scripts/download_data.py --month 2024-06    # a single month
```

The script is resumable — already-downloaded files are skipped — and
streams each file with a progress indicator.

If you prefer to fetch directly from the official source, all files are
also available from the
[NYC TLC Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
Drop them into `data/` following this layout:

```
data/
  taxi_zones.csv                        ← already included in the repo
  yellow_tripdata_2019-01.parquet       ← populated by the downloader
  yellow_tripdata_2019-02.parquet
  ...
  yellow_tripdata_2024-12.parquet
```

You do **not** need every month to test the framework — a single file
(e.g. `yellow_tripdata_2024-06.parquet`) is enough for a demo run.

---

### Step 6 — Run the assessment framework (CLI)

Quick test on a 1 % sample (finishes in seconds):

```bash
python run_all.py --year 2019 --month 1 --sample 0.01
```

Full data for a single month (slower, authoritative):

```bash
python run_all.py --year 2024 --month 1 --full
```

Run a single module only:

```bash
python run_all.py --quality-only        # Junaid's module
python run_all.py --privacy-only        # Sami's module
python run_all.py --security-only       # Jannat's module
```

---

### Step 7 — Launch the interactive dashboard (Iqra's module)

```bash
streamlit run dashboard/Home.py
```

Your browser opens automatically at **http://localhost:8501**.

In the dashboard:

1. Pick a **Year** and **Month** in the left sidebar.
2. Click **"Run Full Assessment"**.
3. Use the page navigation (left sidebar) to drill into Quality, Privacy, Security, or the raw Data Explorer.
4. Scroll to the bottom of **Home** to download a PDF governance report.

To stop the dashboard: press `Ctrl+C` in the terminal.

---

### Step 8 (optional) — Run evaluation experiments

```bash
python evaluation/comparative_analysis.py      # 2019 vs 2024
python evaluation/scalability_benchmarks.py    # O(n) performance
python evaluation/ml_quality_impact.py         # ML with quality defects
python evaluation/manual_validation.py         # Automated vs manual (Cohen's κ)
```

Outputs are written to `outputs/figures/`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `conda: command not found` | Miniconda PATH not set — reopen terminal or use **Anaconda Prompt** (Windows). |
| `ModuleNotFoundError: No module named 'streamlit'` | You forgot `conda activate nyc-taxi-assessment`. |
| `FileNotFoundError: yellow_tripdata_YYYY-MM.parquet` | Parquet file for that month is missing from `data/`. Run `python scripts/download_data.py` to fetch it from the release assets. |
| Streamlit shows a blank page | Check the terminal for errors; stop with `Ctrl+C` and relaunch. |
| PDF export returns plain text | `pip install reportlab` (already in `environment.yml`, but can be force-installed). |

---

## Project Structure

```
nyc-taxi-assessment/
  README.md                          # this file
  .gitignore
  environment.yml                    # Conda spec (Python 3.10) — used in Step 3
  config.py                          # Shared constants, thresholds, paths
  data_loader.py                     # Parquet loading with schema normalisation
  run_all.py                         # CLI entry point for all modules

  data/                              # NYC Taxi Parquet files (gitignored)
    taxi_zones.csv                   # Zone lookup table (tracked in git)

  module1_quality/                   # Junaid Babar — Data Quality Profiling
    quality_profiler.py              # Main orchestrator
    completeness.py                  # Null / missing value analysis
    accuracy.py                      # Outlier and range validation
    consistency.py                   # Fare sum verification, cross-field checks
    timeliness.py                    # Temporal distribution analysis
    visualisations.py                # Chart generation
    report_generator.py              # Text report output
    tests/test_quality.py

  module2_privacy/                   # Sami Ullah — Privacy Risk Detection
    privacy_assessor.py              # Main orchestrator
    pii_classifier.py                # PII field identification
    uniqueness.py                    # Record uniqueness analysis
    k_anonymity.py                   # k-Anonymity measurement
    entropy.py                       # Trajectory and temporal entropy
    linkage_attack.py                # Re-identification simulation
    risk_scorer.py                   # Weighted composite risk
    visualisations.py
    tests/test_privacy.py

  module3_security/                  # Jannat Rafique — Security Compliance
    security_assessor.py             # Main orchestrator
    nist_checklist.py                # NIST CSF 2.0 (6 functions) evaluation
    compliance_matrix.py             # Cross-framework mapping (NIST/GDPR/ISO)
    encryption_checker.py            # File-level encryption scan
    permission_checker.py            # File permission audit
    visualisations.py
    tests/test_security.py

  dashboard/                         # Iqra Aziz — Streamlit Governance Dashboard
    Home.py                          # Executive overview + governance gauge
    pages/
      1_Quality_Assessment.py        # Quality drill-down (Module 1)
      2_Privacy_Risk.py              # Privacy drill-down (Module 2)
      3_Security_Compliance.py       # Security drill-down (Module 3)
      4_Data_Explorer.py             # Raw data exploration
    utils/
      styling.py                     # Shared CSS, colours, Plotly templates
      data_loader.py                 # Cached wrappers for dashboard
      export_reports.py              # PDF report generation

  evaluation/                        # Cross-cutting evaluation
    comparative_analysis.py          # 2019 vs 2024 comparison
    scalability_benchmarks.py        # Performance at 1K-1M rows
    ml_quality_impact.py             # ML experiment with quality defects
    manual_validation.py             # Automated vs manual agreement (Cohen's κ)

  diagrams/                          # Architecture diagrams (.drawio + .png)
  outputs/                           # Generated charts and screenshots
```

---

## How It Works

1. **Data Loading** (`data_loader.py`) — reads Parquet files with schema normalisation across 2019 and 2024 formats.
2. **Module 1: Quality** (Junaid) — profiles data across four dimensions (completeness, accuracy, consistency, timeliness) using vectorised pandas operations.
3. **Module 2: Privacy** (Sami) — evaluates re-identification risk through uniqueness, k-anonymity, entropy, and linkage attack simulation using zone-based quasi-identifiers.
4. **Module 3: Security** (Jannat) — checks compliance against NIST CSF 2.0 (6 functions), GDPR, and ISO 27001 through automated checklists and file-level security scans.
5. **Dashboard** (Iqra) — integrates all three modules into an interactive Streamlit interface with a composite governance score, drill-down pages, and PDF export.

All modules share `config.py` for thresholds and paths, ensuring consistent assessment criteria.

---

## Tech Stack

- **Python 3.10** (Conda-managed)
- **pandas, NumPy, PyArrow, SciPy, scikit-learn** for data processing and ML
- **Streamlit 1.31+** with **Plotly** for the interactive dashboard
- **reportlab** for PDF report generation
- **Dataset:** NYC TLC Yellow Taxi Trip Records (~104 M records across 2019 + 2024)

---

## Key Results

| Metric | Value |
|--------|-------|
| Data Quality Score | 96.8 / 100 |
| Privacy Risk Score | 75.0 / 100 (Critical) |
| Security Compliance | 18.2 % |
| Scalability | O(n) linear — verified 1 K to 1 M rows |
| ML Impact | 12 % RMSE increase with 20 % quality defects |

---

## Academic Context

- **Programme:** MSc IT (Data Analysis), UWS
- **Methodology:** Design Science Research (Hevner et al., 2004)
- **Frameworks:** NIST CSF 2.0, ISO 27001:2022, GDPR
- **Quality model:** Wang and Strong (1996) four-dimension framework
- **Privacy metrics:** de Montjoye et al. (2013), Sweeney (2002)
