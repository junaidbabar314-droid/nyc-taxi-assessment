"""
Data Explorer page — interactive data browsing and filtering.

Provides a direct view into the NYC Yellow Taxi trip data with filtering
by year, month, payment type, and borough (via zone lookups). Supports
schema inspection, descriptive statistics, and CSV export. Follows
Shneiderman's (1996) mantra: overview first, then zoom and filter.

Author : Iqra Aziz (B01802319)

References:
    Shneiderman, B. (1996) 'The eyes have it: a task by data type taxonomy
        for information visualizations', Proc. IEEE Symp. Visual Languages.
    Nielsen, J. (1994) Usability Engineering. Morgan Kaufmann.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# ─── Path setup ─────────────────────────────────────────────────────────
_PHASE2_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE2_ROOT))

from dashboard.utils.styling import apply_custom_css, PLOTLY_TEMPLATE
from dashboard.utils.data_loader import cached_load_month, cached_load_zones
from config import AVAILABLE_YEARS, MONTHS, COLUMNS, PAYMENT_TYPE_MAP

# ─── Page styling (no set_page_config — only Home.py sets it) ──────────
apply_custom_css()


# ─── Sidebar filters ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Data Explorer")
    year = st.selectbox("Year", AVAILABLE_YEARS, key="de_year")
    month = st.selectbox(
        "Month", list(MONTHS),
        format_func=lambda m: f"{m:02d}",
        key="de_month",
    )
    load_btn = st.button("Load Data", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### Filters")

    # Payment type filter
    payment_options = list(PAYMENT_TYPE_MAP.values())
    payment_filter = st.multiselect(
        "Payment Type",
        options=payment_options,
        default=[],
        key="de_payment",
        help="Filter by payment type. Leave empty for all.",
    )

    # Borough filter (from zones)
    _zones_df = pd.DataFrame()
    try:
        _zones_df = cached_load_zones()
        if not _zones_df.empty and "Borough" in _zones_df.columns:
            boroughs = sorted(_zones_df["Borough"].dropna().unique().tolist())
        else:
            boroughs = []
    except Exception:
        boroughs = []

    borough_filter = st.multiselect(
        "Pickup Borough",
        options=boroughs,
        default=[],
        key="de_borough",
        help="Filter by pickup borough (zone-based). Leave empty for all.",
    )

    # Row limit
    max_rows = st.number_input(
        "Display Rows (max)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        key="de_max_rows",
        help="Maximum rows to display in the table.",
    )


# ─── Main content ──────────────────────────────────────────────────────

def main():
    st.markdown("# Data Explorer")
    st.markdown(
        "Browse and filter NYC Yellow Taxi trip data interactively. "
        "Uses zone IDs (1-265) for location data — no GPS coordinates."
    )
    st.markdown("---")

    # Load data
    df = None
    if load_btn:
        try:
            with st.spinner(f"Loading {year}-{month:02d} data..."):
                df = cached_load_month(year, month)
            st.session_state["explorer_df"] = df
            st.session_state["explorer_year"] = year
            st.session_state["explorer_month"] = month
        except FileNotFoundError:
            st.error(f"Data file not found for {year}-{month:02d}.")
            return
        except Exception as exc:
            st.error(f"Error loading data: {exc}")
            return
    else:
        df = st.session_state.get("explorer_df")

    if df is None:
        st.info("Click **Load Data** in the sidebar to begin exploring.")
        return

    ey = st.session_state.get("explorer_year", "?")
    em = st.session_state.get("explorer_month", "?")
    st.success(f"Loaded {len(df):,} records for {ey}-{em:02d}" if isinstance(em, int) else f"Loaded {len(df):,} records")

    # ── Apply filters ───────────────────────────────────────────────────
    filtered = df.copy()

    # Payment type filter
    if payment_filter:
        reverse_map = {v: k for k, v in PAYMENT_TYPE_MAP.items()}
        selected_codes = [reverse_map[p] for p in payment_filter if p in reverse_map]
        if selected_codes and "payment_type" in filtered.columns:
            filtered = filtered[filtered["payment_type"].isin(selected_codes)]

    # Borough filter via zone lookup
    if borough_filter and not _zones_df.empty:
        zone_ids = _zones_df[_zones_df["Borough"].isin(borough_filter)]["LocationID"].tolist()
        if zone_ids and "PULocationID" in filtered.columns:
            filtered = filtered[filtered["PULocationID"].isin(zone_ids)]

    st.markdown(f"**Showing {len(filtered):,} records** (after filters)")

    # ── Row count display ───────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Filtered Rows", f"{len(filtered):,}")
    with col3:
        st.metric("Columns", len(filtered.columns))
    with col4:
        if len(df) > 0:
            st.metric("Filter Rate", f"{len(filtered)/len(df)*100:.1f}%")

    st.markdown("---")

    # ── Tabs: preview, schema, statistics ───────────────────────────────
    tab_preview, tab_schema, tab_stats = st.tabs([
        "Data Preview", "Schema Information", "Basic Statistics",
    ])

    with tab_preview:
        st.markdown(f"Displaying first {min(max_rows, len(filtered)):,} rows.")
        st.dataframe(
            filtered.head(max_rows),
            use_container_width=True,
            height=500,
        )

    with tab_schema:
        st.markdown("### Column Schema")
        schema_data = []
        for col in filtered.columns:
            non_null = filtered[col].notna().sum()
            null_pct = (1 - non_null / len(filtered)) * 100 if len(filtered) > 0 else 0
            unique = filtered[col].nunique()
            schema_data.append({
                "Column": col,
                "Data Type": str(filtered[col].dtype),
                "Non-Null Count": f"{non_null:,}",
                "Null %": f"{null_pct:.2f}%",
                "Unique Values": f"{unique:,}",
                "Sample Value": str(filtered[col].dropna().iloc[0]) if non_null > 0 else "N/A",
            })
        schema_df = pd.DataFrame(schema_data)
        st.dataframe(schema_df, use_container_width=True, hide_index=True, height=500)

        # Expected columns check
        st.markdown("### Expected Columns")
        expected = set(COLUMNS)
        actual = set(filtered.columns)
        missing = expected - actual
        extra = actual - expected
        if missing:
            st.warning(f"Missing expected columns: {', '.join(sorted(missing))}")
        if extra:
            st.info(f"Additional columns present: {', '.join(sorted(extra))}")
        if not missing and not extra:
            st.success("All expected columns present, no extra columns.")

    with tab_stats:
        st.markdown("### Descriptive Statistics")
        # Only describe numeric columns
        numeric_df = filtered.select_dtypes(include=["number"])
        if not numeric_df.empty:
            desc = numeric_df.describe().T
            desc.index.name = "Column"
            st.dataframe(desc, use_container_width=True, height=400)
        else:
            st.info("No numeric columns available for statistics.")

        # Datetime range
        if "tpep_pickup_datetime" in filtered.columns:
            dt = pd.to_datetime(filtered["tpep_pickup_datetime"], errors="coerce")
            valid_dt = dt.dropna()
            if len(valid_dt) > 0:
                st.markdown("### Temporal Range")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Earliest Pickup", str(valid_dt.min()))
                with col_b:
                    st.metric("Latest Pickup", str(valid_dt.max()))

        # Payment type distribution
        if "payment_type" in filtered.columns:
            st.markdown("### Payment Type Distribution")
            pay_counts = filtered["payment_type"].map(PAYMENT_TYPE_MAP).value_counts()
            st.bar_chart(pay_counts)

    # ── CSV download ────────────────────────────────────────────────────
    st.markdown("---")

    @st.cache_data
    def _to_csv(dataframe: pd.DataFrame) -> bytes:
        return dataframe.to_csv(index=False).encode("utf-8")

    download_df = filtered.head(max_rows)
    csv_bytes = _to_csv(download_df)

    st.download_button(
        label=f"Download Filtered Data ({min(max_rows, len(filtered)):,} rows, CSV)",
        data=csv_bytes,
        file_name=f"taxi_data_{year}_{month:02d}_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


main()
