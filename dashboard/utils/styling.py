"""
Dashboard theming and CSS styling utilities.

Provides consistent visual identity across all dashboard pages following
Few's (2006) principles of effective dashboard design: minimal decoration,
clear visual hierarchy, and purposeful use of colour to encode meaning.

References:
    Few, S. (2006) Information Dashboard Design. Analytics Press.
    Nielsen, J. (1994) Usability Engineering. Morgan Kaufmann.
"""

# ─── Colour constants ──────────────────────────────────────────────────
# Consistent palette across all pages; colour encodes semantic meaning
QUALITY_GREEN = "#82b366"
QUALITY_GREEN_DARK = "#6a9a54"
PRIVACY_BLUE = "#6c8ebf"
PRIVACY_BLUE_DARK = "#5a7dad"
SECURITY_RED = "#b85450"
SECURITY_RED_DARK = "#a04440"
WARNING_ORANGE = "#f39c12"
NEUTRAL_GREY = "#95a5a6"
BACKGROUND_DARK = "#1a1a2e"
BACKGROUND_CARD = "#16213e"
TEXT_PRIMARY = "#ecf0f1"
TEXT_SECONDARY = "#bdc3c7"

# Risk-level colour mapping for privacy module
RISK_COLOURS = {
    "Critical": "#b85450",
    "High": "#e67e22",
    "Medium": "#f1c40f",
    "Low": "#82b366",
}

# Compliance score colour mapping for security module
COMPLIANCE_COLOURS = {
    "fail": "#b85450",
    "partial": "#f39c12",
    "pass": "#82b366",
}

# Plotly chart template settings
PLOTLY_TEMPLATE = "plotly_dark"
CHART_FONT = dict(family="Segoe UI, Arial, sans-serif", size=12, color=TEXT_PRIMARY)
CHART_MARGIN = dict(l=40, r=40, t=50, b=40)

# Gauge colour ranges (used in governance score indicator)
GAUGE_STEPS = [
    {"range": [0, 40], "color": "#b85450"},
    {"range": [40, 70], "color": "#f39c12"},
    {"range": [70, 100], "color": "#82b366"},
]


def risk_colour(level: str) -> str:
    """Return hex colour for a given risk level string."""
    return RISK_COLOURS.get(level, NEUTRAL_GREY)


def score_colour(score: float) -> str:
    """Return hex colour for a numeric score (0-100)."""
    if score >= 70:
        return QUALITY_GREEN
    elif score >= 40:
        return WARNING_ORANGE
    return SECURITY_RED


def apply_custom_css():
    """
    Inject custom CSS into the Streamlit app for consistent look and feel.

    Uses st.markdown with unsafe_allow_html to apply styles globally.
    Called once in each page's initialisation block.
    """
    import streamlit as st

    css = """
    <style>
        /* ── Global font & background ─────────────────────────── */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }

        /* ── Metric card styling ──────────────────────────────── */
        div[data-testid="stMetric"] {
            background-color: #16213e;
            border: 1px solid #1f4068;
            border-radius: 10px;
            padding: 15px 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }
        div[data-testid="stMetric"] label {
            color: #bdc3c7;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
        }

        /* ── Sidebar styling ──────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background-color: #0f3460;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stRadio label {
            color: #ecf0f1;
            font-weight: 600;
        }

        /* ── Expander headers ─────────────────────────────────── */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #ecf0f1;
            background-color: #1f4068;
            border-radius: 6px;
        }

        /* ── Tab styling ──────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] button {
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            border-bottom-color: #6c8ebf;
        }

        /* ── Table styling ────────────────────────────────────── */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }

        /* ── Alert boxes ──────────────────────────────────────── */
        .stAlert {
            border-radius: 8px;
        }

        /* ── Download button ──────────────────────────────────── */
        .stDownloadButton > button {
            background-color: #1f4068;
            color: #ecf0f1;
            border: 1px solid #6c8ebf;
            border-radius: 6px;
            font-weight: 600;
        }
        .stDownloadButton > button:hover {
            background-color: #6c8ebf;
            color: #ffffff;
        }

        /* ── Section dividers ─────────────────────────────────── */
        hr {
            border: none;
            border-top: 1px solid #1f4068;
            margin: 1.5rem 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def metric_card_html(label: str, value: str, colour: str = TEXT_PRIMARY) -> str:
    """
    Generate an HTML metric card for use with st.markdown.

    Parameters:
        label:  Metric label text.
        value:  Metric value text.
        colour: Hex colour for the value.

    Returns:
        HTML string for rendering via st.markdown(unsafe_allow_html=True).
    """
    return f"""
    <div style="
        background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
        border: 1px solid #1f4068;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    ">
        <p style="color: #bdc3c7; font-size: 0.8rem; margin: 0 0 5px 0;
                  text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
            {label}
        </p>
        <p style="color: {colour}; font-size: 2rem; margin: 0; font-weight: 700;">
            {value}
        </p>
    </div>
    """
