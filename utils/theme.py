import streamlit as st

# Nord color palette
NORD = {
    # Polar Night (backgrounds)
    "bg_dark": "#2E3440",
    "bg_panel": "#3B4252",
    "bg_hover": "#434C5E",
    "bg_light": "#4C566A",
    # Snow Storm (text)
    "snow_0": "#D8DEE9",
    "snow_1": "#E5E9F0",
    "snow_2": "#ECEFF4",
    # Frost (accents)
    "frost_0": "#8FBCBB",
    "frost_1": "#88C0D0",
    "frost_2": "#81A1C1",
    "frost_3": "#5E81AC",
    # Aurora (semantic)
    "aurora_red": "#BF616A",
    "aurora_orange": "#D08770",
    "aurora_yellow": "#EBCB8B",
    "aurora_green": "#A3BE8C",
    "aurora_purple": "#B48EAD",
}


def plotly_layout_defaults():
    """Return a dict of Plotly layout overrides for consistent Nord theming."""
    return {
        "paper_bgcolor": NORD["bg_dark"],
        "plot_bgcolor": NORD["bg_panel"],
        "font": {"color": NORD["snow_0"], "family": "Inter, sans-serif", "size": 12},
        "xaxis": {
            "gridcolor": "rgba(0,0,0,0)",
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"color": NORD["snow_0"]},
        },
        "yaxis": {
            "gridcolor": "rgba(0,0,0,0)",
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"color": NORD["snow_0"]},
        },
        "colorway": [
            NORD["frost_1"],
            NORD["aurora_green"],
            NORD["aurora_yellow"],
            NORD["aurora_red"],
            NORD["aurora_purple"],
            NORD["aurora_orange"],
            NORD["frost_0"],
            NORD["frost_2"],
        ],
        "margin": dict(l=50, r=20, t=40, b=50),
    }


def inject_custom_css():
    """Inject custom CSS to apply Nord theme overrides to Streamlit."""
    st.markdown(
        f"""
        <style>
        /* ── Global ── */
        .stApp {{
            background-color: {NORD["bg_dark"]};
            color: {NORD["snow_1"]};
            height: 100vh;
            overflow: hidden;
        }}

        /* ── Plotly animation support ── */
        .plotly .scatterlayer .trace .points path {{
            transition: d 0.6s cubic-bezier(0.65, 0, 0.35, 1) !important;
        }}
        .hoverlayer *,
        .plotly .hoverlayer * {{
            transition: none !important;
        }}
        [data-testid="stPlotlyChart"] {{
            contain: layout;
        }}

        /* ── Main container ── */
        .main .block-container {{
            max-width: 100%;
            padding-top: 1rem;
            padding-bottom: 0;
            height: calc(100vh - 1rem);
            overflow-y: auto;
            overflow-x: hidden;
        }}

        [data-testid="stMainBlockContainer"] {{
            border-radius: 16px !important;
            overflow: visible !important;
        }}

        /* ── Sidebar (collapsed) ── */
        section[data-testid="stSidebar"] {{
            background-color: {NORD["bg_panel"]};
        }}

        /* ── Headers ── */
        h1, h2, h3, h4, h5, h6 {{
            color: {NORD["snow_2"]} !important;
        }}

        /* ── Plotly charts with rounded corners ── */
        [data-testid="stPlotlyChart"] {{
            border-radius: 16px !important;
            overflow: hidden !important;
            border: 3px solid {NORD["bg_light"]} !important;
            background-color: {NORD["bg_panel"]} !important;
        }}
        [data-testid="stPlotlyChart"] > div {{
            border-radius: 14px !important;
            overflow: hidden !important;
        }}
        [data-testid="stPlotlyChart"] iframe {{
            border-radius: 14px !important;
        }}
        [data-testid="stPlotlyChart"] .js-plotly-plot {{
            border-radius: 14px !important;
        }}
        [data-testid="stPlotlyChart"] .plotly {{
            border-radius: 14px !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            background-color: {NORD["frost_2"]};
            color: {NORD["snow_2"]};
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.25rem;
            font-weight: 500;
            transition: all 0.25s ease;
        }}
        .stButton > button:hover {{
            background-color: {NORD["frost_3"]};
            color: {NORD["snow_2"]};
            border: none;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(94, 129, 172, 0.3);
        }}

        /* ── Generate button special styling ── */
        .generate-btn .stButton > button {{
            background: linear-gradient(135deg, {NORD["aurora_green"]} 0%, #8faa7a 100%);
            color: {NORD["bg_dark"]};
            font-weight: 700;
            font-size: 1rem;
            padding: 0.6rem 1.5rem;
            border-radius: 10px;
            letter-spacing: 0.02em;
        }}
        .generate-btn .stButton > button:hover {{
            background: linear-gradient(135deg, #8faa7a 0%, {NORD["aurora_green"]} 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(163, 190, 140, 0.4);
        }}

        /* ── Download button ── */
        .stDownloadButton > button {{
            background-color: {NORD["aurora_green"]};
            color: {NORD["bg_dark"]};
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}
        .stDownloadButton > button:hover {{
            background-color: #8faa7a;
            color: {NORD["bg_dark"]};
        }}

        /* ── Select boxes ── */
        .stSelectbox > div > div {{
            background-color: {NORD["bg_panel"]};
            border-color: {NORD["bg_light"]};
            color: {NORD["snow_1"]};
        }}
        .stSelectbox label {{
            color: {NORD["snow_0"]} !important;
            font-size: 0.85rem !important;
        }}

        /* ── Metrics ── */
        [data-testid="stMetric"] {{
            background-color: {NORD["bg_panel"]};
            border: 1px solid {NORD["bg_light"]};
            border-radius: 10px;
            padding: 0.75rem;
        }}
        [data-testid="stMetricLabel"] {{
            color: {NORD["snow_0"]} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {NORD["snow_2"]} !important;
        }}

        /* ── Progress bar ── */
        .stProgress > div > div > div {{
            background-color: {NORD["frost_1"]};
        }}

        /* ── Divider ── */
        hr {{
            border-color: {NORD["bg_light"]};
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {NORD["bg_dark"]};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {NORD["bg_light"]};
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {NORD["frost_3"]};
        }}

        /* ── Hide Streamlit branding ── */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* ── Title area ── */
        .target-title {{
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, {NORD["frost_1"]}, {NORD["frost_0"]});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
            margin-top: 0;
        }}
        .lead-opt-subtitle {{
            font-size: 0.85rem;
            color: {NORD["snow_0"]};
            margin-bottom: 1rem;
            font-family: 'Courier New', monospace;
            word-break: break-all;
        }}
        .lead-opt-label {{
            color: {NORD["frost_2"]};
            font-weight: 600;
            font-family: Inter, sans-serif;
        }}

        /* ── SMILES display ── */
        .smiles-display {{
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            background-color: {NORD["bg_panel"]};
            border: 1px solid {NORD["bg_light"]};
            border-radius: 8px;
            padding: 0.75rem;
            word-break: break-all;
            color: {NORD["snow_1"]};
        }}

        /* ── MolDelta logo ── */
        .moldelta-logo {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {NORD["snow_2"]};
            margin-bottom: 0rem;
            margin-top: 0.5rem;
        }}

        /* ── Enumeration overlay animation ── */
        @keyframes pulse-glow {{
            0%, 100% {{ box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6); }}
            50% {{ box-shadow: 0 8px 32px rgba(94, 129, 172, 0.3); }}
        }}
        .enum-overlay {{
            animation: pulse-glow 2s ease-in-out infinite;
        }}

        /* ── Tagline ── */
        .tagline {{
            font-size: 0.8rem;
            color: {NORD["frost_2"]};
            font-weight: 500;
            letter-spacing: 0.04em;
            margin-bottom: 0.6rem;
            margin-top: -0.1rem;
        }}

        /* ── Step indicator ── */
        .step-indicator {{
            display: flex;
            align-items: center;
            gap: 0.3rem;
            margin-bottom: 0.8rem;
            padding: 0.5rem 0;
        }}
        .step-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.25rem 0.6rem;
            border-radius: 12px;
            transition: all 0.25s ease;
        }}
        .step-badge.active {{
            background: {NORD["frost_2"]};
            color: {NORD["snow_2"]};
        }}
        .step-badge.inactive {{
            background: {NORD["bg_panel"]};
            color: {NORD["bg_light"]};
        }}
        .step-badge.completed {{
            background: {NORD["aurora_green"]};
            color: {NORD["bg_dark"]};
        }}
        .step-arrow {{
            color: {NORD["bg_light"]};
            font-size: 0.7rem;
        }}

        /* ── Control card ── */
        .control-card {{
            background-color: {NORD["bg_panel"]};
            border: 1px solid {NORD["bg_light"]};
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }}

        /* ── Hint text ── */
        .hint-text {{
            font-size: 0.75rem;
            color: {NORD["bg_light"]};
            text-align: center;
            margin-top: 0.4rem;
        }}

        /* ── Smaller tags in multiselect ── */
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
            font-size: 0.68rem;
            max-width: none;
            padding: 0.15rem 0.4rem;
        }}
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] span {{
            font-size: 0.68rem;
            overflow: visible;
            text-overflow: unset;
            white-space: normal;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
