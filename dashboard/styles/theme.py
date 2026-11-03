"""Design tokens and centralized theme management (Light & Dark) for the Supply Chain Dashboard."""

from typing import Any, Dict
import streamlit as st

# ---------------------------------------------------------------------------
# Theme Palettes
# ---------------------------------------------------------------------------

THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "name": "dark",
        "bg_app": "#0A1628",
        "bg_sidebar": "#0D1F3C",
        "bg_card": "#111E35",
        "bg_card_secondary": "#162544",
        "border_color": "#1E3A5F",
        "border_subtle": "#172A45",
        "text_primary": "#F1F5F9",
        "text_muted": "#94A3B8",
        "text_dim": "#64748B",
        "accent_blue": "#3B82F6",
        "accent_green": "#10B981",
        "accent_purple": "#8B5CF6",
        "accent_orange": "#F59E0B",
        "danger_red": "#EF4444",
        "plot_bg": "rgba(17, 30, 53, 0.7)",
        "grid_color": "#1E3A5F",
        "hover_border": "#3B82F6",
        "badge_bg_opacity": "0.18",
    },
    "light": {
        "name": "light",
        "bg_app": "#F8FAFC",
        "bg_sidebar": "#FFFFFF",
        "bg_card": "#FFFFFF",
        "bg_card_secondary": "#F1F5F9",
        "border_color": "#E2E8F0",
        "border_subtle": "#EDF2F7",
        "text_primary": "#0F172A",
        "text_muted": "#64748B",
        "text_dim": "#94A3B8",
        "accent_blue": "#2563EB",
        "accent_green": "#059669",
        "accent_purple": "#7C3AED",
        "accent_orange": "#D97706",
        "danger_red": "#DC2626",
        "plot_bg": "#FFFFFF",
        "grid_color": "#E2E8F0",
        "hover_border": "#2563EB",
        "badge_bg_opacity": "0.12",
    },
}

# Default backwards-compatible constants (Dark palette defaults)
NAVY_BG = THEMES["dark"]["bg_app"]
NAVY_SIDEBAR = THEMES["dark"]["bg_sidebar"]
CARD_BG = THEMES["dark"]["bg_card"]
CARD_BORDER = THEMES["dark"]["border_color"]
ACCENT_BLUE = THEMES["dark"]["accent_blue"]
ACCENT_GREEN = THEMES["dark"]["accent_green"]
ACCENT_PURPLE = THEMES["dark"]["accent_purple"]
ACCENT_ORANGE = THEMES["dark"]["accent_orange"]
DANGER_RED = THEMES["dark"]["danger_red"]
TEXT_PRIMARY = THEMES["dark"]["text_primary"]
TEXT_MUTED = THEMES["dark"]["text_muted"]
TEXT_DIM = THEMES["dark"]["text_dim"]

COLOUR_SEQUENCE = [
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    DANGER_RED,
    "#06B6D4",
    "#F472B6",
    "#A3E635",
]

MODEL_COLOURS = {
    "Random Forest": ACCENT_GREEN,
    "Moving Average": ACCENT_BLUE,
    "Naive": ACCENT_ORANGE,
}

CONFIDENCE_COLOURS = {
    "HIGH": ACCENT_GREEN,
    "MEDIUM": ACCENT_ORANGE,
    "LOW": DANGER_RED,
}


def get_current_theme() -> str:
    """Return the active theme name ('dark' or 'light') stored in session state."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    return st.session_state["theme"]


def get_theme_tokens(theme: str | None = None) -> Dict[str, str]:
    """Return dictionary of color and style tokens for the active or specified theme."""
    t = theme or get_current_theme()
    return THEMES.get(t, THEMES["dark"])


def get_plotly_layout(theme: str | None = None) -> Dict[str, Any]:
    """Return dynamic Plotly layout dict matching the selected theme."""
    t = get_theme_tokens(theme)
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["plot_bg"],
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=t["text_primary"],
            size=12,
        ),
        xaxis=dict(
            gridcolor=t["grid_color"],
            linecolor=t["border_color"],
            tickfont=dict(color=t["text_muted"], size=11),
            title=dict(font=dict(color=t["text_muted"], size=12)),
            zerolinecolor=t["grid_color"],
        ),
        yaxis=dict(
            gridcolor=t["grid_color"],
            linecolor=t["border_color"],
            tickfont=dict(color=t["text_muted"], size=11),
            title=dict(font=dict(color=t["text_muted"], size=12)),
            zerolinecolor=t["grid_color"],
        ),
        margin=dict(l=16, r=16, t=44, b=16),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["text_primary"], size=11),
        ),
    )


# Backwards compatible static dictionary
PLOTLY_LAYOUT = get_plotly_layout("dark")
PLOTLY_TEMPLATE = "plotly_dark"


def inject_css(theme: str | None = None) -> None:
    """Inject responsive, theme-aware global CSS variables and enterprise styling."""
    t = get_theme_tokens(theme)
    is_dark = t["name"] == "dark"
    shadow_card = "0 1px 3px 0 rgba(0, 0, 0, 0.3), 0 1px 2px -1px rgba(0, 0, 0, 0.3)" if is_dark else "0 1px 3px 0 rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.04)"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {{
        --bg-app: {t['bg_app']};
        --bg-sidebar: {t['bg_sidebar']};
        --bg-card: {t['bg_card']};
        --bg-card-secondary: {t['bg_card_secondary']};
        --border-color: {t['border_color']};
        --border-subtle: {t['border_subtle']};
        --text-primary: {t['text_primary']};
        --text-muted: {t['text_muted']};
        --text-dim: {t['text_dim']};
        --accent-blue: {t['accent_blue']};
        --accent-green: {t['accent_green']};
        --accent-purple: {t['accent_purple']};
        --accent-orange: {t['accent_orange']};
        --danger-red: {t['danger_red']};
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: var(--bg-app) !important;
        color: var(--text-primary) !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }}

    [data-testid="stSidebar"] * {{
        color: var(--text-primary) !important;
    }}

    /* Main container padding */
    [data-testid="stMain"] > div {{
        padding: 1.5rem 2.5rem;
        max-width: 1400px;
        margin: 0 auto;
    }}

    /* Enterprise Metric Card styling */
    [data-testid="stMetric"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 1rem 1.25rem !important;
        box-shadow: {shadow_card};
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    [data-testid="stMetric"]:hover {{
        border-color: var(--accent-blue) !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }}

    /* Typography hierarchy */
    h1 {{
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.25rem !important;
    }}
    h2 {{
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.5rem !important;
    }}
    h3 {{
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 0.5rem !important;
    }}

    hr {{
        border-color: var(--border-color) !important;
        margin: 1.25rem 0 !important;
    }}

    /* Plotly chart container */
    [data-testid="stPlotlyChart"] {{
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background: var(--bg-card) !important;
        box-shadow: {shadow_card};
    }}

    /* Tables & Dataframes */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        background: var(--bg-card) !important;
        box-shadow: {shadow_card};
    }}

    /* Form Controls & Labels */
    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stSlider"] label,
    [data-testid="stRadio"] label {{
        color: var(--text-muted) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
    }}

    /* Tabs */
    [data-testid="stTabs"] [role="tab"] {{
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        font-size: 0.9rem !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        color: var(--accent-blue) !important;
        border-bottom-color: var(--accent-blue) !important;
        font-weight: 600 !important;
    }}

    /* Alerts */
    [data-testid="stAlert"] {{
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        border: 1px solid var(--border-color) !important;
    }}

    /* Custom enterprise card */
    .ent-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: {shadow_card};
        margin-bottom: 1rem;
    }}

    .ent-card-header {{
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .ent-card-subtitle {{
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }}

    /* Development / Prototype banner */
    .dev-status-banner {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-orange);
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: {shadow_card};
    }}
    .dev-status-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--accent-orange);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }}
    .dev-status-body {{
        font-size: 0.85rem;
        color: var(--text-muted);
        line-height: 1.5;
        margin: 0;
    }}

    /* Badges */
    .badge {{
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}
    .badge-green  {{ background: rgba(16,185,129,{t['badge_bg_opacity']}); color: var(--accent-green); border: 1px solid rgba(16,185,129,0.3); }}
    .badge-blue   {{ background: rgba(59,130,246,{t['badge_bg_opacity']}); color: var(--accent-blue); border: 1px solid rgba(59,130,246,0.3); }}
    .badge-orange {{ background: rgba(245,158,11,{t['badge_bg_opacity']}); color: var(--accent-orange); border: 1px solid rgba(245,158,11,0.3); }}
    .badge-red    {{ background: rgba(239,68,68,{t['badge_bg_opacity']}); color: var(--danger-red); border: 1px solid rgba(239,68,68,0.3); }}
    .badge-muted  {{ background: rgba(148,163,184,{t['badge_bg_opacity']}); color: var(--text-muted); border: 1px solid var(--border-color); }}

    /* Sidebar Brand Area */
    .sidebar-brand {{
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 1.25rem;
    }}
    .sidebar-brand-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .sidebar-brand-subtitle {{
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 2px;
        font-weight: 400;
    }}

    /* Top-Right Theme Switcher */
    .st-key-top_right_theme {{
        position: fixed !important;
        top: 10px !important;
        right: 60px !important;
        z-index: 999999 !important;
        width: 170px !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    .st-key-top_right_theme div[data-testid="stHorizontalBlock"] {{
        gap: 4px !important;
        align-items: center !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 3px 4px !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.15);
    }}

    .st-key-top_right_theme div[data-testid="stColumn"] {{
        min-width: 0 !important;
        padding: 0 !important;
    }}

    .st-key-top_right_theme button {{
        width: 100% !important;
        border: none !important;
        background: transparent !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        padding: 3px 8px !important;
        min-height: 28px !important;
        height: 28px !important;
        border-radius: 6px !important;
        color: var(--text-muted) !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }}

    .st-key-top_right_theme button:hover {{
        color: var(--text-primary) !important;
        background: var(--bg-card-secondary) !important;
    }}

    .st-key-top_right_theme button[kind="primary"] {{
        background: var(--accent-blue) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
