"""Shared Graphite + Emerald tokens for native light and dark appearances."""

HISTORICAL_DEMAND = "#527D69"
FORECAST = "#21A179"
COMPARISON = "#8BC9A5"
HEALTHY = "#21A179"
WARNING = "#E5A642"
CRITICAL = "#D95D5D"
SECONDARY_COMPARISON = "#8B83F6"

THEMES = {
    "light": {
        "name": "light",
        "bg_app": "#F4F7F5",
        "bg_sidebar": "#FFFFFF",
        "bg_card": "#FFFFFF",
        "bg_card_secondary": "#E8EFEB",
        "border_color": "#D9E3DD",
        "border_subtle": "#D9E3DD",
        "text_primary": "#17231E",
        "text_muted": "#718078",
        "text_dim": "#718078",
        "text_sidebar": "#1D3329",
        "sidebar_accent": "#1D3329",
        "primary_accent": FORECAST,
        "accent_blue": HISTORICAL_DEMAND,
        "accent_green": HEALTHY,
        "accent_purple": SECONDARY_COMPARISON,
        "accent_orange": WARNING,
        "highlight_gold": "#1D3329",
        "danger_red": CRITICAL,
        "historical_demand": HISTORICAL_DEMAND,
        "forecast": FORECAST,
        "comparison": COMPARISON,
        "healthy": HEALTHY,
        "warning": WARNING,
        "critical": CRITICAL,
        "secondary_comparison": SECONDARY_COMPARISON,
        "button_primary_text": "#17231E",
        "plot_bg": "#FFFFFF",
        "grid_color": "#D9E3DD",
        "hover_border": FORECAST,
        "shadow_card": "0 1px 3px rgba(23,35,30,.10), 0 1px 2px rgba(23,35,30,.06)",
        "heatmap_low": "#F4F7F5",
        "heatmap_mid": COMPARISON,
        "heatmap_high": HISTORICAL_DEMAND,
        "fill_blue": "rgba(82,125,105,.12)",
        "fill_purple": "rgba(139,131,246,.16)",
        "chart_series": [
            HISTORICAL_DEMAND,
            FORECAST,
            COMPARISON,
            SECONDARY_COMPARISON,
        ],
    },
    "dark": {
        "name": "dark",
        "bg_app": "#171C1B",
        "bg_sidebar": "#111715",
        "bg_card": "#222B28",
        "bg_card_secondary": "#34443C",
        "border_color": "#3A4840",
        "border_subtle": "#3A4840",
        "text_primary": "#F3F6F3",
        "text_muted": "#A2B0A7",
        "text_dim": "#A2B0A7",
        "text_sidebar": "#F3F6F3",
        "sidebar_accent": FORECAST,
        "primary_accent": FORECAST,
        "accent_blue": HISTORICAL_DEMAND,
        "accent_green": HEALTHY,
        "accent_purple": SECONDARY_COMPARISON,
        "accent_orange": WARNING,
        "highlight_gold": FORECAST,
        "danger_red": CRITICAL,
        "historical_demand": HISTORICAL_DEMAND,
        "forecast": FORECAST,
        "comparison": COMPARISON,
        "healthy": HEALTHY,
        "warning": WARNING,
        "critical": CRITICAL,
        "secondary_comparison": SECONDARY_COMPARISON,
        "button_primary_text": "#171C1B",
        "plot_bg": "#222B28",
        "grid_color": "#3A4840",
        "hover_border": FORECAST,
        "shadow_card": "0 1px 3px rgba(0,0,0,.30), 0 1px 2px rgba(0,0,0,.22)",
        "heatmap_low": "#222B28",
        "heatmap_mid": COMPARISON,
        "heatmap_high": FORECAST,
        "fill_blue": "rgba(82,125,105,.20)",
        "fill_purple": "rgba(139,131,246,.20)",
        "chart_series": [
            HISTORICAL_DEMAND,
            FORECAST,
            COMPARISON,
            SECONDARY_COMPARISON,
        ],
    },
}

# Backward-compatible names used by existing dashboard imports.
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

COLOUR_SEQUENCE = THEMES["dark"]["chart_series"]
MODEL_COLOURS = {
    "Random Forest": THEMES["dark"]["forecast"],
    "Moving Average": THEMES["dark"]["comparison"],
    "Naive": THEMES["dark"]["historical_demand"],
}
CONFIDENCE_COLOURS = {
    "HIGH": THEMES["dark"]["healthy"],
    "MEDIUM": THEMES["dark"]["warning"],
    "LOW": THEMES["dark"]["critical"],
}
