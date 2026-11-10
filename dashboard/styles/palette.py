"""Color tokens and backward-compatible palette constants."""

THEMES = {
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
        "shadow_card": "0 1px 3px rgba(0,0,0,.30), 0 1px 2px rgba(0,0,0,.30)",
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
        "shadow_card": "0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04)",
    },
}

# Existing modules may import these constants directly.
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
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE,
    DANGER_RED, "#06B6D4", "#F472B6", "#A3E635",
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
