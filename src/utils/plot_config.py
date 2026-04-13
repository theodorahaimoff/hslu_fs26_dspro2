# src/utils/plot_config.py
#
# Shared matplotlib and seaborn configuration for the DSPRO2 project.
# Import this module at the top of any notebook or script that produces charts.
# All visual settings are defined here so plots are consistent across the app
# and the report.
#
# Usage:
#   import sys
#   sys.path.append(str(Path.cwd().parent.parent))  # adjust depth as needed
#   from src.utils.plot_config import COIN_COLORS, apply_plot_style, coin_color

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Brand colors (kept in sync with .streamlit/config.toml)
# ---------------------------------------------------------------------------

PRIMARY     = "#5B6EF5"   # indigo -- primary accent
BORDER      = "#A096FF"   # light purple -- borders and grid lines
BACKGROUND  = "#F9F7FC"   # app background
SURFACE     = "#FFFFFF"   # card surface
TEXT        = "#1A1D2E"   # main text
MUTED       = "#6B7280"   # subtitles and axis labels

# One distinct color per coin. BTC is always the primary brand color.
COIN_COLORS = {
    "BTC": PRIMARY,
    "ETH": "#F49D37",
    "SOL": "#C2E812",
    "XRP": "#00A896",
    "BNB": "#E2C2FF",
    "TRX": "#F76F8E",
}


def coin_color(ticker: str) -> str:
    """
    Returns the brand color for a given coin ticker.
    Falls back to MUTED gray for unknown tickers.
    """
    return COIN_COLORS.get(ticker, MUTED)


# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------

def apply_plot_style() -> None:
    """
    Applies the CryptoLens chart style globally.
    Call this once at the top of each notebook or script before any plotting.
    """
    sns.set_style("white", {
        "axes.facecolor": SURFACE,
        "figure.facecolor": BACKGROUND,
        "grid.color": BORDER,
    })

    plt.rcParams.update({
        # Figure
        "figure.facecolor":       BACKGROUND,
        "figure.dpi":             120,
        "figure.figsize":         (12, 4),

        # Axes
        "axes.facecolor":         SURFACE,
        "axes.edgecolor":         BORDER,
        "axes.linewidth":         0.8,
        "axes.grid":              True,
        "axes.spines.top":        False,
        "axes.spines.right":      False,
        "axes.labelcolor":        MUTED,
        "axes.labelsize":         10,
        "axes.titlesize":         12,
        "axes.titleweight":       "semibold",
        "axes.titlecolor":        TEXT,
        "axes.titlelocation":     "left",

        # Grid
        "grid.color":             BORDER,
        "grid.linewidth":         0.5,
        "grid.alpha":             0.5,

        # Ticks
        "xtick.color":            MUTED,
        "ytick.color":            MUTED,
        "xtick.labelsize":        9,
        "ytick.labelsize":        9,
        "xtick.direction":        "out",
        "ytick.direction":        "out",

        # Legend
        "legend.frameon":         True,
        "legend.framealpha":      0.9,
        "legend.edgecolor":       BORDER,
        "legend.fontsize":        9,

        # Lines
        "lines.linewidth":        1.8,
        "lines.antialiased":      True,

        # Font
        "font.family":            "sans-serif",
        "font.size":              10,
    })