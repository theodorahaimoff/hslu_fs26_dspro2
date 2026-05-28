# src/app/app.py
#
# Main entry point for the CryptoLens Streamlit dashboard.
# Run from the project root with:
#   streamlit run src/app/app.py
#
# ============================================================================
# ARCHITECTURE OVERVIEW
# ============================================================================
#
# Layout
# ------
# The page has a sidebar for global user inputs (coin selection, date range)
# and five content tabs, each focusing on one aspect of the analysis:
#   Tab 1 - Price History      : raw prices and CHF 100 growth simulation
#   Tab 2 - Risk Overview      : volatility, drawdown, daily return spread
#   Tab 3 - Coin Independence  : rolling BTC correlation and independence table
#   Tab 4 - Market Conditions  : HMM regime detection, regime stats table
#   Tab 5 - Price Forecast     : LSTM direction signal and accuracy breakdown
#
# Cards
# -----
# Two types of card are used throughout:
#
#   plotly_card()  -- for charts. Renders a Plotly figure inside an <iframe>
#                     so that Plotly.js is isolated from Streamlit's DOM.
#                     The iframe auto-sizes its height via a postMessage to
#                     Streamlit after the chart is drawn (see _RESIZE_SCRIPT).
#
#   st.markdown()  -- for tables and text-only cards. Uses the .card CSS class
#                     defined in the global <style> block injected at startup.
#
# Hidden-tab rendering problem (and fix)
# ---------------------------------------
# Streamlit renders ALL tab content at page load, even inactive tabs.
# Hidden iframes have width=0, so Plotly draws charts against a zero-width
# figure and clips them. Fix: each iframe uses a ResizeObserver on
# document.body. It fires once when the body grows from width=0 to its real
# width, i.e. exactly when the user switches to that tab. A waitForPlotly
# loop ensures Plotly.js is ready before the resize is triggered. After the
# first correct render the observer disconnects, so subsequent tab switches
# use the already-correct chart unchanged.
#
# Design tokens
# -------------
# All brand colors, font sizes, and spacing live in two places:
#   _CARD_CSS        -- injected into every plotly_card iframe
#   st.markdown CSS  -- global Streamlit stylesheet (injected once at startup)
# Both define the same .card / .card-title / .card-subtitle classes so that
# iframe cards and native st.markdown cards look identical.
# Brand palette:
#   #1A1D2E  dark navy   -- primary text
#   #5B6EF5  indigo      -- accent / logo highlight
#   #A096FF  soft purple -- borders
#   #EDE9F7  light lilac -- subcard backgrounds, badge fills
#   #6B7280  slate grey  -- secondary text / subtitles

import re
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.plot_config import COIN_COLORS, REGIME_COLORS, coin_color, apply_plot_style
from src.app.backend import (
    load_close_prices, filter_close, normalize_to_100,
    load_features_long, compute_risk_summary, get_rolling_volatility,
    compute_max_drawdown, get_log_returns_wide,
    get_coin_independence_table, get_rolling_btc_correlation,
    load_hmm_labels, get_regime_label_map, get_current_regime_info,
    get_regime_summary_table,
    load_lstm_predictions, load_baseline_predictions,
    get_btc_with_signal, get_forecast_accuracy_table,
    get_forecast_accuracy_by_regime,
)

apply_plot_style()

# ---------------------------------------------------------------------------
# Plotly brand defaults
# ---------------------------------------------------------------------------
# _PLOTLY_BASE_LAYOUT is applied to every figure via _apply_brand().
# Keeping it here in one place means a single edit propagates to all charts.

_PLOTLY_BASE_LAYOUT = dict(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="sans-serif", size=11, color="#1A1D2E"),
    xaxis=dict(gridcolor="#A096FF", gridwidth=0.5, linecolor="#A096FF"),
    yaxis=dict(gridcolor="#A096FF", gridwidth=0.5, linecolor="#A096FF"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        bgcolor="rgba(255,255,255,0)",
        borderwidth=0,
        font=dict(size=12),
    ),
    margin=dict(l=55, r=10, t=30, b=40),
    hovermode="x unified",
)


def _apply_brand(fig, yaxis_title: str = "", hovermode: str = "x unified") -> None:
    """Apply CryptoLens brand styling to a Plotly figure in place."""
    layout = dict(_PLOTLY_BASE_LAYOUT)
    layout["hovermode"] = hovermode
    layout["yaxis"] = dict(_PLOTLY_BASE_LAYOUT["yaxis"], title=yaxis_title)
    fig.update_layout(**layout)


# ---------------------------------------------------------------------------
# Card CSS (injected into every plotly_card iframe)
# ---------------------------------------------------------------------------
# Uses the standard system font stack so the iframe resolves to the same
# typeface as Streamlit's own page, keeping iframe and native card headings
# visually consistent. Do not add a Google Fonts @import here -- loading an
# external web font into the iframe will make headings render at a different
# weight and size than the Streamlit-native cards below them.

_CARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
.card {
    background: #FFFFFF;
    border: 1px solid #A096FF;
    border-radius: 10px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1A1D2E;
    margin-bottom: 6px;
}
.card-subtitle {
    font-size: 0.88rem;
    color: #6B7280;
    margin-bottom: 14px;
}
"""

# ---------------------------------------------------------------------------
# Iframe resize script
# ---------------------------------------------------------------------------
# Injected into every plotly_card iframe. Handles two problems:
#
# 1. Auto-height: sendHeight() reads card.scrollHeight and posts it to
#    Streamlit via postMessage so the iframe shrinks or grows to fit its
#    content without a scrollbar.
#
# 2. Hidden-tab chart rendering: waitForPlotlyThenResize() polls until
#    Plotly.js is available, then calls Plotly.Plots.resize() on every
#    chart in the iframe. This corrects charts that were drawn at width=0
#    while the tab was hidden.
#
# Timing:
#   - Two immediate setTimeouts handle Tab 1 which is visible on page load.
#   - A ResizeObserver fires when the iframe body gains real width (i.e.
#     the user switches to a hidden tab) and triggers a resize. It
#     disconnects after the first successful resize so later tab switches
#     leave the chart as-is.
#   - A setInterval fallback serves browsers without ResizeObserver.

_RESIZE_SCRIPT = """
<script>
function sendHeight() {
    var card = document.getElementById('main-card');
    if (card) {
        window.parent.postMessage(
            {type: 'streamlit:setFrameHeight', height: card.scrollHeight + 8}, '*'
        );
    }
}

function doResize() {
    var plots = document.querySelectorAll('.js-plotly-plot');
    if (plots.length && window.Plotly) {
        plots.forEach(function(p) { Plotly.Plots.resize(p); });
    }
    sendHeight();
}

function waitForPlotlyThenResize() {
    if (window.Plotly) {
        doResize();
    } else {
        setTimeout(waitForPlotlyThenResize, 150);
    }
}

// Immediate attempts for Tab 1 which is visible on load.
setTimeout(sendHeight, 150);
setTimeout(waitForPlotlyThenResize, 300);

// ResizeObserver fires once when the iframe body gets its real width
// (i.e. the user switches to this tab). Disconnect after first hit
// so subsequent tab switches leave the chart as-is.
if (window.ResizeObserver) {
    var _ro = new ResizeObserver(function(entries) {
        if (entries[0] && entries[0].contentRect.width > 50) {
            _ro.disconnect();
            setTimeout(waitForPlotlyThenResize, 50);
        }
    });
    _ro.observe(document.body);
} else {
    // Fallback: poll for width > 0 then resize once.
    var _poll = setInterval(function() {
        if (document.body.clientWidth > 50) {
            clearInterval(_poll);
            waitForPlotlyThenResize();
        }
    }, 200);
}
</script>
"""


def plotly_card(
    title: str,
    subtitle: str,
    fig,
    chart_height: int = 380,
    note: str = "",
    min_height: int = 0,
) -> None:
    """
    Render a Plotly figure inside a styled white card using an iframe.

    Parameters
    ----------
    title : str
        Bold heading shown at the top of the card.
    subtitle : str
        Muted description shown below the title. Supports basic HTML (e.g. <br>).
    fig : plotly.graph_objects.Figure
        The Plotly figure to render. Its height is set to chart_height.
    chart_height : int
        Height of the Plotly plot area in pixels. Does not affect the overall
        card height directly -- use min_height for that.
    note : str
        Optional HTML string injected below the chart, inside the card.
        Intended for disclaimers or contextual info that belongs inside the card.
    min_height : int
        CSS min-height applied to the card div. The sendHeight script measures
        card.scrollHeight, so the iframe expands to at least this value.
        Useful for keeping side-by-side cards the same height as their neighbour.
    """
    fig.update_layout(height=chart_height)

    chart_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": False},
    )

    note_html = f"\n  {note}" if note else ""
    card_style = f' style="min-height:{min_height}px;"' if min_height else ""
    note_extra = 80 if note else 0
    # Use the larger of (chart height + fixed overhead) and (explicit min height)
    # as the initial iframe height to avoid a flash of scrollbar before
    # the postMessage auto-size fires.
    frame_height = max(chart_height + 140 + note_extra, min_height + 8)

    html = f"""<!DOCTYPE html>
<html><head><style>{_CARD_CSS}</style></head>
<body>
<div class="card" id="main-card"{card_style}>
  <div class="card-title">{title}</div>
  <div class="card-subtitle">{subtitle}</div>
  {chart_html}{note_html}
</div>
{_RESIZE_SCRIPT}
</body></html>"""

    components.html(html, height=frame_height, scrolling=True)


# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------
# The SVG logo is loaded from disk once and cached. Two sizes are prepared:
#   LOGO_SM (26px) -- used in the sidebar
#   LOGO_LG (36px) -- used in the main page header

@st.cache_data
def load_svg_logo(size: int = 32) -> str:
    svg_path = Path(__file__).resolve().parent / "logo.svg"
    raw = svg_path.read_text(encoding="utf-8")
    raw = re.sub(r'width="[^"]*"', f'width="{size}"', raw)
    raw = re.sub(r'height="[^"]*"', f'height="{size}"', raw)
    raw = raw.replace("<svg ", '<svg style="overflow:visible;display:block;" ', 1)
    return raw.strip()


LOGO_SM = load_svg_logo(size=26)
LOGO_LG = load_svg_logo(size=36)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CryptoLens - Cryptocurrencies made easy",
    page_icon="src/app/logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------
# Injected once into the Streamlit page. Defines:
#   .card / .card-title / .card-subtitle  -- shared with _CARD_CSS so that
#       st.markdown cards and plotly_card iframe cards look identical.
#   .card-table                           -- HTML tables rendered inside cards.
#   .asset-badge                          -- purple pill labels (coin tickers,
#       regime tags). Reused across tabs wherever a short label is needed.
#   .app-logo / .app-tagline              -- main page header typography.
#   .disclaimer                           -- small footer text in the sidebar.
#   Streamlit component overrides         -- custom tab strip, metric cards,
#       sidebar background, and dropdown option hiding.

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1600px;
        }
        .stMarkdown p {
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .app-logo {
            font-size: 1.6rem;
            font-weight: 600;
            color: #1A1D2E;
            letter-spacing: -0.02em;
        }
        .app-logo span { color: #5B6EF5; }
        .app-tagline {
            font-size: 0.9rem;
            color: #6B7280;
            margin-bottom: 1.5rem;
        }
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #A096FF;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            color: #6B7280;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.4rem;
            font-weight: 600;
            color: #1A1D2E;
        }
        /* Custom pill-style tab strip */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #ede9f7;
            border-radius: 10px;
            padding: 4px;
            border: none;
            border-bottom: none !important;
        }
        .stTabs [data-baseweb="tab-border"] { display: none !important; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 7px;
            padding: 7px 18px;
            font-size: 0.85rem;
            font-weight: 500;
            color: #6B7280;
            border: none;
            background-color: transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #1A1D2E !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }
        /* Card component -- used by st.markdown cards and mirrored in _CARD_CSS */
        .card {
            background-color: #FFFFFF;
            border: 1px solid #A096FF;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1A1D2E;
            margin-bottom: 6px;
        }
        .card-subtitle {
            font-size: 0.88rem;
            color: #6B7280;
            margin-bottom: 1rem;
        }
        /* Table rendered inside a card via DataFrame.to_html() */
        .card-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            color: #1A1D2E;
        }
        .card-table thead th {
            text-align: left;
            padding: 6px 12px;
            color: #6B7280;
            font-weight: 500;
            border-bottom: 1px solid #A096FF;
        }
        /* Index cells in the tbody (e.g. regime names, model names) should
         * look like bold content rows, not a second header row. */
        .card-table tbody th {
            text-align: left;
            padding: 6px 12px;
            color: #1A1D2E;
            font-weight: 600;
        }
        .card-table td { padding: 6px 12px; }
        .card-table tr:nth-child(even) td { background-color: #f9f7fc; }
        /* Hide the first two default options Streamlit adds to every dropdown */
        [data-baseweb="menu"] [role="option"]:first-child,
        [data-baseweb="menu"] [role="option"]:nth-child(2) {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #A096FF;
        }
        /* Purple pill label -- used for coin tickers and regime tags */
        .asset-badge {
            display: inline-block;
            background-color: #ede9f7;
            color: #5B6EF5;
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 4px;
        }
        .disclaimer {
            font-size: 0.78rem;
            color: #6B7280;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #A096FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar -- global user inputs
# ---------------------------------------------------------------------------
# selected_coins always starts with "BTC" (the reference coin).
# date_range is a two-element list [start, end] of pd.Timestamps.

COIN_NAMES = {
    "BTC": "Bitcoin (BTC)",
    "ETH": "Ethereum (ETH)",
    "SOL": "Solana (SOL)",
    "XRP": "XRP",
    "BNB": "BNB",
    "TRX": "TRON (TRX)",
}

with st.sidebar:
    st.markdown("**Parameters**")
    st.caption("Bitcoin (BTC) is always shown as the reference coin.")
    selected_coins = st.multiselect(
        label="Cryptocurrencies",
        options=[k for k in COIN_NAMES.keys() if k != "BTC"],
        default=["ETH", "SOL"],
        format_func=lambda x: COIN_NAMES[x],
        help="You can select one or more coins to compare against Bitcoin.",
    )
    selected_coins = ["BTC"] + selected_coins

    date_range = st.date_input(
        label="Time period",
        value=[pd.Timestamp("2020-04-10"), pd.Timestamp("2024-12-31")],
        format="DD.MM.YYYY",
        help="Data goes back to April 2020, when all six coins were available.",
    )

    st.markdown("---")
    st.caption("HSLU Informatik")
    st.caption("DSPRO2 FS26")
    st.caption("T. Saliu / T. Haimoff / N. Singh")
    st.markdown(
        '<p class="disclaimer">CryptoLens was built as part of a university student project '
        "and does not constitute financial advice. Always do your own research before investing.</p>",
        unsafe_allow_html=True,
    )

if len(date_range) < 2:
    st.warning("Please select both a start and end date in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# All data is loaded once at startup. Streamlit caches these calls
# automatically via @st.cache_data decorators in backend.py.
#
#   close_df    -- wide DataFrame of daily closing prices (columns = coins)
#   features_df -- long DataFrame of engineered features (one row per coin/day)
#   hmm_df      -- HMM regime label per date (column: hmm_regime)
#   lstm_df     -- LSTM model predictions (columns: pred, proba)
#   baseline_df -- baseline model predictions for comparison (columns: pred, y_true)
#   label_map   -- dict mapping integer regime id -> human-readable label string

close_df    = load_close_prices()
features_df = load_features_long()
hmm_df      = load_hmm_labels()
lstm_df     = load_lstm_predictions()
baseline_df = load_baseline_predictions()

label_map = get_regime_label_map(hmm_df, features_df)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;'
    f'overflow:visible;padding:4px 0 4px 2px;">'
    f'{LOGO_LG}'
    f'<span class="app-logo" style="font-size:1.9rem;">Crypto<span>Lens</span>.</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-tagline" style="font-size:1rem;">Cryptocurrencies made easy.</p>',
    unsafe_allow_html=True,
)

if selected_coins:
    badges = " ".join(
        f'<span class="asset-badge">{COIN_NAMES[c]}</span>' for c in selected_coins
    )
    st.markdown(f"Showing: {badges}", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_overview, tab_risk, tab_independence, tab_conditions, tab_forecast = st.tabs(
    ["Price History", "Risk Overview", "Coin Independence", "Market Conditions", "Price Forecast"]
)

# ===========================================================================
# TAB 1: Price History
# ===========================================================================
# Shows raw closing prices and a CHF 100 growth simulation for the coins
# selected in the sidebar over the chosen date range.
#
# Backend functions used:
#   filter_close()      -- slices close_df to selected_coins and date_range
#   normalize_to_100()  -- rebases prices so the first value = CHF 100

with tab_overview:

    st.markdown(
        "This tab shows you how the price of each coin has changed over time. "
        "Prices are shown in US dollars (USD)."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Coins available", "6")
    col2.metric("Data available since", "April 2020")
    col3.metric("Data available until", pd.Timestamp.today().strftime("%B %Y"))
    st.markdown("<br>", unsafe_allow_html=True)

    filtered_close = filter_close(close_df, selected_coins, date_range)

    # --- Chart: Raw close prices ---
    fig = go.Figure()
    for coin in filtered_close.columns:
        fig.add_trace(go.Scatter(
            x=filtered_close.index,
            y=filtered_close[coin],
            name=COIN_NAMES[coin],
            line=dict(color=coin_color(coin), width=1.8),
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>" + COIN_NAMES[coin] + "</extra>",
        ))
    _apply_brand(fig, yaxis_title="Price (USD)")
    plotly_card(
        title="How have prices changed over time?",
        subtitle=(
            "Each line shows the price of one coin over the selected time period. "
            "Hover over the chart to see exact values. "
            "Click a coin in the legend to hide or show it."
        ),
        fig=fig,
        chart_height=380,
    )

    # --- Chart: CHF 100 normalized growth ---
    # All coins are rebased to 100 at the start of the selected period so
    # percentage growth can be compared on the same axis regardless of price.
    normalized = normalize_to_100(filtered_close)
    fig = go.Figure()
    fig.add_hline(y=100, line=dict(color="#6B7280", width=0.8, dash="dash"))
    for coin in normalized.columns:
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[coin],
            name=COIN_NAMES[coin],
            line=dict(color=coin_color(coin), width=1.8),
            hovertemplate="%{x|%b %d, %Y}<br>CHF %{y:,.0f}<extra>" + COIN_NAMES[coin] + "</extra>",
        ))
    _apply_brand(fig, yaxis_title="Value of CHF 100 invested (CHF)")
    plotly_card(
        title="If you had invested CHF 100 at the start...",
        subtitle=(
            "This shows how much your investment would be worth today if you had put "
            "CHF 100 into each coin at the beginning of the selected period. "
            "Past performance does not guarantee future results."
        ),
        fig=fig,
        chart_height=380,
    )

# ===========================================================================
# TAB 2: Risk Overview
# ===========================================================================
# Quantifies how risky each coin is using three complementary views:
#   1. Summary table  -- single risk label per coin (Low / Medium / High)
#   2. Rolling volatility chart  -- 30-day annualised volatility over time
#   3. Max drawdown bar chart    -- biggest peak-to-trough drop ever recorded
#   4. Daily log return box plot -- spread and outliers of daily returns
#
# Backend functions used:
#   compute_risk_summary()   -- risk label table
#   get_rolling_volatility() -- 30-day rolling vol per coin per date
#   compute_max_drawdown()   -- worst historical drawdown per coin
#   get_log_returns_wide()   -- daily log returns in wide format

with tab_risk:

    st.markdown(
        "All cryptocurrencies carry risk, but some are much more unpredictable than others. "
        "A coin that jumps up 20% one week and drops 30% the next is considered high risk. "
        "A coin with smaller, steadier swings is lower risk. "
        "This tab helps you understand how much each coin's price moves around."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Table: Risk level summary ---
    risk_df = compute_risk_summary(features_df, selected_coins)
    st.markdown(
        '<div class="card">'
        '<div class="card-title">Risk level per coin</div>'
        '<div class="card-subtitle">'
        "We measure risk by looking at how much a coin's price jumps around on a daily basis. "
        "A coin labeled High Risk can lose or gain a large share of its value in a short time. "
        "A coin labeled Lower Risk is more stable, but still far more volatile than a savings account."
        "</div>"
        + risk_df.to_html(index=False, classes="card-table", border=0)
        + "</div>",
        unsafe_allow_html=True,
    )

    # --- Chart: Rolling volatility ---
    vol_wide = get_rolling_volatility(features_df, selected_coins, date_range)
    fig = go.Figure()
    for coin in vol_wide.columns:
        fig.add_trace(go.Scatter(
            x=vol_wide.index,
            y=vol_wide[coin] * 100,
            name=COIN_NAMES[coin],
            line=dict(color=coin_color(coin), width=1.8),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra>" + COIN_NAMES[coin] + "</extra>",
            showlegend=False,
        ))
    _apply_brand(fig, yaxis_title="30-day rolling volatility (%)")
    plotly_card(
        title="How much has each coin jumped around over time?",
        subtitle=(
            "This chart shows how unstable each coin's price has been over rolling 30-day periods. "
            "Tall spikes mean the coin was going through a very unpredictable phase."
        ),
        fig=fig,
        chart_height=380,
    )

    col_left, col_right = st.columns(2)

    # --- Chart: Max drawdown (horizontal bar) ---
    drawdown_df = compute_max_drawdown(close_df, selected_coins)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[c for c in drawdown_df.index],
        x=drawdown_df["Max Drawdown (%)"].values,
        orientation="h",
        marker_color=[coin_color(c) for c in drawdown_df.index],
        hovertemplate="%{y}<br>%{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#6B7280", width=0.8))
    _apply_brand(fig, yaxis_title="")
    fig.update_layout(
        xaxis_title="Max Drawdown (%)",
        showlegend=False,
        hovermode="y unified",
    )
    with col_left:
        plotly_card(
            title="Worst drops ever recorded",
            subtitle=(
                "This shows the biggest drop each coin has experienced from its highest point. "
                "For example, -80% means the coin lost 80% of its peak value at some point."
            ),
            fig=fig,
            chart_height=300,
        )

    # --- Chart: Daily log return box plot ---
    # Box width represents the typical daily return range (IQR).
    # Individual dots beyond the whiskers are single-day outlier moves.
    returns_wide = get_log_returns_wide(features_df, selected_coins, date_range)
    fig = go.Figure()
    for coin in returns_wide.columns:
        fig.add_trace(go.Box(
            y=returns_wide[coin].dropna().values * 100,
            name=COIN_NAMES[coin],
            marker_color=coin_color(coin),
            line_color=coin_color(coin),
            fillcolor=coin_color(coin),
            opacity=0.6,
            boxpoints="outliers",
            hovertemplate="%{y:.3f}%<extra>" + COIN_NAMES[coin] + "</extra>",
        ))
    fig.add_hline(y=0, line=dict(color="#6B7280", width=0.8, dash="dash"))
    _apply_brand(fig, yaxis_title="Daily log return (%)")
    fig.update_layout(hovermode="closest", showlegend=False)
    with col_right:
        plotly_card(
            title="Typical daily price swings",
            subtitle=(
                "This box chart shows the range of typical daily price changes. "
                "A wider box means more unpredictable daily moves. "
                "Dots are unusually large single-day swings."
            ),
            fig=fig,
            chart_height=300,
        )

# ===========================================================================
# TAB 3: Coin Independence
# ===========================================================================
# Shows how closely each altcoin follows Bitcoin, which determines whether
# holding it alongside BTC actually spreads risk.
#
# Backend functions used:
#   get_coin_independence_table()  -- per-coin independence rating table
#   get_rolling_btc_correlation()  -- 30-day rolling Pearson correlation vs BTC

with tab_independence:

    st.markdown(
        "A common reason people buy multiple cryptocurrencies is to spread their risk. "
        "The idea is that if one coin drops, others might not. "
        "But in practice, most coins tend to rise and fall together when Bitcoin moves. "
        "This tab shows you whether the coins you selected actually behave differently from each other."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Table: Independence ratings ---
    indep_df = get_coin_independence_table(features_df, selected_coins)
    st.markdown(
        '<div class="card">'
        '<div class="card-title">Do these coins move together?</div>'
        '<div class="card-subtitle">'
        "Each coin is rated based on its average 30-day correlation with Bitcoin over the full history. "
        "If a coin mostly moves with Bitcoin, buying it alongside Bitcoin does not spread your risk much."
        "</div>"
        + indep_df.to_html(index=False, classes="card-table", border=0)
        + "</div>",
        unsafe_allow_html=True,
    )

    # --- Chart: Rolling BTC correlation ---
    # If no altcoin is selected the chart cannot be drawn (BTC vs BTC = 100%).
    # In that case a placeholder card is shown instead.
    corr_wide = get_rolling_btc_correlation(features_df, selected_coins, date_range)
    if corr_wide.empty:
        st.markdown(
            '<div class="card">'
            '<div class="card-title">How similar have they been over time?</div>'
            '<div class="card-subtitle">'
            "This chart shows how closely each coin has followed Bitcoin over time."
            "</div>"
            '<p style="color:#6B7280;font-size:0.88rem;">'
            "Select at least one coin other than Bitcoin to see this chart."
            "</p></div>",
            unsafe_allow_html=True,
        )
    else:
        fig = go.Figure()
        fig.add_hline(
            y=80,
            line=dict(color="#6B7280", width=0.8, dash="dash"),
            annotation_text="80% threshold",
            annotation_position="bottom right",
        )
        for coin in corr_wide.columns:
            fig.add_trace(go.Scatter(
                x=corr_wide.index,
                y=corr_wide[coin],
                name=COIN_NAMES[coin],
                line=dict(color=coin_color(coin), width=1.8),
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f}%<extra>" + COIN_NAMES[coin] + "</extra>",
                showlegend=False,
            ))
        _apply_brand(fig, yaxis_title="30-day BTC correlation (%)")
        plotly_card(
            title="How similar have they been over time?",
            subtitle=(
                "This chart shows how closely each coin has followed Bitcoin over time. "
                "A value near 100% means the coin moves almost exactly with Bitcoin."
            ),
            fig=fig,
            chart_height=380,
        )

# ===========================================================================
# TAB 4: Market Conditions
# ===========================================================================
# Uses Hidden Markov Model (HMM) regime labels to classify each trading day
# as either High Volatility or Low Volatility / Trending. Shows the current
# regime, a historical chart with colored regime bands, and an average stats
# table per regime.
#
# Backend functions used:
#   get_current_regime_info()  -- current regime label and how long it has lasted
#   get_regime_summary_table() -- avg daily return and avg volatility per regime

with tab_conditions:

    st.markdown(
        "The crypto market changes over time. Sometimes prices rise steadily (uptrend), sometimes they fall (downtrend), and sometimes they are unpredictable. Understanding the current phase can help you decide when might be a good time to buy or wait."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    regime_info = get_current_regime_info(hmm_df, label_map)
    col1, col2 = st.columns(2)
    col1.metric("Current market phase", regime_info["label"])
    col2.metric("Phase duration so far", f"{regime_info['duration_days']} days")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Chart: BTC price with regime bands ---
    # Approach: iterate over the HMM label series and draw a vrect for each
    # consecutive block of the same regime. The BTC price trace is added first
    # so Plotly sets the x-axis type to datetime before the vrects are drawn.
    btc_close = close_df["BTC"].dropna()
    regime_dates = hmm_df.index
    regime_labels_series = hmm_df["hmm_regime"].map(label_map)

    fig = go.Figure()

    # BTC price trace FIRST so Plotly sets x-axis type to datetime
    # before the vrects are drawn against it.
    fig.add_trace(go.Scatter(
        x=btc_close.index,
        y=btc_close.values,
        name="BTC Price",
        line=dict(color=coin_color("BTC"), width=1.5),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>BTC Price</extra>",
        showlegend=False,
    ))

    block_start = regime_dates[0]
    block_regime = regime_labels_series.iloc[0]
    for i in range(1, len(regime_dates)):
        current = regime_labels_series.iloc[i]
        if current != block_regime or i == len(regime_dates) - 1:
            fig.add_vrect(
                x0=block_start,
                x1=regime_dates[i],
                fillcolor=REGIME_COLORS.get(block_regime, "#cccccc"),
                opacity=0.15,
                line_width=0,
                showlegend=False,
            )
            block_start = regime_dates[i]
            block_regime = current

    # Invisible scatter traces so regime colors appear in the legend.
    for label, color in REGIME_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color, opacity=0.4, symbol="square"),
            name=label,
            showlegend=False,
        ))

    _apply_brand(fig, yaxis_title="BTC Price (USD)", hovermode="x")
    plotly_card(
        title="Bitcoin price with market phases highlighted",
        subtitle=(
            "The colored bands show which phase the market was in at each point in time. <br>"
            "Blue line = BTC price, Green = Low Volatility / Trending, Red = High Volatility."
        ),
        fig=fig,
        chart_height=400,
    )

    # --- Table: Regime summary ---
    summary_df = get_regime_summary_table(hmm_df, features_df, label_map)
    summary_display = summary_df.copy()
    summary_display.index.name = None
    summary_display["Avg Daily Return (%)"] = summary_display["Avg Daily Return (%)"].apply(
        lambda x: f"{x:.3f}%"
    )
    summary_display["Avg Volatility (%)"] = summary_display["Avg Volatility (%)"].apply(
        lambda x: f"{x:.3f}%"
    )
    st.markdown(
        '<div class="card">'
        '<div class="card-title">What typically happens in each phase?</div>'
        '<div class="card-subtitle">'
        "This table shows the average daily price change and typical price movement "
        "during each market phase, based on historical data."
        "</div>"
        + summary_display.to_html(index=True, classes="card-table", border=0)
        + "</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 5: Price Forecast
# ===========================================================================
# Shows the output of the LSTM direction model: which days it predicted up
# or down, how often it was correct, how it compares to simple baselines,
# and how its accuracy breaks down by HMM market regime.
#
# Backend functions used:
#   get_btc_with_signal()          -- full BTC price series + test-period signals
#   get_forecast_accuracy_table()  -- accuracy / F1 for LSTM vs baselines
#   get_forecast_accuracy_by_regime() -- per-regime accuracy breakdown

with tab_forecast:

    st.markdown(
        "Based on past price patterns, our model gives a simple prediction of whether Bitcoin's price is likely to go up or down over the next 30 days. It looks at factors like recent returns, volatility, and market trends to make this estimate."
    )

    st.info(
        "This prediction indicates the direction (up or down), not the exact price. It doesn't consider news, regulations, or unexpected events. Use it as a general guideline, not a certainty.",
        icon=None,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Chart: BTC price with direction signal ---
    # Full price history shown as a line. Test-period predictions are overlaid
    # as colored scatter dots (green = predicted up, red = predicted down).
    # The note inside the card flags that the model is BTC-only.
    btc_full, btc_test, signal, proba = get_btc_with_signal(close_df, lstm_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=btc_full.index,
        y=btc_full.values,
        name="BTC Price",
        line=dict(color=coin_color("BTC"), width=1.5),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>BTC Price</extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=btc_test.index[signal == 1],
        y=btc_test.values[signal == 1],
        mode="markers",
        name="Predicted up",
        marker=dict(color=REGIME_COLORS["Low Volatility / Trending"], size=6),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>Predicted up</extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=btc_test.index[signal == 0],
        y=btc_test.values[signal == 0],
        mode="markers",
        name="Predicted down",
        marker=dict(color=REGIME_COLORS["High Volatility"], size=6),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>Predicted down</extra>",
        showlegend=False,
    ))
    _apply_brand(fig, yaxis_title="BTC Price (USD)", hovermode="x")

    # Note rendered inside the chart card below the plot.
    # Uses the same inline SVG info-icon style as the regime subcard notes.
    _BTC_NOTE = (
        '<div style="margin-top:14px; padding-top:12px; border-top:1px solid #EDE9F7;'
        ' display:flex; gap:8px; align-items:flex-start;">'
        '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"'
        ' style="flex-shrink:0; margin-top:2px;">'
        '<circle cx="7" cy="7" r="6.25" stroke="#6B7280" stroke-width="1.25"/>'
        '<text x="7" y="10.8" text-anchor="middle" font-size="8"'
        ' fill="#6B7280" font-family="sans-serif" font-style="italic">i</text>'
        '</svg>'
        '<div style="font-size:0.82rem; color:#6B7280; line-height:1.5;">'
        "This model was trained on Bitcoin data only. "
        "A directional prediction for other coins is not available."
        "</div>"
        "</div>"
    )

    plotly_card(
        title="Bitcoin price with model direction signal",
        subtitle=(
            "This chart shows Bitcoin\u2019s recent price (blue line). The colored dots during the test period (from December 2024 onward) show the model\u2019s prediction: <br>"
            "Green = predicted up over next 30 days, Red = predicted down."
        ),
        fig=fig,
        chart_height=400,
        note=_BTC_NOTE,
    )

    col_left, col_right = st.columns(2)

    # --- Chart: Predicted vs actual direction (left column) ---
    # Each day in the test period is a bar: green if the prediction was correct,
    # red if it was wrong. min_height matches the right column card so both
    # cards are the same height regardless of content.
    y_true = baseline_df["y_true"]
    lstm_preds = lstm_df["pred"].reindex(y_true.index).dropna().astype(int)
    y_true_aligned = y_true.reindex(lstm_preds.index)
    correct = (lstm_preds == y_true_aligned).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=correct.index[correct == 1],
        y=np.ones(correct.sum()),
        name="Correct",
        marker_color=REGIME_COLORS["Low Volatility / Trending"],
        opacity=0.7,
        hovertemplate="%{x|%b %d, %Y}<extra>Correct</extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Bar(
        x=correct.index[correct == 0],
        y=np.ones((correct == 0).sum()),
        name="Incorrect",
        marker_color=REGIME_COLORS["High Volatility"],
        opacity=0.7,
        hovertemplate="%{x|%b %d, %Y}<extra>Incorrect</extra>",
        showlegend=False,
    ))
    _apply_brand(fig, yaxis_title="")
    fig.update_layout(yaxis=dict(visible=False), barmode="overlay")
    with col_left:
        plotly_card(
            title="How accurate has this model been in the past?",
            subtitle=(
                "This chart shows how often the model correctly predicted whether Bitcoin\u2019s price went up or down each day in the past. <br>"
                "Green = the model predicted the correct direction, Red = incorrect."
            ),
            fig=fig,
            chart_height=300,
            min_height=470,
        )

    # --- Table: Accuracy comparison (right column) ---
    # Compares LSTM against three baselines: majority class, momentum/persistence,
    # and logistic regression. min-height keeps both columns the same height.
    accuracy_df = get_forecast_accuracy_table(lstm_df, baseline_df)
    accuracy_display = accuracy_df.copy()
    accuracy_display.index.name = None
    accuracy_display["Accuracy (%)"] = accuracy_display["Accuracy (%)"].apply(
        lambda x: f"{x:.1f}%"
    )
    with col_right:
        st.markdown(
            '<div class="card" style="min-height: 470px;">'
            '<div class="card-title">How does the model compare to simple alternatives?</div>'
            '<div class="card-subtitle">'
            "This table compares how well different methods predict Bitcoin\u2019s price movement:"
            "The accuracy shows how often the method was correct and the F1 score measures how balanced the predictions are between rising and falling prices."
            "Higher numbers mean better predictions. "
            "</div>"
            + accuracy_display.to_html(index=True, classes="card-table", border=0)
            + "</div>",
            unsafe_allow_html=True,
        )

    # --- Per-regime accuracy breakdown ---
    # One outer card with the question as the heading, two subcards inside
    # (one per HMM regime). Regime 0 gets an inline info note because its
    # sample size is small and its high accuracy may not be reliable.
    st.markdown("<br>", unsafe_allow_html=True)

    regime_acc_df = get_forecast_accuracy_by_regime(lstm_df, baseline_df)
    n_regime0 = int(regime_acc_df.loc[0, "n"]) if 0 in regime_acc_df.index else 0

    # Reusable inline SVG info-icon. Rendered in muted grey to match the
    # card-subtitle color so it reads as part of the note text, not a CTA.
    _INFO_ICON = (
        '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" '
        'style="flex-shrink:0; margin-top:2px;">'
        '<circle cx="7" cy="7" r="6.25" stroke="#6B7280" stroke-width="1.25"/>'
        '<text x="7" y="10.8" text-anchor="middle" font-size="8" '
        'fill="#6B7280" font-family="sans-serif" font-style="italic">i</text>'
        '</svg>'
    )

    # Build a subcard HTML string for each regime. Regime 0 gets an extra
    # note div below the stats to warn that the sample is too small to
    # treat the accuracy as a confirmed edge.
    subcard_html = ""
    for regime_id, row in regime_acc_df.iterrows():
        regime_name = label_map.get(int(regime_id), f"Regime {regime_id}")
        note_html = ""
        if int(regime_id) == 0:
            note_html = (
                '<div style="margin-top:14px; padding-top:12px; '
                'border-top:1px solid #EDE9F7; display:flex; gap:8px; align-items:flex-start;">'
                + _INFO_ICON
                + '<div style="font-size:0.82rem; color:#6B7280; line-height:1.5;">'
                f"Uncommon phase &mdash; only {n_regime0} test days, too few to confirm "
                "a real edge. If the pattern holds it would be a high-value signal. "
                "Until more data accumulates, treat it as a lead, not a proven opportunity."
                "</div>"
                "</div>"
            )
        subcard_html += (
            '<div style="background:#F9F7FC; border:1px solid #EDE9F7; '
            'border-radius:8px; padding:16px 18px; flex:1; min-width:0;">'
            f'<div class="asset-badge" style="margin-bottom:8px;">Regime {regime_id}</div>'
            f'<div class="card-title">{regime_name}</div>'
            '<div class="card-subtitle">'
            f"Model performance on the {int(row['n'])} test days "
            "the market spent in this phase."
            "</div>"
            '<div style="display:flex; gap:40px; margin-top:10px;">'
            '<div>'
            '<div style="font-size:1.9rem; font-weight:700; color:#1A1D2E;">'
            f"{row['accuracy']:.1f}%</div>"
            '<div style="font-size:0.8rem; color:#6B7280;">Accuracy</div>'
            '</div>'
            '<div>'
            '<div style="font-size:1.9rem; font-weight:700; color:#1A1D2E;">'
            f"{row['f1_macro']:.3f}</div>"
            '<div style="font-size:0.8rem; color:#6B7280;">F1 macro</div>'
            '</div>'
            '</div>'
            + note_html
            + '</div>'
        )

    st.markdown(
        '<div class="card">'
        '<div class="card-title">'
        "Does the model perform equally well in every market condition?"
        "</div>"
        '<div class="card-subtitle">'
        "It does not, and that is one of its more interesting findings. The cards below "
        "split the model\u2019s test-set accuracy by market phase, the same calm and volatile "
        "phases shown in the Market Conditions tab."
        "</div>"
        '<div style="display:flex; gap:12px; margin-top:4px;">'
        + subcard_html
        + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )