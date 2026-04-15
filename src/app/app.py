# src/app/app.py
#
# Main entry point for the Streamlit dashboard.
# Run from the project root with:
#   streamlit run src/app/app.py
#
# Each tab corresponds to one analysis module. Placeholder sections
# are marked with TODO so they can be filled in as each modeling
# notebook is completed.

import re, sys
import streamlit as st
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.plot_config import COIN_COLORS, coin_color, apply_plot_style

apply_plot_style()

# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------
# The SVG is read from disk and inlined directly into HTML. Streamlit cannot
# serve static files by path reference so inlining is the reliable approach.
# Inkscape-specific elements (sodipodi, namedview) are stripped so only
# the visible path survives.

@st.cache_data
def load_svg_logo(size: int = 32) -> str:
    """
    Reads logo.svg from the project root, strips Inkscape metadata,
    and returns a clean inline SVG string at the requested pixel size.
    """
    svg_path = Path(__file__).resolve().parent / "logo.svg"
    raw = svg_path.read_text(encoding="utf-8")

    raw = re.sub(r'width="[^"]*"', f'width="{size}"', raw)
    raw = re.sub(r'height="[^"]*"', f'height="{size}"', raw)

    raw = raw.replace("<svg ", '<svg style="overflow:visible;display:block;" ', 1)
    return raw.strip()


LOGO_SM = load_svg_logo(size=26)   # sidebar
LOGO_LG = load_svg_logo(size=36)   # main header

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
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /*
         * Font, base colors, and heading sizes are set in .streamlit/config.toml.
         * This block only contains structural styles that Streamlit's config
         * system does not support: layout width, spacing, border-radius,
         * box shadows, and custom component shapes (tab pill, cards, badges).
         *
         * Color reference (kept in sync with config.toml):
         *   backgroundColor:           #f9f7fc
         *   secondaryBackgroundColor:  #FFFFFF
         *   primaryColor:              #5B6EF5
         *   textColor:                 #1A1D2E
         *   borderColor:               #A096FF
         *   muted text (not in config): #6B7280
         */

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* App header */
        .app-logo {
            font-size: 1.6rem;
            font-weight: 600;
            color: #1A1D2E;
            letter-spacing: -0.02em;
        }
        .app-logo span {
            color: #5B6EF5;
        }
        .app-tagline {
            font-size: 0.9rem;
            color: #6B7280;
            margin-bottom: 1.5rem;
        }

        /* Metric cards */
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

        /* Tab strip -- pill shape cannot be set via config.toml */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #ede9f7;
            border-radius: 10px;
            padding: 4px;
            border: none;
            border-bottom: none !important;
        }
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }
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

        /* Section cards */
        .card {
            background-color: #FFFFFF;
            border: 1px solid #A096FF;
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .card-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #1A1D2E;
            margin-bottom: 4px;
        }
        .card-subtitle {
            font-size: 0.82rem;
            color: #6B7280;
            margin-bottom: 1rem;
        }

        /* Placeholder boxes */
        .placeholder-box {
            background-color: #f9f7fc;
            border: 1.5px dashed #A096FF;
            border-radius: 8px;
            padding: 36px 24px;
            text-align: center;
            color: #6B7280;
            font-size: 0.85rem;
            line-height: 1.7;
        }
        .placeholder-box code {
            display: block;
            margin-top: 8px;
            font-size: 0.78rem;
            color: #5B6EF5;
            background: #ede9f7;
            padding: 4px 10px;
            border-radius: 4px;
            width: fit-content;
            margin: 8px auto 0;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #A096FF;
        }

        /* Asset badge */
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

        /* Disclaimer */
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
# Sidebar
# ---------------------------------------------------------------------------

# The rolling window is fixed at 30 days and not exposed to the user.
# It is used internally by the plotting functions in each tab.
ROLLING_WINDOW = 30

COIN_NAMES = {
    "BTC": "Bitcoin (BTC)",
    "ETH": "Ethereum (ETH)",
    "SOL": "Solana (SOL)",
    "XRP": "XRP",
    "BNB": "BNB",
    "TRX": "TRON (TRX)",
}

with st.sidebar:
    # st.markdown(
    #     f'<div style="display:flex;align-items:center;gap:9px;padding:2px 0;">'
    #     f'{LOGO_SM}'
    #     f'<span class="app-logo">Crypto<span>Lens</span>.</span>'
    #     f'</div>',
    #     unsafe_allow_html=True,
    # )
    # st.markdown(
    #     '<div class="app-tagline">Cryptocurrencies made easy.</div>',
    #     unsafe_allow_html=True,
    # st.markdown("---")
    # )


    st.markdown("**Parameters**")
    st.caption("Bitcoin (BTC) is always shown as the reference coin.")
    selected_coins = st.multiselect(
        label="Cryptocurrencies",
        options=[k for k in COIN_NAMES.keys() if k != "BTC"],
        default=["ETH", "SOL"],
        format_func=lambda x: COIN_NAMES[x],
        help="You can select one or more coins to compare against Bitcoin.",
    )

    # BTC is always included as the fixed reference coin.
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
        '<p class="disclaimer">CryptoLens was built as part of a university student project and does not '
        "constitute financial advice. Always do your own research before investing.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;overflow:visible;padding:4px 0 4px 2px;">'
    f'{LOGO_LG}'
    f'<span class="app-logo" style="font-size:1.9rem;">Crypto<span>Lens</span>.</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-tagline" style="font-size:1rem;">'
    "Cryptocurrencies made easy.</p>",
    unsafe_allow_html=True,
)

if selected_coins:
    badges = " ".join(
        f'<span class="asset-badge">{COIN_NAMES[c]}</span>' for c in selected_coins
    )
    st.markdown(f"Showing: {badges}", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_risk, tab_independence, tab_conditions, tab_forecast = st.tabs(
    [
        "Price History",
        "Risk Overview",
        "Coin Independence",
        "Market Conditions",
        "Price Forecast",
    ]
)

# ===========================================================================
# TAB 1: Price History
# ===========================================================================

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

    st.markdown(
        '<div class="card">'
        '<div class="card-title">How have prices changed over time?</div>'
        '<div class="card-subtitle">'
        "Each line shows the price of one coin over the selected time period. "
        "Zoom in on specific periods by adjusting the date range in the sidebar."
        "</div>"
        # TODO: load data/processed/crypto_close_wide_full.csv, filter by selected_coins and date_range, plot with st.line_chart
        '<div class="placeholder-box">Price chart will appear here</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card">'
        '<div class="card-title">If you had invested CHF 100 at the start...</div>'
        '<div class="card-subtitle">'
        "This shows how much your investment would be worth today if you had put "
        "CHF 100 into each coin at the beginning of the selected period. "
        "Past performance does not guarantee future results."
        "</div>"
        # TODO: normalize close prices to 100 at start date, filter by selected_coins and date_range, plot with st.line_chart
        '<div class="placeholder-box">Investment growth chart will appear here</div>'
        "</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 2: Risk Overview
# ===========================================================================

with tab_risk:

    st.markdown(
        "All cryptocurrencies carry risk, but some are much more unpredictable than others. "
        "A coin that jumps up 20% one week and drops 30% the next is considered high risk. "
        "A coin with smaller, steadier swings is lower risk. "
        "This tab helps you understand how much each coin's price moves around."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="card">'
        '<div class="card-title">Risk level per coin</div>'
        '<div class="card-subtitle">'
        "We measure risk by looking at how much a coin's price jumps around on a daily basis. "
        "A coin labeled High Risk can lose or gain a large share of its value in a short time. "
        "A coin labeled Lower Risk is more stable, but still far more volatile than a savings account."
        "</div>"
        # TODO: load rolling volatility from crypto_features_long.csv, compute average per coin,
        # map to Low/Medium/High buckets, display as a simple table or bar chart with st.bar_chart
        '<div class="placeholder-box">Risk level summary will appear here</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card">'
        '<div class="card-title">How much has each coin jumped around over time?</div>'
        '<div class="card-subtitle">'
        "This chart shows how unstable each coin's price has been over rolling 30-day periods. "
        "Tall spikes mean the coin was going through a very unpredictable phase. "
        "Flat periods mean the price was more stable."
        "</div>"
        # TODO: load rolling_volatility column from crypto_features_long.csv,
        # filter by selected_coins and date_range, plot with st.line_chart
        '<div class="placeholder-box">Price stability chart will appear here</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            '<div class="card">'
            '<div class="card-title">Worst drops ever recorded</div>'
            '<div class="card-subtitle">'
            "This shows the biggest drop each coin has experienced from its highest point. "
            "For example, a value of -80% means the coin lost 80% of its peak value at some point."
            "</div>"
            # TODO: compute max drawdown per coin from close prices, display as horizontal bar chart
            '<div class="placeholder-box">Biggest drop chart will appear here</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            '<div class="card">'
            '<div class="card-title">Typical daily price swings</div>'
            '<div class="card-subtitle">'
            "This box chart shows the range of typical daily price changes. "
            "A wider box means more unpredictable daily moves. "
            "Dots outside the box are unusually large single-day swings."
            "</div>"
            # TODO: load log_return column from crypto_features_long.csv,
            # filter by selected_coins, plot as box plot with matplotlib and st.pyplot
            '<div class="placeholder-box">Daily swings chart will appear here</div>'
            "</div>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB 3: Coin Independence
# ===========================================================================

with tab_independence:

    st.markdown(
        "A common reason people buy multiple cryptocurrencies is to spread their risk. "
        "The idea is that if one coin drops, others might not. "
        "But in practice, most coins tend to rise and fall together when Bitcoin moves. "
        "This tab shows you whether the coins you selected actually behave differently from each other."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="card">'
        '<div class="card-title">Do these coins move together?</div>'
        '<div class="card-subtitle">'
        "Each coin is rated from Moves with Bitcoin to Acts Independently. "
        "If a coin mostly moves with Bitcoin, buying it alongside Bitcoin does not spread your risk much. "
        "A coin that acts more independently gives you better protection if Bitcoin drops."
        "</div>"
        # TODO: load cluster_assignments.csv from store/, map cluster labels to plain-language
        # categories (Moves with Bitcoin / Somewhat Independent / Acts Independently),
        # display as a simple labeled table with st.dataframe
        '<div class="placeholder-box">Coin independence ratings will appear here</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card">'
        '<div class="card-title">How similar have they been over time?</div>'
        '<div class="card-subtitle">'
        "This chart shows how closely each coin has followed Bitcoin over time. "
        "A value near 100% means the coin moves almost exactly with Bitcoin. "
        "A lower value means the coin sometimes goes its own way."
        "</div>"
        # TODO: load rolling_corr_btc column from crypto_features_long.csv,
        # convert to percentage, filter by selected_coins and date_range, plot with st.line_chart
        '<div class="placeholder-box">Bitcoin similarity chart will appear here</div>'
        "</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 4: Market Conditions
# ===========================================================================

with tab_conditions:

    st.markdown(
        "The cryptocurrency market is not always the same. "
        "It goes through different phases: periods where prices mostly go up, "
        "periods where they mostly go down, and periods of big unpredictable swings. "
        "Understanding which phase the market is in can help you decide whether now is a "
        "good time to buy, hold, or wait."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # TODO: load hmm_regime_labels.csv from store/, get the label for the most recent date,
    # map to plain-language label and display below
    col1, col2 = st.columns(2)
    col1.metric("Current market phase", "--")
    col2.metric("Phase duration so far", "--")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="card">'
        '<div class="card-title">Bitcoin price with market phases highlighted</div>'
        '<div class="card-subtitle">'
        "The colored bands show which phase the market was in at each point in time. "
        "Green = mostly rising, Red = mostly falling, Yellow = big swings in both directions."
        "</div>"
        # TODO: load hmm_regime_labels.csv and crypto_close_wide_full.csv,
        # plot BTC price as line chart with colored background per regime using matplotlib and st.pyplot
        '<div class="placeholder-box">Market phases chart will appear here</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card">'
        '<div class="card-title">What typically happens in each phase?</div>'
        '<div class="card-subtitle">'
        "This table shows the average price change and typical price swings during each market phase, "
        "based on historical data."
        "</div>"
        # TODO: group feature data by regime label, compute average daily return and avg volatility,
        # convert to readable percentages, display as st.dataframe with renamed columns
        '<div class="placeholder-box">Phase summary table will appear here</div>'
        "</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 5: Price Forecast
# ===========================================================================

with tab_forecast:

    st.markdown(
        "Using patterns from historical price data, this tool produces an estimate "
        "of what each coin's price might look like over the next 30 days. "
    )

    st.info(
        "This forecast is generated by an AI model trained on past price data. "
        "It does not account for news, regulations, or other real-world events. "
        "Use it as a rough guide only.",
        icon=None,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    forecast_asset = st.selectbox(
        label="Which coin do you want to see a forecast for?",
        options=selected_coins if selected_coins else ["BTC"],
        format_func=lambda x: COIN_NAMES[x],
    )

    st.markdown(
        '<div class="card">'
        '<div class="card-title">Estimated price for the next 30 days</div>'
        '<div class="card-subtitle">'
        "The solid line shows past prices. The dashed line shows the model's estimate "
        "for the next 30 days. Actual prices may be higher or lower."
        "</div>"
        # TODO: load store/lstm_predictions.csv for the selected forecast_asset,
        # plot historical close prices as solid line and 30-day forecast as dashed line with matplotlib
        '<div class="placeholder-box">30-day price estimate chart will appear here</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            '<div class="card">'
            '<div class="card-title">How accurate has this model been in the past?</div>'
            '<div class="card-subtitle">'
            "This chart compares the model's estimates against what prices actually did "
            "during a period the model had not seen before. "
            "The closer the two lines are, the better the model performed."
            "</div>"
            # TODO: load store/lstm_predictions.csv test-period columns for selected forecast_asset,
            # plot predicted vs actual as two lines with matplotlib and st.pyplot
            '<div class="placeholder-box">Accuracy comparison chart will appear here</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            '<div class="card">'
            '<div class="card-title">How far off were the estimates on average?</div>'
            '<div class="card-subtitle">'
            "This shows the average difference between the model's estimates and the real prices, "
            "compared to a simple guess of 'tomorrow will be the same as today'. "
            "A lower number means the model was closer to the real price."
            "</div>"
            # TODO: load store/baseline_metrics.csv, display MAE for LSTM and baseline
            # side by side as a simple two-row table with plain column names using st.dataframe
            '<div class="placeholder-box">Accuracy summary will appear here</div>'
            "</div>",
            unsafe_allow_html=True,
        )