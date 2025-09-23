import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import requests, time
import yfinance as yf
from datetime import datetime, timezone
import altair as alt

# Sidebar for About HedgeForge
with st.sidebar:
    st.header("About HedgeForge")
    st.markdown("""
    Built by MohammadMehdi Mehraein – BSc Financial Management Student | Passionate about Financial Engineering, Finance, FinTech & Blockchain
    
    HedgeForge is an interactive simulator for options trading education, focusing on delta/gamma hedging and strategy building. Key features:
    • **Hedging Sims**: Run Black-Scholes Greeks, GBM paths, and real data from Yahoo/Deribit with costs/slippage.
    • **Strategy Builder**: Drag-and-drop multi-leg payoffs (e.g., Iron Condor) with breakevens & metrics.
    • **Mispricing Checker**: Vol smile/term structure analysis for spotting edges in chains.
    
    Open-source on GitHub: [github.com/mohamadmehraein/hedgeforge-hedging] | Follow on X: [@mohamadmehraein](https://x.com/mohamadmehraein) | LinkedIn: [MohammadMehdi Mehraein](https://www.linkedin.com/in/MohammadMehdiMehraein)
    
    Educational only – not financial advice. Inspired by my studies in financial markets and blockchain!
    """)
    st.markdown("---")
    st.markdown("""
    <a href="https://github.com/mohamadmehraein/hedgeforge-hedging"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" height="20" alt="GitHub"> GitHub</a> | 
    <a href="https://x.com/mohamadmehraein"><img src="https://abs.twimg.com/responsive-web/client-web/icon-ios.8f5a1a4f.png" width="20" height="20" alt="X/Twitter"> X/Twitter</a> | 
    <a href="https://www.linkedin.com/in/MohammadMehdiMehraein"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20" height="20" alt="LinkedIn"> LinkedIn</a>
    """, unsafe_allow_html=True)
# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Function to get CSS based on theme (with animation transitions)
def get_css(theme):
    common_transitions = """
        transition: all 0.3s ease !important;
    """
    if theme == 'dark':
        return f"""
        <style>
            /* بدنه اصلی */
            .reportview-container {{
                {common_transitions}
                background: #000000 !important;
                color: white !important;
            }}
            body {{
                {common_transitions}
                background-color: #000000 !important;
                color: white !important;
                font-family: 'Roboto', sans-serif;
            }}
            /* ورودی‌ها (خاکستری تیره) */
            .stNumberInput input, .stTextInput input, .stSelectbox select, .stRadio > div {{
                {common_transitions}
                background-color: #333333 !important;
                color: white !important;
                border: 1px solid #555 !important;
            }}
            /* دکمه‌ها (قرمز) */
            .stButton > button {{
                background-color: red !important;
                color: white !important;
                border: none !important;
            }}
            /* اسلایدرها و رادیوها با اکسنت قرمز */
            .stSlider .st-ck, .stRadio > label > div {{
                accent-color: red !important;
            }}
            /* اکسپندرها، متریک‌ها، جدول‌ها با زمینه سیاه/خاکستری تیره */
            .stExpander, .stMetric, .stDataFrame, .stTable {{
                {common_transitions}
                background-color: #111111 !important;
                color: white !important;
            }}
            .stExpander > div {{
                {common_transitions}
                background-color: #222222 !important;
            }}
            /* چارت‌ها: زمینه سیاه، لیبل سفید، خطوط آبی، grid قرمز */
            .vega-lite {{
                {common_transitions}
                background: #000000 !important;
            }}
            .vega-lite text {{
                fill: white !important;
            }}
            .vega-lite .mark-rule {{
                stroke: red !important;  /* red rules */
            }}
            .vega-lite .mark-line {{
                stroke: blue !important;
            }}
        </style>
        """
    else:  # light theme
        return f"""
        <style>
            /* بدنه اصلی */
            .reportview-container {{
                {common_transitions}
                background: white !important;
                color: black !important;
            }}
            body {{
                {common_transitions}
                background-color: white !important;
                color: black !important;
                font-family: 'Roboto', sans-serif;
            }}
            /* ورودی‌ها (خاکستری روشن) */
            .stNumberInput input, .stTextInput input, .stSelectbox select, .stRadio > div {{
                {common_transitions}
                background-color: #f0f0f0 !important;  /* خاکستری روشن */
                color: black !important;
                border: 1px solid #ccc !important;
            }}
            /* دکمه‌ها (قرمز) */
            .stButton > button {{
                background-color: red !important;
                color: white !important;
                border: none !important;
            }}
            /* اسلایدرها و رادیوها با اکسنت قرمز */
            .stSlider .st-ck, .stRadio > label > div {{
                accent-color: red !important;
            }}
            /* اکسپندرها، متریک‌ها، جدول‌ها با زمینه سفید/خاکستری روشن */
            .stExpander, .stMetric, .stDataFrame, .stTable {{
                {common_transitions}
                background-color: white !important;
                color: black !important;
            }}
            .stExpander > div {{
                {common_transitions}
                background-color: #fafafa !important;  /* خاکستری خیلی روشن */
            }}
            /* چارت‌ها: زمینه سفید، لیبل سیاه، خطوط آبی، grid قرمز */
            .vega-lite {{
                {common_transitions}
                background: white !important;
            }}
            .vega-lite text {{
                fill: black !important;
            }}
            .vega-lite .mark-rule {{
                stroke: red !important;  /* red rules */
            }}
            .vega-lite .mark-line {{
                stroke: blue !important;
            }}
        </style>
        """

# Inject CSS based on current theme
st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# Theme switcher button in top-right
col1, col2 = st.columns([1, 1])
with col2:
    button_label = "🌙 Dark Theme" if st.session_state.theme == 'light' else "☀️ Light Theme"
    if st.button(button_label):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

# Robust Yahoo expiries fetch function (without session to avoid error)
def get_yahoo_expiries(ticker: str, tries: int = 2, delay: float = 0.8) -> list:
    for i in range(tries):
        tk = yf.Ticker(ticker)
        exp = tk.options or []
        if exp:
            return exp
        time.sleep(delay)
    return []

# Deribit API for crypto options (BTC/ETH)
def get_deribit_expiries(currency: str) -> list:
    url = f"https://www.deribit.com/api/v2/public/get_instruments?currency={currency}&kind=option&expired=false"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        instruments = data.get('result', [])
        expiries = sorted(set(inst['expiration_timestamp'] for inst in instruments))
        return [datetime.fromtimestamp(exp / 1000).strftime('%Y-%m-%d') for exp in expiries]
    return []

# Fetch Deribit option chain for selected expiry
def get_deribit_option_chain(currency: str, expiry: str):
    url = f"https://www.deribit.com/api/v2/public/get_instruments?currency={currency}&kind=option&expired=false"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        instruments = [inst for inst in data.get('result', []) if datetime.fromtimestamp(inst['expiration_timestamp'] / 1000).strftime('%Y-%m-%d') == expiry]
        chain = []
        for inst in instruments:
            instrument_name = inst['instrument_name']
            book_url = f"https://www.deribit.com/api/v2/public/get_order_book?instrument_name={instrument_name}"
            book_response = requests.get(book_url)
            if book_response.status_code == 200:
                book_data = book_response.json().get('result', {})
                bid = book_data.get('best_bid_price', np.nan)
                ask = book_data.get('best_ask_price', np.nan)
                mid = (bid + ask) / 2 if not np.isnan(bid) and not np.isnan(ask) else np.nan
                iv = book_data.get('mark_iv', np.nan) / 100.0 if 'mark_iv' in book_data else np.nan
                strike = inst['strike']
                typ = inst['option_type'].capitalize()
                open_interest = book_data.get('open_interest', 0)
                volume = book_data.get('stats', {}).get('volume', 0)
                chain.append({
                    "type": typ,
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "impliedVolatility": iv,
                    "openInterest": open_interest,
                    "volume": volume
                })
        return pd.DataFrame(chain)
    return pd.DataFrame()

# ---- helper: turn an expiry date into year fraction (no timezone bugs)
def years_to_expiry(expiry_date):
    exp_date = pd.Timestamp(expiry_date).date()
    today_utc = datetime.now(timezone.utc).date()
    days = max((exp_date - today_utc).days, 1)
    return days / 365.0

# ------------------------------------------------------------
# App config
# ------------------------------------------------------------
st.set_page_config(page_title="HedgeForge — Delta/Gamma Hedging Simulator", layout="wide")
st.title("HedgeForge — Delta/Gamma Hedging Simulator")
st.caption("Educational only — not investment advice.")

# ------------------------------------------------------------
# Black–Scholes helpers
# ------------------------------------------------------------
def _guard_T(T: float) -> float:
    """Numerical guard to avoid division by zero or sqrt(0)."""
    return max(float(T), 1e-9)

def d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """d1 term for Black–Scholes."""
    T = _guard_T(T)
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(d1_: float, sigma: float, T: float) -> float:
    """d2 = d1 - sigma * sqrt(T)."""
    T = _guard_T(T)
    return d1_ - sigma * np.sqrt(T)

def bs_price(S: float, K: float, r: float, sigma: float, T: float, typ: str = "Call") -> float:
    """Black–Scholes price for European Call/Put (no dividends)."""
    d1_ = d1(S, K, r, sigma, T)
    d2_ = d2(d1_, sigma, T)
    if typ == "Call":
        return S * norm.cdf(d1_) - K * np.exp(-r * T) * norm.cdf(d2_)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2_) - S * norm.cdf(-d1_)

def bs_delta(S: float, K: float, r: float, sigma: float, T: float, typ: str = "Call") -> float:
    """Spot Delta under Black–Scholes."""
    d1_ = d1(S, K, r, sigma, T)
    return norm.cdf(d1_) if typ == "Call" else norm.cdf(d1_) - 1.0

def bs_gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Spot Gamma (same for call/put) under Black–Scholes."""
    T = _guard_T(T)
    d1_ = d1(S, K, r, sigma, T)
    denom = max(S * sigma * np.sqrt(T), 1e-12)
    return np.exp(-0.5 * d1_**2) / (np.sqrt(2.0 * np.pi) * denom)

# ------------------------------------------------------------
# UI — option parameters
# ------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    S0 = st.number_input("Underlying price (S₀)", value=30000.0, step=100.0, min_value=1.0)
    K = st.number_input("Strike (K)", value=32000.0, step=100.0, min_value=1.0)
    T_days = st.number_input("Days to maturity (T)", value=30, min_value=2, max_value=365)
with col2:
    r = st.number_input("Risk-free rate (annual, %)", value=2.0, step=0.1) / 100.0
    sigma = st.number_input("Volatility σ (annual, %)", value=60.0, step=1.0, min_value=1.0, max_value=500.0) / 100.0
    opt_type = st.selectbox("Option type", ["Call", "Put"])

# Greeks panel at t0
T0_years = float(T_days) / 365.0
with st.expander("Greeks (for current inputs at t0)", expanded=False):
    price_0 = bs_price(S0, K, r, sigma, T0_years, opt_type)
    delta_0 = bs_delta(S0, K, r, sigma, T0_years, opt_type)
    gamma_0 = bs_gamma(S0, K, r, sigma, T0_years)
    g1, g2, g3 = st.columns(3)
    g1.metric("Option Price (t0)", f"{price_0:,.4f}")
    g2.metric("Delta (t0)", f"{delta_0:,.4f}")
    g3.metric("Gamma (t0)", f"{gamma_0:,.6f}")

st.markdown("---")

# Data source selection
st.subheader("Price Path")
src = st.radio("Choose source:", ["Simulate GBM", "Upload CSV", "Fetch via Yahoo Finance"], horizontal=True)

uploaded = None
scale_to_S0 = True
# defaults for GBM branch
cost_bps = 10
slip_bps = 5
freq_label = "Daily (252/yr)"
days = T_days
seed = 42

if src == "Upload CSV":
    st.write(
        "Upload a CSV with columns **`date`** (optional) and **`close`** (required). "
        "Each row is treated as one hedge step."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    scale_to_S0 = st.checkbox("Scale price path so first price equals S₀", value=True)

    c1, c2 = st.columns(2)
    with c1:
        cost_bps_upload = st.number_input("Transaction cost (bps) on uploaded path", value=10, min_value=0, max_value=2000)
    with c2:
        slip_bps_upload = st.number_input("Slippage (bps) on uploaded path (underlying only)", value=5, min_value=0, max_value=2000)

elif src == "Simulate GBM":
    st.subheader("Delta-Hedge Simulation Settings (GBM)")
    c1, c2, c3 = st.columns(3)
    with c1:
        cost_bps = st.number_input("Transaction cost (bps)", value=10, min_value=0, max_value=2000)
    with c2:
        slip_bps = st.number_input("Slippage (bps)", value=5, min_value=0, max_value=2000)
    with c3:
        freq_label = st.selectbox("Hedging frequency", ["Daily (252/yr)", "4-hourly (~6/day)", "Hourly (~24/day)"])

    c4, c5 = st.columns(2)
    with c4:
        days = st.number_input("Simulation days", value=T_days, min_value=2, max_value=365)
    with c5:
        seed = st.number_input("Random seed (for GBM)", value=42, min_value=0, step=1)

else:  # Fetch via Yahoo Finance
    st.write("Download real market data from Yahoo Finance and run the hedge on it.")
    c1, c2 = st.columns(2)
    with c1:
        ticker_choice = st.selectbox("Ticker", ["BTC-USD", "ETH-USD", "Custom…"])
        custom_ticker = st.text_input("Custom ticker (e.g., AAPL, SPY)", value="", help="Used only if 'Custom…' is selected.")
    with c2:
        yf_step = st.selectbox("Target hedge step", ["Daily (1d)", "Hourly (1h)", "4-hourly (resampled from 1h)"])

    c3, c4 = st.columns(2)
    with c3:
        period = st.selectbox("Period", ["30d", "60d", "90d", "1y", "2y"])
    with c4:
        save_fetched_csv = st.checkbox("Save fetched CSV for download", value=True)

    scale_to_S0 = st.checkbox("Scale price path so first price equals S₀", value=True)
    cost_bps_yf = st.number_input("Transaction cost (bps) on Yahoo path", value=10, min_value=0, max_value=2000)
    slip_bps_yf = st.number_input("Slippage (bps) on Yahoo path (underlying only)", value=5, min_value=0, max_value=2000)

run = st.button("Run Simulation", type="primary")

# Core simulators
def simulate_gbm(S0: float, sigma: float, days: int, freq_label: str, seed: int):
    """Simulate a GBM price path with a selected hedging frequency."""
    freq_map = {"Daily (252/yr)": (1, "D"), "4-hourly (~6/day)": (6, "4H"), "Hourly (~24/day)": (24, "H")}
    steps_per_day, pd_freq = freq_map[freq_label]
    n_steps = int(days) * steps_per_day
    dt = 1.0 / 252.0 / steps_per_day
    np.random.seed(int(seed))
    S = np.zeros(n_steps, dtype=float)
    S[0] = float(S0)
    for t in range(1, n_steps):
        z = np.random.normal()
        S[t] = S[t - 1] * np.exp((-0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    idx = pd.date_range(periods=n_steps, freq=pd_freq, start=pd.Timestamp.today().normalize())
    return S, idx, dt, n_steps

def simulate_delta_hedge_on_path(S, idx, n_steps, K, r, sigma, T_days, opt_type, cost_bps, slip_bps):
    """Delta hedge on a given price path with explicit fees + slippage (underlying)."""
    T0 = float(T_days) / 365.0
    T_path = np.linspace(T0, 1e-9, n_steps)

    opt_vals = np.array([bs_price(S[t], K, r, sigma, T_path[t], opt_type) for t in range(n_steps)], dtype=float)
    option_pnl = np.diff(opt_vals, prepend=opt_vals[0])

    cost_rate = cost_bps / 10000.0
    slip_rate = slip_bps / 10000.0
    notional = 1.0

    deltas = np.array([bs_delta(S[t], K, r, sigma, T_path[t], opt_type) for t in range(n_steps)], dtype=float)
    hedge_pos = np.zeros(n_steps, dtype=float)
    hedge_trade = np.zeros(n_steps, dtype=float)
    hedge_pnl = np.zeros(n_steps, dtype=float)
    costs = np.zeros(n_steps, dtype=float)

    # t=0
    hedge_pos[0] = -notional * deltas[0]
    exec_p0 = S[0] * (1.0 + np.sign(hedge_pos[0]) * slip_rate)
    fees0 = abs(hedge_pos[0]) * exec_p0 * cost_rate
    slip0 = abs(hedge_pos[0]) * abs(exec_p0 - S[0])
    costs[0] = fees0 + slip0
    hedge_trade[0] = hedge_pos[0]

    for t in range(1, n_steps):
        hedge_pnl[t] = hedge_pos[t - 1] * (S[t] - S[t - 1])
        target = -notional * deltas[t]
        trade_units = target - hedge_pos[t - 1]
        exec_pt = S[t] * (1.0 + np.sign(trade_units) * slip_rate)
        fees_t = abs(trade_units) * exec_pt * cost_rate
        slip_t = abs(trade_units) * abs(exec_pt - S[t])
        costs[t] = fees_t + slip_t
        hedge_pos[t] = hedge_pos[t - 1] + trade_units
        hedge_trade[t] = trade_units

    total_pnl = option_pnl + hedge_pnl - costs

    df = pd.DataFrame(
        {"price": S, "option_value": opt_vals, "option_pnl": option_pnl, "hedge_pnl": hedge_pnl,
         "trade_units": hedge_trade, "costs": costs, "total_pnl": total_pnl},
        index=idx,
    )
    return df

def simulate_delta_gamma_hedge_on_path(
    S, idx, n_steps, K, r, sigma, T_days, opt_type, cost_bps, slip_bps, K_hedge=None
):
    """Delta+Gamma hedge using one additional vanilla option as the gamma instrument."""
    T0 = float(T_days) / 365.0
    T_path = np.linspace(T0, 1e-9, n_steps)

    # Main option path
    opt_vals_main = np.array([bs_price(S[t], K, r, sigma, T_path[t], opt_type) for t in range(n_steps)], dtype=float)
    option_pnl_main = np.diff(opt_vals_main, prepend=opt_vals_main[0])
    deltas_main = np.array([bs_delta(S[t], K, r, sigma, T_path[t], opt_type) for t in range(n_steps)], dtype=float)
    gammas_main = np.array([bs_gamma(S[t], K, r, sigma, T_path[t]) for t in range(n_steps)], dtype=float)

    # Hedge option (call; gamma>0 anyway). Default strike ~ ATM at inception.
    if K_hedge is None:
        K_hedge = float(S[0])
    typ_hedge = "Call"
    opt_vals_hedge = np.array([bs_price(S[t], K_hedge, r, sigma, T_path[t], typ_hedge) for t in range(n_steps)], dtype=float)
    deltas_hedge = np.array([bs_delta(S[t], K_hedge, r, sigma, T_path[t], typ_hedge) for t in range(n_steps)], dtype=float)
    gammas_hedge = np.array([bs_gamma(S[t], K_hedge, r, sigma, T_path[t]) for t in range(n_steps)], dtype=float)

    cost_rate = cost_bps / 10000.0
    slip_rate = slip_bps / 10000.0

    # Positions and accounting
    q_opt = np.zeros(n_steps, dtype=float)     # hedge-option quantity
    u_pos = np.zeros(n_steps, dtype=float)     # underlying units
    trade_q = np.zeros(n_steps, dtype=float)
    trade_u = np.zeros(n_steps, dtype=float)

    pnl_u = np.zeros(n_steps, dtype=float)     # inventory pnl (underlying)
    pnl_q = np.zeros(n_steps, dtype=float)     # pnl from hedge option
    costs = np.zeros(n_steps, dtype=float)

    # t=0: gamma neutral then delta neutral
    q_opt[0] = - gammas_main[0] / max(gammas_hedge[0], 1e-12)
    u_pos[0] = - (deltas_main[0] + q_opt[0] * deltas_hedge[0])

    exec_u0 = S[0] * (1.0 + np.sign(u_pos[0]) * slip_rate)
    fee_u0 = abs(u_pos[0]) * exec_u0 * cost_rate
    slip_u0 = abs(u_pos[0]) * abs(exec_u0 - S[0])
    fee_q0 = abs(q_opt[0]) * opt_vals_hedge[0] * cost_rate
    costs[0] = fee_u0 + slip_u0 + fee_q0
    trade_u[0] = u_pos[0]
    trade_q[0] = q_opt[0]

    # iterate
    for t in range(1, n_steps):
        pnl_u[t] = u_pos[t - 1] * (S[t] - S[t - 1])
        pnl_q[t] = q_opt[t - 1] * (opt_vals_hedge[t] - opt_vals_hedge[t - 1])

        # targets for neutrality
        q_target = - gammas_main[t] / max(gammas_hedge[t], 1e-12)
        u_target = - (deltas_main[t] + q_target * deltas_hedge[t])

        # trades required
        du = u_target - u_pos[t - 1]
        dq = q_target - q_opt[t - 1]

        exec_ut = S[t] * (1.0 + np.sign(du) * slip_rate)
        fee_ut = abs(du) * exec_ut * cost_rate
        slip_ut = abs(du) * abs(exec_ut - S[t])
        fee_qt = abs(dq) * opt_vals_hedge[t] * cost_rate
        costs[t] = fee_ut + slip_ut + fee_qt

        u_pos[t] = u_pos[t - 1] + du
        q_opt[t] = q_opt[t - 1] + dq
        trade_u[t] = du
        trade_q[t] = dq

    total_pnl = option_pnl_main + pnl_u + pnl_q - costs

    df = pd.DataFrame(
        {
            "price": S,
            "option_value": opt_vals_main,
            "option_pnl": option_pnl_main,
            "u_pnl": pnl_u,
            "q_pnl": pnl_q,
            "trade_u": trade_u,
            "trade_q": trade_q,
            "costs": costs,
            "total_pnl": total_pnl,
        },
        index=idx,
    )
    return df

def run_delta_hedge_on_uploaded(
    prices: pd.Series,
    T_days: int,
    opt_type: str,
    K: float,
    r: float,
    sigma: float,
    cost_bps: int = 0,
    slip_bps: int = 0,
):
    """Delta hedge on an uploaded/fetched price series (one step per row) with fees+slippage."""
    S = prices.values.astype(float)
    n_steps = len(S)
    idx = prices.index

    T0 = float(T_days) / 365.0
    T_path = np.linspace(T0, 1e-9, n_steps)

    opt_vals = np.array([bs_price(S[t], K, r, sigma, T_path[t], opt_type) for t in range(n_steps)], dtype=float)
    option_pnl = np.diff(opt_vals, prepend=opt_vals[0])

    deltas = np.array([bs_delta(S[t], K, r, sigma, T_path[t], opt_type) for t in range(n_steps)], dtype=float)

    cost_rate = cost_bps / 10000.0
    slip_rate = slip_bps / 10000.0

    hedge_pos = np.zeros(n_steps, dtype=float)
    hedge_trade = np.zeros(n_steps, dtype=float)
    hedge_pnl = np.zeros(n_steps, dtype=float)
    costs = np.zeros(n_steps, dtype=float)

    # t=0
    hedge_pos[0] = -1.0 * deltas[0]
    exec_p0 = S[0] * (1.0 + np.sign(hedge_pos[0]) * slip_rate)
    fees0 = abs(hedge_pos[0]) * exec_p0 * cost_rate
    slip0 = abs(hedge_pos[0]) * abs(exec_p0 - S[0])
    costs[0] = fees0 + slip0
    hedge_trade[0] = hedge_pos[0]

    for t in range(1, n_steps):
        hedge_pnl[t] = hedge_pos[t - 1] * (S[t] - S[t - 1])
        target = -1.0 * deltas[t]
        trade_units = target - hedge_pos[t - 1]
        exec_pt = S[t] * (1.0 + np.sign(trade_units) * slip_rate)
        fees_t = abs(trade_units) * exec_pt * cost_rate
        slip_t = abs(trade_units) * abs(exec_pt - S[t])
        costs[t] = fees_t + slip_t
        hedge_pos[t] = hedge_pos[t - 1] + trade_units
        hedge_trade[t] = trade_units

    total_pnl = option_pnl + hedge_pnl - costs

    out = pd.DataFrame(
        {
            "price": S,
            "option_value": opt_vals,
            "option_pnl": option_pnl,
            "hedge_pnl": hedge_pnl,
            "trade_units": hedge_trade,
            "costs": costs,
            "total_pnl": total_pnl,
        },
        index=idx,
    )
    return out

# ------------------------------------------------------------
# Yahoo Finance fetch utility
# ------------------------------------------------------------
def fetch_yf_prices(ticker: str, period: str, target_step: str) -> pd.DataFrame:
    """
    Download prices from Yahoo Finance and return a DataFrame with columns: date, close.
    target_step: 'Daily (1d)' | 'Hourly (1h)' | '4-hourly (resampled from 1h)'
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError("yfinance is not installed. Run: pip install yfinance") from e

    if target_step == "Daily (1d)":
        interval = "1d"
    else:
        interval = "1h"

    if interval == "1h" and period not in {"30d", "60d"}:
        period = "60d"

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError("No data returned from Yahoo Finance. Try a shorter period or a different ticker.")

    df = df.copy()
    df = df[["Close"]].rename(columns={"Close": "close"})
    df = df.reset_index()

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    else:
        df = df.rename(columns={"Date": "date"})

    if target_step == "4-hourly (resampled from 1h)":
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").resample("4H").last().dropna().reset_index()

    return df[["date", "close"]]

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if run:
    with st.spinner("Running simulation..."):
        if src == "Simulate GBM":
            S, idx, dt, n_steps = simulate_gbm(S0=S0, sigma=sigma, days=int(days), freq_label=freq_label, seed=int(seed))

            out_delta = simulate_delta_hedge_on_path(
                S=S, idx=idx, n_steps=n_steps, K=K, r=r, sigma=sigma, T_days=int(T_days),
                opt_type=opt_type, cost_bps=int(cost_bps), slip_bps=int(slip_bps)
            )
            cum_nohedge = out_delta["option_pnl"].cumsum()
            cum_delta = out_delta["total_pnl"].cumsum()

            out_dg = simulate_delta_gamma_hedge_on_path(
                S=S, idx=idx, n_steps=n_steps, K=K, r=r, sigma=sigma, T_days=int(T_days),
                opt_type=opt_type, cost_bps=int(cost_bps), slip_bps=int(slip_bps), K_hedge=None
            )
            cum_dg = out_dg["total_pnl"].cumsum()

            pnl_delta = out_delta["total_pnl"].values
            sharpe_delta = float(pnl_delta.mean() / pnl_delta.std(ddof=1)) if pnl_delta.std(ddof=1) > 0 else 0.0
            hedge_err_delta = float(np.sqrt(np.mean((out_delta["option_pnl"].values + out_delta["hedge_pnl"].values) ** 2)))
            eq_delta = cum_delta.values
            run_max_d = np.maximum.accumulate(eq_delta)
            max_dd_delta = float((run_max_d - eq_delta).max())
            turn_delta = float(np.abs(out_delta["trade_units"]).sum())

            pnl_dg = out_dg["total_pnl"].values
            sharpe_dg = float(pnl_dg.mean() / pnl_dg.std(ddof=1)) if pnl_dg.std(ddof=1) > 0 else 0.0
            hedge_err_dg = float(np.sqrt(np.mean((out_dg["option_pnl"].values + out_dg["u_pnl"].values + out_dg["q_pnl"].values) ** 2)))
            eq_dg = cum_dg.values
            run_max_g = np.maximum.accumulate(eq_dg)
            max_dd_dg = float((run_max_g - eq_dg).max())
            turn_dg_u = float(np.abs(out_dg["trade_u"]).sum())

            st.markdown("---")
            st.subheader("Results (Simulated GBM)")
            st.line_chart(pd.DataFrame(
                {"No Hedge (cum P&L)": cum_nohedge, "Delta Hedge (cum P&L)": cum_delta, "Delta+Gamma Hedge (cum P&L)": cum_dg},
                index=idx,
            ))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe (Delta)", f"{sharpe_delta:.2f}")
            c2.metric("Hedge Err RMSE (Delta)", f"{hedge_err_delta:,.2f}")
            c3.metric("Max DD (Delta)", f"{max_dd_delta:,.2f}")
            c4.metric("Turnover U (Delta)", f"{turn_delta:,.2f}")

            d1c, d2c, d3c, d4c = st.columns(4)
            d1c.metric("Sharpe (Δ+Γ)", f"{sharpe_dg:.2f}")
            d2c.metric("Hedge Err RMSE (Δ+Γ)", f"{hedge_err_dg:,.2f}")
            d3c.metric("Max DD (Δ+Γ)", f"{max_dd_dg:,.2f}")
            d4c.metric("Turnover U (Δ+Γ)", f"{turn_dg_u:,.2f}")

            st.markdown("**Last 10 rows (Delta hedge)**")
            st.dataframe(out_delta.tail(10))
            st.download_button(
                "Download CSV — Delta hedge",
                data=out_delta.to_csv(index=True).encode("utf-8"),
                file_name="hedge_results_delta.csv",
                mime="text/csv",
            )

            st.markdown("**Last 10 rows (Delta+Gamma hedge)**")
            st.dataframe(out_dg.tail(10))
            st.download_button(
                "Download CSV — Delta+Gamma hedge",
                data=out_dg.to_csv(index=True).encode("utf-8"),
                file_name="hedge_results_delta_gamma.csv",
                mime="text/csv",
            )

        elif src == "Upload CSV":
            if uploaded is None:
                st.error("Please upload a CSV file with a `close` column.")
                st.stop()
            df = pd.read_csv(uploaded)
            if "close" not in df.columns:
                st.error("CSV must contain a `close` column.")
                st.stop()
            if "date" in df.columns:
                idx = pd.to_datetime(df["date"], errors="coerce")
            else:
                idx = pd.RangeIndex(len(df))
            # Fixed: use .values.flatten() to ensure 1D
            prices = pd.Series(df["close"].astype(float).values.flatten(), index=idx, name="price")
            if scale_to_S0 and len(prices) > 0 and prices.iloc[0] > 0:
                prices = prices * (S0 / prices.iloc[0])

            out = run_delta_hedge_on_uploaded(
                prices=prices,
                T_days=int(T_days),
                opt_type=opt_type,
                K=K,
                r=r,
                sigma=sigma,
                cost_bps=int(cost_bps_upload),
                slip_bps=int(slip_bps_upload),
            )

            cum_nohedge = out["option_pnl"].cumsum()
            cum_hedge = out["total_pnl"].cumsum()

            pnl = out["total_pnl"].values
            sharpe_daily = float(pnl.mean() / pnl.std(ddof=1)) if pnl.std(ddof=1) > 0 else 0.0
            hedge_error_rmse = float(np.sqrt(np.mean((out["option_pnl"].values + out["hedge_pnl"].values) ** 2)))
            equity = cum_hedge.values
            run_max = np.maximum.accumulate(equity)
            max_dd = float((run_max - equity).max())
            turnover_units = float(np.abs(out["trade_units"]).sum())

            st.markdown("---")
            st.subheader("Results (Uploaded CSV)")
            st.line_chart(pd.DataFrame({"No Hedge (cum P&L)": cum_nohedge, "Delta Hedge (cum P&L)": cum_hedge}, index=out.index))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe (daily)", f"{sharpe_daily:.2f}")
            c2.metric("Hedge Error (RMSE)", f"{hedge_error_rmse:,.2f}")
            c3.metric("Max Drawdown", f"{max_dd:,.2f}")
            c4.metric("Turnover (units)", f"{turnover_units:,.2f}")

            st.markdown("**Last 10 rows**")
            st.dataframe(out.tail(10))
            st.download_button(
                "Download CSV — Delta hedge (uploaded path)",
                data=out.to_csv(index=True).encode("utf-8"),
                file_name="hedge_results_uploaded.csv",
                mime="text/csv",
            )

        else:  # Fetch via Yahoo Finance
            ticker = custom_ticker.strip() if ticker_choice == "Custom…" and custom_ticker.strip() else ticker_choice
            if not ticker:
                st.error("Please provide a valid ticker (e.g., BTC-USD, ETH-USD).")
                st.stop()

            try:
                df_yf = fetch_yf_prices(ticker=ticker, period=period, target_step=yf_step)
            except ImportError as e:
                st.error(str(e))
                st.info("Install it in your venv:\n\n`pip install yfinance`")
                st.stop()
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                st.stop()

            idx = pd.to_datetime(df_yf["date"], errors="coerce")
            # Fixed: use .values.flatten() to ensure 1D
            prices = pd.Series(df_yf["close"].astype(float).values.flatten(), index=idx, name="price")
            if scale_to_S0 and len(prices) > 0 and prices.iloc[0] > 0:
                prices = prices * (S0 / prices.iloc[0])

            out = run_delta_hedge_on_uploaded(
                prices=prices,
                T_days=int(T_days),
                opt_type=opt_type,
                K=K,
                r=r,
                sigma=sigma,
                cost_bps=int(cost_bps_yf),
                slip_bps=int(slip_bps_yf),
            )

            cum_nohedge = out["option_pnl"].cumsum()
            cum_hedge = out["total_pnl"].cumsum()

            pnl = out["total_pnl"].values
            sharpe_daily = float(pnl.mean() / pnl.std(ddof=1)) if pnl.std(ddof=1) > 0 else 0.0
            hedge_error_rmse = float(np.sqrt(np.mean((out["option_pnl"].values + out["hedge_pnl"].values) ** 2)))
            equity = cum_hedge.values
            run_max = np.maximum.accumulate(equity)
            max_dd = float((run_max - equity).max())
            turnover_units = float(np.abs(out["trade_units"]).sum())

            st.markdown("---")
            st.subheader(f"Results (Yahoo Finance: {ticker}, {period}, {yf_step})")
            st.line_chart(pd.DataFrame({"No Hedge (cum P&L)": cum_nohedge, "Delta Hedge (cum P&L)": cum_hedge}, index=out.index))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe (daily)", f"{sharpe_daily:.2f}")
            c2.metric("Hedge Error (RMSE)", f"{hedge_error_rmse:,.2f}")
            c3.metric("Max Drawdown", f"{max_dd:,.2f}")
            c4.metric("Turnover (units)", f"{turnover_units:,.2f}")

            st.markdown("**Last 10 rows**")
            st.dataframe(out.tail(10))

            if save_fetched_csv:
                st.download_button(
                    "Download fetched CSV (date,close)",
                    data=df_yf.to_csv(index=False).encode("utf-8"),
                    file_name=f"{ticker.replace('-', '')}_{period}_{'1d' if yf_step=='Daily (1d)' else '1h_or_4h'}.csv",
                    mime="text/csv",
                )
            st.download_button(
                "Download CSV — Delta hedge (Yahoo path)",
                data=out.to_csv(index=True).encode("utf-8"),
                file_name=f"hedge_results_{ticker.replace('-', '')}.csv",
                mime="text/csv",
            )
# ===================== Strategy Builder (Payoff at Expiry) =====================
def render_strategy_builder():
    """Interactive option strategy payoff builder (at expiry)."""
    st.markdown("---")
    st.header("📈 Option Strategy Builder — Payoff at Expiry")
    st.caption("Build multi-leg strategies (Buy/Sell Call/Put/Underlying), then view payoff at expiry. "
               "Premium is a positive number per contract; sign is handled by 'Side'.")

    with st.expander("Global settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            S0_sb = st.number_input("Current underlying price (S₀)", value=30000.0, min_value=0.0, step=100.0)
        with c2:
            opt_mult = st.number_input("Option contract multiplier", value=1, min_value=1, step=1,
                                       help="Set to 100 for equity options; crypto often 1.")
        with c3:
            show_legs = st.toggle("Show individual legs on chart", value=False)

        c4, c5 = st.columns(2)
        with c4:
            lo_mult = st.slider("Price range — lower (% of S₀)", 10, 100, 50,
                                help="Left edge of chart = this % × S₀")
        with c5:
            hi_mult = st.slider("Price range — upper (% of S₀)", 100, 400, 150,
                                help="Right edge of chart = this % × S₀")

    st.caption("Quick presets (optional)")
    preset = st.selectbox(
        "Pick a preset",
        [
            "— none —",
            "Protective Put (Long Stock + Long Put)",
            "Covered Call (Long Stock + Short Call)",
            "Bull Call Spread (Buy Call / Sell Call)",
            "Long Straddle (Buy Call + Buy Put)",
            "Iron Condor (Sell OTM Call/Put, Buy further OTM Call/Put)"
        ],
        index=0,
        label_visibility="collapsed"
    )

    default_df = pd.DataFrame(
        [{"Side":"Buy","Instrument":"Underlying","Strike":S0_sb,"Premium":0.0,"Qty":1.0}]
    )

    if preset != "— none —":
        if preset == "Protective Put (Long Stock + Long Put)":
            default_df = pd.DataFrame([
                {"Side":"Buy","Instrument":"Underlying","Strike":S0_sb,"Premium":0.0,"Qty":1.0},
                {"Side":"Buy","Instrument":"Put","Strike":round(S0_sb,2),"Premium":0.0,"Qty":1.0},
            ])
        elif preset == "Covered Call (Long Stock + Short Call)":
            default_df = pd.DataFrame([
                {"Side":"Buy","Instrument":"Underlying","Strike":S0_sb,"Premium":0.0,"Qty":1.0},
                {"Side":"Sell","Instrument":"Call","Strike":round(S0_sb*1.05,2),"Premium":0.0,"Qty":1.0},
            ])
        elif preset == "Bull Call Spread (Buy Call / Sell Call)":
            default_df = pd.DataFrame([
                {"Side":"Buy","Instrument":"Call","Strike":round(S0_sb*0.95,2),"Premium":0.0,"Qty":1.0},
                {"Side":"Sell","Instrument":"Call","Strike":round(S0_sb*1.05,2),"Premium":0.0,"Qty":1.0},
            ])
        elif preset == "Long Straddle (Buy Call + Buy Put)":
            default_df = pd.DataFrame([
                {"Side":"Buy","Instrument":"Call","Strike":round(S0_sb,2),"Premium":0.0,"Qty":1.0},
                {"Side":"Buy","Instrument":"Put","Strike":round(S0_sb,2),"Premium":0.0,"Qty":1.0},
            ])
        elif preset == "Iron Condor (Sell OTM Call/Put, Buy further OTM Call/Put)":
            default_df = pd.DataFrame([
                {"Side":"Sell","Instrument":"Call","Strike":round(S0_sb*1.05,2),"Premium":0.0,"Qty":1.0},
                {"Side":"Buy","Instrument":"Call","Strike":round(S0_sb*1.10,2),"Premium":0.0,"Qty":1.0},
                {"Side":"Sell","Instrument":"Put","Strike":round(S0_sb*0.95,2),"Premium":0.0,"Qty":1.0},
                {"Side":"Buy","Instrument":"Put","Strike":round(S0_sb*0.90,2),"Premium":0.0,"Qty":1.0},
            ])

    st.write("Add / edit strategy legs (each row is one position):")
    legs_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Side": st.column_config.SelectboxColumn("Side", options=["Buy","Sell"], width="small"),
            "Instrument": st.column_config.SelectboxColumn("Instrument", options=["Call","Put","Underlying"], width="small"),
            "Strike": st.column_config.NumberColumn("Strike", help="Ignored for Underlying", step=0.01, format="%.4f"),
            "Premium": st.column_config.NumberColumn("Premium (per contract, >0)", step=0.01, format="%.4f"),
            "Qty": st.column_config.NumberColumn("Qty (>=0)", step=0.1, format="%.4f"),
        }
    )

    if legs_df.empty:
        st.info("Add at least one leg to see the payoff.")
        return

    errors = []
    for i, r in legs_df.iterrows():
        side = str(r.get("Side","")).strip()
        inst = str(r.get("Instrument","")).strip()
        K = float(r.get("Strike", 0) or 0)
        prem = float(r.get("Premium", 0) or 0)
        qty = float(r.get("Qty", 0) or 0)
        if side not in ("Buy","Sell"):
            errors.append(f"Row {i+1}: Side must be Buy/Sell.")
        if inst not in ("Call","Put","Underlying"):
            errors.append(f"Row {i+1}: Instrument must be Call/Put/Underlying.")
        if inst in ("Call","Put") and K <= 0:
            errors.append(f"Row {i+1}: Strike must be > 0 for options.")
        if prem < 0:
            errors.append(f"Row {i+1}: Premium must be non-negative.")
        if qty < 0:
            errors.append(f"Row {i+1}: Qty must be non-negative (use Side=Sell for short).")
    if S0_sb <= 0:
        errors.append("S₀ must be > 0.")
    if lo_mult >= hi_mult:
        errors.append("Lower % must be strictly less than Upper % in price range.")

    if errors:
        st.error("Please fix the following before plotting:\n\n- " + "\n- ".join(errors))
        return

    S_min = max(0.0, S0_sb * (lo_mult/100.0))
    S_max = S0_sb * (hi_mult/100.0)
    S_grid = np.linspace(S_min, S_max, 501)

    def leg_payoff_one(row, S):
        side = row["Side"]
        inst = row["Instrument"]
        K = float(row["Strike"])
        prem = float(row["Premium"])
        q = float(row["Qty"])
        sign = 1.0 if side == "Buy" else -1.0

        if inst == "Call":
            base = np.maximum(S - K, 0.0) - prem
            factor = opt_mult
            return q * sign * base * factor
        elif inst == "Put":
            base = np.maximum(K - S, 0.0) - prem
            factor = opt_mult
            return q * sign * base * factor
        else:
            return q * sign * (S - S0_sb)

    total_pnl = np.zeros_like(S_grid)
    leg_curves = []
    leg_labels = []

    for i, row in legs_df.iterrows():
        pnl = leg_payoff_one(row, S_grid)
        total_pnl += pnl
        if show_legs:
            label = f"{row['Side']} {row['Instrument']}"
            if row["Instrument"] in ("Call","Put"):
                label += f" K={float(row['Strike']):.2f}, prem={float(row['Premium']):.2f}, q={float(row['Qty']):g}"
            else:
                label += f" q={float(row['Qty']):g}"
            leg_curves.append(pd.DataFrame({"Price": S_grid, "PnL": pnl, "Leg": label}))
            leg_labels.append(label)

    breakevens = []
    y = total_pnl
    for i in range(1, len(S_grid)):
        y1, y2 = y[i-1], y[i]
        if y1 == 0.0:
            breakevens.append(float(S_grid[i-1]))
        elif (y1 < 0 and y2 > 0) or (y1 > 0 and y2 < 0):
            x1, x2 = S_grid[i-1], S_grid[i]
            x0 = x1 - y1 * (x2 - x1) / (y2 - y1)
            breakevens.append(float(x0))

    max_profit = float(np.max(total_pnl))
    max_profit_S = float(S_grid[np.argmax(total_pnl)])
    max_loss = float(np.min(total_pnl))
    max_loss_S = float(S_grid[np.argmin(total_pnl)])

    df_total = pd.DataFrame({"Price": S_grid, "Total_PnL": total_pnl})
    base = alt.Chart(df_total).mark_line().encode(x='Price:Q', y='Total_PnL:Q')
    zero_line = alt.Chart(pd.DataFrame({"y":[0.0]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
    s0_rule = alt.Chart(pd.DataFrame({"x":[S0_sb]})).mark_rule(opacity=0.25).encode(x="x:Q")
    layers = [base, zero_line, s0_rule]
    if breakevens:
        be_df = pd.DataFrame({'x': breakevens})
        rule = alt.Chart(be_df).mark_rule(color='red').encode(x='x:Q')
        label = alt.Chart(be_df.assign(label=be_df['x'].apply(lambda x: f"{x:.0f}"))).mark_text(dy=-10).encode(x='x:Q', text='label:N')
        layers.append(rule)
        layers.append(label)
    if show_legs and leg_curves:
        legs_df_plot = pd.concat(leg_curves, ignore_index=True)
        legs_layer = alt.Chart(legs_df_plot).mark_line(opacity=0.5).encode(x="Price:Q", y="PnL:Q", color=alt.Color("Leg:N", legend=alt.Legend(title="Legs")))
        layers.append(legs_layer)
    chart = alt.layer(*layers).properties(height=360)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Break-even(s)",
                  ", ".join([f"{b:,.2f}" for b in breakevens]) if breakevens else "—")
    with c2:
        st.metric("Max Profit", f"{max_profit:,.2f}", help=f"At S ≈ {max_profit_S:,.2f}")
    with c3:
        st.metric("Max Loss", f"{max_loss:,.2f}", help=f"At S ≈ {max_loss_S:,.2f}")

    with st.expander("Show payoff table / download CSV"):
        st.dataframe(df_total.tail(20), use_container_width=True)
        st.download_button(
            "Download payoff CSV",
            data=df_total.to_csv(index=False).encode("utf-8"),
            file_name="strategy_payoff.csv",
            mime="text/csv"
        )
        st.download_button(
            "Download legs as CSV",
            data=legs_df.to_csv(index=False).encode("utf-8"),
            file_name="strategy_legs.csv",
            mime="text/csv"
        )

st.markdown("---")
render_strategy_builder()

# =========================
# Phase 2: Mispricing Checker (beta)
# =========================
def render_mispricing_checker():
    """Mispricing checker."""
    import datetime as _dt
    import numpy as _np
    import pandas as _pd
    import yfinance as _yf
    import altair as _alt
    from scipy.stats import norm as _norm
    from scipy.optimize import brentq as _brentq

    st.markdown("---")
    st.header("Mispricing Checker (beta)")

    def _bs_d1(S, K, r, sigma, T):
        return (_np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * _np.sqrt(T))

    def _bs_d2(d1, sigma, T):
        return d1 - sigma * _np.sqrt(T)

    def _bs_price(S, K, r, sigma, T, typ="Call"):
        d1 = _bs_d1(S, K, r, sigma, T)
        d2 = _bs_d2(d1, sigma, T)
        if typ == "Call":
            return S * _norm.cdf(d1) - K * _np.exp(-r * T) * _norm.cdf(d2)
        else:
            return K * _np.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1)

    def _intrinsic(S, K, typ):
        return max(S - K, 0.0) if typ == "Call" else max(K - S, 0.0)

    def _iv_from_price(mkt, S, K, r, T, typ):
        try:
            intr = _intrinsic(S, K, typ)
            if mkt <= intr + 1e-8 or T <= 0:
                return _np.nan
            f = lambda s: _bs_price(S, K, r, s, T, typ) - mkt
            return float(_brentq(f, 1e-6, 5.0, maxiter=100, xtol=1e-8))
        except Exception:
            return _np.nan

    def _clean_mid(bid, ask, last):
        if bid > 0 and ask > 0 and ask >= bid:
            return 0.5 * (bid + ask)
        if last > 0:
            return float(last)
        return _np.nan

    def _load_underlying_price(ticker):
        tk = _yf.Ticker(ticker)
        hist = tk.history(period="5d", interval="1d")
        if hist is None or hist.empty:
            raise RuntimeError("Failed to load underlying price.")
        return float(hist["Close"].iloc[-1])

    with st.expander("Data & Filters", expanded=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            ticker = st.text_input("Ticker (equity/ETF with options on Yahoo)", value="AAPL")
            rf_pct = st.number_input("Risk-free (annual, %)", value=2.00, step=0.25, min_value=0.0)
        with c2:
            period_for_rv = st.selectbox("Realized Vol lookback", ["20d", "30d", "60d"], index=1)
            mny_band = st.slider("Moneyness band for smile fit (|ln(K/S)| ≤)", 0.05, 0.60, 0.35, 0.05)
        with c3:
            min_oi = st.number_input("Min Open Interest", value=1, step=1, min_value=0)
            max_spread_pct = st.number_input("Max spread % of mid", value=50.0, step=5.0, min_value=0.0)

        side_filter = st.multiselect("Option side(s)", ["Call", "Put"], default=["Call", "Put"])

    # Check if ticker is crypto
    is_crypto = ticker.upper() in ['BTC-USD', 'ETH-USD']
    currency = 'BTC' if ticker.upper() == 'BTC-USD' else 'ETH' if ticker.upper() == 'ETH-USD' else None

    if is_crypto:
        expiries = get_deribit_expiries(currency)
    else:
        expiries = get_yahoo_expiries(ticker)

    if not expiries:
        st.info("No option chain for this ticker. Try AAPL, MSFT, BTC-USD, ETH-USD or wait/try again.")
        return

    e1, e2 = st.columns([2, 1])
    with e1:
        expiry = st.selectbox("Expiry", expiries, index=0)
    with e2:
        show_download = st.checkbox("Enable CSV download", value=True)

    try:
        S0 = _load_underlying_price(ticker)
    except Exception as exc:
        st.error(f"Failed to load underlying price: {exc}")
        return

    rf = rf_pct / 100.0
    now = pd.Timestamp.now(tz="UTC")
    exp_dt = pd.to_datetime(expiry)
    if exp_dt.tzinfo is None:
        exp_dt = exp_dt.tz_localize("UTC")
    else:
        exp_dt = exp_dt.tz_convert("UTC")
    exp_dt = exp_dt + pd.Timedelta(hours=16)
    T_years = max((exp_dt - now).total_seconds() / (365.0 * 24 * 3600.0), 1.0 / 365.0)

    try:
        rv_hist = _yf.download(ticker, period="6mo", interval="1d", progress=False)
        rv = np.nan
        if not rv_hist.empty:
            logret = np.log(rv_hist["Close"]).diff()
            look = int(period_for_rv.strip("d"))
            rv = float(logret.tail(look).std(ddof=1) * np.sqrt(252))
    except Exception:
        rv = np.nan

    if is_crypto:
        chain_df = get_deribit_option_chain(currency, expiry)
    else:
        tk = _yf.Ticker(ticker)
        try:
            ch = tk.option_chain(expiry)
            calls = ch.calls.copy()
            puts = ch.puts.copy()
        except Exception as exc:
            st.error(f"Failed to load option chain: {exc}")
            return

        def _prep(df, typ):
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            df["type"] = typ
            for col in ["bid", "ask", "lastPrice", "openInterest", "volume", "impliedVolatility"]:
                if col not in df.columns:
                    df[col] = np.nan
            df["mid"] = [_clean_mid(b, a, l) for b, a, l in zip(df["bid"], df["ask"], df["lastPrice"])]
            df["spread_pct"] = np.where(df["mid"] > 0, (df["ask"] - df["bid"]) / df["mid"] * 100.0, np.nan)
            df["moneyness"] = np.log(df["strike"] / S0)
            df["intrinsic"] = [_intrinsic(S0, k, typ) for k in df["strike"]]
            iv_solver = [_iv_from_price(m, S0, float(k), rf, T_years, typ) if np.isfinite(m) else np.nan for k, m in zip(df["strike"], df["mid"])]
            df["iv_market"] = np.where(np.isfinite(iv_solver), iv_solver, df["impliedVolatility"].astype(float))
            return df

        calls_p = _prep(calls, "Call")
        puts_p = _prep(puts, "Put")
        chain_df = pd.concat([calls_p, puts_p], ignore_index=True)

    chain = chain_df[pd.notna(chain_df["mid"])]
    chain = chain[chain["openInterest"].fillna(0) >= min_oi]
    chain = chain[chain["spread_pct"].fillna(1e9) <= max_spread_pct]
    if side_filter:
        chain = chain[chain["type"].isin(side_filter)]
    chain = chain.sort_values(["type", "strike"]).reset_index(drop=True)

    if chain.empty:
        st.warning("No options left after filters. Relax filters or pick another expiry.")
        return

    fit_df = chain[abs(chain["moneyness"]) <= mny_band].copy()
    fit_df = fit_df[pd.notna(fit_df["iv_market"]) & (fit_df["iv_market"] > 0)]
    if len(fit_df) >= 5:
        w = 1.0 / np.clip(fit_df["spread_pct"].fillna(50.0).values, 1.0, 200.0)
        x = fit_df["moneyness"].values
        y = fit_df["iv_market"].values
        coef = np.polyfit(x, y, deg=2, w=w)
        chain["iv_fitted"] = np.polyval(coef, chain["moneyness"].values)
        chain["iv_fitted"] = chain["iv_fitted"].clip(lower=0.01, upper=3.0)
    else:
        chain["iv_fitted"] = np.nan

    fair_vals = [ _bs_price(S0, float(k), rf, float(ivf), T_years, typ) if pd.notna(ivf) and ivf > 0 else np.nan
                 for typ, k, ivf in zip(chain["type"], chain["strike"], chain["iv_fitted"]) ]
    chain["fair"] = fair_vals
    chain["edge"] = chain["mid"] - chain["fair"]
    chain["edge_pct"] = chain["edge"] / chain["mid"] * 100.0

    atm_row = chain.iloc[(chain["strike"] - S0).abs().argsort()[:1]]
    atm_iv = float(atm_row["iv_market"].values[0]) if not atm_row.empty and pd.notna(atm_row["iv_market"].values[0]) else np.nan

    term_rows = []
    for e in expiries[:8]:
        try:
            if is_crypto:
                # For crypto, skip term structure for simplicity or implement if needed
                continue
            ch_e = tk.option_chain(e)
            df_e = pd.concat([ch_e.calls.assign(type="Call"), ch_e.puts.assign(type="Put")], ignore_index=True)
            df_e["mid"] = [_clean_mid(b, a, l) for b, a, l in zip(df_e["bid"], df_e["ask"], df_e["lastPrice"])]
            df_e = df_e[pd.notna(df_e["mid"])]
            if df_e.empty:
                continue
            k_atm = float(df_e.iloc[(df_e["strike"] - S0).abs().argsort()[:1]]["strike"].values[0])
            atm_slice = df_e[(df_e["strike"] == k_atm) & (df_e["type"] == "Call")]
            if atm_slice.empty:
                atm_slice = df_e[df_e["strike"] == k_atm]
            m = float(atm_slice["mid"].iloc[0])
            T_e = max((pd.to_datetime(e) + pd.Timedelta(hours=16) - now).total_seconds() / (365.0 * 24 * 3600.0), 1.0 / 365.0)
            iv_e = _iv_from_price(m, S0, k_atm, rf, T_e, "Call")
            if not pd.notna(iv_e):
                iv_e = float(atm_slice["impliedVolatility"].iloc[0]) if "impliedVolatility" in atm_slice.columns else np.nan
            term_rows.append({"expiry": e, "T_days": int(T_e * 365), "atm_iv": iv_e})
        except Exception:
            continue
    term_df = pd.DataFrame(term_rows)

    m1, m2, m3 = st.columns(3)
    m1.metric("Underlying (S₀)", f"{S0:,.2f}")
    m2.metric("ATM IV (this expiry)", f"{atm_iv*100:.2f}%" if pd.notna(atm_iv) else "—")
    m3.metric(f"Realized Vol ({period_for_rv})", f"{rv*100:.2f}%" if pd.notna(rv) else "—")

    st.subheader("Vol Smile (market vs. fitted)")
    sm_df = chain[["moneyness", "iv_market", "iv_fitted", "type", "strike"]].copy()
    sm_df["iv_market_%"] = sm_df["iv_market"] * 100.0
    sm_df["iv_fitted_%"] = sm_df["iv_fitted"] * 100.0
    scatter = alt.Chart(sm_df).mark_circle(size=60).encode(
        x=alt.X("moneyness:Q", title="log(K/S₀)"),
        y=alt.Y("iv_market_%:Q", title="IV market (%)"),
        color="type:N",
        tooltip=["type", "strike", alt.Tooltip("iv_market_%:Q", format=".2f")]
    )
    if pd.notna(sm_df["iv_fitted"].dropna().mean() if len(sm_df)>0 else np.nan):
        fitline = alt.Chart(sm_df).transform_loess("moneyness", "iv_fitted_%", groupby=["type"], bandwidth=0.8).mark_line(size=2).encode(x="moneyness:Q", y="iv_fitted_%:Q", color="type:N")
        st.altair_chart((scatter + fitline).interactive(), use_container_width=True)
    else:
        st.altair_chart(scatter.interactive(), use_container_width=True)

    if not term_df.empty:
        st.subheader("Term Structure (ATM IV across expiries)")
        term_df["atm_iv_%"] = term_df["atm_iv"] * 100.0
        ts = alt.Chart(term_df).mark_line(point=True).encode(
            x=alt.X("T_days:Q", title="Time to expiry (days)"),
            y=alt.Y("atm_iv_%:Q", title="ATM IV (%)"),
            tooltip=["expiry", "T_days", alt.Tooltip("atm_iv_%:Q", format=".2f")]
        )
        st.altair_chart(ts.interactive(), use_container_width=True)

    st.subheader("Option Chain (filtered) with Fair Value & Edge")
    view_cols = ["type", "strike", "bid", "ask", "mid", "openInterest", "volume",
                 "moneyness", "iv_market", "iv_fitted", "fair", "edge", "edge_pct", "spread_pct"]
    show = chain[view_cols].copy()
    show = show.sort_values("edge", ascending=False)
    show.rename(columns={
        "openInterest": "OI",
        "volume": "Vol",
        "iv_market": "IV_mkt",
        "iv_fitted": "IV_fit",
        "edge": "Edge($)",
        "edge_pct": "Edge(%)",
        "spread_pct": "Spread(%)"
    }, inplace=True)
    fmt_cols_pct = ["IV_mkt", "IV_fit"]
    for c in fmt_cols_pct:
        show[c] = show[c] * 100.0
    st.dataframe(show.round({
        "bid": 2, "ask": 2, "mid": 2, "moneyness": 4, "IV_mkt": 2, "IV_fit": 2,
        "fair": 2, "Edge($)": 2, "Edge(%)": 2, "Spread(%)": 1
    }), use_container_width=True, hide_index=True)

    if show_download:
        st.download_button(
            "Download CSV — Mispricing table",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_{expiry}_mispricing.csv",
            mime="text/csv"
        )

render_mispricing_checker()
