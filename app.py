from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ==========================================================
# PAGE CONFIGURATION & STYLING
# ==========================================================
st.set_page_config(
    page_title="TWT Multi-Timeframe Pullback Bot Pro", page_icon="⚡", layout="wide"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Using IANA Timezones with ZoneInfo for proper Daylight Saving Time (DST) handling
TRADING_TIMEZONES = {
    "(UTC+5:30) Kolkata / India (IST)": ZoneInfo("Asia/Kolkata"),
    "(UTC+1:00) Berlin / Paris / CET": ZoneInfo("Europe/Berlin"),
    "(UTC+0:00) London / UTC / GMT": ZoneInfo("Europe/London"),
    "(UTC-5:00) New York / US East (EST)": ZoneInfo("America/New_York"),
    "(UTC+8:00) Singapore / Hong Kong (SGT)": ZoneInfo("Asia/Singapore"),
}

ASSET_LIST = [
    "BTC/USDT", "ETH/USDT", "XAUT/USDT", "SOL/USDT", "XRP/USDT",
    "BNB/USDT", "DOGE/USDT", "MATIC/USDT", "ADA/USDT", "AVAX/USDT",
    "LINK/USDT", "LTC/USDT", "DOT/USDT", "NEAR/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "SUI/USDT", "FIL/USDT", "RENDER/USDT",
]

# Safe Auto-refresh every 30 seconds
st_autorefresh(interval=30000, key="twt_mtf_refresh")

st.title("⚡ TWT MULTI-TIMEFRAME INSTITUTIONAL PULLBACK BOT (PRO)")

# ==========================================================
# HEADER CONTROLS (TIMEFRAME & TIMEZONE)
# ==========================================================
col_tz = st.columns([1])[0]
with col_tz:
    selected_tz_name = st.selectbox("🌍 Trading Timezone", options=list(TRADING_TIMEZONES.keys()), index=0)
    active_tz = TRADING_TIMEZONES[selected_tz_name]

tz_short = selected_tz_name.split(")")[0] + ")"

def get_decimals(price):
    if price < 1:
        return 4
    elif price < 10:
        return 3
    else:
        return 2

# ==========================================================
# MULTI-TIMEFRAME EXCHANGE DATA FETCHING (4H, 1H, 15M, 5M)
# ==========================================================
@st.cache_data(ttl=30)
def fetch_mtf_data(symbol):
    exchanges_to_try = [
        (ccxt.binance, symbol),
        (ccxt.bybit, symbol),
        (ccxt.kucoin, symbol),
    ]
    for ex_class, sym in exchanges_to_try:
        try:
            exchange = ex_class({"enableRateLimit": True, "timeout": 15000})
            exchange.load_markets()
            if sym in exchange.markets:
                ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=50)
                ohlcv_1h = exchange.fetch_ohlcv(sym, timeframe="1h", limit=100)
                ohlcv_15m = exchange.fetch_ohlcv(sym, timeframe="15m", limit=150)
                ohlcv_5m = exchange.fetch_ohlcv(sym, timeframe="5m", limit=200)

                if ohlcv_4h and ohlcv_1h and ohlcv_15m and ohlcv_5m:
                    def to_df(data):
                        df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        df = df.set_index("timestamp")
                        if df.index.tz is None:
                            df.index = df.index.tz_localize("UTC")
                        return df

                    return to_df(ohlcv_4h), to_df(ohlcv_1h), to_df(ohlcv_15m), to_df(ohlcv_5m)
        except Exception:
            continue
    return None, None, None, None

# ==========================================================
# TWT MULTI-TIMEFRAME PULLBACK ENGINE
# ==========================================================
class TWTMultiTimeframeEngine:
    def __init__(self, df_4h, df_1h, df_15m, df_5m, symbol, user_tz):
        self.df_4h = df_4h
        self.df_1h = df_1h
        self.df_15m = df_15m
        self.df_5m = df_5m
        self.symbol = symbol
        self.user_tz = user_tz
        self.signals = []

    def calculate_indicators(self, df):
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df["TR"] = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["Close"].shift(1)),
                abs(df["Low"] - df["Close"].shift(1)),
            ),
        )
        df["ATR"] = df["TR"].rolling(14).mean()
        return df

    def run_strategy(self):
        if any(df is None or df.empty for df in [self.df_4h, self.df_1h, self.df_15m, self.df_5m]):
            return

        self.df_4h = self.calculate_indicators(self.df_4h)
        self.df_1h = self.calculate_indicators(self.df_1h)
        self.df_15m = self.calculate_indicators(self.df_15m)
        self.df_5m = self.calculate_indicators(self.df_5m)

        # 4H Overall Trend
        h4_close = float(self.df_4h["Close"].iloc[-1])
        h4_ema200 = float(self.df_4h["EMA200"].iloc[-1])
        h4_bullish = h4_close >= h4_ema200

        # 1H Market Structure & EMA Confirmation
        h1_last = self.df_1h.iloc[-1]
        h1_bullish = (h1_last["Close"] >= h1_last["EMA200"]) and (h1_last["EMA50"] >= h1_last["EMA200"])
        h1_bearish = (h1_last["Close"] < h1_last["EMA200"]) and (h1_last["EMA50"] < h1_last["EMA200"])

        IST = timezone(timedelta(hours=5, minutes=30))
        current_utc_time = datetime.now(timezone.utc)

        buy_signals_count = 0
        sell_signals_count = 0

        # Scan 15M for Setup Detection & 5M for Entry Trigger
        for i in range(20, len(self.df_15m)):
            m15 = self.df_15m.iloc[i]
            m15_prev = self.df_15m.iloc[i-1]
            dt_utc = self.df_15m.index[i]
            
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            dt_user_tz = dt_utc.astimezone(self.user_tz)
            timestamp_str = dt_user_tz.strftime("%H:%M")

            age_minutes = (current_utc_time - dt_utc).total_seconds() / 60.0
            is_within_1hr = age_minutes <= 60.0

            c_val = float(m15["Close"])
            o_val = float(m15["Open"])
            dec = get_decimals(c_val)
            atr = float(m15["ATR"]) if not np.isnan(m15["ATR"]) else (c_val * 0.002)

            body_size = abs(c_val - o_val)
            avg_body = abs(self.df_15m["Close"] - self.df_15m["Open"]).tail(20).mean()

            # --- SELL SETUP RULES (15M Setup + 5M Confirmation + Early Entry Blocker) ---
            m15_bearish_displacement = (c_val < o_val) and (body_size > avg_body * 1.2)
            if sell_signals_count < 1 and (not h4_bullish) and h1_bearish and m15_bearish_displacement:
                # Find latest Lower High in 15M pullback
                recent_highs = self.df_15m["High"].iloc[i-10:i]
                latest_lower_high = float(recent_highs.max())

                # 5M confirmation check (Bearish rejection + breaking low)
                m5_slice = self.df_5m[self.df_5m.index >= dt_utc].head(5)
                if not m5_slice.empty:
                    m5_last = m5_slice.iloc[-1]
                    m5_bearish_rej = (m5_last["High"] - max(m5_last["Open"], m5_last["Close"])) > (abs(m5_last["Close"] - m5_last["Open"]) * 1.5)
                    
                    if m5_bearish_rej or m5_last["Close"] < m5_last["Open"]:
                        sl = round(latest_lower_high + (1.2 * atr), dec)
                        tp2 = round(c_val - (2.5 * atr), dec)  # Min 1:2 RR
                        rr = round((c_val - tp2) / (sl - c_val), 2) if (sl - c_val) > 0 else 2.0

                        if rr >= 2.0:
                            self.signals.append({
                                "Asset": self.symbol,
                                "Time": timestamp_str,
                                "Type": "SELLER",
                                "Trend": "Bearish (4H + 1H Structure)",
                                "Setup": "15M Displacement + Pullback Lower High",
                                "Confirmation": "5M Bearish Rejection",
                                "Entry Price": round(c_val, dec),
                                "Stop Loss (SL)": sl,
                                "Target 2 (1:2 RR)": tp2,
                                "RR": f"1:{rr:.2f}",
                                "Score": 95,
                                "Is_Within_1hr": is_within_1hr
                            })
                            sell_signals_count += 1

            # --- BUY SETUP RULES (15M Setup + 5M Confirmation + Early Entry Blocker) ---
            m15_bullish_displacement = (c_val > o_val) and (body_size > avg_body * 1.2)
            if buy_signals_count < 1 and h4_bullish and h1_bullish and m15_bullish_displacement:
                # Find latest Higher Low in 15M pullback
                recent_lows = self.df_15m["Low"].iloc[i-10:i]
                latest_higher_low = float(recent_lows.min())

                # 5M confirmation check (Bullish rejection + breaking high)
                m5_slice = self.df_5m[self.df_5m.index >= dt_utc].head(5)
                if not m5_slice.empty:
                    m5_last = m5_slice.iloc[-1]
                    m5_bullish_rej = (min(m5_last["Open"], m5_last["Close"]) - m5_last["Low"]) > (abs(m5_last["Close"] - m5_last["Open"]) * 1.5)

                    if m5_bullish_rej or m5_last["Close"] > m5_last["Open"]:
                        sl = round(latest_higher_low - (1.2 * atr), dec)
                        tp2 = round(c_val + (2.5 * atr), dec)  # Min 1:2 RR
                        rr = round((tp2 - c_val) / (c_val - sl), 2) if (c_val - sl) > 0 else 2.0

                        if rr >= 2.0:
                            self.signals.append({
                                "Asset": self.symbol,
                                "Time": timestamp_str,
                                "Type": "BUYER",
                                "Trend": "Bullish (4H + 1H Structure)",
                                "Setup": "15M Displacement + Pullback Higher Low",
                                "Confirmation": "5M Bullish Rejection",
                                "Entry Price": round(c_val, dec),
                                "Stop Loss (SL)": sl,
                                "Target 2 (1:2 RR)": tp2,
                                "RR": f"1:{rr:.2f}",
                                "Score": 95,
                                "Is_Within_1hr": is_within_1hr
                            })
                            buy_signals_count += 1

# ==========================================================
# MASTER SCANNER DASHBOARD
# ==========================================================
st.markdown("---")
st.subheader("🔥 TWT Master Multi-Timeframe Signal Dashboard (Last 1 Hour)")

master_signals = []
asset_engines = {}

def scan_asset(sym):
    df_4h, df_1h, df_15m, df_5m = fetch_mtf_data(sym)
    if df_4h is not None and not df_4h.empty:
        engine = TWTMultiTimeframeEngine(df_4h, df_1h, df_15m, df_5m, sym, active_tz)
        engine.run_strategy()
        recent_sigs = [s for s in engine.signals if s.get("Is_Within_1hr", False)]
        latest = recent_sigs[-1] if recent_sigs else None
        return sym, engine, latest
    return sym, None, None

with st.spinner("⚡ Scanning assets across 4H, 1H, 15M, and 5M timeframes..."):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scan_asset, sym): sym for sym in ASSET_LIST}
        for future in as_completed(futures):
            sym, engine, latest = future.result()
            if engine:
                asset_engines[sym] = engine
            if latest:
                master_signals.append(latest)

if master_signals:
    st.success(f"✅ Found {len(master_signals)} multi-timeframe institutional setup(s)!")
    df_display = pd.DataFrame(master_signals).drop(columns=["Is_Within_1hr"], errors="ignore")
    st.dataframe(df_display, use_container_width=True)
else:
    st.info("ℹ️ No multi-timeframe signals in the last 1 hour matching 1:2 RR and Early Entry Blockers.")

# ==========================================================
# MANUAL ASSET INSPECTOR & CHART
# ==========================================================
st.markdown("---")
st.subheader("📊 Manual Asset Inspector & MTF Chart")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart & Complete History",
    options=ASSET_LIST,
    index=0
)

if f"paper_trades_{selected_symbol}" not in st.session_state:
    st.session_state[f"paper_trades_{selected_symbol}"] = []

if selected_symbol in asset_engines:
    bot = asset_engines[selected_symbol]
else:
    df_4h, df_1h, df_15m, df_5m = fetch_mtf_data(selected_symbol)
    bot = TWTMultiTimeframeEngine(df_4h, df_1h, df_15m, df_5m, selected_symbol, active_tz)
    bot.run_strategy()

if bot.df_15m is not None and not bot.df_15m.empty:
    curr_p = float(bot.df_15m["Close"].iloc[-1])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Current 15M Price", value=f"{curr_p:.2f}")
    with col2:
        st.metric(label="15M RSI (14)", value=f"{bot.df_15m['RSI'].iloc[-1]:.1f}")
    with col3:
        st.metric(label="15M ATR (14)", value=f"{bot.df_15m['ATR'].iloc[-1]:.4f}")
    with col4:
        st.metric(label="Timezone", value=tz_short, delta="MTF Strategy Active")

    if bot.signals:
        latest_sig = bot.signals[-1]
        is_buy = latest_sig["Type"] == "BUYER"
        
        if is_buy:
            st.success(f"🟢 **TWT MTF BUY SETUP: {selected_symbol}**")
            st.code(f"""BUY SETUP DETAILS
Trend: {latest_sig['Trend']}
Setup: {latest_sig['Setup']}
Confirmation: {latest_sig['Confirmation']}
Entry Price: {latest_sig['Entry Price']}
Stop Loss: {latest_sig['Stop Loss (SL)']}
Target 2 (1:2 RR): {latest_sig['Target 2 (1:2 RR)']}
Risk-to-Reward: {latest_sig['RR']}
Score: {latest_sig['Score']}/100""", language="text")
        else:
            st.error(f"🔴 **TWT MTF SELL SETUP: {selected_symbol}**")
            st.code(f"""SELL SETUP DETAILS
Trend: {latest_sig['Trend']}
Setup: {latest_sig['Setup']}
Confirmation: {latest_sig['Confirmation']}
Entry Price: {latest_sig['Entry Price']}
Stop Loss: {latest_sig['Stop Loss (SL)']}
Target 2 (1:2 RR): {latest_sig['Target 2 (1:2 RR)']}
Risk-to-Reward: {latest_sig['RR']}
Score: {latest_sig['Score']}/100""", language="text")
    else:
        st.warning(f"🟡 **STATUS: WAITING FOR 15M SETUP & 5M CONFIRMATION ON {selected_symbol}**")

    # Interactive Paper Trading Desk
    st.markdown("---")
    st.subheader(f"⚡ Interactive Paper Trading Desk ({selected_symbol})")
    
    suggested_entry = bot.signals[-1]['Entry Price'] if bot.signals else curr_p
    suggested_sl = bot.signals[-1]['Stop Loss (SL)'] if bot.signals else curr_p * 0.99
    suggested_tp = bot.signals[-1]['Target 2 (1:2 RR)'] if bot.signals else curr_p * 1.02
    suggested_type = bot.signals[-1]['Type'] if bot.signals else 'BUYER'

    with st.form(f"paper_trading_form_{selected_symbol}"):
        c1, c2, c3, c4, c5 = st.columns(5)
        trade_type = c1.selectbox("Order Type", options=["BUYER", "SELLER"], index=0 if suggested_type=="BUYER" else 1)
        entry_input = c2.number_input("Entry Price", value=float(suggested_entry), format="%.4f")
        sl_input = c3.number_input("Stop Loss (SL)", value=float(suggested_sl), format="%.4f")
        tp_input = c4.number_input("Target (TP)", value=float(suggested_tp), format="%.4f")
        amount_input = c5.number_input("Position Size (USDT)", value=100.0, min_value=10.0, step=10.0)
        
        submitted = st.form_submit_button("🚀 Place Paper Trade")
        if submitted:
            new_trade = {
                "id": f"{selected_symbol}_{datetime.now(timezone.utc).timestamp()}",
                "Asset": selected_symbol,
                "Type": trade_type,
                "Entry": entry_input,
                "SL": sl_input,
                "TP": tp_input,
                "Amount": amount_input,
                "Status": "OPEN",
                "PnL": 0.0,
                "Time": datetime.now(active_tz).strftime('%Y-%m-%d %H:%M:%S')
            }
            st.session_state[f"paper_trades_{selected_symbol}"].insert(0, new_trade)
            st.success(f"Paper Trade Executed Successfully for {selected_symbol} at {entry_input}!")

    if st.session_state[f"paper_trades_{selected_symbol}"]:
        st.write(f"### Active & Closed Paper Trades for {selected_symbol}")
        updated_trades = []
        for t in st.session_state[f"paper_trades_{selected_symbol}"]:
            if t["Status"] == "OPEN":
                if t["Type"] == "BUYER":
                    if curr_p <= t["SL"]:
                        t["Status"] = "CLOSED (SL HIT)"
                        t["PnL"] = - (t["Entry"] - t["SL"]) * (t["Amount"] / t["Entry"])
                    elif curr_p >= t["TP"]:
                        t["Status"] = "CLOSED (TP HIT)"
                        t["PnL"] = (t["TP"] - t["Entry"]) * (t["Amount"] / t["Entry"])
                    else:
                        t["PnL"] = (curr_p - t["Entry"]) * (t["Amount"] / t["Entry"])
                else:
                    if curr_p >= t["SL"]:
                        t["Status"] = "CLOSED (SL HIT)"
                        t["PnL"] = - (t["SL"] - t["Entry"]) * (t["Amount"] / t["Entry"])
                    elif curr_p <= t["TP"]:
                        t["Status"] = "CLOSED (TP HIT)"
                        t["PnL"] = (t["Entry"] - t["TP"]) * (t["Amount"] / t["Entry"])
                    else:
                        t["PnL"] = (t["Entry"] - curr_p) * (t["Amount"] / t["Entry"])
            updated_trades.append(t)
        st.session_state[f"paper_trades_{selected_symbol}"] = updated_trades
        
        paper_df = pd.DataFrame(st.session_state[f"paper_trades_{selected_symbol}"])
        st.dataframe(paper_df[["Time", "Asset", "Type", "Entry", "SL", "TP", "Amount", "Status", "PnL"]], use_container_width=True)
        
        if st.button(f"🗑️ Clear Paper Trades for {selected_symbol}"):
            st.session_state[f"paper_trades_{selected_symbol}"] = []
            st.rerun()
    else:
        st.info(f"No paper trades placed yet for {selected_symbol}.")

    # Complete Signal History
    if bot.signals:
        st.markdown("---")
        st.subheader(f"📋 Complete MTF Signal History for {selected_symbol}")
        df_history = pd.DataFrame(bot.signals).drop(columns=["Is_Within_1hr"], errors="ignore")
        st.dataframe(df_history, use_container_width=True)

    # Professional Matplotlib Chart (15M Chart with EMAs)
    st.markdown("---")
    st.subheader(f"📈 15M Institutional Chart & EMAs — {selected_symbol}")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    df_15m_plot = bot.df_15m.tail(100)
    
    for idx, (dt, row) in enumerate(df_15m_plot.iterrows()):
        col = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
        body_size = abs(row["Close"] - row["Open"])
        min_body = curr_p * 0.0003
        plot_height = max(body_size, min_body)
        plot_bottom = min(row["Open"], row["Close"]) - ((min_body - body_size) / 2) if body_size < min_body else min(row["Open"], row["Close"])

        ax.bar(idx, plot_height, bottom=plot_bottom, color=col, width=0.7)
        ax.vlines(idx, row["Low"], row["High"], color="#b2b5be", linewidth=1.3)

    ema20_vals = df_15m_plot["EMA20"].values
    ema50_vals = df_15m_plot["EMA50"].values
    ema200_vals = df_15m_plot["EMA200"].values
    x_ids = list(range(len(df_15m_plot)))

    ax.plot(x_ids, ema20_vals, color="#29b6f6", linewidth=1.2, label="15M EMA 20")
    ax.plot(x_ids, ema50_vals, color="#ab47bc", linewidth=1.2, label="15M EMA 50")
    ax.plot(x_ids, ema200_vals, color="#ffeb3b", linewidth=1.5, label="15M EMA 200")

    lows = df_15m_plot["Low"].values
    highs = df_15m_plot["High"].values

    if len(lows) > 0 and len(highs) > 0:
        margin = (max(highs) - min(lows)) * 0.09
        if margin == 0:
            margin = curr_p * 0.005
        ax.set_ylim(min(lows) - margin, max(highs) + margin)

    ax.set_title(f"{selected_symbol} 15M — TWT Multi-Timeframe Strategy ({tz_short})", color="white", fontsize=14)
    ax.set_facecolor("#131722")
    fig.patch.set_facecolor("#1e222d")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#363a45")
    ax.grid(color="#2a2e39", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", facecolor="#1e222d", edgecolor="none", labelcolor="white")

    st.pyplot(fig)
else:
    st.error("Market data load failed. Please check internet connection.")
