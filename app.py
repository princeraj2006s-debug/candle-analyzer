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
    page_title="TWT Institutional Pullback Strategy Pro", page_icon="⚡", layout="wide"
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
st_autorefresh(interval=30000, key="twt_institutional_refresh")

st.title("⚡ TWT INSTITUTIONAL PULLBACK STRATEGY (PRO)")

# ==========================================================
# HEADER CONTROLS (TIMEFRAME & TIMEZONE)
# ==========================================================
col_tf, col_tz = st.columns([1, 2])
with col_tf:
    selected_tf = st.selectbox("⏱️ Timeframe", options=["5m", "15m", "30m", "1h"], index=1)
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
# ROBUST EXCHANGE DATA FETCHING
# ==========================================================
@st.cache_data(ttl=30)
def fetch_institutional_data(symbol, tf):
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
                ohlcv_tf = exchange.fetch_ohlcv(sym, timeframe=tf, limit=200)
                ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=50)
                if ohlcv_tf and ohlcv_4h:
                    df = pd.DataFrame(ohlcv_tf, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df = df.set_index("timestamp")

                    df_4h = pd.DataFrame(ohlcv_4h, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                    df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], unit="ms")
                    df_4h = df_4h.set_index("timestamp")
                    
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    if df_4h.index.tz is None:
                        df_4h.index = df_4h.index.tz_localize("UTC")

                    return df, df_4h
        except Exception:
            continue
    return None, None

# ==========================================================
# TWT INSTITUTIONAL PULLBACK ENGINE
# ==========================================================
class TWTInstitutionalPullbackEngine:
    def __init__(self, df, df_4h, symbol, user_tz):
        self.df = df
        self.df_4h = df_4h
        self.symbol = symbol
        self.user_tz = user_tz
        self.candles = []
        self.signals = []

    def calculate_and_scan(self):
        if self.df is None or self.df.empty:
            return

        # 1. Indicators: EMA 20, 50, 200, RSI 14, ATR 14
        self.df["EMA20"] = self.df["Close"].ewm(span=20, adjust=False).mean()
        self.df["EMA50"] = self.df["Close"].ewm(span=50, adjust=False).mean()
        self.df["EMA200"] = self.df["Close"].ewm(span=200, adjust=False).mean()

        delta = self.df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df["RSI"] = 100 - (100 / (1 + rs))

        self.df["TR"] = np.maximum(
            self.df["High"] - self.df["Low"],
            np.maximum(
                abs(self.df["High"] - self.df["Close"].shift(1)),
                abs(self.df["Low"] - self.df["Close"].shift(1)),
            ),
        )
        self.df["ATR"] = self.df["TR"].rolling(14).mean()

        # 4H Overall Direction Benchmark
        self.df_4h["EMA200"] = self.df_4h["Close"].ewm(span=200, adjust=False).mean()
        h4_price = float(self.df_4h["Close"].iloc[-1])
        h4_ema200 = float(self.df_4h["EMA200"].iloc[-1])
        global_trend_bullish = h4_price >= h4_ema200

        IST = timezone(timedelta(hours=5, minutes=30))
        current_utc_time = datetime.now(timezone.utc)

        buy_count_today = 0
        sell_count_today = 0

        for idx, row in enumerate(self.df.iterrows()):
            dt_utc = row[0]
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            dt_ist = dt_utc.astimezone(IST)
            dt_user_tz = dt_utc.astimezone(self.user_tz)
            timestamp_str = dt_user_tz.strftime("%H:%M")

            r = row[1]
            o, h, l, c, v = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]), float(r["Volume"])
            ema20 = float(r["EMA20"]) if not pd.isna(r["EMA20"]) else c
            ema50 = float(r["EMA50"]) if not pd.isna(r["EMA50"]) else c
            ema200 = float(r["EMA200"]) if not pd.isna(r["EMA200"]) else c
            rsi = float(r["RSI"]) if not pd.isna(r["RSI"]) else 50.0
            atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)

            age_minutes = (current_utc_time - dt_utc).total_seconds() / 60.0
            is_within_1hr = age_minutes <= 60.0

            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            rng = h - l
            upper_wick_pct = (upper_wick / rng * 100) if rng > 0 else 0
            lower_wick_pct = (lower_wick / rng * 100) if rng > 0 else 0

            self.candles.append({
                "id": idx + 1,
                "time": timestamp_str,
                "dt_ist": dt_ist,
                "dt_user": dt_user_tz,
                "color": "Green" if c >= o else "Red",
                "open": o, "high": h, "low": l, "close": c, "volume": v,
                "atr": atr, "rsi": rsi,
                "upper_wick_pct": upper_wick_pct,
                "lower_wick_pct": lower_wick_pct,
                "is_within_1hr": is_within_1hr
            })

            if idx < 20:
                continue

            dec = get_decimals(c)

            # Pullback zone confirmation: Price retraces into EMA 20-50 zone
            in_pullback_zone = min(ema20, ema50) <= c <= max(ema20, ema50)

            # BUY Entry Rules: 4H price > EMA200, EMA50 >= EMA200, Pullback Zone, Lower Wick 45-57%
            if buy_count_today < 1 and global_trend_bullish and ema50 >= ema200 and in_pullback_zone and (45.0 <= lower_wick_pct <= 57.0):
                sl = round(l - (1.2 * atr), dec)
                tp1 = round(c + (1.2 * atr), dec)
                tp2 = round(c + (2.5 * atr), dec) # Minimum 1:2 RR Target
                rr = round((tp2 - c) / (c - sl), 2) if (c - sl) > 0 else 2.0

                self.signals.append({
                    "Asset": self.symbol,
                    "Time": timestamp_str,
                    "Candle #": idx + 1,
                    "Type": "BUYER",
                    "Trend": "Bullish (4H + EMA 50/200)",
                    "Zone": "EMA 20-50 Pullback Zone",
                    "Score": 95,
                    "WickRejection": f"Lower Wick {lower_wick_pct:.1f}% (45-57%)",
                    "Entry Price": round(c, dec),
                    "Stop Loss (SL)": sl,
                    "Target 1": tp1,
                    "Target 2 (1:2 RR)": tp2,
                    "RR": f"1:{rr:.2f}",
                    "Status": "VALID",
                    "Is_Within_1hr": is_within_1hr
                })
                buy_count_today += 1

            # SELL Entry Rules: 4H price < EMA200, EMA50 <= EMA200, Pullback Zone, Upper Wick 45-57%
            elif sell_count_today < 1 and (not global_trend_bullish) and ema50 <= ema200 and in_pullback_zone and (45.0 <= upper_wick_pct <= 57.0):
                sl = round(h + (1.2 * atr), dec)
                tp1 = round(c - (1.2 * atr), dec)
                tp2 = round(c - (2.5 * atr), dec) # Minimum 1:2 RR Target
                rr = round((c - tp2) / (sl - c), 2) if (sl - c) > 0 else 2.0

                self.signals.append({
                    "Asset": self.symbol,
                    "Time": timestamp_str,
                    "Candle #": idx + 1,
                    "Type": "SELLER",
                    "Trend": "Bearish (4H + EMA 50/200)",
                    "Zone": "EMA 20-50 Pullback Zone",
                    "Score": 95,
                    "WickRejection": f"Upper Wick {upper_wick_pct:.1f}% (45-57%)",
                    "Entry Price": round(c, dec),
                    "Stop Loss (SL)": sl,
                    "Target 1": tp1,
                    "Target 2 (1:2 RR)": tp2,
                    "RR": f"1:{rr:.2f}",
                    "Status": "VALID",
                    "Is_Within_1hr": is_within_1hr
                })
                sell_count_today += 1

# ==========================================================
# MASTER SCANNER DASHBOARD
# ==========================================================
st.markdown("---")
st.subheader("🔥 TWT Master Institutional Signal Dashboard (Last 1 Hour)")

master_signals = []
asset_engines = {}

def scan_asset(sym):
    df, df_4h = fetch_institutional_data(sym, selected_tf)
    if df is not None and not df.empty:
        engine = TWTInstitutionalPullbackEngine(df, df_4h, sym, active_tz)
        engine.calculate_and_scan()
        recent_sigs = [s for s in engine.signals if s.get("Is_Within_1hr", False)]
        latest = recent_sigs[-1] if recent_sigs else None
        return sym, engine, latest
    return sym, None, None

with st.spinner("⚡ Scanning assets with TWT Institutional Pullback Strategy..."):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scan_asset, sym): sym for sym in ASSET_LIST}
        for future in as_completed(futures):
            sym, engine, latest = future.result()
            if engine:
                asset_engines[sym] = engine
            if latest:
                master_signals.append(latest)

if master_signals:
    st.success(f"✅ Found {len(master_signals)} institutional setup(s)!")
    df_display = pd.DataFrame(master_signals).drop(columns=["Is_Within_1hr"], errors="ignore")
    st.dataframe(df_display, use_container_width=True)
else:
    st.info("ℹ️ No institutional signals in the last 1 hour. Waiting for valid pullback & rejection setups...")

# ==========================================================
# MANUAL ASSET INSPECTOR & CHART
# ==========================================================
st.markdown("---")
st.subheader("📊 Manual Asset Inspector & Institutional Chart")

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
    df, df_4h = fetch_institutional_data(selected_symbol, selected_tf)
    bot = TWTInstitutionalPullbackEngine(df, df_4h, selected_symbol, active_tz)
    bot.calculate_and_scan()

if bot.df is not None and not bot.df.empty:
    curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Current Price", value=f"{curr_p:.2f}")
    with col2:
        st.metric(label="Current RSI (14)", value=f"{bot.candles[-1]['rsi']:.1f}")
    with col3:
        st.metric(label="Current ATR (14)", value=f"{bot.candles[-1]['atr']:.4f}")
    with col4:
        st.metric(label="Timezone", value=tz_short, delta="Institutional Strategy Active")

    if bot.signals:
        latest_sig = bot.signals[-1]
        is_buy = latest_sig["Type"] == "BUYER"
        
        if is_buy:
            st.success(f"🟢 **TWT INSTITUTIONAL BUY SETUP: {selected_symbol}**")
            st.code(f"""BUY SETUP DETAILS
Trend: {latest_sig['Trend']}
Zone: {latest_sig['Zone']}
Rejection: {latest_sig['WickRejection']}
Entry Price: {latest_sig['Entry Price']}
Stop Loss: {latest_sig['Stop Loss (SL)']}
Target 2 (1:2 RR): {latest_sig['Target 2 (1:2 RR)']}
Risk-to-Reward: {latest_sig['RR']}
Score: {latest_sig['Score']}/100 (High-Quality Setup)""", language="text")
        else:
            st.error(f"🔴 **TWT INSTITUTIONAL SELL SETUP: {selected_symbol}**")
            st.code(f"""SELL SETUP DETAILS
Trend: {latest_sig['Trend']}
Zone: {latest_sig['Zone']}
Rejection: {latest_sig['WickRejection']}
Entry Price: {latest_sig['Entry Price']}
Stop Loss: {latest_sig['Stop Loss (SL)']}
Target 2 (1:2 RR): {latest_sig['Target 2 (1:2 RR)']}
Risk-to-Reward: {latest_sig['RR']}
Score: {latest_sig['Score']}/100 (High-Quality Setup)""", language="text")
    else:
        st.warning(f"🟡 **STATUS: WAITING FOR PULLBACK IN EMA 20-50 ZONE ON {selected_symbol}**")

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
        st.subheader(f"📋 Complete Signal History for {selected_symbol}")
        df_history = pd.DataFrame(bot.signals).drop(columns=["Is_Within_1hr"], errors="ignore")
        st.dataframe(df_history, use_container_width=True)

    # Professional Matplotlib Chart with EMAs
    st.markdown("---")
    st.subheader(f"📈 Institutional Chart & EMAs — {selected_symbol}")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    for c in bot.candles:
        col = "#26a69a" if c["color"] == "Green" else "#ef5350"
        body_size = abs(c["close"] - c["open"])
        min_body = curr_p * 0.0003
        plot_height = max(body_size, min_body)
        plot_bottom = min(c["open"], c["close"]) - ((min_body - body_size) / 2) if body_size < min_body else min(c["open"], c["close"])

        ax.bar(c["id"], plot_height, bottom=plot_bottom, color=col, width=0.7)
        ax.vlines(c["id"], c["low"], c["high"], color="#b2b5be", linewidth=1.3)

    # Plot EMAs
    ema20_vals = bot.df["EMA20"].values
    ema50_vals = bot.df["EMA50"].values
    ema200_vals = bot.df["EMA200"].values
    x_ids = [c["id"] for c in bot.candles]

    ax.plot(x_ids, ema20_vals, color="#29b6f6", linewidth=1.2, label="EMA 20")
    ax.plot(x_ids, ema50_vals, color="#ab47bc", linewidth=1.2, label="EMA 50")
    ax.plot(x_ids, ema200_vals, color="#ffeb3b", linewidth=1.5, label="EMA 200")

    lows = [c["low"] for c in bot.candles]
    highs = [c["high"] for c in bot.candles]

    if lows and highs:
        margin = (max(highs) - min(lows)) * 0.09
        if margin == 0:
            margin = curr_p * 0.005
        ax.set_ylim(min(lows) - margin, max(highs) + margin)

    for sig in bot.signals:
        is_b = sig["Type"] == "BUYER"
        bg_col = "#00e676" if is_b else "#ff5252"
        txt = " B " if is_b else " S "
        ax.text(
            sig["Candle #"], sig["Entry Price"], txt,
            color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_col, alpha=0.9),
            ha="center", va="center",
        )

    ax.set_title(f"{selected_symbol} {selected_tf} — TWT Institutional Pullback Strategy ({tz_short})", color="white", fontsize=14)
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
