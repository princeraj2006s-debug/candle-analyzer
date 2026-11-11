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
    page_title="TWT Advanced Pullback Bot Pro", page_icon="⚡", layout="wide"
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
    "BTC/USDT",
    "ETH/USDT",
    "XAUT/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "DOGE/USDT",
    "MATIC/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "DOT/USDT",
    "NEAR/USDT",
    "APT/USDT",
    "ARB/USDT",
    "OP/USDT",
    "SUI/USDT",
    "FIL/USDT",
    "RENDER/USDT",
]

# Safe Auto-refresh every 30 seconds
count = st_autorefresh(interval=30000, key="twt_advanced_refresh")

st.title("⚡ TWT ADVANCED BUYER & SELLER PULLBACK ENTRY SYSTEM (PRO)")

# ==========================================================
# HEADER CONTROLS (TIMEFRAME & TIMEZONE)
# ==========================================================
col_tf, col_tz = st.columns([1, 2])
with col_tf:
    selected_tf = st.selectbox("⏱️ Timeframe", options=["5m", "15m", "30m", "1h"], index=0)
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
# ROBUST EXCHANGE DATA FETCHING WITH MULTI-EXCHANGE FALLBACK
# ==========================================================
@st.cache_data(ttl=30)
def fetch_twt_delta_data(symbol, tf):
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
                ohlcv_tf = exchange.fetch_ohlcv(sym, timeframe=tf, limit=100)
                ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=25)
                ohlcv_6h = exchange.fetch_ohlcv(sym, timeframe="6h", limit=25)

                if ohlcv_tf and ohlcv_4h:
                    df_5 = pd.DataFrame(ohlcv_tf, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                    df_5["timestamp"] = pd.to_datetime(df_5["timestamp"], unit="ms")
                    df_5 = df_5.set_index("timestamp")

                    df_4 = pd.DataFrame(ohlcv_4h, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                    df_4["timestamp"] = pd.to_datetime(df_4["timestamp"], unit="ms")
                    df_4 = df_4.set_index("timestamp")

                    df_6 = pd.DataFrame(ohlcv_6h, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]) if ohlcv_6h else df_4.copy()
                    df_6["timestamp"] = pd.to_datetime(df_6["timestamp"], unit="ms")
                    df_6 = df_6.set_index("timestamp")
                    
                    if df_5.index.tz is None:
                        df_5.index = df_5.index.tz_localize("UTC")
                    if df_4.index.tz is None:
                        df_4.index = df_4.index.tz_localize("UTC")
                    if df_6.index.tz is None:
                        df_6.index = df_6.index.tz_localize("UTC")

                    return df_5, df_4, df_6
        except Exception:
            continue
    return None, None, None

# ==========================================================
# TWT ADVANCED PULLBACK ENGINE (STRICT WICK 45-57% & TREND MAP)
# ==========================================================
class TWTSafeSecondRedEngine:
    def __init__(self, df_5, df_4, df_6, symbol, user_tz):
        self.df_5 = df_5
        self.df_4 = df_4
        self.df_6 = df_6
        self.symbol = symbol
        self.user_tz = user_tz
        self.candles = []
        self.signals = []
        self.resistance_lvl = 0
        self.support_lvl = 0

    def calculate_and_scan(self):
        if self.df_5 is None or self.df_5.empty:
            return

        IST = timezone(timedelta(hours=5, minutes=30))
        current_utc_time = datetime.now(timezone.utc)

        # 1. Calculate Key Levels
        if self.df_4 is not None and not self.df_4.empty:
            idx_utc = self.df_4.index
            df_4_ist = idx_utc.tz_convert(IST)
            filtered_4h = self.df_4[df_4_ist.hour == 1]
            self.resistance_lvl = float(filtered_4h["High"].iloc[-1]) if not filtered_4h.empty else float(self.df_4["High"].iloc[-1])

        if self.df_6 is not None and not self.df_6.empty:
            idx_utc_6 = self.df_6.index
            df_6_ist = idx_utc_6.tz_convert(IST)
            filtered_6h = self.df_6[df_6_ist.hour == 5]
            self.support_lvl = float(filtered_6h["Low"].iloc[-1]) if not filtered_6h.empty else float(self.df_6["Low"].iloc[-1])
        elif self.df_4 is not None:
            self.support_lvl = float(self.df_4["Low"].iloc[-1])

        # 2. Indicators Calculation (EMA 20 for Trend Determination)
        self.df_5["TR"] = np.maximum(
            self.df_5["High"] - self.df_5["Low"],
            np.maximum(
                abs(self.df_5["High"] - self.df_5["Close"].shift(1)),
                abs(self.df_5["Low"] - self.df_5["Close"].shift(1)),
            ),
        )
        self.df_5["ATR"] = self.df_5["TR"].rolling(14).mean()
        self.df_5["EMA20"] = self.df_5["Close"].ewm(span=20, adjust=False).mean()

        for idx, row in enumerate(self.df_5.iterrows()):
            dt_utc = row[0]
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            dt_ist = dt_utc.astimezone(IST)
            dt_user_tz = dt_utc.astimezone(self.user_tz)
            timestamp_str = dt_user_tz.strftime("%H:%M")

            r = row[1]
            o, h, l, c, v = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]), float(r["Volume"])
            atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)
            ema20 = float(r["EMA20"]) if not pd.isna(r["EMA20"]) else c

            age_minutes = (current_utc_time - dt_utc).total_seconds() / 60.0
            is_within_1hr = age_minutes <= 60.0

            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            rng = h - l
            upper_wick_pct = (upper_wick / rng * 100) if rng > 0 else 0
            lower_wick_pct = (lower_wick / rng * 100) if rng > 0 else 0

            # Trend definition: Price > EMA20 is Uptrend, Price < EMA20 is Downtrend
            trend = "Uptrend" if c >= ema20 else "Downtrend"

            self.candles.append({
                "id": idx + 1,
                "time": timestamp_str,
                "dt_ist": dt_ist,
                "dt_user": dt_user_tz,
                "color": "Green" if c >= o else "Red",
                "open": o, "high": h, "low": l, "close": c, "volume": v,
                "atr": atr,
                "trend": trend,
                "upper_wick_pct": upper_wick_pct,
                "lower_wick_pct": lower_wick_pct,
                "is_within_1hr": is_within_1hr,
            })

        # --- STRICT 45%-57% WICK & TREND-MAPPED ENGINE (MAX 1 BUY & 1 SELL) ---
        buy_signal_found = False
        sell_signal_found = False

        for i in range(2, len(self.candles)):
            c = self.candles[i]
            prev2 = self.candles[i-2]
            dec = get_decimals(c["close"])
            atr = c["atr"] if not np.isnan(c["atr"]) else (c["close"] * 0.002)

            # 1. SELLER ENTRY: Strictly in DOWNREND when Upper Wick is between 45% and 57%
            if not sell_signal_found and c["trend"] == "Downtrend" and (45.0 <= prev2["upper_wick_pct"] <= 57.0):
                sl = round(c["high"] + (1.2 * atr), dec)
                tp1 = round(c["close"] - (1.2 * atr), dec)
                tp2 = round(c["close"] - (2.5 * atr), dec)
                tp3 = round(c["close"] - (4.0 * atr), dec)
                rr = round((c["close"] - tp1) / (sl - c["close"]), 2) if (sl - c["close"]) > 0 else 2.0

                self.signals.append({
                    "Asset": self.symbol,
                    "Time": c["time"],
                    "Candle #": c["id"],
                    "Type": "SELLER",
                    "Trend": "Downtrend",
                    "Pullback": "Pullback 3",
                    "State": "SIGNAL_VALID",
                    "SellerStrength": "Confirmed",
                    "BuyerWeakness": "Confirmed (Strict 45-57% Upper Wick)",
                    "WickRejection": f"Upper Wick {prev2['upper_wick_pct']:.1f}%",
                    "Confirmation": "Valid Downtrend Seller Rejection",
                    "Entry Price": round(c["close"], dec),
                    "Stop Loss (SL)": sl,
                    "TP 1": tp1,
                    "TP 2": tp2,
                    "TP 3": tp3,
                    "Target": tp1,
                    "RR": f"1:{rr:.2f}",
                    "Score": 95,
                    "Status": "VALID",
                    "Is_Within_1hr": c["is_within_1hr"]
                })
                sell_signal_found = True

            # 2. BUYER ENTRY: Strictly in UPTREND when Lower Wick is between 45% and 57%
            elif not buy_signal_found and c["trend"] == "Uptrend" and (45.0 <= prev2["lower_wick_pct"] <= 57.0):
                sl = round(c["low"] - (1.2 * atr), dec)
                tp1 = round(c["close"] + (1.2 * atr), dec)
                tp2 = round(c["close"] + (2.5 * atr), dec)
                tp3 = round(c["close"] + (4.0 * atr), dec)
                rr = round((tp1 - c["close"]) / (c["close"] - sl), 2) if (c["close"] - sl) > 0 else 2.0

                self.signals.append({
                    "Asset": self.symbol,
                    "Time": c["time"],
                    "Candle #": c["id"],
                    "Type": "BUYER",
                    "Trend": "Uptrend",
                    "Pullback": "Pullback 3",
                    "State": "SIGNAL_VALID",
                    "BuyerStrength": "Confirmed",
                    "SellerWeakness": "Confirmed (Strict 45-57% Lower Wick)",
                    "WickRejection": f"Lower Wick {prev2['lower_wick_pct']:.1f}%",
                    "Confirmation": "Valid Uptrend Buyer Rebound",
                    "Entry Price": round(c["close"], dec),
                    "Stop Loss (SL)": sl,
                    "TP 1": tp1,
                    "TP 2": tp2,
                    "TP 3": tp3,
                    "Target": tp1,
                    "RR": f"1:{rr:.2f}",
                    "Score": 95,
                    "Status": "VALID",
                    "Is_Within_1hr": c["is_within_1hr"]
                })
                buy_signal_found = True

# ==========================================================
# MASTER SCANNER (Global Feed for Top Dashboard)
# ==========================================================
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (Last 1 Hour)")

master_signals = []
asset_engines = {}

def scan_single_asset(sym):
    df_5, df_4, df_6 = fetch_twt_delta_data(sym, selected_tf)
    if df_5 is not None and not df_5.empty:
        engine = TWTSafeSecondRedEngine(df_5, df_4, df_6, sym, active_tz)
        engine.calculate_and_scan()
        recent_sigs = [s for s in engine.signals if s.get("Is_Within_1hr", False)]
        latest = recent_sigs[-1] if recent_sigs else None
        return sym, engine, latest
    return sym, None, None

with st.spinner("⚡ Scanning assets with Strict 45-57% Wick Engine..."):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scan_single_asset, sym): sym for sym in ASSET_LIST}
        for future in as_completed(futures):
            sym, engine, latest = future.result()
            if engine:
                asset_engines[sym] = engine
            if latest:
                master_signals.append(latest)

if master_signals:
    st.success(f"✅ Found {len(master_signals)} high-conviction setup(s)!")
    df_display = pd.DataFrame(master_signals).drop(columns=["Is_Within_1hr"], errors="ignore")
    st.dataframe(df_display, use_container_width=True)
else:
    st.info("ℹ️ No high-conviction signals matching 45-57% wick rule in the last 1 hour.")

# ==========================================================
# MANUAL ASSET SELECTOR & CHART VIEWER (ISOLATED)
# ==========================================================
st.markdown("---")
st.subheader("📊 Manual Asset Inspector & Chart")

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
    df_5, df_4, df_6 = fetch_twt_delta_data(selected_symbol, selected_tf)
    bot = TWTSafeSecondRedEngine(df_5, df_4, df_6, selected_symbol, active_tz)
    bot.calculate_and_scan()

if bot.df_5 is not None and not bot.df_5.empty:
    curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="4H Resistance", value=f"{bot.resistance_lvl:.2f}")
    with col2:
        st.metric(label="6H Support", value=f"{bot.support_lvl:.2f}")
    with col3:
        st.metric(label="Current Market Price", value=f"{curr_p:.2f}")
    with col4:
        st.metric(label="Timezone", value=tz_short, delta="45-57% Wick Rule Active")

    if bot.signals:
        latest_sig = bot.signals[-1]
        is_buy = latest_sig["Type"] == "BUYER"
        
        if is_buy:
            st.success(f"🟢 **BUYER SETUP DETECTED FOR {selected_symbol}**")
            st.code(f"""BUYER SETUP DETECTED
Trend: {latest_sig['Trend']}
Pullback: {latest_sig['Pullback']}
State: {latest_sig['State']}
Buyer Strength: {latest_sig['BuyerStrength']}
Seller Weakness: {latest_sig['SellerWeakness']}
Wick Rejection: {latest_sig['WickRejection']}
Confirmation: {latest_sig['Confirmation']}
Entry: {latest_sig['Entry Price']}
Stop Loss: {latest_sig['Stop Loss (SL)']}
Target: {latest_sig['Target']}
Risk-to-Reward: {latest_sig['RR']}
Signal Status: VALID (Score: {latest_sig['Score']}/100)""", language="text")
        else:
            st.error(f"🔴 **SELLER SETUP DETECTED FOR {selected_symbol}**")
            st.code(f"""SELLER SETUP DETECTED
Trend: {latest_sig['Trend']}
Pullback: {latest_sig['Pullback']}
State: {latest_sig['State']}
Seller Strength: {latest_sig['SellerStrength']}
Buyer Weakness: {latest_sig['BuyerWeakness']}
Wick Rejection: {latest_sig['WickRejection']}
Confirmation: {latest_sig['Confirmation']}
Entry: {latest_sig['Entry Price']}
Stop Loss: {latest_sig['Stop Loss (SL)']}
Target: {latest_sig['Target']}
Risk-to-Reward: {latest_sig['RR']}
Signal Status: VALID (Score: {latest_sig['Score']}/100)""", language="text")
    else:
        st.warning(f"🟡 **STATUS: WAITING FOR VALID 45-57% WICK SETUP ON {selected_symbol}**")

    # ==========================================================
    # INTERACTIVE PAPER TRADING DESK (ISOLATED)
    # ==========================================================
    st.markdown("---")
    st.subheader(f"⚡ Interactive Paper Trading Desk ({selected_symbol})")
    
    suggested_entry = bot.signals[-1]['Entry Price'] if bot.signals else curr_p
    suggested_sl = bot.signals[-1]['Stop Loss (SL)'] if bot.signals else curr_p * 0.99
    suggested_tp = bot.signals[-1]['Target'] if bot.signals else curr_p * 1.02
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

    # ==========================================================
    # COMPLETE SIGNAL HISTORY FOR SELECTED ASSET
    # ==========================================================
    if bot.signals:
        st.markdown("---")
        st.subheader(f"📋 Complete Signal History for {selected_symbol}")
        df_history = pd.DataFrame(bot.signals).drop(columns=["Is_Within_1hr"], errors="ignore")
        st.dataframe(df_history, use_container_width=True)

    # ==========================================================
    # PROFESSIONAL MATPLOTLIB CANDLESTICK CHART
    # ==========================================================
    st.markdown("---")
    st.subheader(f"📈 Professional Candlestick Chart — {selected_symbol}")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    for c in bot.candles:
        col = "#26a69a" if c["color"] == "Green" else "#ef5350"
        body_size = abs(c["close"] - c["open"])
        min_body = curr_p * 0.0003
        plot_height = max(body_size, min_body)
        plot_bottom = min(c["open"], c["close"]) - ((min_body - body_size) / 2) if body_size < min_body else min(c["open"], c["close"])

        ax.bar(c["id"], plot_height, bottom=plot_bottom, color=col, width=0.7)
        ax.vlines(c["id"], c["low"], c["high"], color="#b2b5be", linewidth=1.3)

    if bot.resistance_lvl > 0:
        ax.axhline(bot.resistance_lvl, color="#ffeb3b", linestyle="--", linewidth=2, label="4H Resistance")
        ax.text(2, bot.resistance_lvl, f"  4H Resistance ({bot.resistance_lvl:.1f})", color="#ffeb3b", weight="bold", fontsize=10)

    if bot.support_lvl > 0:
        ax.axhline(bot.support_lvl, color="#26a69a", linestyle="--", linewidth=2, label="6H Support")
        ax.text(2, bot.support_lvl, f"  6H Support ({bot.support_lvl:.1f})", color="#26a69a", weight="bold", fontsize=10)

    lows = [c["low"] for c in bot.candles]
    highs = [c["high"] for c in bot.candles]
    if bot.resistance_lvl > 0: highs.append(bot.resistance_lvl)
    if bot.support_lvl > 0: lows.append(bot.support_lvl)

    if lows and highs:
        margin = (max(highs) - min(lows)) * 0.09
        if margin == 0: margin = curr_p * 0.005
        ax.set_ylim(min(lows) - margin, max(highs) + margin)

    for sig in bot.signals:
        is_b = sig["Type"] == "BUYER"
        bg_col = "#00e676" if is_b else "#ff5252"
        txt = " B " if is_b else " S "
        ax.text(
            sig["Candle #"], sig["Entry Price"], txt,
            color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_col, alpha=0.9),
            ha="center", va="center"
        )

    ax.set_title(f"{selected_symbol} {selected_tf} — TWT Advanced Pullback Bot Pro ({tz_short})", color="white", fontsize=14)
    ax.set_facecolor("#131722")
    fig.patch.set_facecolor("#1e222d")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#363a45")
    ax.grid(color="#2a2e39", linestyle="--", linewidth=0.5)

    st.pyplot(fig)
else:
    st.error("Market data load failed. Please check internet connection.")
