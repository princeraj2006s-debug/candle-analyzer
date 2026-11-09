import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta, timezone

import ccxt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWT Delta Pro by TUCKYAA", page_icon="⚡", layout="wide"
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------------------------------
# TIMEZONES & ASSETS
# ---------------------------------------------------------
TRADING_TIMEZONES = {
    "(UTC+5:30) Kolkata / India (IST)": timezone(timedelta(hours=5, minutes=30)),
    "(UTC+1:00) Berlin / Paris / CET": timezone(timedelta(hours=1)),
    "(UTC+0:00) London / UTC / GMT": timezone.utc,
    "(UTC-5:00) New York / US East (EST)": timezone(timedelta(hours=-5)),
    "(UTC+8:00) Singapore / Hong Kong (SGT)": timezone(timedelta(hours=8)),
}

ASSET_LIST = [
    "BTC/USDT", "ETH/USDT", "XAUT/USDT", "SOL/USDT", "XRP/USDT",
    "BNB/USDT", "DOGE/USDT", "MATIC/USDT", "ADA/USDT", "AVAX/USDT",
    "LINK/USDT", "LTC/USDT", "DOT/USDT", "NEAR/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "SUI/USDT", "FIL/USDT", "RENDER/USDT",
]

# FIX 1: refresh every 60s (was 30s). If a scan takes longer than the refresh
# interval, the page restarts before it finishes and never shows anything.
count = st_autorefresh(interval=60000, key="twt_delta_refresh")

st.title("⚡ TRADE WITH TUCKYAA (TUCKYAA) — Precision Retest & Pullback Engine")

col_tf, col_tz = st.columns([1, 2])
with col_tf:
    selected_tf = st.selectbox("⏱️ Timeframe", options=["5m", "15m", "30m", "1h"], index=0)
with col_tz:
    selected_tz_name = st.selectbox("🌍 Trading Timezone", options=list(TRADING_TIMEZONES.keys()), index=0)
    active_tz = TRADING_TIMEZONES[selected_tz_name]

tz_short = selected_tz_name.split(")")[0] + ")"

st.sidebar.markdown("### 🤖 TWT Ultimate Price Action Panel")
st.sidebar.markdown("---")
server_time_target = datetime.now(active_tz)
st.sidebar.text(f"Server time          ———  {server_time_target.strftime('%H:%M')}")
st.sidebar.text("Support Ref (6H)   ———  05:30 IST Reference Low")
st.sidebar.text("Resistance Ref (4H) ———  01:30 IST Reference High")
st.sidebar.text("No-Trade Zone      ———  01:00 ➔ 06:00 IST")
st.sidebar.markdown("---")
st.sidebar.text("Exchange Platform  — Delta Exchange (20 Assets)")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Trading Strategy")
st.sidebar.text("Logic            : Double Retest + Pullback + Zone Logic")
st.sidebar.text("Take profit      : Multi-TP (TP1, TP2, TP3)")
st.sidebar.text("Stop loss        : ATR Managed")


def get_decimals(price):
    if price < 1:
        return 4
    elif price < 10:
        return 3
    else:
        return 2


# ---------------------------------------------------------
# DATA FETCHING
# FIX 2: exchanges are created ONCE and reused (before: load_markets() was
# called again for every asset). Blocked exchanges (Binance/Bybit often block
# Streamlit Cloud's US servers) are skipped for 10 minutes instead of timing
# out on every asset.
# ---------------------------------------------------------
EXCHANGE_ORDER = ["kucoin", "okx", "gateio", "mexc", "bybit", "binance"]


@st.cache_resource
def dead_exchanges():
    return {}


@st.cache_resource(ttl=3600, show_spinner=False)
def get_exchange(name):
    ex = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 10000})
    ex.load_markets()
    return ex


def _to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_twt_delta_data(symbol, tf):
    variants = [symbol]
    if symbol == "RENDER/USDT":
        variants = ["RENDER/USDT", "RNDR/USDT"]

    dead = dead_exchanges()

    for name in EXCHANGE_ORDER:
        if _time.time() - dead.get(name, 0) < 600:
            continue
        try:
            ex = get_exchange(name)
        except Exception:
            dead[name] = _time.time()
            continue

        for sym in variants:
            if sym not in ex.markets:
                continue
            try:
                o_tf = ex.fetch_ohlcv(sym, timeframe=tf, limit=140)
                o_4 = ex.fetch_ohlcv(sym, timeframe="4h", limit=25)
            except Exception:
                continue
            try:
                o_6 = ex.fetch_ohlcv(sym, timeframe="6h", limit=25)
            except Exception:
                o_6 = None

            if not o_tf or not o_4:
                continue

            df_5 = _to_df(o_tf)
            df_4 = _to_df(o_4)
            df_6 = _to_df(o_6) if o_6 else df_4.copy()  # FIX 3: old code crashed here

            if symbol == "XAUT/USDT":
                for d in (df_5, df_4, df_6):
                    for col in ["Open", "High", "Low", "Close"]:
                        d[col] *= 1.0025

            return df_5, df_4, df_6
    return None, None, None


# ---------------------------------------------------------
# ENGINE
# ---------------------------------------------------------
class TWTPrecisionEngine:
    """
    BUY: green wick touches support (retest 1) -> 2 red above support ->
         2 green cross red high -> red pullback -> green confirm -> BUY.
         Support closed below -> nearest demand zone, same sequence.
    SELL: mirror (red touches resistance, 2 green below, 2 red cross,
          weak green pullback, red confirm). Resistance broken -> supply zone.
    """

    TOUCH_TOL_ATR = 0.3
    MAX_PIERCE_ATR = 0.5
    MAX_AGE_PUSH = 12
    MAX_AGE_CROSS = 10
    MAX_AGE_PULLBACK = 8
    WEAK_BODY_ATR = 0.6

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

    @staticmethod
    def _fresh():
        return {"state": 0, "age": 0, "ref": 0.0, "pb_open": 0.0}

    def _long_machine(self, cs, support, require_weak=False):
        hits = []
        st_ = self._fresh()
        mode = "support"
        lvl = support
        lvl_top = support
        brk = None

        for i in range(2, len(cs)):
            c, p = cs[i], cs[i - 1]
            atr = c["atr"]
            tol = self.TOUCH_TOL_ATR * atr
            pierce = self.MAX_PIERCE_ATR * atr
            is_green = c["color"] == "Green"
            is_red = not is_green

            if mode == "support" and c["close"] < support:
                mode = "demand"
                lvl = None
                brk = i
                st_ = self._fresh()
                continue

            if mode == "demand":
                if c["close"] > support:
                    mode = "support"
                    lvl = lvl_top = support
                    st_ = self._fresh()
                    continue
                if lvl is not None and c["close"] < lvl - pierce:
                    lvl = None
                    st_ = self._fresh()
                if lvl is None and i >= 4:
                    j = i - 2
                    if j >= brk and j >= 2:
                        lj = cs[j]["low"]
                        if (
                            lj < support
                            and lj < cs[j - 1]["low"]
                            and lj < cs[j - 2]["low"]
                            and lj < cs[j + 1]["low"]
                            and lj < cs[j + 2]["low"]
                        ):
                            lvl = lj
                            lvl_top = max(cs[j]["open"], cs[j]["close"])
                            st_ = self._fresh()
                            continue
                if lvl is None:
                    continue

            s = st_["state"]

            if s == 0:
                if (
                    is_green
                    and c["low"] <= lvl_top + tol
                    and c["low"] >= lvl - pierce
                    and c["close"] > lvl
                ):
                    st_ = self._fresh()
                    st_["state"] = 1
                continue

            st_["age"] += 1

            if c["close"] <= lvl:
                st_ = self._fresh()
                continue

            if s == 1:
                if st_["age"] > self.MAX_AGE_PUSH:
                    st_ = self._fresh()
                    continue
                if is_red and p["color"] == "Red" and c["low"] > lvl and p["low"] > lvl:
                    st_["ref"] = max(c["high"], p["high"])
                    st_["state"] = 2
                    st_["age"] = 0

            elif s == 2:
                if st_["age"] > self.MAX_AGE_CROSS:
                    st_ = self._fresh()
                    continue
                if is_green and p["color"] == "Green" and c["close"] > st_["ref"]:
                    st_["state"] = 3
                    st_["age"] = 0

            elif s == 3:
                if st_["age"] > self.MAX_AGE_PULLBACK:
                    st_ = self._fresh()
                    continue
                if is_red and c["low"] > lvl:
                    body = abs(c["close"] - c["open"])
                    if (not require_weak) or body <= self.WEAK_BODY_ATR * atr:
                        st_["pb_open"] = c["open"]
                        st_["state"] = 4
                        st_["age"] = 0

            elif s == 4:
                if st_["age"] > 3:
                    st_ = self._fresh()
                    continue
                if is_green and c["close"] > st_["pb_open"]:
                    hits.append({"idx": i, "mode": mode, "lvl": lvl, "lvl_top": lvl_top})
                    st_ = self._fresh()
                elif is_red and c["low"] > lvl:
                    st_["pb_open"] = c["open"]

        return hits

    def _mirror(self):
        return [
            {
                "id": c["id"],
                "open": -c["open"],
                "high": -c["low"],
                "low": -c["high"],
                "close": -c["close"],
                "atr": c["atr"],
                "color": "Red" if c["color"] == "Green" else "Green",
            }
            for c in self.candles
        ]

    def calculate_and_scan(self):
        if self.df_5 is None or self.df_5.empty:
            return

        IST = timezone(timedelta(hours=5, minutes=30))
        current_utc_time = datetime.now(timezone.utc)

        if self.df_4 is not None and not self.df_4.empty:
            idx_utc = self.df_4.index.tz_localize("UTC") if self.df_4.index.tz is None else self.df_4.index
            df_4_ist = idx_utc.tz_convert(IST)
            filtered_4h = self.df_4[df_4_ist.hour == 1]
            self.resistance_lvl = float(filtered_4h["High"].iloc[-1]) if not filtered_4h.empty else float(self.df_4["High"].iloc[-1])

        if self.df_6 is not None and not self.df_6.empty:
            idx_utc_6 = self.df_6.index.tz_localize("UTC") if self.df_6.index.tz is None else self.df_6.index
            df_6_ist = idx_utc_6.tz_convert(IST)
            filtered_6h = self.df_6[df_6_ist.hour == 5]
            self.support_lvl = float(filtered_6h["Low"].iloc[-1]) if not filtered_6h.empty else float(self.df_6["Low"].iloc[-1])
        elif self.df_4 is not None:
            self.support_lvl = float(self.df_4["Low"].iloc[-1])

        self.df_5["TR"] = np.maximum(
            self.df_5["High"] - self.df_5["Low"],
            np.maximum(
                abs(self.df_5["High"] - self.df_5["Close"].shift(1)),
                abs(self.df_5["Low"] - self.df_5["Close"].shift(1)),
            ),
        )
        self.df_5["ATR"] = self.df_5["TR"].rolling(14).mean()

        for idx, row in enumerate(self.df_5.iterrows()):
            dt_utc = row[0]
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            dt_ist = dt_utc.astimezone(IST)
            dt_user_tz = dt_utc.astimezone(self.user_tz)

            r = row[1]
            o, h, l, c, v = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]), float(r["Volume"])
            atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)
            age_minutes = (current_utc_time - dt_utc).total_seconds() / 60.0

            self.candles.append({
                "id": idx + 1,
                "time": dt_user_tz.strftime("%H:%M"),
                "dt_ist": dt_ist,
                "dt_user": dt_user_tz,
                "color": "Green" if c >= o else "Red",
                "open": o, "high": h, "low": l, "close": c,
                "volume": v, "atr": atr,
                "is_within_1hr": age_minutes <= 60.0,
            })

        if len(self.candles) < 20:
            return

        buy_signals, sell_signals = [], []

        for hit in self._long_machine(self.candles, self.support_lvl, require_weak=False):
            c = self.candles[hit["idx"]]
            if time(1, 0) <= c["dt_ist"].time() <= time(6, 0):
                continue
            atr = c["atr"]
            dec = get_decimals(c["close"])
            is_zone = hit["mode"] == "demand"
            buy_signals.append({
                "Asset": self.symbol,
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "Demand Zone Retest + Pullback Long" if is_zone else "Support Double Retest + Pullback Long",
                "Action": "Buy / Long (B) 🟢",
                "Entry Price": round(c["close"], dec),
                "Stop Loss (SL)": round(hit["lvl"] - ((1.0 if is_zone else 1.2) * atr), dec),
                "TP 1": round(c["close"] + 1.2 * atr, dec),
                "TP 2": round(c["close"] + 2.5 * atr, dec),
                "TP 3": round(c["close"] + 4.0 * atr, dec),
                "Is_Within_1hr": c["is_within_1hr"],
            })

        for hit in self._long_machine(self._mirror(), -self.resistance_lvl, require_weak=True):
            c = self.candles[hit["idx"]]
            if time(1, 0) <= c["dt_ist"].time() <= time(6, 0):
                continue
            atr = c["atr"]
            dec = get_decimals(c["close"])
            is_zone = hit["mode"] == "demand"
            real_lvl = -hit["lvl"]
            sell_signals.append({
                "Asset": self.symbol,
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "Supply Zone Retest + Pullback Short" if is_zone else "Resistance Double Retest + Pullback Short",
                "Action": "Sell / Short (S) 🔴",
                "Entry Price": round(c["close"], dec),
                "Stop Loss (SL)": round(real_lvl + ((1.0 if is_zone else 1.2) * atr), dec),
                "TP 1": round(c["close"] - 1.2 * atr, dec),
                "TP 2": round(c["close"] - 2.5 * atr, dec),
                "TP 3": round(c["close"] - 3.8 * atr, dec),
                "Is_Within_1hr": c["is_within_1hr"],
            })

        if buy_signals:
            self.signals.extend(buy_signals[-2:])
        if sell_signals:
            self.signals.extend(sell_signals[-2:])


# ---------------------------------------------------------
# MASTER SCANNER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (Last 1 Hour)")

master_signals = []
asset_engines = {}


def scan_single_asset(sym):
    try:
        df_5, df_4, df_6 = fetch_twt_delta_data(sym, selected_tf)
        if df_5 is not None and not df_5.empty:
            engine = TWTPrecisionEngine(df_5, df_4, df_6, sym, active_tz)
            engine.calculate_and_scan()
            recent_sigs = [s for s in engine.signals if s.get("Is_Within_1hr", False)]
            latest = recent_sigs[-1] if recent_sigs else None
            return sym, engine, latest
    except Exception:
        pass
    return sym, None, None


with st.spinner("⚡ Scanning all 20 assets with precision retest rules..."):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scan_single_asset, sym): sym for sym in ASSET_LIST}
        for future in as_completed(futures):
            sym, engine, latest = future.result()
            if engine:
                asset_engines[sym] = engine
            if latest:
                master_signals.append(latest)

st.caption(f"Data loaded for {len(asset_engines)}/{len(ASSET_LIST)} assets")

if master_signals:
    st.success(f"✅ Found {len(master_signals)} active precision setup(s)!")
    st.dataframe(pd.DataFrame(master_signals).drop(columns=["Is_Within_1hr"], errors="ignore"))
else:
    st.info("ℹ️ No active setups in the last 1 hour. Waiting for exact double retest & pullback sequence...")

# ---------------------------------------------------------
# MANUAL ASSET INSPECTOR
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Manual Asset Inspector & Chart")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart & Complete History", options=ASSET_LIST, index=0
)

if selected_symbol in asset_engines:
    bot = asset_engines[selected_symbol]
else:
    df_5, df_4, df_6 = fetch_twt_delta_data(selected_symbol, selected_tf)
    bot = TWTPrecisionEngine(df_5, df_4, df_6, selected_symbol, active_tz)
    bot.calculate_and_scan()

if bot.df_5 is not None and not bot.df_5.empty and bot.candles:
    curr_p = bot.candles[-1]["close"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="4H Resistance (01:30 High)", value=f"{bot.resistance_lvl:.2f}")
    with col2:
        st.metric(label="6H Support (05:30 Low)", value=f"{bot.support_lvl:.2f}")
    with col3:
        st.metric(label="Current Market Price", value=f"{curr_p:.2f}")
    with col4:
        st.metric(label="Timezone", value=tz_short, delta="Precision Engine Active")

    recent_signals = [s for s in bot.signals if s.get("Is_Within_1hr", False)]

    if recent_signals:
        latest_sig = recent_signals[-1]
        is_buy = "Buy" in latest_sig["Action"]
        msg = (
            f"{'🟢' if is_buy else '🔴'} **TWT PRECISION SETUP: {latest_sig['Action']}**\n\n"
            f"🕒 **Candle Time ({tz_short}):** `{latest_sig['Time']}` (Candle #{latest_sig['Candle #']})\n"
            f"- **Setup:** {latest_sig['Setup']}\n"
            f"- **Entry Price:** `{latest_sig['Entry Price']}`\n"
            f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
            f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:** `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`"
        )
        st.warning("⚠️ **Don't use after 40 minutes!**")
        if is_buy:
            st.success(msg)
        else:
            st.error(msg)
    else:
        st.warning(
            "🟡 **STATUS: WAITING FOR PRECISION SETUP**\n\n"
            "📌 *Instructions:* Monitoring double retests, pullbacks, and zone confirmations."
        )

    if bot.signals:
        st.markdown("---")
        st.subheader(f"📋 Complete Signal History for {selected_symbol} (Precision Engine)")
        st.dataframe(pd.DataFrame(bot.signals).drop(columns=["Is_Within_1hr"], errors="ignore"))

    # Chart
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    for c in bot.candles:
        col = "#26a69a" if c["color"] == "Green" else "#ef5350"
        body_size = abs(c["close"] - c["open"])
        min_body = curr_p * 0.0003
        plot_height = max(body_size, min_body)
        plot_bottom = (
            min(c["open"], c["close"]) - ((min_body - body_size) / 2)
            if body_size < min_body
            else min(c["open"], c["close"])
        )
        ax.bar(c["id"], plot_height, bottom=plot_bottom, color=col, width=0.7)
        ax.vlines(c["id"], c["low"], c["high"], color="#b2b5be", linewidth=1.3)

    ax.axhline(bot.resistance_lvl, color="#ffeb3b", linestyle="--", linewidth=2)
    ax.text(2, bot.resistance_lvl, f"  4H Resistance ({bot.resistance_lvl:.1f})",
            color="#ffeb3b", weight="bold", fontsize=10)
    ax.axhline(bot.support_lvl, color="#26a69a", linestyle="--", linewidth=2)
    ax.text(2, bot.support_lvl, f"  6H Support ({bot.support_lvl:.1f})",
            color="#26a69a", weight="bold", fontsize=10)

    lows = [c["low"] for c in bot.candles]
    highs = [c["high"] for c in bot.candles]
    if bot.resistance_lvl > 0:
        highs.append(bot.resistance_lvl)
    if bot.support_lvl > 0:
        lows.append(bot.support_lvl)
    margin = (max(highs) - min(lows)) * 0.09
    if margin == 0:
        margin = curr_p * 0.005
    ax.set_ylim(min(lows) - margin, max(highs) + margin)

    for sig in bot.signals:
        tag = "B" if "Buy" in sig["Action"] else "S"
        tag_color = "#00e676" if tag == "B" else "#ff1744"
        ax.text(sig["Candle #"], sig["Entry Price"], f" {tag} ", color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=tag_color, alpha=0.9))

    ax.set_title(f"{selected_symbol} {selected_tf} — TRADE WITH TUCKYAA Precision Engine ({tz_short})",
                 color="white", fontsize=14)
    ax.set_facecolor("#131722")
    fig.patch.set_facecolor("#1e222d")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#363a45")
    ax.grid(color="#2a2e39", linestyle="--", linewidth=0.5)

    st.pyplot(fig)
    plt.close(fig)
else:
    st.error("Market data load failed. Exchanges may be blocking the server — check the app logs.")
