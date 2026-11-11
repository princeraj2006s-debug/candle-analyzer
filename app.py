from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="TWT Multi-Asset Recent vs History Bot",
    page_icon="⚡",
    layout="wide",
)

# Hide Streamlit Default Elements
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Safe Auto-refresh every 30 seconds
count = st_autorefresh(interval=30000, key="twt_multi_refresh")

st.title("⚡ TRADE WITH TUCKYAA — Recent Dashboard & History Engine")

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Multi-Asset Configurations")
st.sidebar.markdown("---")

IANA_TIMEZONES = [
    "Asia/Kolkata",
    "UTC",
    "America/New_York",
    "Europe/London",
    "Asia/Singapore",
]
selected_tz_name = st.sidebar.selectbox(
    "🌍 Trading Timezone (IANA)", options=IANA_TIMEZONES, index=0
)
active_tz = ZoneInfo(selected_tz_name)

selected_tf = st.sidebar.selectbox(
    "⏱️ Timeframe", options=["5m", "15m", "30m", "1h"], index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Display Rules")
st.sidebar.text("Master Dashboard : Recent Live Trades (Last 1 hr)")
st.sidebar.text("Asset Inspector  : Complete History & Chart")
st.sidebar.text("Demand Midpoint  ➔ Green ➔ BUY")
st.sidebar.text("Supply Midpoint  ➔ Red   ➔ SELL")

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


def get_decimals(price):
  if price < 1:
    return 4
  elif price < 10:
    return 3
  else:
    return 2


# ---------------------------------------------------------
# DATA FETCHING MODULE WITH CCXT
# ---------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_multi_asset_ohlcv(symbol, timeframe="5m", limit=300):
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
        ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
        if ohlcv:
          df = pd.DataFrame(
              ohlcv,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
          df = df.set_index("timestamp")
          return df
    except Exception:
      continue
  return None


# ---------------------------------------------------------
# ENGINE WITH RECENT VS HISTORY SEPARATION
# ---------------------------------------------------------
class TWTRecentHistoryEngine:

  def __init__(self, df, symbol, user_tz):
    self.df = df
    self.symbol = symbol
    self.user_tz = user_tz
    self.candles = []
    self.recent_signals = []
    self.history = []
    self.demand_box = None
    self.supply_box = None
    self.demand_mid = 0.0
    self.supply_mid = 0.0

  def calculate_and_scan(self):
    if self.df is None or self.df.empty:
      return

    current_utc_time = datetime.now(timezone.utc)

    self.df["TR"] = np.maximum(
        self.df["High"] - self.df["Low"],
        np.maximum(
            abs(self.df["High"] - self.df["Close"].shift(1)),
            abs(self.df["Low"] - self.df["Close"].shift(1)),
        ),
    )
    self.df["ATR"] = self.df["TR"].rolling(14).mean()

    for idx, row in enumerate(self.df.iterrows()):
      dt_utc = row[0]
      if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
      dt_user_tz = dt_utc.astimezone(self.user_tz)
      timestamp_str = dt_user_tz.strftime("%H:%M")

      r = row[1]
      o, h, l, c, v = (
          float(r["Open"]),
          float(r["High"]),
          float(r["Low"]),
          float(r["Close"]),
          float(r["Volume"]),
      )
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)

      age_minutes = (current_utc_time - dt_utc).total_seconds() / 60.0
      is_within_1hr = age_minutes <= 60.0

      self.candles.append({
          "id": idx + 1,
          "time": timestamp_str,
          "dt_user": dt_user_tz,
          "date": dt_user_tz.date(),
          "color": "Green" if c >= o else "Red",
          "open": o,
          "high": h,
          "low": l,
          "close": c,
          "volume": v,
          "atr": atr,
          "is_within_1hr": is_within_1hr,
      })

    if not self.candles:
      return

    latest_date = self.candles[-1]["date"]
    today_candles = [c for c in self.candles if c["date"] == latest_date]
    if not today_candles:
      today_candles = self.candles

    all_lows = [c["low"] for c in today_candles]
    all_highs = [c["high"] for c in today_candles]

    if not all_lows or not all_highs:
      return

    min_price = min(all_lows)
    max_price = max(all_highs)
    total_range = max_price - min_price if max_price > min_price else 1.0

    demand_zone_low = min_price
    demand_zone_high = min_price + (total_range * 0.22)
    self.demand_box = (demand_zone_low, demand_zone_high)
    self.demand_mid = (demand_zone_low + demand_zone_high) / 2.0

    supply_zone_high = max_price
    supply_zone_low = max_price - (total_range * 0.22)
    self.supply_box = (supply_zone_low, supply_zone_high)
    self.supply_mid = (supply_zone_low + supply_zone_high) / 2.0

    best_buy = None
    best_sell = None
    all_detected = []

    for c in today_candles:
      atr = c["atr"]
      dec = get_decimals(c["close"])

      if best_buy is None and c["low"] <= self.demand_mid and c["color"] == "Green":
        sl = round(demand_zone_low - (1.0 * atr), dec)
        tp1 = round(c["close"] + (1.2 * atr), dec)
        tp2 = round(c["close"] + (2.5 * atr), dec)
        tp3 = round(c["close"] + (4.0 * atr), dec)

        best_buy = {
            "Asset": self.symbol,
            "Date": str(c["date"]),
            "Time": c["time"],
            "Candle #": c["id"],
            "Type": "BUY 🟢",
            "Setup": "Demand Midpoint Bounce",
            "Entry Price": round(c["close"], dec),
            "Stop Loss (SL)": sl,
            "TP 1": tp1,
            "TP 2": tp2,
            "TP 3": tp3,
            "Is_Within_1hr": c["is_within_1hr"],
        }
        all_detected.append(best_buy)

      if best_sell is None and c["high"] >= self.supply_mid and c["color"] == "Red":
        sl = round(supply_zone_high + (1.0 * atr), dec)
        tp1 = round(c["close"] - (1.2 * atr), dec)
        tp2 = round(c["close"] - (2.5 * atr), dec)
        tp3 = round(c["close"] - (4.0 * atr), dec)

        best_sell = {
            "Asset": self.symbol,
            "Date": str(c["date"]),
            "Time": c["time"],
            "Candle #": c["id"],
            "Type": "SELL 🔴",
            "Setup": "Supply Midpoint Rejection",
            "Entry Price": round(c["close"], dec),
            "Stop Loss (SL)": sl,
            "TP 1": tp1,
            "TP 2": tp2,
            "TP 3": tp3,
            "Is_Within_1hr": c["is_within_1hr"],
        }
        all_detected.append(best_sell)

    self.history = all_detected
    # Master dashboard shows ONLY recent signals formed within the last 1 hour
    self.recent_signals = [s for s in all_detected if s.get("Is_Within_1hr", False)]


# ---------------------------------------------------------
# MASTER MULTI-ASSET SCANNER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Recent Live Dashboard (20+ Assets)")

master_recent = []
asset_engines = {}


def scan_asset(sym):
  df = fetch_multi_asset_ohlcv(sym, selected_tf, limit=300)
  if df is not None and not df.empty:
    engine = TWTRecentHistoryEngine(df, sym, active_tz)
    engine.calculate_and_scan()
    return sym, engine, engine.recent_signals
  return sym, None, []


with st.spinner("⚡ Parallel scanning 20+ assets for recent live trades..."):
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(scan_asset, sym): sym for sym in ASSET_LIST}
    for future in as_completed(futures):
      sym, engine, recent_sigs = future.result()
      if engine:
        asset_engines[sym] = engine
      if recent_sigs:
        master_recent.extend(recent_sigs)

if master_recent:
  st.success(f"✅ Found {len(master_recent)} recent active trade(s) in the last 1 hour!")
  df_display = pd.DataFrame(master_recent).drop(columns=["Is_Within_1hr"], errors="ignore")
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No recent trades in the last 1 hour across assets. Check individual asset history below.")

# ---------------------------------------------------------
# MANUAL ASSET INSPECTOR & CHART VIEWER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector & History")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart & Complete History",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df = fetch_multi_asset_ohlcv(selected_symbol, selected_tf, limit=300)
  bot = TWTRecentHistoryEngine(df, selected_symbol, active_tz)
  bot.calculate_and_scan()

if bot.df is not None and not bot.df.empty:
  curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label=f"Current {selected_symbol} Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Total Locked Entries", value=len(bot.history))
  with col3:
    st.metric(label="Timezone", value=selected_tz_name.split(")")[0] + ")")

  if bot.history:
    st.markdown(f"### 📋 Complete History Feed for {selected_symbol}")
    df_history = pd.DataFrame(bot.history).drop(columns=["Is_Within_1hr"], errors="ignore")
    st.dataframe(df_history, width="stretch")
  else:
    st.info("ℹ️ No historical entries for today on this asset yet.")

  # --- CLEAN CANDLESTICK CHART RENDERING ---
  fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
  for c in bot.candles:
    col = "#26a69a" if c["close"] >= c["open"] else "#ef5350"
    lower_val = min(c["open"], c["close"])
    body_height = abs(c["close"] - c["open"])
    if body_height == 0:
      body_height = curr_p * 0.0005

    ax.bar(c["id"], body_height, bottom=lower_val, color=col, width=0.6, edgecolor=col)
    ax.vlines(c["id"], c["low"], c["high"], color=col, linewidth=1.2)

  if bot.demand_box:
    d_low, d_high = bot.demand_box
    ax.axhspan(d_low, d_high, color="#26a69a", alpha=0.25, label="Demand Zone")
    ax.axhline(bot.demand_mid, color="#00e676", linestyle=":", linewidth=1.5, label="Demand Midpoint")

  if bot.supply_box:
    s_low, s_high = bot.supply_box
    ax.axhspan(s_low, s_high, color="#ef5350", alpha=0.25, label="Supply Zone")
    ax.axhline(bot.supply_mid, color="#ff5252", linestyle=":", linewidth=1.5, label="Supply Midpoint")

  lows = [c["low"] for c in bot.candles]
  highs = [c["high"] for c in bot.candles]
  if lows and highs:
    margin = (max(highs) - min(lows)) * 0.09
    if margin == 0:
      margin = curr_p * 0.005
    ax.set_ylim(min(lows) - margin, max(highs) + margin)

  for sig in bot.history:
    is_b = "BUY" in sig["Type"]
    bg_col = "#00e676" if is_b else "#ff5252"
    txt = " B " if is_b else " S "
    ax.text(
        sig["Candle #"],
        sig["Entry Price"],
        txt,
        color="white",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_col, alpha=0.9),
        ha="center",
        va="center",
    )

  ax.set_title(
      f"{selected_symbol} {selected_tf} — Recent vs History Visualizer",
      color="white",
      fontsize=14,
  )
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
  st.error("Market data fetch failed. Please check your internet connection.")
