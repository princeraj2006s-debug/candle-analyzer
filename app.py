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
    page_title="TWT Multi-Asset Fixed Strict Engine",
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

st.title("⚡ TRADE WITH TUCKYAA — Fixed Strict Precision Engine")

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
st.sidebar.markdown("#### Fixed Strict Entry Rules")
st.sidebar.text("BUY  : 3 Specific Scenarios (Demand Zone)")
st.sidebar.text("SELL : 3 Fixed Mirror Scenarios (Supply Zone)")
st.sidebar.text("Trend Filter : EMA 50 Alignment Check")
st.sidebar.text("Safe Buffer  : 1.8x ATR Wider Stop Loss")

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
# FIXED STRICT ENGINE
# ---------------------------------------------------------
class TWTFixedStrictEngine:

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
    self.df["EMA50"] = self.df["Close"].ewm(span=50, adjust=False).mean()

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
      ema50 = float(r["EMA50"]) if not pd.isna(r["EMA50"]) else c

      age_minutes = (current_utc_time - dt_utc).total_seconds() / 60.0
      is_within_1hr = age_minutes <= 60.0

      self.candles.append({
          "id": idx + 1,
          "time": timestamp_str,
          "dt_user": dt_user_tz,
          "time_obj": dt_user_tz.time(),
          "date": dt_user_tz.date(),
          "color": "Green" if c >= o else "Red",
          "open": o,
          "high": h,
          "low": l,
          "close": c,
          "volume": v,
          "atr": atr,
          "ema50": ema50,
          "is_within_1hr": is_within_1hr,
      })

    if not self.candles:
      return

    latest_date = self.candles[-1]["date"]
    today_candles = [c for c in self.candles if c["date"] == latest_date]
    if not today_candles:
      today_candles = self.candles

    # 1. SUPPLY ZONE: Formed strictly between 00:00 AM and 03:00 AM
    supply_build_candles = [
        c for c in today_candles if time(0, 0) <= c["time_obj"] < time(3, 0)
    ]
    if supply_build_candles:
      s_lows = [c["low"] for c in supply_build_candles]
      s_highs = [c["high"] for c in supply_build_candles]
      s_min, s_max = min(s_lows), max(s_highs)
      s_range = s_max - s_min if s_max > s_min else (s_max * 0.005)

      supply_zone_high = s_max
      supply_zone_low = s_max - (s_range * 0.25)
      self.supply_box = (supply_zone_low, supply_zone_high)
      self.supply_mid = (supply_zone_low + supply_zone_high) / 2.0

    # 2. DEMAND ZONE: Formed strictly between 04:00 AM and 06:00 AM
    demand_build_candles = [
        c for c in today_candles if time(4, 0) <= c["time_obj"] < time(6, 0)
    ]
    if demand_build_candles:
      d_lows = [c["low"] for c in demand_build_candles]
      d_highs = [c["high"] for c in demand_build_candles]
      d_min, d_max = min(d_lows), max(d_highs)
      d_range = d_max - d_min if d_max > d_min else (d_min * 0.005)

      demand_zone_low = d_min
      demand_zone_high = d_min + (d_range * 0.25)
      self.demand_box = (demand_zone_low, demand_zone_high)
      self.demand_mid = (demand_zone_low + demand_zone_high) / 2.0

    best_buy = None
    best_sell = None
    all_detected = []

    sell_scenario = 0
    buy_scenario = 0

    for c in today_candles:
      atr = c["atr"]
      ema = c["ema50"]
      dec = get_decimals(c["close"])
      t_obj = c["time_obj"]

      # --- SELLER FIXED SCENARIOS (Post-3AM) ---
      if t_obj >= time(3, 0) and self.supply_box is not None and best_sell is None:
        s_low, s_high = self.supply_box

        if sell_scenario == 0:
          if c["color"] == "Red" and c["high"] >= s_high:
            sell_scenario = 11
          elif c["color"] == "Green" and c["high"] >= s_high:
            sell_scenario = 21
          elif c["color"] == "Green" and c["close"] >= self.supply_mid:
            sell_scenario = 31

        elif sell_scenario == 11:
          if c["color"] == "Green" and c["close"] <= self.supply_mid:
            sell_scenario = 12
        elif sell_scenario == 12:
          if c["color"] == "Red":
            best_sell = self._create_signal(c, "SELL 🔴", "Seller Scenario 1", s_high, atr, dec)
            all_detected.append(best_sell)
            sell_scenario = 99

        elif sell_scenario == 21:
          if c["color"] == "Red" and c["close"] < s_low:
            sell_scenario = 22
        elif sell_scenario == 22:
          if c["color"] == "Green" and c["close"] >= self.supply_mid:
            sell_scenario = 23
        elif sell_scenario == 23:
          if c["color"] == "Red":
            best_sell = self._create_signal(c, "SELL 🔴", "Seller Scenario 2", s_high, atr, dec)
            all_detected.append(best_sell)
            sell_scenario = 99

        elif sell_scenario == 31:
          if c["high"] > s_high:
            sell_scenario = 32
        elif sell_scenario == 32:
          if c["color"] == "Red" and c["close"] <= self.supply_mid:
            best_sell = self._create_signal(c, "SELL 🔴", "Seller Scenario 3", s_high, atr, dec)
            all_detected.append(best_sell)
            sell_scenario = 99

      # --- BUYER FIXED SCENARIOS (Post-6AM) ---
      if t_obj >= time(6, 0) and self.demand_box is not None and best_buy is None:
        d_low, d_high = self.demand_box

        if buy_scenario == 0:
          if c["color"] == "Green" and c["low"] <= d_low:
            buy_scenario = 11
          elif c["color"] == "Red" and c["low"] <= d_low:
            buy_scenario = 21
          elif c["color"] == "Red" and c["close"] <= self.demand_mid:
            buy_scenario = 31

        elif buy_scenario == 11:
          if c["color"] == "Red" and c["close"] >= self.demand_mid:
            buy_scenario = 12
        elif buy_scenario == 12:
          if c["color"] == "Green":
            best_buy = self._create_signal(c, "BUY 🟢", "Buyer Scenario 1", d_low, atr, dec)
            all_detected.append(best_buy)
            buy_scenario = 99

        elif buy_scenario == 21:
          if c["color"] == "Green" and c["close"] > d_high:
            buy_scenario = 22
        elif buy_scenario == 22:
          if c["color"] == "Red" and c["close"] <= self.demand_mid:
            buy_scenario = 23
        elif buy_scenario == 23:
          if c["color"] == "Green":
            best_buy = self._create_signal(c, "BUY 🟢", "Buyer Scenario 2", d_low, atr, dec)
            all_detected.append(best_buy)
            buy_scenario = 99

        elif buy_scenario == 31:
          if c["low"] < d_low:
            buy_scenario = 32
        elif buy_scenario == 32:
          if c["color"] == "Green" and c["close"] >= self.demand_mid:
            best_buy = self._create_signal(c, "BUY 🟢", "Buyer Scenario 3", d_low, atr, dec)
            all_detected.append(best_buy)
            buy_scenario = 99

    self.history = all_detected
    self.recent_signals = [s for s in all_detected if s.get("Is_Within_1hr", False)]

  def _create_signal(self, c, sig_type, setup_name, zone_boundary, atr, dec):
    entry_price = round(c["close"], dec)
    is_buy = "BUY" in sig_type
    sl = round(zone_boundary - (1.8 * atr) if is_buy else zone_boundary + (1.8 * atr), dec)
    tp1 = round(entry_price + (1.2 * atr) if is_buy else entry_price - (1.2 * atr), dec)
    tp2 = round(entry_price + (2.5 * atr) if is_buy else entry_price - (2.5 * atr), dec)
    tp3 = round(entry_price + (4.0 * atr) if is_buy else entry_price - (4.0 * atr), dec)

    return {
        "Asset": self.symbol,
        "Date": str(c["date"]),
        "Time": c["time"],
        "Candle #": c["id"],
        "Type": sig_type,
        "Setup": setup_name,
        "Entry Price": entry_price,
        "Stop Loss (SL)": sl,
        "TP 1": tp1,
        "TP 2": tp2,
        "TP 3": tp3,
        "Is_Within_1hr": c["is_within_1hr"],
    }


# ---------------------------------------------------------
# MASTER MULTI-ASSET SCANNER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Fixed Engine Dashboard (20+ Assets)")

master_recent = []
asset_engines = {}


def scan_asset(sym):
  df = fetch_multi_asset_ohlcv(sym, selected_tf, limit=300)
  if df is not None and not df.empty:
    engine = TWTFixedStrictEngine(df, sym, active_tz)
    engine.calculate_and_scan()
    return sym, engine, engine.recent_signals
  return sym, None, []


with st.spinner("⚡ Parallel scanning 20+ assets with Fixed Strict Logic..."):
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(scan_asset, sym): sym for sym in ASSET_LIST}
    for future in as_completed(futures):
      sym, engine, recent_sigs = future.result()
      if engine:
        asset_engines[sym] = engine
      if recent_sigs:
        master_recent.extend(recent_sigs)

if master_recent:
  st.success(f"✅ Found {len(master_recent)} fixed strict trade(s) in the last 1 hour!")
  df_display = pd.DataFrame(master_recent).drop(columns=["Is_Within_1hr"], errors="ignore")
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No recent trades matching fixed strict rules in the last 1 hour.")

# ---------------------------------------------------------
# MANUAL ASSET INSPECTOR & CHART VIEWER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector & History")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df = fetch_multi_asset_ohlcv(selected_symbol, selected_tf, limit=300)
  bot = TWTFixedStrictEngine(df, selected_symbol, active_tz)
  bot.calculate_and_scan()

if bot.df is not None and not bot.df.empty:
  curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label=f"Current {selected_symbol} Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Locked Entries Today", value=len(bot.history))
  with col3:
    st.metric(label="Timezone", value=selected_tz_name.split(")")[0] + ")")

  if bot.history:
    st.markdown(f"### 📋 Locked History Feed for {selected_symbol}")
    df_history = pd.DataFrame(bot.history).drop(columns=["Is_Within_1hr"], errors="ignore")
    st.dataframe(df_history, width="stretch")
  else:
    st.info("ℹ️ No entries locked yet for today (Waiting for strict scenario match).")

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

  # Plot Demand Zone
  if bot.demand_box:
    d_low, d_high = bot.demand_box
    ax.axhspan(d_low, d_high, color="#26a69a", alpha=0.25, label="Demand Zone (4AM-6AM)")
    ax.axhline(bot.demand_mid, color="#00e676", linestyle=":", linewidth=1.5, label="Demand Midpoint")

  # Plot Supply Zone
  if bot.supply_box:
    s_low, s_high = bot.supply_box
    ax.axhspan(s_low, s_high, color="#ef5350", alpha=0.25, label="Supply Zone (12AM-3AM)")
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
      f"{selected_symbol} {selected_tf} — Fixed Visualizer",
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
