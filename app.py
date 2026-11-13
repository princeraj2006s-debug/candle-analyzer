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
    page_title="TWT Dual 4H Zone & 5M Entry Bot",
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

st.title("⚡ TRADE WITH TUCKYAA — Dual 4H Zone (01:30 & 05:30) Engine")

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

# Fixed Scan Timeframe for Entries
selected_tf = "5m"
st.sidebar.text("⏱️ Entry Timeframe : 5m (Fixed)")
st.sidebar.text("📊 Zone Timeframe  : 4h (01:30 & 05:30)")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Dual Anchor Rules")
st.sidebar.text("Zone 1 : 01:30 AM 4H High/Low (Post 01:30)")
st.sidebar.text("Zone 2 : 05:30 AM 4H High/Low (Post 05:30)")
st.sidebar.text("BUY    : 5M Green Close in Demand Zone")
st.sidebar.text("SELL   : 5M Red Close in Supply Zone")

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
def fetch_data(symbol):
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
        # Fetch 4H data for Zones
        ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=50)
        # Fetch 5M data for Entries
        ohlcv_5m = exchange.fetch_ohlcv(sym, timeframe="5m", limit=300)
        if ohlcv_4h and ohlcv_5m:
          df_4h = pd.DataFrame(ohlcv_4h, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
          df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], unit="ms", utc=True)
          df_4h = df_4h.set_index("timestamp")

          df_5m = pd.DataFrame(ohlcv_5m, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
          df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], unit="ms", utc=True)
          df_5m = df_5m.set_index("timestamp")
          return df_4h, df_5m
    except Exception:
      continue
  return None, None


# ---------------------------------------------------------
# DUAL ANCHOR MULTI-TIMEFRAME ENGINE
# ---------------------------------------------------------
class TWTDualAnchorEngine:

  def __init__(self, df_4h, df_5m, symbol, user_tz):
    self.df_4h = df_4h
    self.df_5m = df_5m
    self.symbol = symbol
    self.user_tz = user_tz
    self.candles_5m = []
    self.history = []
    self.signals = []
    # Anchor 1 (01:30 AM)
    self.demand_box_1 = None
    self.supply_box_1 = None
    # Anchor 2 (05:30 AM)
    self.demand_box_2 = None
    self.supply_box_2 = None

  def _build_zones(self, anchor_candle):
    high_val = anchor_candle["high"]
    low_val = anchor_candle["low"]
    rng = high_val - low_val
    zone_thickness = (rng * 0.3) if rng > 0 else (high_val * 0.002)

    supply_box = (high_val - zone_thickness, high_val)
    demand_box = (low_val, low_val + zone_thickness)
    return demand_box, supply_box

  def calculate_and_scan(self):
    if self.df_4h is None or self.df_4h.empty or self.df_5m is None or self.df_5m.empty:
      return

    current_utc_time = datetime.now(timezone.utc)

    # 1. Process 4H Candles
    candles_4h = []
    for row in self.df_4h.iterrows():
      dt_utc = row[0]
      dt_user = dt_utc.astimezone(self.user_tz)
      r = row[1]
      candles_4h.append({
          "date": dt_user.date(),
          "time_obj": dt_user.time(),
          "dt_user": dt_user,
          "high": float(r["High"]),
          "low": float(r["Low"])
      })

    latest_date = candles_4h[-1]["date"] if candles_4h else datetime.now().date()
    today_4h = [c for c in candles_4h if c["date"] == latest_date]
    if not today_4h:
      today_4h = candles_4h

    # Anchor 1: ~01:30 AM
    anchor_1 = min(
        today_4h,
        key=lambda x: abs(datetime.combine(latest_date, x["time_obj"]) - datetime.combine(latest_date, time(1, 30))),
        default=today_4h[0]
    )
    anchor_dt_1 = anchor_1["dt_user"]
    self.demand_box_1, self.supply_box_1 = self._build_zones(anchor_1)

    # Anchor 2: ~05:30 AM (Only if 05:30 candle has formed)
    anchor_2_candidates = [c for c in today_4h if c["time_obj"] >= time(5, 0)]
    anchor_2 = None
    anchor_dt_2 = None
    if anchor_2_candidates:
      anchor_2 = min(
          anchor_2_candidates,
          key=lambda x: abs(datetime.combine(latest_date, x["time_obj"]) - datetime.combine(latest_date, time(5, 30)))
      )
      anchor_dt_2 = anchor_2["dt_user"]
      self.demand_box_2, self.supply_box_2 = self._build_zones(anchor_2)

    # 2. Process 5M Candles for Entries
    self.df_5m["TR"] = np.maximum(
        self.df_5m["High"] - self.df_5m["Low"],
        np.maximum(
            abs(self.df_5m["High"] - self.df_5m["Close"].shift(1)),
            abs(self.df_5m["Low"] - self.df_5m["Close"].shift(1)),
        ),
    )
    self.df_5m["ATR"] = self.df_5m["TR"].rolling(14).mean()

    for idx, row in enumerate(self.df_5m.iterrows()):
      dt_utc = row[0]
      dt_user = dt_utc.astimezone(self.user_tz)
      r = row[1]
      o, h, l, c, v = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]), float(r["Volume"])
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)
      age_min = (current_utc_time - dt_utc).total_seconds() / 60.0

      self.candles_5m.append({
          "id": idx + 1,
          "time": dt_user.strftime("%H:%M"),
          "date": dt_user.date(),
          "dt_user": dt_user,
          "color": "Green" if c >= o else "Red",
          "open": o, "high": h, "low": l, "close": c,
          "atr": atr,
          "is_within_1hr": age_min <= 60.0
      })

    today_5m = [c for c in self.candles_5m if c["date"] == latest_date]
    if not today_5m:
      today_5m = self.candles_5m

    # Track 1 Buy & 1 Sell per Anchor Zone
    buy_1, sell_1 = None, None
    buy_2, sell_2 = None, None

    for c in today_5m:
      atr = c["atr"]
      dec = get_decimals(c["close"])

      # --- ZONE 1 SCAN (Post 01:30 AM) ---
      if c["dt_user"] > anchor_dt_1 and self.demand_box_1 and self.supply_box_1:
        if not buy_1 and c["color"] == "Green" and self.demand_box_1[0] <= c["close"] <= self.demand_box_1[1]:
          sl = round(self.demand_box_1[0] - (1.5 * atr), dec)
          buy_1 = self._format_signal(c, "BUY 🟢", "01:30 Zone Demand Close", sl, atr, dec, is_buy=True)

        if not sell_1 and c["color"] == "Red" and self.supply_box_1[0] <= c["close"] <= self.supply_box_1[1]:
          sl = round(self.supply_box_1[1] + (1.5 * atr), dec)
          sell_1 = self._format_signal(c, "SELL 🔴", "01:30 Zone Supply Close", sl, atr, dec, is_buy=False)

      # --- ZONE 2 SCAN (Post 05:30 AM) ---
      if anchor_dt_2 and c["dt_user"] > anchor_dt_2 and self.demand_box_2 and self.supply_box_2:
        if not buy_2 and c["color"] == "Green" and self.demand_box_2[0] <= c["close"] <= self.demand_box_2[1]:
          sl = round(self.demand_box_2[0] - (1.5 * atr), dec)
          buy_2 = self._format_signal(c, "BUY 🟢", "05:30 Zone Demand Close", sl, atr, dec, is_buy=True)

        if not sell_2 and c["color"] == "Red" and self.supply_box_2[0] <= c["close"] <= self.supply_box_2[1]:
          sl = round(self.supply_box_2[1] + (1.5 * atr), dec)
          sell_2 = self._format_signal(c, "SELL 🔴", "05:30 Zone Supply Close", sl, atr, dec, is_buy=False)

    collected = [sig for sig in [buy_1, sell_1, buy_2, sell_2] if sig is not None]
    # Sort chronologically by Candle #
    collected.sort(key=lambda x: x["Candle #"])

    self.history = collected
    self.signals = [s for s in collected if s.get("Is_Within_1hr", False)]

  def _format_signal(self, c, sig_type, setup_name, sl, atr, dec, is_buy):
    entry = round(c["close"], dec)
    return {
        "Asset": self.symbol,
        "Date": str(c["date"]),
        "Time": c["time"],
        "Candle #": c["id"],
        "Type": sig_type,
        "Setup": setup_name,
        "Entry Price": entry,
        "Stop Loss (SL)": sl,
        "TP 1": round(entry + (1.2 * atr) if is_buy else entry - (1.2 * atr), dec),
        "TP 2": round(entry + (2.5 * atr) if is_buy else entry - (2.5 * atr), dec),
        "TP 3": round(entry + (4.0 * atr) if is_buy else entry - (4.0 * atr), dec),
        "Is_Within_1hr": c["is_within_1hr"],
    }


# ---------------------------------------------------------
# MASTER SCANNER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (20+ Assets)")

master_signals = []
asset_engines = {}


def scan_asset(sym):
  df_4h, df_5m = fetch_data(sym)
  if df_4h is not None and df_5m is not None:
    engine = TWTDualAnchorEngine(df_4h, df_5m, sym, active_tz)
    engine.calculate_and_scan()
    return sym, engine, engine.signals
  return sym, None, []


with st.spinner("⚡ Scanning 20+ assets (01:30 & 05:30 4H Zones + 5M Entries)..."):
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(scan_asset, sym): sym for sym in ASSET_LIST}
    for future in as_completed(futures):
      sym, engine, sigs = future.result()
      if engine:
        asset_engines[sym] = engine
      if sigs:
        master_signals.extend(sigs)

if master_signals:
  st.success(f"✅ Found {len(master_signals)} active setup(s) across assets!")
  df_display = pd.DataFrame(master_signals).drop(columns=["Is_Within_1hr"], errors="ignore")
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No active 5M zone-close signals in the last 1 hour.")

# ---------------------------------------------------------
# MANUAL ASSET INSPECTOR & CHART VIEWER (5M Chart)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector (5M Chart with Dual 4H Zones)")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df_4h, df_5m = fetch_data(selected_symbol)
  bot = TWTDualAnchorEngine(df_4h, df_5m, selected_symbol, active_tz)
  bot.calculate_and_scan()

if bot.candles_5m:
  curr_p = bot.candles_5m[-1]["close"]

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label=f"Current {selected_symbol} Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Locked Signals Today", value=len(bot.history))
  with col3:
    st.metric(label="Timezone", value=selected_tz_name)

  if bot.history:
    st.markdown(f"### 📋 Locked Entries Feed for {selected_symbol}")
    df_history = pd.DataFrame(bot.history).drop(columns=["Is_Within_1hr"], errors="ignore")
    st.dataframe(df_history, width="stretch")

  # --- PLOT 5M CANDLESTICK CHART ---
  fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
  for c in bot.candles_5m:
    col = "#26a69a" if c["close"] >= c["open"] else "#ef5350"
    lower_val = min(c["open"], c["close"])
    body_height = abs(c["close"] - c["open"])
    if body_height == 0:
      body_height = curr_p * 0.0005

    ax.bar(c["id"], body_height, bottom=lower_val, color=col, width=0.6, edgecolor=col)
    ax.vlines(c["id"], c["low"], c["high"], color=col, linewidth=1.2)

  # Plot 01:30 AM Anchor Zones
  if bot.demand_box_1:
    ax.axhspan(bot.demand_box_1[0], bot.demand_box_1[1], color="#26a69a", alpha=0.22, label="Demand 1 (01:30 Low)")
  if bot.supply_box_1:
    ax.axhspan(bot.supply_box_1[0], bot.supply_box_1[1], color="#ef5350", alpha=0.22, label="Supply 1 (01:30 High)")

  # Plot 05:30 AM Anchor Zones
  if bot.demand_box_2:
    ax.axhspan(bot.demand_box_2[0], bot.demand_box_2[1], color="#00e676", alpha=0.22, label="Demand 2 (05:30 Low)")
  if bot.supply_box_2:
    ax.axhspan(bot.supply_box_2[0], bot.supply_box_2[1], color="#ff9100", alpha=0.22, label="Supply 2 (05:30 High)")

  lows = [c["low"] for c in bot.candles_5m]
  highs = [c["high"] for c in bot.candles_5m]
  if lows and highs:
    margin = (max(highs) - min(lows)) * 0.09
    if margin == 0:
      margin = curr_p * 0.005
    ax.set_ylim(min(lows) - margin, max(highs) + margin)

  for sig in bot.history:
    is_b = "BUY" in sig["Type"]
    bg_col = "#00e676" if is_b else "#ff5252"
    txt = " B1 " if ("BUY" in sig["Type"] and "01:30" in sig["Setup"]) else \
          " B2 " if ("BUY" in sig["Type"] and "05:30" in sig["Setup"]) else \
          " S1 " if ("SELL" in sig["Type"] and "01:30" in sig["Setup"]) else " S2 "
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
      f"{selected_symbol} 5m Chart — Dual 4H Anchor Zones (01:30 & 05:30)",
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
  st.error("Market data fetch failed or empty.")
