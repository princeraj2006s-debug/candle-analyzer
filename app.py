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
    page_title="TWT 4H Zone & 5M Entry Precision Bot",
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

st.title("⚡ TRADE WITH TUCKYAA — 4H Zone & 5M Entry Engine")

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
st.sidebar.text("📊 Zone Timeframe  : 4h (Anchor ~01:30)")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Rules")
st.sidebar.text("Supply Zone : 4H Anchor High")
st.sidebar.text("Demand Zone : 4H Anchor Low")
st.sidebar.text("Entries Only After : 01:30 AM Anchor Candle")
st.sidebar.text("BUY  : 5M Green Close in Demand Zone (1/day)")
st.sidebar.text("SELL : 5M Red Close in Supply Zone (1/day)")

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
# MULTI-TIMEFRAME ENGINE
# ---------------------------------------------------------
class TWTMultiTFEngine:

  def __init__(self, df_4h, df_5m, symbol, user_tz):
    self.df_4h = df_4h
    self.df_5m = df_5m
    self.symbol = symbol
    self.user_tz = user_tz
    self.candles_5m = []
    self.history = []
    self.signals = []
    self.demand_box = None
    self.supply_box = None

  def calculate_and_scan(self):
    if self.df_4h is None or self.df_4h.empty or self.df_5m is None or self.df_5m.empty:
      return

    current_utc_time = datetime.now(timezone.utc)

    # 1. Calculate 4H Anchor Zones
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

    if candles_4h:
      latest_date = candles_4h[-1]["date"]
    else:
      latest_date = datetime.now().date()

    today_4h = [c for c in candles_4h if c["date"] == latest_date]
    if not today_4h:
      today_4h = candles_4h

    # Find anchor candle closest to 01:30 AM
    anchor_target = time(1, 30)
    anchor_4h = min(
        today_4h,
        key=lambda x: abs(datetime.combine(latest_date, x["time_obj"]) - datetime.combine(latest_date, anchor_target)),
        default=today_4h[0]
    )

    anchor_dt = anchor_4h["dt_user"]

    s_high = anchor_4h["high"]
    s_low = s_high - ((anchor_4h["high"] - anchor_4h["low"]) * 0.3)
    self.supply_box = (s_low, s_high)

    d_low = anchor_4h["low"]
    d_high = d_low + ((anchor_4h["high"] - anchor_4h["low"]) * 0.3)
    self.demand_box = (d_low, d_high)

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

    first_buy = None
    first_sell = None

    for c in today_5m:
      # STRICT RULE: Entries only allowed AFTER the 01:30 AM anchor candle time
      if c["dt_user"] <= anchor_dt:
        continue

      atr = c["atr"]
      dec = get_decimals(c["close"])

      # BUY Entry: 5M Green candle close inside Demand Zone
      if not first_buy and c["color"] == "Green" and self.demand_box[0] <= c["close"] <= self.demand_box[1]:
        sl = round(self.demand_box[0] - (1.5 * atr), dec)
        first_buy = {
            "Asset": self.symbol,
            "Date": str(c["date"]),
            "Time": c["time"],
            "Candle #": c["id"],
            "Type": "BUY 🟢",
            "Setup": "5M Demand Zone Green Close",
            "Entry Price": round(c["close"], dec),
            "Stop Loss (SL)": sl,
            "TP 1": round(c["close"] + (1.2 * atr), dec),
            "TP 2": round(c["close"] + (2.5 * atr), dec),
            "TP 3": round(c["close"] + (4.0 * atr), dec),
            "Is_Within_1hr": c["is_within_1hr"],
        }

      # SELL Entry: 5M Red candle close inside Supply Zone
      if not first_sell and c["color"] == "Red" and self.supply_box[0] <= c["close"] <= self.supply_box[1]:
        sl = round(self.supply_box[1] + (1.5 * atr), dec)
        first_sell = {
            "Asset": self.symbol,
            "Date": str(c["date"]),
            "Time": c["time"],
            "Candle #": c["id"],
            "Type": "SELL 🔴",
            "Setup": "5M Supply Zone Red Close",
            "Entry Price": round(c["close"], dec),
            "Stop Loss (SL)": sl,
            "TP 1": round(c["close"] - (1.2 * atr), dec),
            "TP 2": round(c["close"] - (2.5 * atr), dec),
            "TP 3": round(c["close"] - (4.0 * atr), dec),
            "Is_Within_1hr": c["is_within_1hr"],
        }

    collected = []
    if first_buy:
      collected.append(first_buy)
    if first_sell:
      collected.append(first_sell)

    self.history = collected
    self.signals = [s for s in collected if s.get("Is_Within_1hr", False)]


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
    engine = TWTMultiTFEngine(df_4h, df_5m, sym, active_tz)
    engine.calculate_and_scan()
    return sym, engine, engine.signals
  return sym, None, []


with st.spinner("⚡ Scanning 20+ assets (4H Zones + 5M Entries after 01:30)..."):
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
  st.info("ℹ️ No active 5M zone-close signals after 01:30 AM in the last 1 hour.")

# ---------------------------------------------------------
# MANUAL ASSET INSPECTOR & CHART VIEWER (5M Chart)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector (5M Chart with 4H Zones)")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df_4h, df_5m = fetch_data(selected_symbol)
  bot = TWTMultiTFEngine(df_4h, df_5m, selected_symbol, active_tz)
  bot.calculate_and_scan()

if bot.candles_5m:
  curr_p = bot.candles_5m[-1]["close"]

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label=f"Current {selected_symbol} Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Locked Signals Today", value=len(bot.history))
  with col3:
    st.metric(label="Timezone", value=selected_tz_name.split(")")[0] + ")")

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

  # Plot 4H Anchor Zones on 5M Chart
  if bot.demand_box:
    d_low, d_high = bot.demand_box
    ax.axhspan(d_low, d_high, color="#26a69a", alpha=0.25, label="Demand Zone (4H Anchor Low)")

  if bot.supply_box:
    s_low, s_high = bot.supply_box
    ax.axhspan(s_low, s_high, color="#ef5350", alpha=0.25, label="Supply Zone (4H Anchor High)")

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
      f"{selected_symbol} 5m Chart — 4H Anchor Zones & Entries",
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
