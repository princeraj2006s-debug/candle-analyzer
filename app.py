from datetime import datetime, time, timedelta, timezone
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWT BTC Mid-Zone Precision Engine", page_icon="⚡", layout="wide"
)

# Hide Streamlit Header, Footer, Menu & GitHub Icon via CSS
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------------------------------
# TIMEZONES & EXCLUSIVE BTC/USDT ASSET LIST
# ---------------------------------------------------------
TRADING_TIMEZONES = {
    "(UTC+5:30) Kolkata / India (IST)": timezone(
        timedelta(hours=5, minutes=30)
    ),
    "(UTC+1:00) Berlin / Paris / CET": timezone(timedelta(hours=1)),
    "(UTC+0:00) London / UTC / GMT": timezone.utc,
    "(UTC-5:00) New York / US East (EST)": timezone(timedelta(hours=-5)),
    "(UTC+8:00) Singapore / Hong Kong (SGT)": timezone(timedelta(hours=8)),
}

ASSET_LIST = ["BTC/USDT"]

# Safe Auto-refresh every 30 seconds
count = st_autorefresh(interval=30000, key="twt_btc_refresh")

st.title("⚡ TRADE WITH TUCKYAA (BTC PRO) — Mid-Zone Precision Engine")

col_tf, col_tz = st.columns([1, 2])
with col_tf:
  selected_tf = st.selectbox(
      "⏱️ Timeframe", options=["5m", "15m", "30m", "1h"], index=0
  )
with col_tz:
  selected_tz_name = st.selectbox(
      "🌍 Trading Timezone", options=list(TRADING_TIMEZONES.keys()), index=0
  )
  active_tz = TRADING_TIMEZONES[selected_tz_name]

tz_short = selected_tz_name.split(")")[0] + ")"

# Sidebar Panel
st.sidebar.markdown("### 🤖 TWT BTC Strategy Panel")
st.sidebar.markdown("---")
server_time_target = datetime.now(active_tz)
st.sidebar.text(
    f"Server time          ———  {server_time_target.strftime('%H:%M')}"
)
st.sidebar.text("Active Asset       ———  BTC/USDT Only")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Mid-Zone Rules")
st.sidebar.text("Demand Midpoint ➔ Green Candle ➔ 1 Buy Entry")
st.sidebar.text("Supply Midpoint ➔ Red Candle   ➔ 1 Sell Entry")
st.sidebar.text("Risk Mgmt       ➔ ATR Managed SL & Multi-TP")


def get_decimals(price):
  if price < 1:
    return 4
  elif price < 10:
    return 3
  else:
    return 2


@st.cache_data(ttl=30)
def fetch_btc_data(symbol, tf):
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
        ohlcv_tf = exchange.fetch_ohlcv(sym, timeframe=tf, limit=180)
        if ohlcv_tf:
          df = pd.DataFrame(
              ohlcv_tf,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
          df = df.set_index("timestamp")
          return df
    except Exception:
      continue
  return None


class TWTMidZoneEngine:

  def __init__(self, df, symbol, user_tz):
    self.df = df
    self.symbol = symbol
    self.user_tz = user_tz
    self.candles = []
    self.signals = []
    self.history = []
    self.demand_box = None
    self.supply_box = None
    self.demand_mid = 0.0
    self.supply_mid = 0.0

  def calculate_and_scan(self):
    if self.df is None or self.df.empty:
      return

    IST = timezone(timedelta(hours=5, minutes=30))
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
      dt_ist = dt_utc.astimezone(IST)
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
          "dt_ist": dt_ist,
          "dt_user": dt_user_tz,
          "color": "Green" if c >= o else "Red",
          "open": o,
          "high": h,
          "low": l,
          "close": c,
          "volume": v,
          "atr": atr,
          "is_within_1hr": is_within_1hr,
      })

    all_lows = [c["low"] for c in self.candles]
    all_highs = [c["high"] for c in self.candles]
    
    if not all_lows or not all_highs:
      return

    min_price = min(all_lows)
    max_price = max(all_highs)
    total_range = max_price - min_price

    # Structural Zones (~20% range each) & their Midpoints
    demand_zone_low = min_price
    demand_zone_high = min_price + (total_range * 0.22)
    self.demand_box = (demand_zone_low, demand_zone_high)
    self.demand_mid = (demand_zone_low + demand_zone_high) / 2.0

    supply_zone_high = max_price
    supply_zone_low = max_price - (total_range * 0.22)
    self.supply_box = (supply_zone_low, supply_zone_high)
    self.supply_mid = (supply_zone_low + supply_zone_high) / 2.0

    latest_buy = None
    latest_sell = None

    # Scan backwards from latest candles to find exact Mid-Zone touches
    for c in reversed(self.candles):
      atr = c["atr"]
      dec = get_decimals(c["close"])

      # BUY: Price dipped to Demand Midpoint or lower, and formed a Green candle
      if not latest_buy and c["low"] <= self.demand_mid and c["color"] == "Green":
        sl = round(demand_zone_low - (1.0 * atr), dec)
        tp1 = round(c["close"] + (1.2 * atr), dec)
        tp2 = round(c["close"] + (2.5 * atr), dec)
        tp3 = round(c["close"] + (4.0 * atr), dec)

        latest_buy = {
            "Asset": self.symbol,
            "Time": c["time"],
            "Candle #": c["id"],
            "Type": "BUY 🟢",
            "Setup": "Demand Midpoint Bounce",
            "Action": "Buy / Long (Demand Mid) 🟢",
            "Entry Price": round(c["close"], dec),
            "Stop Loss (SL)": sl,
            "TP 1": tp1,
            "TP 2": tp2,
            "TP 3": tp3,
            "Is_Within_1hr": c["is_within_1hr"],
        }

      # SELL: Price rose to Supply Midpoint or higher, and formed a Red candle
      if not latest_sell and c["high"] >= self.supply_mid and c["color"] == "Red":
        sl = round(supply_zone_high + (1.0 * atr), dec)
        tp1 = round(c["close"] - (1.2 * atr), dec)
        tp2 = round(c["close"] - (2.5 * atr), dec)
        tp3 = round(c["close"] - (4.0 * atr), dec)

        latest_sell = {
            "Asset": self.symbol,
            "Time": c["time"],
            "Candle #": c["id"],
            "Type": "SELL 🔴",
            "Setup": "Supply Midpoint Rejection",
            "Action": "Sell / Short (Supply Mid) 🔴",
            "Entry Price": round(c["close"], dec),
            "Stop Loss (SL)": sl,
            "TP 1": tp1,
            "TP 2": tp2,
            "TP 3": tp3,
            "Is_Within_1hr": c["is_within_1hr"],
        }

      if latest_buy and latest_sell:
        break

    collected = []
    if latest_buy:
      collected.append(latest_buy)
    if latest_sell:
      collected.append(latest_sell)

    self.history = collected
    self.signals = [s for s in collected if s.get("Is_Within_1hr", False)]


# ---------------------------------------------------------
# EXECUTION & DASHBOARD FOR BTC/USDT ONLY
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 BTC/USDT Mid-Zone Precision Dashboard")

df_btc = fetch_btc_data("BTC/USDT", selected_tf)
bot = TWTMidZoneEngine(df_btc, "BTC/USDT", active_tz)
bot.calculate_and_scan()

if bot.signals:
  st.success(f"✅ Active Mid-Zone Setup(s) Found!")
  df_display = pd.DataFrame(bot.signals).drop(
      columns=["Is_Within_1hr"], errors="ignore"
  )
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No active signal in the last 1 hour. Exact mid-zone entries shown in history below.")

# --- MANUAL INSPECTOR & CHART VIEWER ---
st.markdown("---")
st.subheader("📊 BTC/USDT Chart & Mid-Zone Entries")

if bot.df is not None and not bot.df.empty:
  curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label="Current BTC Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Active Entries Displayed", value=len(bot.history))
  with col3:
    st.metric(label="Timezone", value=tz_short, delta="Mid-Zone Engine Active")

  if bot.history:
    st.markdown("### 📋 Exact Mid-Zone Entries Feed (1 Buy & 1 Sell)")
    df_history = pd.DataFrame(bot.history).drop(
        columns=["Is_Within_1hr"], errors="ignore"
    )
    st.dataframe(df_history, width="stretch")

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

  # Plot Demand and Supply Zones & Midlines
  if bot.demand_box:
    d_low, d_high = bot.demand_box
    ax.axhspan(d_low, d_high, color="#26a69a", alpha=0.25, label="Demand Zone (Buy Support)")
    ax.axhline(bot.demand_mid, color="#00e676", linestyle=":", linewidth=1.5, label="Demand Midpoint")

  if bot.supply_box:
    s_low, s_high = bot.supply_box
    ax.axhspan(s_low, s_high, color="#ef5350", alpha=0.25, label="Supply Zone (Sell Resistance)")
    ax.axhline(bot.supply_mid, color="#ff5252", linestyle=":", linewidth=1.5, label="Supply Midpoint")

  lows = [c["low"] for c in bot.candles]
  highs = [c["high"] for c in bot.candles]
  if lows and highs:
    margin = (max(highs) - min(lows)) * 0.09
    if margin == 0:
      margin = curr_p * 0.005
    ax.set_ylim(min(lows) - margin, max(highs) + margin)

  # Plot exact mid-zone signals on chart
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
      f"BTC/USDT {selected_tf} — Mid-Zone Precision Engine ({tz_short})",
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
  st.error("BTC market data load failed. Please check internet connection.")
