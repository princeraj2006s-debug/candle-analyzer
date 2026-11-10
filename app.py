from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta, timezone
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWT Delta Pro by TUCKYAA", page_icon="⚡", layout="wide"
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
# GLOBAL TRADING TIMEZONES & COMPLETE 20+ ASSETS LIST
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
count = st_autorefresh(interval=30000, key="twt_delta_refresh")

st.title(
    "⚡ TRADE WITH TUCKYAA (PRO) — Engulfing Seller & Refined Buyer Engine"
)

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
st.sidebar.markdown("### 🤖 TWT Ultimate Price Action Panel")
st.sidebar.markdown("---")
server_time_target = datetime.now(active_tz)
st.sidebar.text(
    f"Server time          ———  {server_time_target.strftime('%H:%M')}"
)
st.sidebar.text("Support Ref (6H)   ———  05:30 IST Candle Low")
st.sidebar.text("Resistance Ref (4H) ———  01:30 IST Candle High")
st.sidebar.text("No-Trade Zone      ———  01:00 ➔ 06:00 IST")
st.sidebar.markdown("---")
st.sidebar.text("Exchange Platform  — Delta Exchange (20 Assets)")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Strategy Settings")
st.sidebar.text("Sell Rule        : 3 Green Candles + Engulfing Red Candle Entry")
st.sidebar.text("Buy Rule         : 7 Red Exhaustion + Weak Dip + Recovery")
st.sidebar.text("Strict Lock      : Single First Entry Only")


def get_decimals(price):
  if price < 1:
    return 4
  elif price < 10:
    return 3
  else:
    return 2


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
          df_5 = pd.DataFrame(
              ohlcv_tf,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df_5["timestamp"] = pd.to_datetime(df_5["timestamp"], unit="ms")
          df_5 = df_5.set_index("timestamp")

          df_4 = pd.DataFrame(
              ohlcv_4h,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df_4["timestamp"] = pd.to_datetime(df_4["timestamp"], unit="ms")
          df_4 = df_4.set_index("timestamp")

          df_6 = (
              pd.DataFrame(
                  ohlcv_6h,
                  columns=[
                      "timestamp",
                      "Open",
                      "High",
                      "Low",
                      "Close",
                      "Volume",
                  ],
              )
              if ohlcv_6h
              else df_4.copy()
          )
          df_6["timestamp"] = pd.to_datetime(df_6["timestamp"], unit="ms")
          df_6 = df_6.set_index("timestamp")
          return df_5, df_4, df_6
    except Exception:
      continue
  return None, None, None


class TWTEngulfingEngine:

  def __init__(self, df_5, df_4, df_6, symbol, user_tz):
    self.df_5 = df_5
    self.df_4 = df_4
    self.df_6 = df_6
    self.symbol = symbol
    self.user_tz = user_tz
    self.candles = []
    self.signals = []
    self.history_signals = []
    self.resistance_lvl = 0
    self.support_lvl = 0

  def calculate_and_scan(self):
    if self.df_5 is None or self.df_5.empty:
      return

    IST = timezone(timedelta(hours=5, minutes=30))
    current_utc_time = datetime.now(timezone.utc)

    # 1. Calculate Key Levels
    if self.df_4 is not None and not self.df_4.empty:
      dt_index_4 = (
          self.df_4.index
          if isinstance(self.df_4.index, pd.DatetimeIndex)
          else pd.to_datetime(self.df_4.index)
      )
      idx_utc = (
          dt_index_4.tz_localize("UTC")
          if dt_index_4.tz is None
          else dt_index_4
      )
      df_4_ist = idx_utc.tz_convert(IST)
      filtered_4h = self.df_4[df_4_ist.hour == 1]
      self.resistance_lvl = (
          float(filtered_4h["High"].iloc[-1])
          if not filtered_4h.empty
          else float(self.df_4["High"].iloc[-1])
      )

    if self.df_6 is not None and not self.df_6.empty:
      dt_index_6 = (
          self.df_6.index
          if isinstance(self.df_6.index, pd.DatetimeIndex)
          else pd.to_datetime(self.df_6.index)
      )
      idx_utc_6 = (
          dt_index_6.tz_localize("UTC")
          if dt_index_6.tz is None
          else dt_index_6
      )
      df_6_ist = idx_utc_6.tz_convert(IST)
      filtered_6h = self.df_6[df_6_ist.hour == 5]
      self.support_lvl = (
          float(filtered_6h["Low"].iloc[-1])
          if not filtered_6h.empty
          else float(self.df_6["Low"].iloc[-1])
      )
    elif self.df_4 is not None and not self.df_4.empty:
      self.support_lvl = float(self.df_4["Low"].iloc[-1])

    # 2. Indicators Calculation (ATR)
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

    found_signal = None
    is_major_coin = self.symbol in ["BTC/USDT", "ETH/USDT"]
    zone_multiplier = 3.5 if is_major_coin else 2.5

    for i in range(len(self.candles)):
      c = self.candles[i]
      if time(1, 0) <= c["dt_ist"].time() <= time(6, 0):
        continue

      atr = c["atr"] if not np.isnan(c["atr"]) else (c["close"] * 0.002)
      dec = get_decimals(c["close"])

      # -------------------------------------------------------------
      # 1. BUY LOGIC: 7 Red Exhaustion + Weak Dip + Recovery Breakout
      # -------------------------------------------------------------
      if c["close"] <= self.support_lvl + (atr * zone_multiplier):
        if i >= 7:
          red_count = 0
          for k in range(i - 6, i + 1):
            if self.candles[k]["color"] == "Red":
              red_count += 1
          
          if red_count >= 6:
            for m in range(i + 1, min(i + 5, len(self.candles))):
              cm = self.candles[m]
              if cm["color"] == "Green" and cm["close"] > self.candles[m - 1]["high"]:
                sl = round(cm["low"] - (1.2 * atr), dec)
                tp1 = round(cm["close"] + (1.2 * atr), dec)
                tp2 = round(cm["close"] + (2.5 * atr), dec)
                tp3 = round(cm["close"] + (4.0 * atr), dec)

                found_signal = {
                    "Asset": self.symbol,
                    "Time": cm["time"],
                    "Candle #": cm["id"],
                    "Type": "BUY 🟢",
                    "Setup": "7-Red Exhaustion & Recovery",
                    "Action": "Buy / Long (B) [Exhaustion Bounce] 🟢",
                    "Entry Price": round(cm["close"], dec),
                    "Stop Loss (SL)": sl,
                    "TP 1": tp1,
                    "TP 2": tp2,
                    "TP 3": tp3,
                    "Is_Within_1hr": cm["is_within_1hr"],
                }
                break
        if found_signal:
          break

      # -------------------------------------------------------------
      # 2. SELLER LOGIC: 3 Green Candles followed by an Engulfing Red Candle
      # -------------------------------------------------------------
      if i >= 3:
        # Check if previous 3 candles (i-3, i-2, i-1) are ALL GREEN
        c1 = self.candles[i - 3]
        c2 = self.candles[i - 2]
        c3 = self.candles[i - 1]
        curr = self.candles[i] # Current candle must be RED and engulf previous Green (c3)

        if c1["color"] == "Green" and c2["color"] == "Green" and c3["color"] == "Green":
          if curr["color"] == "Red":
            # Check Bearish Engulfing: Current Red body completely covers previous Green body
            prev_green_body_top = max(c3["open"], c3["close"])
            prev_green_body_bot = min(c3["open"], c3["close"])
            curr_red_body_top = max(curr["open"], curr["close"])
            curr_red_body_bot = min(curr["open"], curr["close"])

            is_engulfing = (curr_red_body_top >= prev_green_body_top) and (curr_red_body_bot <= prev_green_body_bot)

            if is_engulfing:
              sl = round(curr["high"] + (1.5 * atr), dec)
              tp1 = round(curr["close"] - (1.2 * atr), dec)
              tp2 = round(curr["close"] - (2.5 * atr), dec)
              tp3 = round(curr["close"] - (4.0 * atr), dec)

              found_signal = {
                  "Asset": self.symbol,
                  "Time": curr["time"],
                  "Candle #": curr["id"],
                  "Type": "SELL 🔴",
                  "Setup": "3 Green Candles + Engulfing Red Reversal",
                  "Action": "Sell / Short (S) [3-Green Engulfing] 🔴",
                  "Entry Price": round(curr["close"], dec),
                  "Stop Loss (SL)": sl,
                  "TP 1": tp1,
                  "TP 2": tp2,
                  "TP 3": tp3,
                  "Is_Within_1hr": curr["is_within_1hr"],
              }
              break

      if found_signal:
        break

    if found_signal:
      self.signals = [found_signal]
      self.history_signals = [found_signal]
    else:
      self.signals = []
      self.history_signals = []


# ---------------------------------------------------------
# MASTER SCANNER (Global Feed for Top Dashboard)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (Last 1 Hour)")

master_signals = []
asset_engines = {}


def scan_single_asset(sym):
  df_5, df_4, df_6 = fetch_twt_delta_data(sym, selected_tf)
  if df_5 is not None and not df_5.empty:
    engine = TWTEngulfingEngine(df_5, df_4, df_6, sym, active_tz)
    engine.calculate_and_scan()
    recent_sigs = [s for s in engine.signals if s.get("Is_Within_1hr", False)]
    latest = recent_sigs[0] if recent_sigs else None
    return sym, engine, latest
  return sym, None, None


with st.spinner("⚡ Scanning with Engulfing Seller Logic..."):
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {
        executor.submit(scan_single_asset, sym): sym for sym in ASSET_LIST
    }
    for future in as_completed(futures):
      sym, engine, latest = future.result()
      if engine:
        asset_engines[sym] = engine
      if latest:
        master_signals.append(latest)

if master_signals:
  st.success(f"✅ Found {len(master_signals)} active setup(s)!")
  df_display = pd.DataFrame(master_signals).drop(
      columns=["Is_Within_1hr"], errors="ignore"
  )
  st.dataframe(df_display, width="stretch")
else:
  st.info(
      "ℹ️ No active signals in the last 1 hour. Waiting for setup triggers..."
  )

# ---------------------------------------------------------
# MANUAL ASSET SELECTOR & CHART VIEWER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Manual Asset Inspector & Chart")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart & Engulfing Entry",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df_5, df_4, df_6 = fetch_twt_delta_data(selected_symbol, selected_tf)
  bot = TWTEngulfingEngine(
      df_5, df_4, df_6, selected_symbol, active_tz
  )
  bot.calculate_and_scan()

if bot.df_5 is not None and not bot.df_5.empty:
  curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric(
        label="4H Resistance (01:30 High)", value=f"{bot.resistance_lvl:.2f}"
    )
  with col2:
    st.metric(label="6H Support (05:30 Low)", value=f"{bot.support_lvl:.2f}")
  with col3:
    st.metric(label="Current Market Price", value=f"{curr_p:.2f}")
  with col4:
    st.metric(label="Timezone", value=tz_short, delta="Engulfing Logic Active")

  if bot.signals:
    latest_sig = bot.signals[0]
    is_buy = "BUY" in latest_sig["Type"]
    entry_p = latest_sig["Entry Price"]
    signal_time = latest_sig["Time"]

    st.warning("⚠️ **Don't use after 40 minutes!**")

    if is_buy:
      st.success(
          f"🟢 **TWT BUY SETUP: {latest_sig['Action']}**\n\n"
          f"🕒 **Candle Time ({tz_short}):** `{signal_time}` (Candle"
          f" #{latest_sig['Candle #']})\n"
          f"- **Entry Price:** `{entry_p}`\n"
          f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
          f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:**"
          f" `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`"
      )
    else:
      st.error(
          f"🔴 **TWT SELL SETUP: {latest_sig['Action']}**\n\n"
          f"🕒 **Candle Time ({tz_short}):** `{signal_time}` (Candle"
          f" #{latest_sig['Candle #']})\n"
          f"- **Entry Price:** `{entry_p}`\n"
          f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
          f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:**"
          f" `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`"
      )
  else:
    st.warning(
        "🟡 **STATUS: WAITING FOR SETUP**\n\n"
        "📌 *Instructions:* Monitoring 3-Green followed by Engulfing Red for Sell, and 7-Red exhaustion for Buy."
    )

  # Complete Single History Table
  if bot.history_signals:
    st.markdown("---")
    st.subheader(
        f"📋 Single Entry History for {selected_symbol}"
    )
    df_history = pd.DataFrame(bot.history_signals).drop(
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

  ax.axhline(
      bot.resistance_lvl,
      color="#ffeb3b",
      linestyle="--",
      linewidth=2,
      label="4H Resistance",
  )
  ax.text(
      2,
      bot.resistance_lvl,
      f"  4H Resistance ({bot.resistance_lvl:.2f})",
      color="#ffeb3b",
      weight="bold",
      fontsize=10,
  )

  ax.axhline(
      bot.support_lvl,
      color="#26a69a",
      linestyle="--",
      linewidth=2,
      label="6H Support",
  )
  ax.text(
      2,
      bot.support_lvl,
      f"  6H Support ({bot.support_lvl:.2f})",
      color="#26a69a",
      weight="bold",
      fontsize=10,
  )

  lows = [c["low"] for c in bot.candles]
  highs = [c["high"] for c in bot.candles]
  if bot.resistance_lvl > 0:
    highs.append(bot.resistance_lvl)
  if bot.support_lvl > 0:
    lows.append(bot.support_lvl)

  if lows and highs:
    margin = (max(highs) - min(lows)) * 0.09
    if margin == 0:
      margin = curr_p * 0.005
    ax.set_ylim(min(lows) - margin, max(highs) + margin)

  for sig in bot.signals:
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
      f"{selected_symbol} 5m — TRADE WITH TUCKYAA Engulfing Engine ({tz_short})",
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

  st.pyplot(fig)
else:
  st.error("Market data load failed. Please check internet connection.")
