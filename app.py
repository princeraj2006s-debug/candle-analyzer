from datetime import datetime, time, timedelta, timezone
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
    page_title="TWT Professional BTC ORB Trading Bot",
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
count = st_autorefresh(interval=30000, key="twt_orb_refresh")

st.title("⚡ TRADE WITH TUCKYAA — Professional BTC ORB & Retest Engine")

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION (Strategy Parameters)
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ ORB Engine Configurations")
st.sidebar.markdown("---")

# Timezones
TRADING_TIMEZONES = {
    "(UTC+5:30) Kolkata / India (IST)": timezone(
        timedelta(hours=5, minutes=30)
    ),
    "(UTC+0:00) London / UTC / GMT": timezone.utc,
    "(UTC-5:00) New York / US East (EST)": timezone(timedelta(hours=-5)),
    "(UTC+8:00) Singapore / Hong Kong (SGT)": timezone(timedelta(hours=8)),
}
selected_tz_name = st.sidebar.selectbox(
    "🌍 Trading Timezone", options=list(TRADING_TIMEZONES.keys()), index=0
)
active_tz = TRADING_TIMEZONES[selected_tz_name]
tz_short = selected_tz_name.split(")")[0] + ")"

selected_tf = st.sidebar.selectbox(
    "⏱️ Entry Timeframe", options=["5m", "15m", "30m", "1h"], index=0
)
orb_duration_mins = st.sidebar.slider(
    "⏳ Opening Range Duration (Minutes)", min=15, max=120, value=75, step=5
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Filters & Risk Management")
use_ema_filter = st.sidebar.checkbox("Enable EMA 200 Trend Filter", value=True)
use_atr_buffer = st.sidebar.checkbox("Enable ATR Buffer for Stop Loss", value=True)
risk_reward_ratio = st.sidebar.slider(
    "🎯 Risk / Reward Target Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5
)
max_trades_per_day = st.sidebar.number_input(
    "Max Trades Per Day", min_value=1, max_value=5, value=1
)


# ---------------------------------------------------------
# DATA FETCHING MODULE WITH CCXT
# ---------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_btc_ohlcv(timeframe="5m", limit=200):
  exchanges_to_try = [
      (ccxt.binance, "BTC/USDT"),
      (ccxt.bybit, "BTC/USDT"),
      (ccxt.kucoin, "BTC/USDT"),
  ]

  for ex_class, symbol in exchanges_to_try:
    try:
      exchange = ex_class({"enableRateLimit": True, "timeout": 15000})
      exchange.load_markets()
      if symbol in exchange.markets:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if ohlcv:
          df = pd.DataFrame(
              ohlcv,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
          df = df.set_index("timestamp")
          return df
    except Exception:
      continue
  return None


# ---------------------------------------------------------
# TECHNICAL INDICATORS & ORB ENGINE
# ---------------------------------------------------------
class ProfessionalORBEngine:

  def __init__(self, df, orb_duration, user_tz):
    self.df = df
    self.orb_duration = orb_duration
    self.user_tz = user_tz
    self.orb_high = 0.0
    self.orb_low = 0.0
    self.orb_mid = 0.0
    self.signals = []
    self.history = []

  def calculate_indicators(self):
    if self.df is None or self.df.empty:
      return
    # EMA 200 Calculation
    self.df["EMA200"] = self.df["Close"].ewm(span=200, adjust=False).mean()

    # ATR Calculation
    self.df["TR"] = np.maximum(
        self.df["High"] - self.df["Low"],
        np.maximum(
            abs(self.df["High"] - self.df["Close"].shift(1)),
            abs(self.df["Low"] - self.df["Close"].shift(1)),
        ),
    )
    self.df["ATR"] = self.df["TR"].rolling(14).mean()

  def identify_opening_range(self):
    if self.df is None or self.df.empty:
      return
    # Group candles by calendar day and find the high/low of the first N minutes
    first_timestamp = self.df.index[0]
    session_start_time = first_timestamp.normalize() + timedelta(
        hours=9, minutes=30
    )  # Market open reference
    session_end_time = session_start_time + timedelta(
        minutes=self.orb_duration
    )

    orb_candles = self.df[
        (self.df.index >= session_start_time)
        & (self.df.index < session_end_time)
    ]

    if not orb_candles.empty:
      self.orb_high = float(orb_candles["High"].max())
      self.orb_low = float(orb_candles["Low"].min())
    else:
      # Fallback to first few candles if specific session time window is missing in data feed
      fallback_count = max(
          3, int(self.orb_duration / 5)
      )  v_slice = self.df.iloc[:fallback_count]
      self.orb_high = float(v_slice["High"].max())
      self.orb_low = float(v_slice["Low"].min())

    self.orb_mid = (self.orb_high + self.orb_low) / 2.0

  def scan_breakout_and_retest(self, use_ema, use_atr, rr_target):
    if self.df is None or len(self.df) < 20:
      return

    self.calculate_indicators()
    self.identify_opening_range()

    found_signal = None
    current_utc_time = datetime.now(timezone.utc)

    for i in range(
        int(self.orb_duration / 5) + 1, len(self.df)
    ):  # Start scanning after ORB period
      c = self.df.iloc[i]
      prev_c = self.df.iloc[i - 1]
      dt_utc = self.df.index[i]
      if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
      dt_user = dt_utc.astimezone(self.user_tz)

      close = float(c["Close"])
      open_p = float(c["Open"])
      high = float(c["High"])
      low = float(c["Low"])
      ema200 = float(c["EMA200"]) if not np.isnan(c["EMA200"]) else close
      atr = float(c["ATR"]) if not np.isnan(c["ATR"]) else (close * 0.002)

      # Age check (within last 1 hour)
      is_recent = (current_utc_time - dt_utc).total_seconds() / 60.0 <= 60.0

      # -------------------------------------------------
      # BULLISH BREAKOUT & RETEST SETUP
      # -------------------------------------------------
      breakout_condition = close > self.orb_high and prev_c["Close"] <= self.orb_high
      ema_cond_bull = (not use_ema) or (close > ema200)

      if breakout_condition and ema_cond_bull:
        # Check retest or continuation momentum confirmation
        sl = (
            round(self.orb_low - (atr * 0.5), 2)
            if use_atr
            else round(self.orb_low, 2)
        )
        risk = close - sl
        if risk <= 0:
          continue
        target = round(close + (risk * rr_target), 2)

        found_signal = {
            "Type": "BUY 🟢",
            "Setup": "ORB Bullish Breakout & Continuation",
            "Entry Price": round(close, 2),
            "Stop Loss": sl,
            "Target": target,
            "R:R Ratio": f"1:{rr_target}",
            "Time": dt_user.strftime("%H:%M"),
            "Reason": "Clean close above ORB High with momentum confirmation.",
            "Is_Recent": is_recent,
        }
        break

      # -------------------------------------------------
      # BEARISH BREAKOUT & RETEST SETUP
      # -------------------------------------------------
      bear_breakout = close < self.orb_low and prev_c["Close"] >= self.orb_low
      ema_cond_bear = (not use_ema) or (close < ema200)

      if bear_breakout and ema_cond_bear:
        sl = (
            round(self.orb_high + (atr * 0.5), 2)
            if use_atr
            else round(self.orb_high, 2)
        )
        risk = sl - close
        if risk <= 0:
          continue
        target = round(close - (risk * rr_target), 2)

        found_signal = {
            "Type": "SELL 🔴",
            "Setup": "ORB Bearish Breakout & Continuation",
            "Entry Price": round(close, 2),
            "Stop Loss": sl,
            "Target": target,
            "R:R Ratio": f"1:{rr_target}",
            "Time": dt_user.strftime("%H:%M"),
            "Reason": "Clean close below ORB Low with momentum confirmation.",
            "Is_Recent": is_recent,
        }
        break

    if found_signal:
      self.signals = [found_signal]
      self.history = [found_signal]
    else:
      self.signals = []


# ---------------------------------------------------------
# EXECUTION & DASHBOARD INTERFACE
# ---------------------------------------------------------
df_btc = fetch_btc_ohlcv(timeframe=selected_tf, limit=150)
bot = ProfessionalORBEngine(df_btc, orb_duration_mins, active_tz)
bot.scan_breakout_and_retest(use_ema_filter, use_atr_buffer, risk_reward_ratio)

recent_signals = [s for s in bot.signals if s.get("Is_Recent", False)]

col_d1, col_d2, col_d3, col_d4 = st.columns(4)
with col_d1:
  st.metric(label="🟢 Opening Range High (ORB)", value=f"${bot.orb_high:,.2f}")
with col_d2:
  st.metric(label="🔴 Opening Range Low (ORB)", value=f"${bot.orb_low:,.2f}")
with col_d3:
  st.metric(label="⚪ Opening Range Mid", value=f"${bot.orb_mid:,.2f}")
with col_d4:
  current_price = (
      float(bot.df["Close"].iloc[-1])
      if bot.df is not None and not bot.df.empty
      else 0.0
  )
  st.metric(label="⚡ Current BTC Price", value=f"${current_price:,.2f}")

st.markdown("---")

if recent_signals:
  sig = recent_signals[0]
  st.success(f"✅ **ACTIVE ORB SETUP GENERATED: {sig['Type']}**")

  col_s1, col_s2, col_s3 = st.columns(3)
  with col_s1:
    st.info(f"**Setup Type:** {sig['Setup']}\n\n**Entry Time:** {sig['Time']}")
  with col_s2:
    st.warning(
        f"**Suggested Entry:** ${sig['Entry Price']}\n\n**Stop Loss:**"
        f" ${sig['Stop Loss']}"
    )
  with col_s3:
    st.success(
        f"**Target:** ${sig['Target']}\n\n**Risk/Reward:** {sig['R:R Ratio']}"
    )
else:
  st.info(
      "ℹ️ **Status:** Scanning market structure. Waiting for clean candle"
      " close breakout confirmation beyond the 75-min Opening Range."
  )

# ---------------------------------------------------------
# MATPLOTLIB PROFESSIONAL CHART RENDERER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Professional ORB Chart & Retest Visualizer")

if bot.df is not None and not bot.df.empty:
  fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

  # Plot Candlesticks
  for idx, row in enumerate(bot.df.iterrows()):
    r = row[1]
    col = "#26a69a" if r["Close"] >= r["Open"] else "#ef5350"
    lower_v = min(r["Open"], r["Close"])
    body_h = abs(r["Close"] - r["Open"])
    if body_h == 0:
      body_h = r["Close"] * 0.0005

    ax.bar(
        idx, body_h, bottom=lower_v, color=col, width=0.6, edgecolor=col
    )
    ax.vlines(idx, r["Low"], r["High"], color=col, linewidth=1.2)

  # Plot EMA 200 Line if available
  if "EMA200" in bot.df.columns:
    ax.plot(
        range(len(bot.df)),
        bot.df["EMA200"],
        color="#29b6f6",
        linewidth=1.5,
        label="EMA 200 Trend",
    )

  # Plot ORB High & Low Lines
  if bot.orb_high > 0:
    ax.axhline(
        bot.orb_high,
        color="#00e676",
        linestyle="--",
        linewidth=2,
        label=f"ORB High ({bot.orb_high:,.2f})",
    )
    ax.axhline(
        bot.orb_low,
        color="#ff5252",
        linestyle="--",
        linewidth=2,
        label=f"ORB Low ({bot.orb_low:,.2f})",
    )
    ax.axhline(
        bot.orb_mid,
        color="#ab47bc",
        linestyle=":",
        linewidth=1.5,
        label="ORB Midpoint",
    )

  ax.set_title(
      f"BTC/USDT {selected_tf} — Professional ORB Engine ({tz_short})",
      color="white",
      fontsize=14,
      weight="bold",
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
  st.error("Failed to load BTC chart data. Please check internet connection.")
