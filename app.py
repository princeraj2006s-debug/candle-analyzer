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
    page_title="TWT Professional Multi-Asset ORB Retest Bot",
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

st.title("⚡ TRADE WITH TUCKYAA — Multi-Asset ORB Breakout & Retest Engine")

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION (Strategy Parameters & Controls)
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Multi-Asset ORB Configurations")
st.sidebar.markdown("---")

# IANA Timezones with automatic DST handling
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

# Configurable Session Opening Time
session_open_time = st.sidebar.time_input(
    "⏰ Session Opening Time", value=time(9, 30)
)

selected_tf = st.sidebar.selectbox(
    "⏱️ Entry Timeframe", options=["5m", "15m", "30m", "1h"], index=0
)
orb_duration_mins = st.sidebar.slider(
    "⏳ Opening Range Duration (Minutes)", min_value=15, max_value=120, value=75, step=5
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Strategy Filters & Risk Management")
use_ema_filter = st.sidebar.checkbox("Enable EMA 200 Trend Filter", value=True)
min_breakout_body_atr = st.sidebar.slider(
    "📏 Min Breakout Body (% of ATR)", min_value=0.1, max_value=2.0, value=0.5, step=0.1
)
max_retest_candles = st.sidebar.slider(
    "🔍 Max Candles Allowed for Retest", min_value=5, max_value=50, value=15, step=1
)
retest_tolerance_atr = st.sidebar.slider(
    "🎯 Retest Tolerance (% of ATR)", min_value=0.1, max_value=1.0, value=0.3, step=0.1
)
use_atr_buffer = st.sidebar.checkbox("Enable ATR Buffer for Stop Loss", value=True)
risk_reward_ratio = st.sidebar.slider(
    "🎯 Risk / Reward Target Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5
)
fee_slippage_pct = st.sidebar.slider(
    "💸 Fee & Slippage Assumption (%)", min_value=0.0, max_value=0.5, value=0.05, step=0.01
)

# Complete 20+ Asset List
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


# ---------------------------------------------------------
# DATA FETCHING MODULE WITH CCXT (Timezone-Aware UTC)
# ---------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_multi_asset_ohlcv(symbol, timeframe="5m", limit=500):
  exchanges_to_try = [
      (ccxt.binance, symbol),
      (ccxt.bybit, symbol),
      (ccxt.kucoin, symbol),
  ]

  tf_delta_map = {
      "5m": timedelta(minutes=5),
      "15m": timedelta(minutes=15),
      "30m": timedelta(minutes=30),
      "1h": timedelta(hours=1),
  }
  tf_delta = tf_delta_map.get(timeframe, timedelta(minutes=5))
  now_utc = datetime.now(timezone.utc)

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

          # Remove incomplete current live candle if unclosed
          if not df.empty and (df.index[-1] + tf_delta > now_utc):
            df = df.iloc[:-1]

          return df
    except Exception:
      continue
  return None


# ---------------------------------------------------------
# TECHNICAL INDICATORS & PROFESSIONAL ORB RETEST ENGINE
# ---------------------------------------------------------
class ProfessionalORBRetestEngine:

  def __init__(self, df, symbol, orb_duration, user_tz, session_time, timeframe):
    self.df = df
    self.symbol = symbol
    self.orb_duration = orb_duration
    self.user_tz = user_tz
    self.session_time = session_time
    self.timeframe = timeframe
    self.orb_high = 0.0
    self.orb_low = 0.0
    self.orb_mid = 0.0
    self.orb_available = False
    self.signals = []
    self.history = []
    
    self.diag = {
        "session_available": False,
        "orb_complete": False,
        "breakout_found": False,
        "retest_found": False,
        "rejection_confirmed": False,
        "confirmation_completed": False,
        "ema_filter_passed": False,
        "risk_validation_passed": False,
        "final_signal_status": "Pending scan...",
        "rejection_reason": "Initializing scan..."
    }

  def calculate_indicators(self):
    if self.df is None or self.df.empty:
      return
    self.df["EMA200"] = self.df["Close"].ewm(span=200, adjust=False).mean()

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
      self.diag["rejection_reason"] = "DataFrame is empty or unavailable."
      return

    tf_mins_map = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
    tf_mins = tf_mins_map.get(self.timeframe, 5)

    if self.orb_duration % tf_mins != 0 or self.orb_duration < tf_mins:
      self.diag["rejection_reason"] = f"Incompatible ORB duration ({self.orb_duration}m) for timeframe ({tf_mins}m)."
      return

    expected_candles = int(self.orb_duration / tf_mins)

    df_local = self.df.copy()
    df_local["time_tz"] = df_local.index.tz_convert(self.user_tz)
    df_local["date"] = df_local["time_tz"].dt.date

    if df_local.empty:
      return

    latest_date = df_local["date"].iloc[-1]
    session_start_local = datetime.combine(latest_date, self.session_time, tzinfo=self.user_tz)
    session_end_local = session_start_local + timedelta(minutes=self.orb_duration)

    self.diag["session_available"] = True

    mask = (df_local["time_tz"] >= session_start_local) & (df_local["time_tz"] < session_end_local)
    orb_candles = df_local[mask]

    if len(orb_candles) >= expected_candles:
      self.orb_high = float(orb_candles["High"].max())
      self.orb_low = float(orb_candles["Low"].min())
      self.orb_mid = (self.orb_high + self.orb_low) / 2.0
      self.orb_available = True
      self.diag["orb_complete"] = True
      self.diag["rejection_reason"] = "ORB successfully calculated from completed session candles."
    else:
      self.orb_available = False
      self.diag["orb_complete"] = False
      self.diag["rejection_reason"] = f"WAITING FOR SESSION: Found {len(orb_candles)}/{expected_candles} completed session candles."

  def scan_breakout_and_retest(self, use_ema, min_body_atr, max_retest, retest_tol, use_atr, rr_target, fee_slip):
    if self.df is None or len(self.df) < 50:
      return

    self.calculate_indicators()
    self.identify_opening_range()

    if not self.orb_available:
      return

    df_local = self.df.copy()
    df_local["time_tz"] = df_local.index.tz_convert(self.user_tz)
    df_local["date"] = df_local["time_tz"].dt.date
    latest_date = df_local["date"].iloc[-1]

    session_start_local = datetime.combine(latest_date, self.session_time, tzinfo=self.user_tz)
    session_end_local = session_start_local + timedelta(minutes=self.orb_duration)

    valid_signals = []
    current_utc_time = datetime.now(timezone.utc)

    for i in range(len(df_local)):
      c_time = df_local["time_tz"].iloc[i]
      if c_time < session_end_local:
        continue

      c = df_local.iloc[i]
      close = float(c["Close"])
      open_p = float(c["Open"])
      high = float(c["High"])
      low = float(c["Low"])
      body = abs(close - open_p)
      
      ema200 = float(c["EMA200"])
      atr = float(c["ATR"]) if not np.isnan(c["ATR"]) else (close * 0.002)

      is_recent = (current_utc_time - c.name).total_seconds() / 60.0 <= 60.0
      min_body_req = atr * min_body_atr

      # -------------------------------------------------
      # BULLISH BREAKOUT, RETEST & REJECTION CONFIRMATION
      # -------------------------------------------------
      if close > self.orb_high and body >= min_body_req:
        self.diag["breakout_found"] = True

        for j in range(i + 1, min(i + max_retest + 1, len(df_local))):
          retest_c = df_local.iloc[j]
          r_low = float(retest_c["Low"])
          r_high = float(retest_c["High"])
          r_close = float(retest_c["Close"])
          r_open = float(retest_c["Open"])

          tolerance = atr * retest_tol
          is_retest = r_low <= (self.orb_high + tolerance) and r_low >= (self.orb_high - tolerance * 2)

          if is_retest:
            self.diag["retest_found"] = True

            is_rejection_candle = r_close > r_open and (r_open - r_low) > (r_close - r_open)
            if is_rejection_candle:
              self.diag["rejection_confirmed"] = True

              if j + 1 < len(df_local):
                conf_c = df_local.iloc[j + 1]
                if float(conf_c["Close"]) > r_high:
                  self.diag["confirmation_completed"] = True

                  if use_ema:
                    if np.isnan(conf_c["EMA200"]):
                      self.diag["ema_filter_passed"] = False
                      self.diag["final_signal_status"] = "Rejected"
                      self.diag["rejection_reason"] = "EMA 200 unavailable for confirmation candle."
                      continue
                    if float(conf_c["Close"]) <= float(conf_c["EMA200"]):
                      self.diag["ema_filter_passed"] = False
                      self.diag["final_signal_status"] = "Rejected"
                      self.diag["rejection_reason"] = "Signal rejected: Failed EMA 200 bullish filter."
                      continue
                  self.diag["ema_filter_passed"] = True

                  swing_low = min(r_low, float(conf_c["Low"]))
                  sl = round(swing_low - (atr * 0.5), 2) if use_atr else round(swing_low, 2)
                  entry_price = float(conf_c["Close"]) * (1.0 + (fee_slip / 100.0))

                  if sl >= entry_price:
                    self.diag["risk_validation_passed"] = False
                    self.diag["final_signal_status"] = "Rejected"
                    self.diag["rejection_reason"] = "Signal rejected: Invalid BUY Stop Loss calculation (SL >= Entry)."
                    continue
                  self.diag["risk_validation_passed"] = True

                  risk = entry_price - sl
                  target = round(entry_price + (risk * rr_target) * (1.0 - (fee_slip / 100.0)), 2)

                  self.diag["final_signal_status"] = "Accepted"
                  self.diag["rejection_reason"] = "Signal successfully validated and accepted."

                  valid_signals.append({
                      "Asset": self.symbol,
                      "Type": "BUY 🟢",
                      "Setup": "ORB Breakout + Retest + Rejection Confirmation",
                      "Entry Price": round(entry_price, 2),
                      "Stop Loss": sl,
                      "Target": target,
                      "R:R Ratio": f"1:{rr_target}",
                      "Time": conf_c["time_tz"].strftime("%H:%M"),
                      "Reason": "Valid bullish setup confirmed.",
                      "Is_Recent": is_recent,
                      "Direction": "BUY"
                  })
                  break

      # -------------------------------------------------
      # BEARISH BREAKOUT, RETEST & REJECTION CONFIRMATION
      # -------------------------------------------------
      elif close < self.orb_low and body >= min_body_req:
        self.diag["breakout_found"] = True

        for j in range(i + 1, min(i + max_retest + 1, len(df_local))):
          retest_c = df_local.iloc[j]
          r_low = float(retest_c["Low"])
          r_high = float(retest_c["High"])
          r_close = float(retest_c["Close"])
          r_open = float(retest_c["Open"])

          tolerance = atr * retest_tol
          is_retest = r_high >= (self.orb_low - tolerance) and r_high <= (self.orb_low + tolerance * 2)

          if is_retest:
            self.diag["retest_found"] = True

            is_rejection_candle = r_close < r_open and (r_high - r_open) > (r_open - r_close)
            if is_rejection_candle:
              self.diag["rejection_confirmed"] = True

              if j + 1 < len(df_local):
                conf_c = df_local.iloc[j + 1]
                if float(conf_c["Close"]) < r_low:
                  self.diag["confirmation_completed"] = True

                  if use_ema:
                    if np.isnan(conf_c["EMA200"]):
                      self.diag["ema_filter_passed"] = False
                      self.diag["final_signal_status"] = "Rejected"
                      self.diag["rejection_reason"] = "EMA 200 unavailable for confirmation candle."
                      continue
                    if float(conf_c["Close"]) >= float(conf_c["EMA200"]):
                      self.diag["ema_filter_passed"] = False
                      self.diag["final_signal_status"] = "Rejected"
                      self.diag["rejection_reason"] = "Signal rejected: Failed EMA 200 bearish filter."
                      continue
                  self.diag["ema_filter_passed"] = True

                  swing_high = max(r_high, float(conf_c["High"]))
                  sl = round(swing_high + (atr * 0.5), 2) if use_atr else round(swing_high, 2)
                  entry_price = float(conf_c["Close"]) * (1.0 - (fee_slip / 100.0))

                  if sl <= entry_price:
                    self.diag["risk_validation_passed"] = False
                    self.diag["final_signal_status"] = "Rejected"
                    self.diag["rejection_reason"] = "Signal rejected: Invalid SELL Stop Loss calculation (SL <= Entry)."
                    continue
                  self.diag["risk_validation_passed"] = True

                  risk = sl - entry_price
                  target = round(entry_price - (risk * rr_target) * (1.0 - (fee_slip / 100.0)), 2)

                  self.diag["final_signal_status"] = "Accepted"
                  self.diag["rejection_reason"] = "Signal successfully validated and accepted."

                  valid_signals.append({
                      "Asset": self.symbol,
                      "Type": "SELL 🔴",
                      "Setup": "ORB Breakout + Retest + Rejection Confirmation",
                      "Entry Price": round(entry_price, 2),
                      "Stop Loss": sl,
                      "Target": target,
                      "R:R Ratio": f"1:{rr_target}",
                      "Time": conf_c["time_tz"].strftime("%H:%M"),
                      "Reason": "Valid bearish setup confirmed.",
                      "Is_Recent": is_recent,
                      "Direction": "SELL"
                  })
                  break

    if valid_signals:
      unique_signals = []
      seen_directions = set()
      for sig in reversed(valid_signals):
        if sig["Direction"] not in seen_directions:
          seen_directions.add(sig["Direction"])
          unique_signals.append(sig)

      self.signals = unique_signals
      self.history = valid_signals
    else:
      self.signals = []


# ---------------------------------------------------------
# MASTER SCANNER (Global Feed for Top Dashboard - 20+ Assets)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (Last 1 Hour — 20+ Assets)")

master_signals = []
asset_engines = {}


def scan_single_asset(sym):
  df = fetch_multi_asset_ohlcv(sym, selected_tf, limit=500)
  if df is not None and not df.empty:
    engine = ProfessionalORBRetestEngine(df, sym, orb_duration_mins, active_tz, session_open_time, selected_tf)
    engine.scan_breakout_and_retest(
        use_ema_filter, min_breakout_body_atr, max_retest_candles, retest_tolerance_atr, use_atr_buffer, risk_reward_ratio, fee_slippage_pct
    )
    recent_sigs = [s for s in engine.signals if s.get("Is_Recent", False)]
    latest = recent_sigs[0] if recent_sigs else None
    return sym, engine, latest
  return sym, None, None


with st.spinner("⚡ Scanning all 20+ assets with Professional ORB Retest Engine..."):
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
  st.success(f"✅ Found {len(master_signals)} active setup(s) across 20+ assets!")
  df_display = pd.DataFrame(master_signals).drop(
      columns=["Is_Recent", "Direction"], errors="ignore"
  )
  st.dataframe(df_display, width="stretch")
else:
  st.info(
      "ℹ️ No active signals in the last 1 hour across 20+ assets. Waiting for session and breakout confirmation..."
  )

# ---------------------------------------------------------
# MANUAL ASSET SELECTOR & CHART VIEWER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Manual Asset Inspector & Chart")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Clean Chart & Protected Entry",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df = fetch_multi_asset_ohlcv(selected_symbol, selected_tf, limit=500)
  bot = ProfessionalORBRetestEngine(df, selected_symbol, orb_duration_mins, active_tz, session_open_time, selected_tf)
  bot.scan_breakout_and_retest(
      use_ema_filter, min_breakout_body_atr, max_retest_candles, retest_tolerance_atr, use_atr_buffer, risk_reward_ratio, fee_slippage_pct
  )

# Top Metrics Dashboard Display for Selected Asset
col_d1, col_d2, col_d3, col_d4 = st.columns(4)
with col_d1:
  if bot.orb_available:
    st.metric(label="🟢 Opening Range High (ORB)", value=f"${bot.orb_high:,.2f}")
  else:
    st.metric(label="🟢 Opening Range High (ORB)", value="WAITING")
with col_d2:
  if bot.orb_available:
    st.metric(label="🔴 Opening Range Low (ORB)", value=f"${bot.orb_low:,.2f}")
  else:
    st.metric(label="🔴 Opening Range Low (ORB)", value="WAITING")
with col_d3:
  if bot.orb_available:
    st.metric(label="⚪ Opening Range Mid", value=f"${bot.orb_mid:,.2f}")
  else:
    st.metric(label="⚪ Opening Range Mid", value="WAITING")
with col_d4:
  current_price = (
      float(bot.df["Close"].iloc[-1])
      if bot.df is not None and not bot.df.empty
      else 0.0
  )
  st.metric(label=f"⚡ Current {selected_symbol} Price", value=f"${current_price:,.2f}")

st.markdown("---")

recent_signals = [s for s in bot.signals if s.get("Is_Recent", False)]

if not bot.orb_available:
  st.warning(f"⚠️ **WAITING FOR SESSION:** Opening Range for {selected_symbol} is unavailable or incomplete for the current session window.")
elif recent_signals:
  sig = recent_signals[0]
  st.success(f"✅ **ACTIVE RETEST SETUP FOR {selected_symbol}: {sig['Type']}**")

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
      f"ℹ️ **Status for {selected_symbol}:** Scanning market. Waiting for ORB breakout followed by a clean level retest and rejection confirmation."
  )

# Single Entry History Table for Selected Asset
if bot.history:
  st.markdown("---")
  st.subheader(f"📋 Confirmed Entry History Feed for {selected_symbol}")
  df_history = pd.DataFrame(bot.history).drop(
      columns=["Is_Recent", "Direction"], errors="ignore"
  )
  st.dataframe(df_history, width="stretch")

# ---------------------------------------------------------
# DIAGNOSTIC DASHBOARD (Debugging & Execution Tracking)
# ---------------------------------------------------------
with st.expander(f"🛠️ Diagnostic & Execution Status Dashboard ({selected_symbol})", expanded=False):
  diag_col1, diag_col2, diag_col3 = st.columns(3)
  with diag_col1:
    st.text(f"Session Available: {'✅' if bot.diag['session_available'] else '❌'}")
    st.text(f"ORB Complete: {'✅' if bot.diag['orb_complete'] else '❌'}")
    st.text(f"Breakout Found: {'✅' if bot.diag['breakout_found'] else '❌'}")
  with diag_col2:
    st.text(f"Retest Found: {'✅' if bot.diag['retest_found'] else '❌'}")
    st.text(f"Rejection Confirmed: {'✅' if bot.diag['rejection_confirmed'] else '❌'}")
    st.text(f"Confirmation Completed: {'✅' if bot.diag['confirmation_completed'] else '❌'}")
  with diag_col3:
    st.text(f"EMA Filter Passed: {'✅' if bot.diag['ema_filter_passed'] else '❌'}")
    st.text(f"Risk Validation Passed: {'✅' if bot.diag['risk_validation_passed'] else '❌'}")
    st.text(f"Final Status: {bot.diag['final_signal_status']}")
  st.markdown(f"**System Note / Rejection Reason:** `{bot.diag['rejection_reason']}`")

# ---------------------------------------------------------
# MATPLOTLIB PROFESSIONAL CHART RENDERER
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"📊 Professional ORB Chart & Retest Visualizer ({selected_symbol})")

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

  # Plot ORB High & Low Lines if available
  if bot.orb_available:
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
      f"{selected_symbol} {selected_tf} — Professional Multi-Asset ORB Engine ({selected_tz_name})",
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
  st.error("Failed to load market chart data. Please check internet connection.")
