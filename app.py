from datetime import datetime, time, timedelta, timezone
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWT BTC Fixed Demand Zone Strategy by TUCKYAA", page_icon="⚡", layout="wide"
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

st.title("⚡ TRADE WITH TUCKYAA (BTC PRO) — Fixed Demand Zone Engine")

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
st.sidebar.markdown("#### Fixed Zone Rules")
st.sidebar.text("01:00 ➔ 11:00 : Supply Zone (Fixed Upper Box) ➔ SELL ONLY")
st.sidebar.text("13:00 ➔ 22:00 : Demand Zone (Fixed Lower Box) ➔ BUY ONLY")
st.sidebar.text("Pattern      : Box Test ➔ Red Drop ➔ Green Overlap ➔ Red Pullback ➔ Green Entry")


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
    except Exception:
      continue
  return None


class TWTFixedDemandZoneEngine:

  def __init__(self, df, symbol, user_tz):
    self.df = df
    self.symbol = symbol
    self.user_tz = user_tz
    self.candles = []
    self.signals = []
    self.zone_box = None

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

    found_signal = None

    # Find a FIXED Demand Zone (First prominent swing low in the session chart history)
    # We lock this zone so it never shifts around on newer candles
    session_lows = [c["low"] for c in self.candles[:120]] if len(self.candles) >= 120 else [c["low"] for c in self.candles]
    if not session_lows:
      return
    
    fixed_zone_low = min(session_lows)
    fixed_zone_high = fixed_zone_low + (self.candles[0]["atr"] * 1.8)
    self.zone_box = (fixed_zone_low, fixed_zone_high, 0, len(self.candles))

    # Scan for pattern after the fixed zone is established
    for i in range(15, len(self.candles) - 2):
      c = self.candles[i]
      t_ist = c["dt_ist"].time()

      is_morning_session = time(1, 0) <= t_ist <= time(11, 0)
      is_evening_session = time(13, 0) <= t_ist <= time(22, 0)

      if not is_evening_session:  # Focusing on Evening Session BUY Setup from Fixed Demand Zone
        continue

      atr = c["atr"]
      dec = get_decimals(c["close"])

      # Pattern Check:
      # 1. Price touched or crossed near the fixed demand zone
      # 2. Followed by 2-3 red candles moving down/crossing
      # 3. Followed by a Green candle overlapping the red candles
      # 4. Followed by 1 Red pullback candle
      # 5. Followed by a Medium Green entry candle
      
      if i >= 4:
        c_curr = self.candles[i]     # Potential Entry Green Candle
        c_prev1 = self.candles[i-1]  # Red Pullback Candle
        c_prev2 = self.candles[i-2]  # Strong Green Overlap Candle
        c_prev3 = self.candles[i-3]  # Red Drop Candle 2
        c_prev4 = self.candles[i-4]  # Red Drop Candle 1

        # Check if recent price action interacted with the fixed demand zone box
        near_zone = any(
            self.candles[idx]["low"] <= fixed_zone_high 
            for idx in range(max(0, i-10), i)
        )

        if near_zone:
          # Sequence check: Red drop -> Green overlap -> Red pullback -> Green entry
          is_red_drop = (c_prev4["color"] == "Red" and c_prev3["color"] == "Red")
          is_green_overlap = (c_prev2["color"] == "Green" and c_prev2["close"] > c_prev3["open"])
          is_red_pullback = (c_prev1["color"] == "Red")
          is_green_entry = (c_curr["color"] == "Green" and c_curr["close"] > c_prev1["open"])

          if is_red_drop and is_green_overlap and is_red_pullback and is_green_entry:
            sl = round(fixed_zone_low - (1.0 * atr), dec)
            tp1 = round(c_curr["close"] + (1.2 * atr), dec)
            tp2 = round(c_curr["close"] + (2.5 * atr), dec)
            tp3 = round(c_curr["close"] + (4.0 * atr), dec)

            found_signal = {
                "Asset": self.symbol,
                "Time": c_curr["time"],
                "Candle #": c_curr["id"],
                "Type": "BUY 🟢",
                "Setup": "Fixed Demand Zone + Pullback Bounce",
                "Action": "Buy / Long (Fixed Demand Zone) 🟢",
                "Entry Price": round(c_curr["close"], dec),
                "Stop Loss (SL)": sl,
                "TP 1": tp1,
                "TP 2": tp2,
                "TP 3": tp3,
                "Is_Within_1hr": c_curr["is_within_1hr"],
            }
            break

    if found_signal:
      self.signals = [found_signal]
    else:
      self.signals = []


# ---------------------------------------------------------
# EXECUTION & DASHBOARD FOR BTC/USDT ONLY
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 BTC/USDT Fixed Demand Zone Dashboard")

df_btc = fetch_btc_data("BTC/USDT", selected_tf)
bot = TWTFixedDemandZoneEngine(df_btc, "BTC/USDT", active_tz)
bot.calculate_and_scan()

master_signals = [s for s in bot.signals if s.get("Is_Within_1hr", False)]

if master_signals:
  st.success("✅ Active Fixed Demand Zone BTC Setup Found!")
  df_display = pd.DataFrame(master_signals).drop(
      columns=["Is_Within_1hr"], errors="ignore"
  )
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No active signal in the last 1 hour. Waiting for exact fixed zone pullback pattern...")

# --- MANUAL INSPECTOR & CHART VIEWER ---
st.markdown("---")
st.subheader("📊 BTC/USDT Chart & Fixed Zone Inspector")

if bot.df is not None and not bot.df.empty:
  curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label="Current BTC Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Session Window", value="Active Trading Session")
  with col3:
    st.metric(label="Timezone", value=tz_short, delta="Fixed Zone Engine Active")

  if bot.signals:
    latest_sig = bot.signals[0]
    is_buy = "BUY" in latest_sig["Type"]
    entry_p = latest_sig["Entry Price"]
    signal_time = latest_sig["Time"]

    st.warning("⚠️ **Don't use after 40 minutes!**")

    if is_buy:
      st.success(
          f"🟢 **TWT BTC BUY SETUP: {latest_sig['Action']}**\n\n"
          f"🕒 **Time ({tz_short}):** `{signal_time}` (Candle"
          f" #{latest_sig['Candle #']})\n"
          f"- **Entry Price:** `{entry_p}`\n"
          f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
          f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:**"
          f" `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`"
      )
    else:
      st.error(
          f"🔴 **TWT BTC SELL SETUP: {latest_sig['Action']}**\n\n"
          f"🕒 **Time ({tz_short}):** `{signal_time}` (Candle"
          f" #{latest_sig['Candle #']})\n"
          f"- **Entry Price:** `{entry_p}`\n"
          f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
          f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:**"
          f" `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`"
      )
  else:
    st.warning(
        "🟡 **STATUS: MONITORING FIXED DEMAND ZONE**\n\n"
        "📌 *Instructions:* Waiting for the red drop, green overlap, red pullback and green entry pattern at the fixed demand box."
    )

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

  # Plot Fixed Zone Box
  if bot.zone_box:
    z_low, z_high, _, _ = bot.zone_box
    ax.axhspan(z_low, z_high, color="#26a69a", alpha=0.3, label="Fixed Demand Zone")

  lows = [c["low"] for c in bot.candles]
  highs = [c["high"] for c in bot.candles]
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
      f"BTC/USDT {selected_tf} — Fixed Demand Engine ({tz_short})",
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
  st.error("BTC market data load failed. Please check internet connection.")
