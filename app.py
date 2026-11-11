from datetime import datetime, time, timedelta, timezone
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWT BTC Multi-Candle Retracement Strategy by TUCKYAA", page_icon="⚡", layout="wide"
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

st.title("⚡ TRADE WITH TUCKYAA (BTC PRO) — Retracement Zone Entry Engine")

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
st.sidebar.markdown("#### Retracement Rules")
st.sidebar.text("01:00 ➔ 11:00 : 3 Red Zone ➔ Green Touch ➔ Red ➔ 2-3 Green ➔ SELL ONLY")
st.sidebar.text("13:00 ➔ 22:00 : 3 Green Zone ➔ Red Touch ➔ Green ➔ 2-3 Red ➔ BUY ONLY")
st.sidebar.text("Risk Mgmt    : ATR Managed SL & Multi-TP")


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
        ohlcv_tf = exchange.fetch_ohlcv(sym, timeframe=tf, limit=160)
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


class TWT3CandleZoneEngine:

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

    # Scan from the most recent candles backwards or forwards properly
    for i in range(len(self.candles) - 1, 7, -1):
      c = self.candles[i]
      t_ist = c["dt_ist"].time()

      is_morning_session = time(1, 0) <= t_ist <= time(11, 0)
      is_evening_session = time(13, 0) <= t_ist <= time(22, 0)

      if not is_morning_session and not is_evening_session:
        continue

      atr = c["atr"]
      dec = get_decimals(c["close"])

      # ==========================================================
      # 1. MORNING SESSION (01:00 to 11:00) ➔ SELL SETUP ONLY
      # ==========================================================
      if is_morning_session:
        for start in range(max(0, i - 25), i - 5):
          c1 = self.candles[start]
          c2 = self.candles[start + 1]
          c3 = self.candles[start + 2]

          if c1["color"] == "Red" and c2["color"] == "Red" and c3["color"] == "Red":
            zone_high = max(c1["high"], c2["high"], c3["high"])
            zone_low = min(c1["low"], c2["low"], c3["low"])

            for k in range(start + 3, i - 3):
              touch_c = self.candles[k]
              if touch_c["color"] == "Green" and touch_c["high"] >= zone_low and touch_c["close"] <= zone_high:
                if k + 1 < len(self.candles) and self.candles[k + 1]["color"] == "Red":
                  green_count = 0
                  end_idx = k + 1
                  for g in range(k + 2, min(k + 5, len(self.candles))):
                    if self.candles[g]["color"] == "Green":
                      green_count += 1
                      end_idx = g
                    else:
                      break

                  if green_count >= 2:
                    if end_idx + 1 < len(self.candles):
                      entry_c = self.candles[end_idx + 1]
                      if entry_c["color"] == "Red":
                        sl = round(zone_high + (1.2 * atr), dec)
                        tp1 = round(entry_c["close"] - (1.2 * atr), dec)
                        tp2 = round(entry_c["close"] - (2.5 * atr), dec)
                        tp3 = round(entry_c["close"] - (4.0 * atr), dec)

                        self.zone_box = (zone_low, zone_high, start, start + 2)
                        found_signal = {
                            "Asset": self.symbol,
                            "Time": entry_c["time"],
                            "Candle #": entry_c["id"],
                            "Type": "SELL 🔴",
                            "Setup": "3-Red Zone + Retracement + Red Entry",
                            "Action": "Sell / Short (S) [Retracement Entry] 🔴",
                            "Entry Price": round(entry_c["close"], dec),
                            "Stop Loss (SL)": sl,
                            "TP 1": tp1,
                            "TP 2": tp2,
                            "TP 3": tp3,
                            "Is_Within_1hr": entry_c["is_within_1hr"],
                        }
                        break
            if found_signal:
              break

      # ==========================================================
      # 2. EVENING SESSION (13:00 to 22:00) ➔ BUY SETUP ONLY
      # ==========================================================
      if is_evening_session:
        for start in range(max(0, i - 25), i - 5):
          c1 = self.candles[start]
          c2 = self.candles[start + 1]
          c3 = self.candles[start + 2]

          if c1["color"] == "Green" and c2["color"] == "Green" and c3["color"] == "Green":
            zone_high = max(c1["high"], c2["high"], c3["high"])
            zone_low = min(c1["low"], c2["low"], c3["low"])

            for k in range(start + 3, i - 3):
              touch_c = self.candles[k]
              if touch_c["color"] == "Red" and touch_c["low"] <= zone_high and touch_c["close"] >= zone_low:
                if k + 1 < len(self.candles) and self.candles[k + 1]["color"] == "Green":
                  red_count = 0
                  end_idx = k + 1
                  for r_idx in range(k + 2, min(k + 5, len(self.candles))):
                    if self.candles[r_idx]["color"] == "Red":
                      red_count += 1
                      end_idx = r_idx
                    else:
                      break

                  if red_count >= 2:
                    if end_idx + 1 < len(self.candles):
                      entry_c = self.candles[end_idx + 1]
                      if entry_c["color"] == "Green":
                        sl = round(zone_low - (1.2 * atr), dec)
                        tp1 = round(entry_c["close"] + (1.2 * atr), dec)
                        tp2 = round(entry_c["close"] + (2.5 * atr), dec)
                        tp3 = round(entry_c["close"] + (4.0 * atr), dec)

                        self.zone_box = (zone_low, zone_high, start, start + 2)
                        found_signal = {
                            "Asset": self.symbol,
                            "Time": entry_c["time"],
                            "Candle #": entry_c["id"],
                            "Type": "BUY 🟢",
                            "Setup": "3-Green Zone + Retracement + Green Entry",
                            "Action": "Buy / Long (B) [Retracement Entry] 🟢",
                            "Entry Price": round(entry_c["close"], dec),
                            "Stop Loss (SL)": sl,
                            "TP 1": tp1,
                            "TP 2": tp2,
                            "TP 3": tp3,
                            "Is_Within_1hr": entry_c["is_within_1hr"],
                        }
                        break
            if found_signal:
              break

      if found_signal:
        break

    if found_signal:
      self.signals = [found_signal]
    else:
      self.signals = []


# ---------------------------------------------------------
# EXECUTION & DASHBOARD FOR BTC/USDT ONLY
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 BTC/USDT Live Strategy Dashboard")

df_btc = fetch_btc_data("BTC/USDT", selected_tf)
bot = TWT3CandleZoneEngine(df_btc, "BTC/USDT", active_tz)
bot.calculate_and_scan()

master_signals = [s for s in bot.signals if s.get("Is_Within_1hr", False)]

if master_signals:
  st.success("✅ Active Retracement Zone BTC Setup Found!")
  df_display = pd.DataFrame(master_signals).drop(
      columns=["Is_Within_1hr"], errors="ignore"
  )
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No active signal in the last 1 hour. Waiting for exact session time and retracement sequence on BTC...")

# --- MANUAL INSPECTOR & CHART VIEWER ---
st.markdown("---")
st.subheader("📊 BTC/USDT Chart & Zone Inspector")

if bot.df is not None and not bot.df.empty:
  curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label="Current BTC Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Session Window", value="Active Trading Session")
  with col3:
    st.metric(label="Timezone", value=tz_short, delta="Retracement Engine Active")

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
        "🟡 **STATUS: SCANNING BTC FOR EVENING BUY SETUP**\n\n"
        "📌 *Instructions:* Waiting for 3 Green candles zone -> Red touch -> Green -> 2-3 Red candles -> Buy Entry."
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

  # Plot Zone Box if detected
  if bot.zone_box:
    z_low, z_high, start_idx, end_idx = bot.zone_box
    ax.axhspan(z_low, z_high, xmin=start_idx/len(bot.candles), xmax=end_idx/len(bot.candles), color="#ab47bc", alpha=0.3, label="3-Candle Zone")

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
      f"BTC/USDT {selected_tf} — TWT Retracement Engine ({tz_short})",
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
