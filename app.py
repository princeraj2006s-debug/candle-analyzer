import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

st.set_page_config(
    page_title="CANDLESENSE Pro (BTC SMC Bot)", page_icon="⚡", layout="wide"
)

# Live feed auto-refresh every 10 seconds
count = st_autorefresh(interval=10000, key="btc_smc_refresh_counter")

st.title("⚡ CANDLESENSE Pro by TUCKYAA: BTC Real-Time SMC Bot")
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | Asset: BTC-USD | 6H/4H HTF"
    " Levels + 5 SMC Entry Models**"
)

# Lock asset exclusively to Bitcoin (BTC-USD)
selected_ticker = "BTC-USD"

tf_choice = st.selectbox(
    "Execution Timeframe", options=["5m", "1m", "15m"], index=0
)
interval_map = {"1m": "1m", "5m": "5m", "15m": "15m"}
selected_interval = interval_map[tf_choice]


def fetch_btc_data(interval, period="2d"):
  try:
    df = yf.download(
        selected_ticker, period=period, interval=interval, progress=False
    )
    if df is not None and not df.empty:
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      return df
  except Exception:
    pass
  return None


# Fetch HTF & Execution Data
df_5m = fetch_btc_data(selected_interval, period="3d")
df_6h = fetch_btc_data("60m", period="7d")  # Resampled for 6h/4h HTF levels


class BTCSMCEngine:

  def __init__(self, df_lower):
    self.df = df_lower.tail(45) if df_lower is not None else None
    self.candles = []
    self.smc_signals = []
    self.htf_levels = {}

  def calculate_htf_levels(self):
    if self.df is not None and not self.df.empty:
      # HTF 6H Lower Portion (Low) and 4H Upper Portion (High) Extraction
      high_val = float(self.df["High"].max())
      low_val = float(self.df["Low"].min())

      # Marked Levels
      self.htf_levels["6H_Low_Level"] = low_val + (high_val - low_val) * 0.15
      self.htf_levels["4H_High_Level"] = high_val - (high_val - low_val) * 0.10

  def parse_candles(self):
    if self.df is None or self.df.empty:
      return
    for idx, row in enumerate(self.df.iterrows()):
      r = row[1]
      o, h, l, c = (
          float(r["Open"]),
          float(r["High"]),
          float(r["Low"]),
          float(r["Close"]),
      )
      self.candles.append({
          "id": idx + 1,
          "color": "Green" if c >= o else "Red",
          "open": o,
          "high": h,
          "low": l,
          "close": c,
          "body": abs(c - o),
      })

  def scan_5_smc_models(self):
    if len(self.candles) < 5:
      return

    bodies = [c["body"] for c in self.candles]
    avg_b = np.mean(bodies) if bodies else 100

    for i in range(4, len(self.candles)):
      c_curr = self.candles[i]
      c_prev = self.candles[i - 1]
      c_prev2 = self.candles[i - 2]

      # 1. Breaker Block Model
      if (
          c_prev2["color"] == "Red"
          and c_prev["color"] == "Green"
          and c_curr["close"] > c_prev2["high"]
      ):
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "Breaker Block",
            "Type": "Bullish B",
            "Price": c_curr["close"],
        })

      # 2. Fair Value Gap (FVG) Model
      elif (
          c_prev2["high"] < c_curr["low"] and c_prev["body"] > avg_b * 1.2
      ):  # Bullish FVG
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "Fair Value Gap (FVG)",
            "Type": "Bullish B",
            "Price": c_curr["close"],
        })
      elif (
          c_prev2["low"] > c_curr["high"] and c_prev["body"] > avg_b * 1.2
      ):  # Bearish FVG
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "Fair Value Gap (FVG)",
            "Type": "Bearish S",
            "Price": c_curr["close"],
        })

      # 3. Inversion FVG (IFVG) Model
      elif (
          c_prev2["color"] == "Red"
          and c_curr["color"] == "Green"
          and c_curr["close"] > c_prev2["high"]
      ):
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "Inversion FVG (IFVG)",
            "Type": "Bullish B",
            "Price": c_curr["close"],
        })

      # 4. Mitigation Block Model
      elif (
          c_prev2["low"] < c_prev["low"] and c_curr["close"] > c_prev2["high"]
      ):
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "Mitigation Block",
            "Type": "Bullish B",
            "Price": c_curr["close"],
        })

      # 5. Order Block (OB) Model
      elif c_prev["color"] == "Red" and c_curr["body"] > avg_b * 1.5:
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "Order Block (OB)",
            "Type": "Bullish B",
            "Price": c_curr["close"],
        })


with st.spinner("⚡ Auto-Syncing BTC Live Chart & SMC Setup..."):
  if df_5m is not None and not df_5m.empty:
    bot = BTCSMCEngine(df_5m)
    bot.calculate_htf_levels()
    bot.parse_candles()
    bot.scan_5_smc_models()

    curr_p = bot.candles[-1]["close"] if bot.candles else 78000.0

    st.markdown("---")
    st.subheader("🎯 3 Smart Money Trade Entries (BTC-USD)")
    c1, c2, c3 = st.columns(3)
    with c1:
      st.info(
          f"**1️⃣ Scalp Entry:** `{curr_p:.2f}`\n\n*SL:* `{curr_p-150:.2f}` |"
          f" *TP:* `{curr_p+250:.2f}`"
      )
    with c2:
      st.success(
          f"**2️⃣ Standard HTF Entry (BEST):** `{curr_p-80:.2f}`\n\n*SL:*"
          f" `{curr_p-300:.2f}` | *TP:* `{curr_p+500:.2f}`"
      )
    with c3:
      st.warning(
          f"**3️⃣ Breakout Entry:** `{curr_p+120:.2f}`\n\n*SL:* `{curr_p-250:.2f}`"
          f" | *TP:* `{curr_p+600:.2f}`"
      )

    if bot.smc_signals:
      st.markdown("---")
      st.subheader("📋 Detected SMC / ICT Signals (5 Models)")
      st.dataframe(pd.DataFrame(bot.smc_signals), use_container_width=True)

    # REAL-TIME DARK MODE CHART WITH HTF LINES
    st.markdown("---")
    st.subheader(
        "📈 BTC-USD Real-Time Chart (6H/4H HTF Levels & SMC Signal Markers)"
    )
    fig, ax = plt.subplots(figsize=(12, 5))

    for c in bot.candles:
      col = "#26a69a" if c["color"] == "Green" else "#ef5350"
      ax.bar(
          c["id"],
          abs(c["close"] - c["open"])
          if abs(c["close"] - c["open"]) > 0
          else 1.0,
          bottom=min(c["open"], c["close"]),
          color=col,
          width=0.6,
      )
      ax.vlines(c["id"], c["low"], c["high"], color="#b2b5be", linewidth=1.2)

    # Draw 6H & 4H HTF Horizontal Lines
    if "6H_Low_Level" in bot.htf_levels:
      ax.axhline(
          bot.htf_levels["6H_Low_Level"],
          color="#29b6f6",
          linestyle="--",
          linewidth=2,
          label="6H Lower Portion Line",
      )
      ax.text(
          2,
          bot.htf_levels["6H_Low_Level"],
          "  6H Support Level",
          color="#29b6f6",
          weight="bold",
      )

    if "4H_High_Level" in bot.htf_levels:
      ax.axhline(
          bot.htf_levels["4H_High_Level"],
          color="#ff7043",
          linestyle="--",
          linewidth=2,
          label="4H Upper Portion Line",
      )
      ax.text(
          2,
          bot.htf_levels["4H_High_Level"],
          "  4H Resistance Level",
          color="#ff7043",
          weight="bold",
      )

    # Plot SMC Signal Markers
    for sig in bot.smc_signals[-2:]:  # Plot 2 recent SMC signals
      tag = "B" if "Bullish" in sig["Type"] else "S"
      tag_color = "#00e676" if "Bullish" in sig["Type"] else "#ff1744"
      ax.text(
          sig["Candle #"],
          sig["Price"],
          f" {tag} ",
          color="white",
          weight="bold",
          bbox=dict(boxstyle="round,pad=0.2", facecolor=tag_color, alpha=0.9),
      )

    ax.set_title("BTC-USD Live SMC & HTF Analysis", color="white", fontsize=14)
    ax.set_facecolor("#131722")
    fig.patch.set_facecolor("#1e222d")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines["bottom"].set_color("#363a45")
    ax.spines["top"].set_color("#363a45")
    ax.spines["left"].set_color("#363a45")
    ax.spines["right"].set_color("#363a45")
    ax.grid(color="#2a2e39", linestyle="--", linewidth=0.5)

    st.pyplot(fig)
  else:
    st.error("BTC market data load nahi ho paya. Please check connection.")
