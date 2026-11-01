import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

st.set_page_config(
    page_title="CANDLESENSE Pro by TUCKYAA", page_icon="⚡", layout="wide"
)

# Live auto-refresh every 10 seconds
count = st_autorefresh(interval=10000, key="tuckyaa_btc_refresh")

st.title(
    "⚡ CANDLESENSE Pro by TUCKYAA: BTC Real-Time HTF 6H/4H & SMC Bot"
)
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | Asset: BTC-USD | Exact 6H"
    " (2:00 AM Low) & 4H (00:00 AM High) Levels**"
)

selected_ticker = "BTC-USD"
tf_choice = st.selectbox(
    "Execution Timeframe", options=["5m", "1m", "15m"], index=0
)
interval_map = {"1m": "1m", "5m": "5m", "15m": "15m"}
selected_interval = interval_map[tf_choice]


def fetch_btc_data(interval, period="3d"):
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


df_5m = fetch_btc_data(selected_interval, period="3d")
df_6h = fetch_btc_data("1h", period="5d")  # Used for precise 6h/4h time matching


class TuckyasExactHTFEngine:

  def __init__(self, df_lower, df_higher):
    self.df = df_lower.tail(45) if df_lower is not None else None
    self.df_htf = df_higher if df_higher is not None else None
    self.candles = []
    self.smc_signals = []
    self.htf_levels = {}

  def calculate_exact_htf_levels(self):
    if self.df_htf is not None and not self.df_htf.empty:
      # Filter for 2:00 AM candle for 6H support level (lower portion/low)
      df_2am = self.df_htf[self.df_htf.index.hour == 2]
      if not df_2am.empty:
        self.htf_levels["6H_Support"] = float(df_2am["Low"].iloc[-1])
      else:
        self.htf_levels["6H_Support"] = float(self.df_htf["Low"].min())

      # Filter for 00:00 AM (midnight) candle for 4H resistance level (upper portion/high)
      df_00am = self.df_htf[self.df_htf.index.hour == 0]
      if not df_00am.empty:
        self.htf_levels["4H_Resistance"] = float(df_00am["High"].iloc[-1])
      else:
        self.htf_levels["4H_Resistance"] = float(self.df_htf["High"].max())
    else:
      # Fallback if htf data is missing
      self.htf_levels["6H_Support"] = 77764.5
      self.htf_levels["4H_Resistance"] = 78009.5

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

  def scan_smc_entries(self):
    if len(self.candles) < 5:
      return
    bodies = [c["body"] for c in self.candles]
    avg_b = np.mean(bodies) if bodies else 100

    for i in range(4, len(self.candles)):
      c_curr = self.candles[i]
      c_prev = self.candles[i - 1]

      # Simple SMC Bearish/Bullish Rejection based on HTF levels
      if (
          c_curr["high"] >= self.htf_levels.get("4H_Resistance", 78009.5)
          and c_curr["color"] == "Red"
      ):
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "4H Resistance Rejection",
            "Type": "Bearish S",
            "Price": c_curr["close"],
        })
      elif (
          c_curr["low"] <= self.htf_levels.get("6H_Support", 77764.5)
          and c_curr["color"] == "Green"
      ):
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "6H Support Rebound",
            "Type": "Bullish B",
            "Price": c_curr["close"],
        })


with st.spinner("⚡ Syncing exact 6H & 4H HTF levels for TUCKYAA..."):
  if df_5m is not None and not df_5m.empty:
    bot = TuckyasExactHTFEngine(df_5m, df_6h)
    bot.calculate_exact_htf_levels()
    bot.parse_candles()
    bot.scan_smc_entries()

    curr_p = bot.candles[-1]["close"] if bot.candles else 77912.0

    st.markdown("---")
    st.subheader("🎯 TUCKYAA Smart Money Trade Entries (BTC-USD)")
    c1, c2, c3 = st.columns(3)
    with c1:
      st.info(
          f"**1️⃣ Scalp Entry:** `{curr_p:.2f}`\n\n*SL:* `{curr_p-120:.2f}` |"
          f" *TP:* `{curr_p+200:.2f}`"
      )
    with c2:
      st.success(
          f"**2️⃣ Standard HTF Entry (BEST):** `{curr_p-60:.2f}`\n\n*SL:*"
          f" `{curr_p-250:.2f}` | *TP:* `{curr_p+400:.2f}`"
      )
    with c3:
      st.warning(
          f"**3️⃣ Breakout Entry:** `{curr_p+100:.2f}`\n\n*SL:* `{curr_p-200:.2f}`"
          f" | *TP:* `{curr_p+500:.2f}`"
      )

    if bot.smc_signals:
      st.markdown("---")
      st.subheader("📋 Detected HTF Rejection Signals")
      st.dataframe(pd.DataFrame(bot.smc_signals), use_container_width=True)

    # TRADINGVIEW / DELTA EXCHANGE STYLE DARK CHART PLOT
    st.markdown("---")
    st.subheader(
        "📈 Real-Time Chart for TUCKYAA (Exact 6H & 4H HTF Level Lines)"
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

    # Plot Exact 6H Support Line (2:00 AM Candle Low)
    s_lvl = bot.htf_levels["6H_Support"]
    ax.axhline(
        s_lvl,
        color="#26a69a",
        linestyle="-",
        linewidth=2,
        label="6h Support Line",
    )
    ax.text(
        2,
        s_lvl,
        f"  6h Support ({s_lvl:.1f})",
        color="#26a69a",
        weight="bold",
        fontsize=10,
    )

    # Plot Exact 4H Resistance Line (00:00 AM Candle High)
    r_lvl = bot.htf_levels["4H_Resistance"]
    ax.axhline(
        r_lvl,
        color="#ff7043",
        linestyle="-",
        linewidth=2,
        label="4h Resistance Line",
    )
    ax.text(
        2,
        r_lvl,
        f"  4h Resistance ({r_lvl:.1f})",
        color="#ff7043",
        weight="bold",
        fontsize=10,
    )

    # Plot SMC Signal Tags
    for sig in bot.smc_signals[-2:]:
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

    ax.set_title(
        f"BTC-USD Live Chart — TUCKYAA HTF Strategy", color="white", fontsize=14
    )
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
