import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

st.set_page_config(
    page_title="CANDLESENSE Pro by TUCKYAA", page_icon="⚡", layout="wide"
)

# Live auto-refresh every 10 seconds (TradingView Live Feed Style)
count = st_autorefresh(interval=10000, key="tuckyaa_5m_execution_refresh")

st.title(
    "⚡ CANDLESENSE Pro by TUCKYAA: BTC 5-Minute Execution Bot & HTF Zones"
)
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | Asset: BTC-USD | Chart"
    " Timeframe: 5m | 6H & 4H HTF Levels Mapped**"
)

selected_ticker = "BTC-USD"


@st.cache_data(ttl=10)
def fetch_execution_data():
  df_5m = yf.download(
      selected_ticker, period="2d", interval="5m", progress=False
  )
  df_htf = yf.download(
      selected_ticker, period="5d", interval="60m", progress=False
  )
  if df_5m is not None and not df_5m.empty:
    if isinstance(df_5m.columns, pd.MultiIndex):
      df_5m.columns = df_5m.columns.get_level_values(0)
  if df_htf is not None and not df_htf.empty:
    if isinstance(df_htf.columns, pd.MultiIndex):
      df_htf.columns = df_htf.columns.get_level_values(0)
  return df_5m, df_htf


df_5m, df_htf = fetch_execution_data()


class Tuckyas5MinExecutionBot:

  def __init__(self, df_5m, df_htf):
    self.df = df_5m.tail(50) if df_5m is not None else None
    self.df_htf = df_htf if df_htf is not None else None
    self.candles = []
    self.htf_levels = {}
    self.signals = []

  def calculate_htf_zones(self):
    if self.df_htf is not None and not self.df_htf.empty:
      # 6H support level: 2:00 AM candle low
      df_2am = self.df_htf[self.df_htf.index.hour == 2]
      if not df_2am.empty:
        self.htf_levels["6H_Support"] = float(df_2am["Low"].iloc[-1])
      else:
        self.htf_levels["6H_Support"] = float(self.df_htf["Low"].min())

      # 4H resistance level: 00:00 AM candle high
      df_00am = self.df_htf[self.df_htf.index.hour == 0]
      if not df_00am.empty:
        self.htf_levels["4H_Resistance"] = float(df_00am["High"].iloc[-1])
      else:
        self.htf_levels["4H_Resistance"] = float(self.df_htf["High"].max())
    else:
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

  def scan_5m_entries(self):
    if len(self.candles) < 5:
      return
    sup = self.htf_levels.get("6H_Support", 0)
    res = self.htf_levels.get("4H_Resistance", 0)

    for i in range(len(self.candles)):
      c = self.candles[i]
      # Check reaction near 4H resistance
      if abs(c["high"] - res) <= 40 or (
          c["high"] >= res and c["close"] < res
      ):
        self.signals.append({
            "Candle #": c["id"],
            "Setup": "4H Resistance Rejection",
            "Action": "Sell / Short (S)",
            "Price": c["high"],
        })
      # Check reaction near 6H support
      elif abs(c["low"] - sup) <= 40 or (c["low"] <= sup and c["close"] > sup):
        self.signals.append({
            "Candle #": c["id"],
            "Setup": "6H Support Bounce",
            "Action": "Buy / Long (B)",
            "Price": c["low"],
        })


with st.spinner("⚡ Loading 5m BTC chart with 6H & 4H HTF levels..."):
  if df_5m is not None and not df_5m.empty:
    bot = Tuckyas5MinExecutionBot(df_5m, df_htf)
    bot.calculate_htf_zones()
    bot.parse_candles()
    bot.scan_5m_entries()

    curr_p = bot.candles[-1]["close"] if bot.candles else 77912.0

    st.markdown("---")
    st.subheader("🎯 TUCKYAA 5-Minute Trade Entries (BTC-USD)")
    c1, c2, c3 = st.columns(3)
    with c1:
      st.info(
          f"**1️⃣ Scalp Entry:** `{curr_p:.2f}`\n\n*SL:* `{curr_p-100:.2f}` |"
          f" *TP:* `{curr_p+180:.2f}`"
      )
    with c2:
      st.success(
          f"**2️⃣ HTF Zone Re-Test (BEST):** `{curr_p-50:.2f}`\n\n*SL:*"
          f" `{curr_p-200:.2f}` | *TP:* `{curr_p+350:.2f}`"
      )
    with c3:
      st.warning(
          f"**3️⃣ Breakout Entry:** `{curr_p+80:.2f}`\n\n*SL:* `{curr_p-180:.2f}`"
          f" | *TP:* `{curr_p+400:.2f}`"
      )

    if bot.signals:
      st.markdown("---")
      st.subheader("📋 Detected 5m Zone Reactions")
      st.dataframe(pd.DataFrame(bot.signals[-4:]), use_container_width=True)

    # PLOTTING 5M CHART WITH HTF LINES
    st.markdown("---")
    st.subheader(
        "📈 BTC-USD 5-Minute Execution Chart (HTF 6H & 4H Mapped)"
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

    # Plot 6H Support Line
    sup_lvl = bot.htf_levels["6H_Support"]
    ax.axhline(
        sup_lvl,
        color="#26a69a",
        linestyle="-",
        linewidth=2,
        label="6H Support",
    )
    ax.text(
        2,
        sup_lvl,
        f"  6H Support ({sup_lvl:.1f})",
        color="#26a69a",
        weight="bold",
        fontsize=10,
    )

    # Plot 4H Resistance Line
    res_lvl = bot.htf_levels["4H_Resistance"]
    ax.axhline(
        res_lvl,
        color="#ff7043",
        linestyle="-",
        linewidth=2,
        label="4H Resistance",
    )
    ax.text(
        2,
        res_lvl,
        f"  4H Resistance ({res_lvl:.1f})",
        color="#ff7043",
        weight="bold",
        fontsize=10,
    )

    # Plot Signal Tags
    for sig in bot.signals[-3:]:
      tag = "B" if "Long" in sig["Action"] else "S"
      tag_color = "#00e676" if "Long" in sig["Action"] else "#ff1744"
      ax.text(
          sig["Candle #"],
          sig["Price"],
          f" {tag} ",
          color="white",
          weight="bold",
          bbox=dict(boxstyle="round,pad=0.2", facecolor=tag_color, alpha=0.9),
      )

    ax.set_title(
        "BTC-USD 5m Chart — TUCKYAA HTF Zones Execution",
        color="white",
        fontsize=14,
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
    st.error("Data load nahi ho paya.")
