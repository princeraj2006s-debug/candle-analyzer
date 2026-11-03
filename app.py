import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWP ORB Pro by TUCKYAA", page_icon="⚡", layout="wide"
)

# Live auto-refresh every 10 seconds
count = st_autorefresh(interval=10000, key="twp_orb_refresh")

st.title("⚡ TWP ORB Bot by TUCKYAA: Opening Range Breakout Strategy")
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | Asset: BTC/USDT (CCXT)"
    " | Strategy: Opening Range Breakout (TWP Style)**"
)

selected_symbol = "BTC/USDT"


@st.cache_data(ttl=10)
def fetch_twp_data():
  try:
    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 15000})
    ohlcv = exchange.fetch_ohlcv(selected_symbol, timeframe="5m", limit=50)
    if ohlcv and len(ohlcv) > 0:
      df = pd.DataFrame(
          ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
      )
      df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
      df.set_index("timestamp", inplace=True)
      return df
  except Exception:
    pass
  return None


df_5m = fetch_twp_data()


class TWPORBEngine:

  def __init__(self, df):
    self.df = df if df is not None else None
    self.candles = []
    self.signals = []
    self.or_high = 0
    self.or_low = 0

  def calculate_orb_levels(self):
    if self.df is not None and len(self.df) >= 10:
      # Take first 3 candles of the fetched session as the Opening Range (Monitoring Range)
      monitoring_df = self.df.head(3)
      self.or_high = float(monitoring_df["High"].max())
      self.or_low = float(monitoring_df["Low"].min())

  def parse_and_scan(self):
    if self.df is None or self.df.empty:
      return

    # Calculate EMA 20 for trend filter
    self.df["EMA"] = self.df["Close"].ewm(span=20, adjust=False).mean()
    # Calculate ATR for SL/TP
    self.df["TR"] = np.maximum(
        self.df["High"] - self.df["Low"],
        np.maximum(
            abs(self.df["High"] - self.df["Close"].shift(1)),
            abs(self.df["Low"] - self.df["Close"].shift(1)),
        ),
    )
    self.df["ATR"] = self.df["TR"].rolling(14).mean()

    for idx, row in enumerate(self.df.iterrows()):
      r = row[1]
      o, h, l, c = (
          float(r["Open"]),
          float(r["High"]),
          float(r["Low"]),
          float(r["Close"]),
      )
      ema = float(r["EMA"]) if not pd.isna(r["EMA"]) else c
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else 100.0

      self.candles.append({
          "id": idx + 1,
          "color": "Green" if c >= o else "Red",
          "open": o,
          "high": h,
          "low": l,
          "close": c,
          "ema": ema,
          "atr": atr,
      })

    # Scan breakouts after monitoring range (from index 3 onwards)
    for i in range(3, len(self.candles)):
      c = self.candles[i]
      # Bullish Breakout above OR High
      if c["close"] > self.or_high and self.candles[i - 1]["close"] <= self.or_high:
        self.signals.append({
            "Candle #": c["id"],
            "Setup": "ORB Bullish Breakout",
            "Action": "Buy / Long (B)",
            "Price": c["close"],
            "SL": c["close"] - (1.5 * c["atr"]),
            "TP": c["close"] + (2.5 * c["atr"]),
        })
      # Bearish Breakout below OR Low
      elif c["close"] < self.or_low and self.candles[i - 1]["close"] >= self.or_low:
        self.signals.append({
            "Candle #": c["id"],
            "Setup": "ORB Bearish Breakout",
            "Action": "Sell / Short (S)",
            "Price": c["close"],
            "SL": c["close"] + (1.5 * c["atr"]),
            "TP": c["close"] - (2.5 * c["atr"]),
        })


with st.spinner("⚡ Loading TWP ORB Strategy Engine..."):
  if df_5m is not None and not df_5m.empty:
    bot = TWPORBEngine(df_5m)
    bot.calculate_orb_levels()
    bot.parse_and_scan()

    curr_p = bot.candles[-1]["close"] if bot.candles else 78000.0

    st.markdown("---")
    st.subheader("🎯 TWP ORB Trade Setup & Live Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
      st.info(
          f"**Opening Range High:** `{bot.or_high:.2f}`\n\n*Status:* Active"
          " Resistance"
      )
    with c2:
      st.success(
          f"**Opening Range Low:** `{bot.or_low:.2f}`\n\n*Status:* Active"
          " Support"
      )
    with c3:
      st.warning(
          f"**Current Price:** `{curr_p:.2f}`\n\n*Strategy:* Breakout Mode"
      )

    if bot.signals:
      st.markdown("---")
      st.subheader("📋 Detected Opening Range Breakout Signals")
      st.dataframe(pd.DataFrame(bot.signals[-4:]), use_container_width=True)

    st.markdown("---")
    st.subheader("📈 BTC/USDT 5m Chart — TWP Opening Range Breakout Strategy")
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

    # Plot OR High Line
    ax.axhline(
        bot.or_high,
        color="#ff7043",
        linestyle="--",
        linewidth=2,
        label="OR High (Resistance)",
    )
    ax.text(
        2,
        bot.or_high,
        f"  OR High ({bot.or_high:.1f})",
        color="#ff7043",
        weight="bold",
        fontsize=10,
    )

    # Plot OR Low Line
    ax.axhline(
        bot.or_low,
        color="#26a69a",
        linestyle="--",
        linewidth=2,
        label="OR Low (Support)",
    )
    ax.text(
        2,
        bot.or_low,
        f"  OR Low ({bot.or_low:.1f})",
        color="#26a69a",
        weight="bold",
        fontsize=10,
    )

    # Plot Signal Markers
    for sig in bot.signals[-3:]:
      tag = "B" if "Buy" in sig["Action"] else "S"
      tag_color = "#00e676" if tag == "B" else "#ff1744"
      ax.text(
          sig["Candle #"],
          sig["Price"],
          f" {tag} ",
          color="white",
          weight="bold",
          bbox=dict(boxstyle="round,pad=0.2", facecolor=tag_color, alpha=0.9),
      )

    ax.set_title(
        "BTC/USDT 5m — TWP Opening Range Breakout", color="white", fontsize=14
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
    st.error("CCXT data load nahi ho paya. Please check connection.")
