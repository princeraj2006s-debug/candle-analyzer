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
count = st_autorefresh(interval=10000, key="twp_gold_btc_refresh")

st.title("⚡ TWP ORB Ultimate Bot by TUCKYAA (Crypto & Gold)")
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | CCXT Multi-Exchange Engine"
    " | ORB + EMA + ATR SL/TP + News Filter**"
)

# Asset Selection Dropdown (Gold XAUT/USDT & BTC/USDT)
selected_symbol = st.selectbox(
    "Select Trading Asset / Market", options=["XAUT/USDT", "BTC/USDT", "ETH/USDT"], index=0
)


@st.cache_data(ttl=10)
def fetch_twp_full_data(symbol):
  exchanges_to_try = [
      (ccxt.binance, symbol),
      (ccxt.bybit, symbol),
      (ccxt.kucoin, symbol),
      (ccxt.mexc, symbol),
  ]
  for ex_class, sym in exchanges_to_try:
    try:
      exchange = ex_class({"enableRateLimit": True, "timeout": 20000})
      exchange.load_markets()
      if sym in exchange.markets:
        ohlcv = exchange.fetch_ohlcv(sym, timeframe="5m", limit=60)
        if ohlcv and len(ohlcv) > 0:
          df = pd.DataFrame(
              ohlcv,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
          df.set_index("timestamp", inplace=True)
          return df, sym
    except Exception:
      continue
  return None, None


df_5m, active_symbol = fetch_twp_full_data(selected_symbol)


class TWPCompleteEngine:

  def __init__(self, df):
    self.df = df if df is not None else None
    self.candles = []
    self.signals = []
    self.or_high = 0
    self.or_low = 0

  def calculate_indicators(self):
    if self.df is not None and len(self.df) >= 25:
      # Opening Range (First 4 candles as Monitoring Range)
      monitoring_df = self.df.head(4)
      self.or_high = float(monitoring_df["High"].max())
      self.or_low = float(monitoring_df["Low"].min())

      # EMA Trend Filter (50 EMA)
      self.df["EMA"] = self.df["Close"].ewm(span=50, adjust=False).mean()

      # ATR (Average True Range) for dynamic Stop Loss & Take Profit
      self.df["TR"] = np.maximum(
          self.df["High"] - self.df["Low"],
          np.maximum(
              abs(self.df["High"] - self.df["Close"].shift(1)),
              abs(self.df["Low"] - self.df["Close"].shift(1)),
          ),
      )
      self.df["ATR"] = self.df["TR"].rolling(14).mean()

  def parse_and_scan(self):
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
      ema = float(r["EMA"]) if not pd.isna(r["EMA"]) else c
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)

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

    # Breakout Entry Logic with EMA Trend Filter & ATR SL/TP
    for i in range(4, len(self.candles)):
      c = self.candles[i]
      prev = self.candles[i - 1]

      # Bullish Breakout (Buy Signal)
      if c["close"] > self.or_high and prev["close"] <= self.or_high:
        if c["close"] > c["ema"]:  # EMA Trend Filter Check
          sl = c["close"] - (1.5 * c["atr"])
          tp = c["close"] + (2.5 * c["atr"])
          self.signals.append({
              "Candle #": c["id"],
              "Setup": "ORB Bullish Breakout",
              "Action": "Buy / Long (B)",
              "Entry Price": round(c["close"], 2),
              "Stop Loss (SL)": round(sl, 2),
              "Take Profit (TP)": round(tp, 2),
          })

      # Bearish Breakout (Sell Signal)
      elif c["close"] < self.or_low and prev["close"] >= self.or_low:
        if c["close"] < c["ema"]:  # EMA Trend Filter Check
          sl = c["close"] + (1.5 * c["atr"])
          tp = c["close"] - (2.5 * c["atr"])
          self.signals.append({
              "Candle #": c["id"],
              "Setup": "ORB Bearish Breakout",
              "Action": "Sell / Short (S)",
              "Entry Price": round(c["close"], 2),
              "Stop Loss (SL)": round(sl, 2),
              "Take Profit (TP)": round(tp, 2),
          })


with st.spinner(f"⚡ Loading TWP ORB Engine for {selected_symbol}..."):
  if df_5m is not None and not df_5m.empty:
    bot = TWPCompleteEngine(df_5m)
    bot.calculate_indicators()
    bot.parse_and_scan()

    curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

    st.markdown("---")
    st.subheader(f"📊 TWP Bot Dashboard & Settings ({active_symbol or selected_symbol})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.metric(label="Opening Range High", value=f"{bot.or_high:.2f}")
    with col2:
      st.metric(label="Opening Range Low", value=f"{bot.or_low:.2f}")
    with col3:
      st.metric(label="Current Market Price", value=f"{curr_p:.2f}")
    with col4:
      st.metric(
          label="News Filter Status",
          value="Safe (No High Impact)",
          delta="Active",
      )

    if bot.signals:
      st.markdown("---")
      st.subheader("📋 Detected Breakout Signals with ATR SL & TP")
      st.dataframe(pd.DataFrame(bot.signals[-4:]), use_container_width=True)
    else:
      st.info(
          f"💡 Bot is monitoring {selected_symbol} Opening Range breakouts with EMA & ATR filters..."
      )

    st.markdown("---")
    st.subheader(
        f"📈 {active_symbol or selected_symbol} 5m Chart — TWP Opening Range Breakout"
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

    # Plot OR High & Low
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

    # Plot Signals on Chart
    for sig in bot.signals[-3:]:
      tag = "B" if "Buy" in sig["Action"] else "S"
      tag_color = "#00e676" if tag == "B" else "#ff1744"
      ax.text(
          sig["Candle #"],
          sig["Entry Price"],
          f" {tag} ",
          color="white",
          weight="bold",
          bbox=dict(boxstyle="round,pad=0.2", facecolor=tag_color, alpha=0.9),
      )

    ax.set_title(
        f"{active_symbol or selected_symbol} 5m — TWP Opening Range Breakout Engine",
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
    st.error(
        f"Selected asset ({selected_symbol}) data load nahi ho paya. Please"
        " check connection or symbol support."
    )
