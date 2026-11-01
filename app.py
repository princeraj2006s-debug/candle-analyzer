import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="CANDLESENSE Pro by TUCKYAA", page_icon="⚡", layout="wide"
)

# Live auto-refresh every 10 seconds (TradingView Live Feed Style)
count = st_autorefresh(interval=10000, key="tuckyaa_ccxt_strict_refresh")

st.title("⚡ CANDLESENSE Pro by TUCKYAA: BTC CCXT Strict Real-Time Bot")
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | CCXT Engine Powered | 5m"
    " Execution | Exact Delta-Matched Levels**"
)


@st.cache_data(ttl=10)
def fetch_strict_ccxt_data():
  # Multi-exchange CCXT fallback to ensure 100% uptime on Streamlit Cloud
  exchanges_to_try = [
      (ccxt.binance, "BTC/USDT"),
      (ccxt.kraken, "BTC/USD"),
      (ccxt.coinbase, "BTC/USD"),
  ]

  for ex_class, symbol in exchanges_to_try:
    try:
      exchange = ex_class({
          "enableRateLimit": True,
          "timeout": 15000,
      })
      ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=50)
      if ohlcv and len(ohlcv) > 0:
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df, symbol
    except Exception:
      continue
  return None, None


df_5m, active_symbol = fetch_strict_ccxt_data()


class TuckyasCCXTBot:

  def __init__(self, df_5m):
    self.df = df_5m.tail(45) if df_5m is not None else None
    self.candles = []
    self.htf_levels = {}
    self.signals = []

  def set_exact_levels(self):
    # Exact Delta Exchange matched levels (4H Resistance: 78009.5, 6H Support: 77764.5)
    self.htf_levels["4H_Resistance"] = 78009.5
    self.htf_levels["6H_Support"] = 77764.5

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

  def scan_signals(self):
    if len(self.candles) < 5:
      return
    sup = self.htf_levels["6H_Support"]
    res = self.htf_levels["4H_Resistance"]
    bodies = [c["body"] for c in self.candles]
    avg_b = np.mean(bodies) if bodies else 50

    for i in range(4, len(self.candles)):
      c = self.candles[i]
      c_prev = self.candles[i - 1]
      c_prev2 = self.candles[i - 2]

      # Zone Reactions & SMC 5 Models Detection
      if c["high"] >= res - 10 and c["close"] < res:
        self.signals.append({
            "Candle #": c["id"],
            "Setup": "4H Resistance Rejection",
            "Action": "Sell / Short (S)",
            "Price": c["high"],
        })
      elif c["low"] <= sup + 10 and c["close"] > sup:
        self.signals.append({
            "Candle #": c["id"],
            "Setup": "6H Support Bounce",
            "Action": "Buy / Long (B)",
            "Price": c["low"],
        })
      elif (
          c_prev2["color"] == "Red"
          and c_prev["color"] == "Green"
          and c["close"] > c_prev2["high"]
      ):
        self.signals.append({
            "Candle #": c["id"],
            "Setup": "Breaker Block / IFVG",
            "Action": "Buy / Long (B)",
            "Price": c["close"],
        })
      elif c_prev2["high"] < c["low"] and c_prev["body"] > avg_b * 1.2:
        self.signals.append({
            "Candle #": c["id"],
            "Setup": "Fair Value Gap (FVG)",
            "Action": "Buy / Long (B)",
            "Price": c["close"],
        })
      elif c_prev["color"] == "Red" and c["body"] > avg_b * 1.5:
        self.signals.append({
            "Candle #": c["id"],
            "Setup": "Order Block (OB)",
            "Action": "Buy / Long (B)",
            "Price": c["close"],
        })


with st.spinner(
    f"⚡ Fetching CCXT Real-Time Clean Data ({active_symbol or 'BTC'})..."
):
  if df_5m is not None and not df_5m.empty:
    bot = TuckyasCCXTBot(df_5m)
    bot.set_exact_levels()
    bot.parse_candles()
    bot.scan_signals()

    curr_p = bot.candles[-1]["close"] if bot.candles else 77929.5

    st.markdown("---")
    st.subheader(f"🎯 TUCKYAA 5-Minute Trade Entries ({active_symbol})")
    c1, c2, c3 = st.columns(3)
    with c1:
      st.info(
          f"**1️⃣ Scalp Entry:** `{curr_p:.2f}`\n\n*SL:* `{curr_p-80:.2f}` |"
          f" *TP:* `{curr_p+150:.2f}`"
      )
    with c2:
      st.success(
          f"**2️⃣ HTF Zone Re-Test (BEST):** `{curr_p-40:.2f}`\n\n*SL:*"
          f" `{curr_p-150:.2f}` | *TP:* `{curr_p+300:.2f}`"
      )
    with c3:
      st.warning(
          f"**3️⃣ Breakout Entry:** `{curr_p+60:.2f}`\n\n*SL:* `{curr_p-120:.2f}`"
          f" | *TP:* `{curr_p+350:.2f}`"
      )

    if bot.signals:
      st.markdown("---")
      st.subheader("📋 Detected SMC & HTF Signals")
      st.dataframe(pd.DataFrame(bot.signals[-4:]), use_container_width=True)

    st.markdown("---")
    st.subheader(
        f"📈 {active_symbol} 5m Chart — TUCKYAA CCXT Live Execution"
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

    for sig in bot.signals[-3:]:
      tag = "B" if "Buy" in sig["Action"] or "Long" in sig["Action"] else "S"
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
        f"{active_symbol} 5m Chart — TUCKYAA CCXT Execution Strategy",
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
        "CCXT data load nahi ho paya. Please check your internet connection or"
        " requirements.txt."
    )
