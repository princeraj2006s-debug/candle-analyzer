import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="CANDLESENSE Pro by TUCKYAA", page_icon="⚡", layout="wide"
)

# Live feed auto-refresh every 10 seconds
count = st_autorefresh(interval=10000, key="tuckyaa_ccxt_btc_refresh")

st.title(
    "⚡ CANDLESENSE Pro by TUCKYAA: BTC CCXT Real-Time SMC & HTF Bot"
)
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | Asset: BTC/USDT (Binance"
    " Clean Data) | 5m Execution | Exact 6H/4H Levels**"
)

selected_symbol = "BTC/USDT"


@st.cache_data(ttl=10)
def fetch_clean_crypto_data():
  try:
    exchange = ccxt.binance({"enableRateLimit": True})
    # Fetch 5-minute candles for execution chart
    ohlcv_5m = exchange.fetch_ohlcv(selected_symbol, timeframe="5m", limit=50)
    df_5m = pd.DataFrame(
        ohlcv_5m, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], unit="ms")
    df_5m.set_index("timestamp", inplace=True)

    # Fetch 1-hour candles for precise 6H (2:00 AM) and 4H (00:00 AM) levels
    ohlcv_1h = exchange.fetch_ohlcv(selected_symbol, timeframe="1h", limit=72)
    df_1h = pd.DataFrame(
        ohlcv_1h, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"], unit="ms")
    df_1h.set_index("timestamp", inplace=True)

    return df_5m, df_1h
  except Exception as e:
    return None, None


df_5m, df_1h = fetch_clean_crypto_data()


class TuckyasCCXTSMCEngine:

  def __init__(self, df_5m, df_1h):
    self.df = df_5m if df_5m is not None else None
    self.df_htf = df_1h if df_1h is not None else None
    self.candles = []
    self.htf_levels = {}
    self.smc_signals = []

  def calculate_exact_htf_levels(self):
    if self.df_htf is not None and not self.df_htf.empty:
      # 6H Support: Low of the 2:00 AM candle
      df_2am = self.df_htf[self.df_htf.index.hour == 2]
      if not df_2am.empty:
        self.htf_levels["6H_Support"] = float(df_2am["Low"].iloc[-1])
      else:
        self.htf_levels["6H_Support"] = float(self.df_htf["Low"].min())

      # 4H Resistance: High of the 00:00 AM (midnight) candle
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

  def scan_5_smc_models(self):
    if len(self.candles) < 5:
      return

    bodies = [c["body"] for c in self.candles]
    avg_b = np.mean(bodies) if bodies else 50
    sup = self.htf_levels["6H_Support"]
    res = self.htf_levels["4H_Resistance"]

    for i in range(4, len(self.candles)):
      c_curr = self.candles[i]
      c_prev = self.candles[i - 1]
      c_prev2 = self.candles[i - 2]

      # Zone Reaction Check + 5 SMC Models
      if c_curr["high"] >= res - 10 and c_curr["close"] < res:
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "4H Resistance Rejection",
            "Type": "Bearish S",
            "Price": c_curr["high"],
        })
      elif c_curr["low"] <= sup + 10 and c_curr["close"] > sup:
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "6H Support Bounce",
            "Type": "Bullish B",
            "Price": c_curr["low"],
        })
      elif (
          c_prev2["color"] == "Red"
          and c_prev["color"] == "Green"
          and c_curr["close"] > c_prev2["high"]
      ):
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "Breaker Block / IFVG",
            "Type": "Bullish B",
            "Price": c_curr["close"],
        })
      elif (
          c_prev2["high"] < c_curr["low"] and c_prev["body"] > avg_b * 1.2
      ):
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "Fair Value Gap (FVG)",
            "Type": "Bullish B",
            "Price": c_curr["close"],
        })
      elif c_prev["color"] == "Red" and c_curr["body"] > avg_b * 1.5:
        self.smc_signals.append({
            "Candle #": c_curr["id"],
            "Model": "Order Block (OB)",
            "Type": "Bullish B",
            "Price": c_curr["close"],
        })


with st.spinner("⚡ Fetching Clean Binance CCXT Feed for TUCKYAA..."):
  if df_5m is not None and not df_5m.empty:
    bot = TuckyasCCXTSMCEngine(df_5m, df_1h)
    bot.calculate_exact_htf_levels()
    bot.parse_candles()
    bot.scan_5_smc_models()

    curr_p = bot.candles[-1]["close"] if bot.candles else 77929.5

    st.markdown("---")
    st.subheader("🎯 TUCKYAA 5-Minute Trade Entries (BTC/USDT)")
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

    if bot.smc_signals:
      st.markdown("---")
      st.subheader("📋 Detected SMC & HTF Reactions")
      st.dataframe(pd.DataFrame(bot.smc_signals[-4:]), use_container_width=True)

    # TRADINGVIEW / DELTA EXCHANGE STYLE DARK CHART PLOT
    st.markdown("---")
    st.subheader(
        "📈 BTC/USDT 5m Clean Chart — TUCKYAA Live Execution (CCXT Powered)"
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

    # Plot 6H Support Line (2:00 AM Low)
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

    # Plot 4H Resistance Line (00:00 AM High)
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

    # Plot Signal Markers
    for sig in bot.smc_signals[-3:]:
      tag = "B" if "Bullish" in sig["Type"] or "Bounce" in sig["Model"] else "S"
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
        "BTC/USDT 5m Chart — TUCKYAA CCXT Bot", color="white", fontsize=14
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
    st.error("Binance CCXT data load nahi ho paya. Please check connection.")
