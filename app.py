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
count = st_autorefresh(interval=10000, key="twp_smart_rolling_refresh")

st.title("⚡ TWP ORB Ultimate Bot by TUCKYAA (Smart Rolling Engine)")
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | CCXT Engine | Dynamic Range +"
    " Zero Confusion**"
)

# Asset Selection Dropdown
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
        ohlcv = exchange.fetch_ohlcv(sym, timeframe="5m", limit=80)
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


class TWPSmartRollingEngine:

  def __init__(self, df, symbol):
    self.df = df if df is not None else None
    self.symbol = symbol
    self.candles = []
    self.signals = []
    self.or_high = 0
    self.or_low = 0

  def calculate_indicators(self):
    if self.df is not None and len(self.df) >= 30:
      # SMART ROLLING RANGE: Use recent consolidation range (last 12 candles window) for dynamic breakout
      recent_window = self.df.tail(24)
      monitoring_df = recent_window.head(6) # Reference session window

      self.or_high = float(monitoring_df["High"].max())
      self.or_low = float(monitoring_df["Low"].min())

      # Ensure current price has breathing room from levels
      curr = float(self.df["Close"].iloc[-1])
      if curr > self.or_high + (curr * 0.005):
        # If market broke way above, shift reference range up
        self.or_high = float(recent_window.tail(10)["High"].min())
        self.or_low = float(recent_window.tail(10)["Low"].max())

      self.df["EMA"] = self.df["Close"].ewm(span=20, adjust=False).mean()

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
      timestamp_str = row[0].strftime("%H:%M")
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
          "time": timestamp_str,
          "color": "Green" if c >= o else "Red",
          "open": o,
          "high": h,
          "low": l,
          "close": c,
          "ema": ema,
          "atr": atr,
      })

    # Clean Breakout Logic with EMA momentum confirmation
    for i in range(4, len(self.candles)):
      c = self.candles[i]
      prev = self.candles[i - 1]

      # Bullish Breakout
      if c["close"] > self.or_high and prev["close"] <= self.or_high:
        if c["close"] > c["ema"]:
          sl = c["close"] - (1.5 * c["atr"])
          tp = c["close"] + (2.5 * c["atr"])
          self.signals.append({
              "Time": c["time"],
              "Candle #": c["id"],
              "Setup": "Dynamic Bullish Breakout",
              "Action": "Buy / Long (B)",
              "Entry Price": round(c["close"], 2),
              "Stop Loss (SL)": round(sl, 2),
              "Take Profit (TP)": round(tp, 2),
          })

      # Bearish Breakout
      elif c["close"] < self.or_low and prev["close"] >= self.or_low:
        if c["close"] < c["ema"]:
          sl = c["close"] + (1.5 * c["atr"])
          tp = c["close"] - (2.5 * c["atr"])
          self.signals.append({
              "Time": c["time"],
              "Candle #": c["id"],
              "Setup": "Dynamic Bearish Breakout",
              "Action": "Sell / Short (S)",
              "Entry Price": round(c["close"], 2),
              "Stop Loss (SL)": round(sl, 2),
              "Take Profit (TP)": round(tp, 2),
          })


with st.spinner(f"⚡ Loading Smart Engine for {selected_symbol}..."):
  if df_5m is not None and not df_5m.empty:
    bot = TWPSmartRollingEngine(df_5m, selected_symbol)
    bot.calculate_indicators()
    bot.parse_and_scan()

    curr_p = bot.candles[-1]["close"] if bot.candles else 0.0
    curr_atr = bot.candles[-1]["atr"] if bot.candles else (curr_p * 0.002)

    st.markdown("---")
    st.subheader(f"📊 Live Dashboard ({active_symbol or selected_symbol})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.metric(label="Dynamic Range High", value=f"{bot.or_high:.2f}")
    with col2:
      st.metric(label="Dynamic Range Low", value=f"{bot.or_low:.2f}")
    with col3:
      st.metric(label="Current Price", value=f"{curr_p:.2f}")
    with col4:
      st.metric(label="Trend Status", value="Active", delta="Live")

    # Single Clear Action Card
    st.markdown("---")
    st.subheader("🎯 Live Market Direction & Action")

    if curr_p > bot.or_high:
      buy_sl = round(curr_p - (1.5 * curr_atr), 2)
      buy_tp = round(curr_p + (2.5 * curr_atr), 2)
      st.success(
          "🟢 **BULLISH SETUP: BUY / LONG (B)**\n\n"
          f"- **Reason:** Market Range High ({bot.or_high:.2f}) ke upar nikal chuka hai aur momentum bullish hai!\n"
          f"- **Entry Price:** `{curr_p:.2f}`\n"
          f"- **Stop Loss (SL):** `{buy_sl}`\n"
          f"- **Take Profit (TP):** `{buy_tp}`"
      )
    elif curr_p < bot.or_low:
      sell_sl = round(curr_p + (1.5 * curr_atr), 2)
      sell_tp = round(curr_p - (2.5 * curr_atr), 2)
      st.error(
          "🔴 **BEARISH SETUP: SELL / SHORT (S)**\n\n"
          f"- **Reason:** Market Range Low ({bot.or_low:.2f}) ke neeche gir chuka hai!\n"
          f"- **Entry Price:** `{curr_p:.2f}`\n"
          f"- **Stop Loss (SL):** `{sell_sl}`\n"
          f"- **Take Profit (TP):** `{sell_tp}`"
      )
    else:
      st.warning(
          "🟡 **HOLD / RANGE BOUND (WAIT)**\n\n"
          f"- Market abhi range ({bot.or_low:.2f} - {bot.or_high:.2f}) ke beech me hai.\n"
          "- **Action:** Jab tak proper breakout na ho, tab tak wait karo!"
      )

    if bot.signals:
      st.markdown("---")
      st.subheader("📋 Recent Breakout Signals History")
      st.dataframe(pd.DataFrame(bot.signals[-4:]), use_container_width=True)
    else:
      st.info(f"💡 Monitoring {selected_symbol} for active breakout...")

    st.markdown("---")
    st.subheader(f"📈 {active_symbol or selected_symbol} 5m Chart — Dynamic View")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    for c in bot.candles:
      col = "#26a69a" if c["color"] == "Green" else "#ef5350"
      body_size = abs(c["close"] - c["open"])
      min_body = (
          curr_p * 0.0008
          if "XAUT" in selected_symbol or "ETH" in selected_symbol
          else curr_p * 0.0003
      )
      plot_height = max(body_size, min_body)

      if body_size < min_body:
        plot_bottom = min(c["open"], c["close"]) - (
            (min_body - body_size) / 2
        )
      else:
        plot_bottom = min(c["open"], c["close"])

      ax.bar(
          c["id"],
          plot_height,
          bottom=plot_bottom,
          color=col,
          width=0.7,
      )
      ax.vlines(c["id"], c["low"], c["high"], color="#b2b5be", linewidth=1.3)

    ax.axhline(
        bot.or_high,
        color="#ff7043",
        linestyle="--",
        linewidth=2,
        label="Range High",
    )
    ax.text(
        2,
        bot.or_high,
        f"  Range High ({bot.or_high:.1f})",
        color="#ff7043",
        weight="bold",
        fontsize=10,
    )

    ax.axhline(
        bot.or_low,
        color="#26a69a",
        linestyle="--",
        linewidth=2,
        label="Range Low",
    )
    ax.text(
        2,
        bot.or_low,
        f"  Range Low ({bot.or_low:.1f})",
        color="#26a69a",
        weight="bold",
        fontsize=10,
    )

    lows = [c["low"] for c in bot.candles]
    highs = [c["high"] for c in bot.candles]
    if bot.or_high > 0:
      highs.append(bot.or_high)
    if bot.or_low > 0:
      lows.append(bot.or_low)

    if lows and highs:
      margin = (max(highs) - min(lows)) * 0.09
      if margin == 0:
        margin = curr_p * 0.005
      ax.set_ylim(min(lows) - margin, max(highs) + margin)

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
        f"{active_symbol or selected_symbol} 5m — Dynamic Smart Engine",
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
    st.error("Data load nahi ho paya. Please check connection.")
