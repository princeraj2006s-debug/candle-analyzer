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
count = st_autorefresh(interval=10000, key="twp_locked_signal_refresh")

st.title("⚡ TWP ORB Ultimate Bot by TUCKYAA (Locked Entry Engine)")
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | CCXT Engine | Locked Targets +"
    " No Fluctuation**"
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


class TWPLockedEngine:

  def __init__(self, df, symbol):
    self.df = df if df is not None else None
    self.symbol = symbol
    self.candles = []
    self.signals = []
    self.or_high = 0
    self.or_low = 0

  def calculate_indicators(self):
    if self.df is not None and len(self.df) >= 30:
      current_date = self.df.index[-1].date()
      day_df = self.df[self.df.index.date == current_date]
      monitoring_df = day_df.head(4) if len(day_df) >= 4 else self.df.head(4)

      self.or_high = float(monitoring_df["High"].max())
      self.or_low = float(monitoring_df["Low"].min())

      self.df["EMA"] = self.df["Close"].ewm(span=50, adjust=False).mean()

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

    # Strict Confirmed Breakout Logic
    for i in range(5, len(self.candles)):
      c = self.candles[i]
      prev1 = self.candles[i - 1]

      if c["close"] > self.or_high and prev1["close"] > self.or_high:
        if c["close"] > c["ema"] and c["close"] > c["open"]:
          sl = c["close"] - (1.5 * c["atr"])
          tp = c["close"] + (2.5 * c["atr"])
          if not any(s["Setup"] == "Confirmed Buy" and abs(s["Candle #"] - c["id"]) < 5 for s in self.signals):
            self.signals.append({
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "Confirmed Buy",
                "Action": "Buy / Long (B)",
                "Entry Price": round(c["close"], 2),
                "Stop Loss (SL)": round(sl, 2),
                "Take Profit (TP)": round(tp, 2),
            })

      elif c["close"] < self.or_low and prev1["close"] < self.or_low:
        if c["close"] < c["ema"] and c["close"] < c["open"]:
          sl = c["close"] + (1.5 * c["atr"])
          tp = c["close"] - (2.5 * c["atr"])
          if not any(s["Setup"] == "Confirmed Sell" and abs(s["Candle #"] - c["id"]) < 5 for s in self.signals):
            self.signals.append({
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "Confirmed Sell",
                "Action": "Sell / Short (S)",
                "Entry Price": round(c["close"], 2),
                "Stop Loss (SL)": round(sl, 2),
                "Take Profit (TP)": round(tp, 2),
            })


with st.spinner(f"⚡ Loading Locked Engine for {selected_symbol}..."):
  if df_5m is not None and not df_5m.empty:
    bot = TWPLockedEngine(df_5m, selected_symbol)
    bot.calculate_indicators()
    bot.parse_and_scan()

    curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

    st.markdown("---")
    st.subheader(f"📊 Live Dashboard ({active_symbol or selected_symbol})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.metric(label="Fixed OR High", value=f"{bot.or_high:.2f}")
    with col2:
      st.metric(label="Fixed OR Low", value=f"{bot.or_low:.2f}")
    with col3:
      st.metric(label="Current Price", value=f"{curr_p:.2f}")
    with col4:
      st.metric(label="Status", value="Locked & Active", delta="Stable")

    # Locked Single Signal Card
    st.markdown("---")
    st.subheader("🎯 Locked Active Trade Setup")

    if bot.signals:
      latest_sig = bot.signals[-1]
      is_buy = "Buy" in latest_sig["Action"]
      
      if is_buy:
        st.success(
            f"🟢 **LOCKED SETUP: {latest_sig['Action']}**\n\n"
            f"- **Breakout Time:** `{latest_sig['Time']}` (Candle #{latest_sig['Candle #']})\n"
            f"- **Locked Entry Price:** `{latest_sig['Entry Price']}`\n"
            f"- **Locked Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
            f"- **Locked Take Profit (TP):** `{latest_sig['Take Profit (TP)']}`\n\n"
            "👉 *Ye entry price tab tak locked rahegi jab tak target ya SL hit nahi hota!*"
        )
      else:
        st.error(
            f"🔴 **LOCKED SETUP: {latest_sig['Action']}**\n\n"
            f"- **Breakout Time:** `{latest_sig['Time']}` (Candle #{latest_sig['Candle #']})\n"
            f"- **Locked Entry Price:** `{latest_sig['Entry Price']}`\n"
            f"- **Locked Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
            f"- **Locked Take Profit (TP):** `{latest_sig['Take Profit (TP)']}`\n\n"
            "👉 *Ye entry price tab tak locked rahegi jab tak target ya SL hit nahi hota!*"
        )
    else:
      st.warning(
          "🟡 **HOLD / WAIT (NO LOCKED SIGNAL YET)**\n\n"
          "- Abhi koi bhi confirmed breakout nahi hua hai.\n"
          "- **Action:** Fuzool me entry mat lo, jab tak proper locked signal na aaye, wait karo!"
      )

    if bot.signals:
      st.markdown("---")
      st.subheader("📋 Locked Signal History")
      st.dataframe(pd.DataFrame(bot.signals[-4:]), use_container_width=True)

    st.markdown("---")
    st.subheader(f"📈 {active_symbol or selected_symbol} 5m Chart — Locked View")
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
        label="OR High",
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
        label="OR Low",
    )
    ax.text(
        2,
        bot.or_low,
        f"  OR Low ({bot.or_low:.1f})",
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
        f"{active_symbol or selected_symbol} 5m — Locked Signal Engine",
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
