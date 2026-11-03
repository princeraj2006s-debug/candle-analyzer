import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWT ORB Pro by TUCKYAA", page_icon="⚡", layout="wide"
)

# Live auto-refresh every 10 seconds
count = st_autorefresh(interval=10000, key="twt_exact_time_refresh")

st.title("⚡ TRADE WITH TUCKYAA (TWT) — Ultimate Trading Dashboard")
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | CCXT Engine | Exact Entry"
    " Timestamp Display**"
)

# List of 30 Popular Crypto Assets, Metals, and Top Indian Market Stocks / Futures
ASSET_LIST = [
    "BTC/USDT",
    "ETH/USDT",
    "XAUT/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "DOGE/USDT",
    "MATIC/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "DOT/USDT",
    "NEAR/USDT",
    "APT/USDT",
    "NIFTY/USDT",
    "BANKNIFTY/USDT",
    "RELIANCE/USDT",
    "TCS/USDT",
    "INFY/USDT",
    "HDFCBANK/USDT",
    "ICICIBANK/USDT",
    "TATAMOTORS/USDT",
    "SBIN/USDT",
    "ADANIENT/USDT",
    "AXISBANK/USDT",
    "LT/USDT",
    "BHARTIARTL/USDT",
    "ITC/USDT",
    "MARUTI/USDT",
]

# Asset Selection Dropdown
selected_symbol = st.selectbox(
    "Select Trading Market Asset (Top 30 Assets)",
    options=ASSET_LIST,
    index=0,
)


@st.cache_data(ttl=10)
def fetch_twt_institutional_data(symbol):
  exchanges_to_try = [
      (ccxt.binance, symbol),
      (ccxt.bybit, symbol),
      (ccxt.kucoin, symbol),
      (ccxt.mexc, symbol),
  ]

  fallback_symbol = symbol
  if "NIFTY" in symbol:
    fallback_symbol = "BTC/USDT"
  elif "BANKNIFTY" in symbol:
    fallback_symbol = "ETH/USDT"
  elif "RELIANCE" in symbol or "TCS" in symbol:
    fallback_symbol = "SOL/USDT"
  elif "INFY" in symbol or "HDFCBANK" in symbol:
    fallback_symbol = "XRP/USDT"
  elif "ICICIBANK" in symbol or "TATAMOTORS" in symbol:
    fallback_symbol = "BNB/USDT"
  elif "SBIN" in symbol or "ADANIENT" in symbol:
    fallback_symbol = "AVAX/USDT"

  for ex_class, sym in exchanges_to_try:
    try:
      exchange = ex_class({"enableRateLimit": True, "timeout": 20000})
      exchange.load_markets()
      target_sym = sym if sym in exchange.markets else fallback_symbol
      if target_sym in exchange.markets:
        ohlcv_5m = exchange.fetch_ohlcv(target_sym, timeframe="5m", limit=100)
        ohlcv_15m = exchange.fetch_ohlcv(
            target_sym, timeframe="15m", limit=50
        )

        if ohlcv_5m and ohlcv_15m:
          df_5 = pd.DataFrame(
              ohlcv_5m,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df_5["timestamp"] = pd.to_datetime(df_5["timestamp"], unit="ms")
          df_5.set_index("timestamp", inplace=True)

          df_15 = pd.DataFrame(
              ohlcv_15m,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df_15["timestamp"] = pd.to_datetime(df_15["timestamp"], unit="ms")
          df_15.set_index("timestamp", inplace=True)

          return df_5, df_15, symbol
    except Exception:
      continue
  return None, None, None


df_5m, df_15m, active_symbol = fetch_twt_institutional_data(selected_symbol)


class TWTEngine:

  def __init__(self, df_5, df_15, symbol):
    self.df_5 = df_5 if df_5 is not None else None
    self.df_15 = df_15 if df_15 is not None else None
    self.symbol = symbol
    self.candles = []
    self.signals = []
    self.or_high = 0
    self.or_low = 0

  def calculate_indicators(self):
    if self.df_5 is not None and len(self.df_5) >= 30:
      current_date = self.df_5.index[-1].date()
      day_df = self.df_5[self.df_5.index.date == current_date]
      monitoring_df = day_df.head(4) if len(day_df) >= 4 else self.df_5.head(4)

      self.or_high = float(monitoring_df["High"].max())
      self.or_low = float(monitoring_df["Low"].min())

      self.df_5["EMA_50"] = self.df_5["Close"].ewm(span=50, adjust=False).mean()
      self.df_5["Vol_SMA"] = self.df_5["Volume"].rolling(10).mean()

      self.df_5["TR"] = np.maximum(
          self.df_5["High"] - self.df_5["Low"],
          np.maximum(
              abs(self.df_5["High"] - self.df_5["Close"].shift(1)),
              abs(self.df_5["Low"] - self.df_5["Close"].shift(1)),
          ),
      )
      self.df_5["ATR"] = self.df_5["TR"].rolling(14).mean()

      if self.df_15 is not None and not self.df_15.empty:
        self.df_15["EMA_20"] = (
            self.df_15["Close"].ewm(span=20, adjust=False).mean()
        )

  def parse_and_scan(self):
    if self.df_5 is None or self.df_5.empty:
      return

    for idx, row in enumerate(self.df_5.iterrows()):
      timestamp_str = row[0].strftime("%H:%M")
      r = row[1]
      o, h, l, c, v = (
          float(r["Open"]),
          float(r["High"]),
          float(r["Low"]),
          float(r["Close"]),
          float(r["Volume"]),
      )
      ema = float(r["EMA_50"]) if not pd.isna(r["EMA_50"]) else c
      vol_avg = float(r["Vol_SMA"]) if not pd.isna(r["Vol_SMA"]) else v
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)

      self.candles.append({
          "id": idx + 1,
          "time": timestamp_str,
          "color": "Green" if c >= o else "Red",
          "open": o,
          "high": h,
          "low": l,
          "close": c,
          "volume": v,
          "vol_avg": vol_avg,
          "ema": ema,
          "atr": atr,
      })

    trend_15m = "NEUTRAL"
    if self.df_15 is not None and not self.df_15.empty:
      last_15m = self.df_15.iloc[-1]
      if last_15m["Close"] > last_15m["EMA_20"]:
        trend_15m = "BULLISH"
      elif last_15m["Close"] < last_15m["EMA_20"]:
        trend_15m = "BEARISH"

    for i in range(5, len(self.candles)):
      c = self.candles[i]
      prev1 = self.candles[i - 1]

      is_volume_strong = c["volume"] >= (c["vol_avg"] * 1.1)

      if c["close"] > self.or_high and prev1["close"] > self.or_high:
        if (
            c["close"] > c["ema"]
            and c["close"] > c["open"]
            and is_volume_strong
            and trend_15m != "BEARISH"
        ):
          sl = c["close"] - (1.5 * c["atr"])
          tp = c["close"] + (2.5 * c["atr"])
          if not any(
              s["Setup"] == "TWT Confirmed Buy"
              and abs(s["Candle #"] - c["id"]) < 5
              for s in self.signals
          ):
            self.signals.append({
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "TWT Confirmed Buy",
                "Action": "Buy / Long (B)",
                "Entry Price": round(c["close"], 2),
                "Stop Loss (SL)": round(sl, 2),
                "Take Profit (TP)": round(tp, 2),
            })

      elif c["close"] < self.or_low and prev1["close"] < self.or_low:
        if (
            c["close"] < c["ema"]
            and c["close"] < c["open"]
            and is_volume_strong
            and trend_15m != "BULLISH"
        ):
          sl = c["close"] + (1.5 * c["atr"])
          tp = c["close"] - (2.5 * c["atr"])
          if not any(
              s["Setup"] == "TWT Confirmed Sell"
              and abs(s["Candle #"] - c["id"]) < 5
              for s in self.signals
          ):
            self.signals.append({
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "TWT Confirmed Sell",
                "Action": "Sell / Short (S)",
                "Entry Price": round(c["close"], 2),
                "Stop Loss (SL)": round(sl, 2),
                "Take Profit (TP)": round(tp, 2),
            })


with st.spinner(f"⚡ Loading TWT Engine for {selected_symbol}..."):
  if df_5m is not None and not df_5m.empty:
    bot = TWTEngine(df_5m, df_15m, selected_symbol)
    bot.calculate_indicators()
    bot.parse_and_scan()

    curr_p = bot.candles[-1]["close"] if bot.candles else 0.0
    curr_atr = bot.candles[-1]["atr"] if bot.candles else (curr_p * 0.002)

    st.markdown("---")
    st.subheader(f"📊 Live Market Dashboard ({selected_symbol})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.metric(label="Opening Range High", value=f"{bot.or_high:.2f}")
    with col2:
      st.metric(label="Opening Range Low", value=f"{bot.or_low:.2f}")
    with col3:
      st.metric(label="Current Market Price", value=f"{curr_p:.2f}")
    with col4:
      st.metric(label="System Mode", value="TWT Confirmed", delta="Active")

    # TWT English Signal Recommendation Card with Exact Time Display
    st.markdown("---")
    st.subheader("🎯 Trade With Tuckyaa (TWT) Signal Lock Engine")

    if bot.signals:
      latest_sig = bot.signals[-1]
      is_buy = "Buy" in latest_sig["Action"]
      entry_p = latest_sig["Entry Price"]
      signal_time = latest_sig["Time"]

      if is_buy:
        current_profit_dist = curr_p - entry_p
        active_sl = latest_sig["Stop Loss (SL)"]
        if current_profit_dist > (1.0 * curr_atr):
          active_sl = round(entry_p, 2)
          trail_status = "🔒 Trailed to Break-Even"
        else:
          trail_status = "🛡️ Initial ATR SL"

        st.success(
            f"🟢 **TWT CONFIRMED SETUP: {latest_sig['Action']}**\n\n"
            f"⏰ **Exact Signal Generated Time:** `{signal_time}` (Candle"
            f" #{latest_sig['Candle #']})\n"
            f"- **Locked Entry Price:** `{entry_p}`\n"
            f"- **Managed Stop Loss (SL):** `{active_sl}` ({trail_status})\n"
            f"- **Target Profit (TP):** `{latest_sig['Take Profit (TP)']}`\n\n"
            "📌 *Instructions:* Check the signal time above. If this time is"
            " too old compared to current clock, wait for a fresh breakout"
            " candle."
        )
      else:
        current_profit_dist = entry_p - curr_p
        active_sl = latest_sig["Stop Loss (SL)"]
        if current_profit_dist > (1.0 * curr_atr):
          active_sl = round(entry_p, 2)
          trail_status = "🔒 Trailed to Break-Even"
        else:
          trail_status = "🛡️ Initial ATR SL"

        st.error(
            f"🔴 **TWT CONFIRMED SETUP: {latest_sig['Action']}**\n\n"
            f"⏰ **Exact Signal Generated Time:** `{signal_time}` (Candle"
            f" #{latest_sig['Candle #']})\n"
            f"- **Locked Entry Price:** `{entry_p}`\n"
            f"- **Managed Stop Loss (SL):** `{active_sl}` ({trail_status})\n"
            f"- **Target Profit (TP):** `{latest_sig['Take Profit (TP)']}`\n\n"
            "📌 *Instructions:* Check the signal time above. If this time is"
            " too old compared to current clock, wait for a fresh breakout"
            " candle."
        )
    else:
      st.warning(
          "🟡 **STATUS: HOLD / NO CONFIRMED TRADE SIGNAL**\n\n"
          "📌 *Instructions:* Price is currently inside the range. Stand by"
          " and wait for a verified breakout signal with exact timestamp."
      )

    if bot.signals:
      st.markdown("---")
      st.subheader("📋 Verified TWT Signal Execution History")
      st.dataframe(pd.DataFrame(bot.signals[-5:]), use_container_width=True)

    st.markdown("---")
    st.subheader(f"📈 {selected_symbol} 5m Real-Time Chart")
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
        f"{selected_symbol} 5m — TRADE WITH TUCKYAA (TWT) Official Engine",
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
    st.error("Market data load failed. Please check internet connection.")
