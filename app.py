from datetime import datetime, timedelta, timezone
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWT Delta Pro by TUCKYAA", page_icon="⚡", layout="wide"
)

# Live auto-refresh every 10 seconds
count = st_autorefresh(interval=10000, key="twt_delta_refresh")

st.title(
    "⚡ TRADE WITH TUCKYAA (TUCKYAA) — Delta Exchange 4H/6H Resistance/Support"
    " Dashboard"
)
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | 6H Support & 4H Resistance +"
    " Verified Retest & Rejection Engine + Multi-TP**"
)

# Delta Exchange 20 Crypto & Metal Assets (No Indian Stocks)
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
    "ARB/USDT",
    "OP/USDT",
    "SUI/USDT",
    "FIL/USDT",
    "RENDER/USDT",
]

# Sidebar Session Box Panel
st.sidebar.markdown("### 🤖 TWT Expert Panel")
st.sidebar.markdown("---")
st.sidebar.text(
    f"Server time          ———  {datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%H:%M')}"
)
st.sidebar.text("Support Ref (6H)   ———  05:30 Candle Low")
st.sidebar.text("Resistance Ref (4H) ———  01:30 Candle High")
st.sidebar.markdown("---")
st.sidebar.text("Exchange Platform  — Delta Exchange (20 Assets)")
st.sidebar.text("Spread:  4 (Max. = 50)")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Trading Settings")
st.sidebar.text("Entry method     : Resistance Rejection & Retest")
st.sidebar.text("Take profit        : Multi-TP (TP1, TP2, TP3)")
st.sidebar.text("Stop loss          : ATR Managed")

# Asset Selection Dropdown
selected_symbol = st.selectbox(
    "Select Delta Exchange Asset", options=ASSET_LIST, index=0
)


@st.cache_data(ttl=10)
def fetch_twt_delta_data(symbol):
  exchanges_to_try = [
      (ccxt.binance, symbol),
      (ccxt.bybit, symbol),
      (ccxt.kucoin, symbol),
  ]

  for ex_class, sym in exchanges_to_try:
    try:
      exchange = ex_class({"enableRateLimit": True, "timeout": 20000})
      exchange.load_markets()
      if sym in exchange.markets:
        ohlcv_5m = exchange.fetch_ohlcv(sym, timeframe="5m", limit=100)
        ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=20)
        ohlcv_6h = exchange.fetch_ohlcv(sym, timeframe="6h", limit=20)

        if ohlcv_5m and ohlcv_4h:
          df_5 = pd.DataFrame(
              ohlcv_5m,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df_5["timestamp"] = pd.to_datetime(df_5["timestamp"], unit="ms")
          df_5 = df_5.set_index("timestamp")

          df_4 = pd.DataFrame(
              ohlcv_4h,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df_4["timestamp"] = pd.to_datetime(df_4["timestamp"], unit="ms")
          df_4 = df_4.set_index("timestamp")

          df_6 = (
              pd.DataFrame(
                  ohlcv_6h,
                  columns=[
                      "timestamp",
                      "Open",
                      "High",
                      "Low",
                      "Close",
                      "Volume",
                  ],
              )
              if ohlcv_6h
              else df_4.copy()
          )
          df_6["timestamp"] = pd.to_datetime(df_6["timestamp"], unit="ms")
          df_6 = df_6.set_index("timestamp")

          return df_5, df_4, df_6, symbol
    except Exception:
      continue
  return None, None, None, None


df_5m, df_4h, df_6h, active_symbol = fetch_twt_delta_data(selected_symbol)


class TWTDeltaEngine:

  def __init__(self, df_5, df_4, df_6, symbol):
    self.df_5 = df_5 if df_5 is not None else None
    self.df_4 = df_4 if df_4 is not None else None
    self.df_6 = df_6 if df_6 is not None else None
    self.symbol = symbol
    self.candles = []
    self.signals = []
    self.resistance_lvl = 0
    self.support_lvl = 0

  def calculate_indicators(self):
    if self.df_5 is not None and len(self.df_5) >= 30:
      if self.df_4 is not None and not self.df_4.empty:
        self.resistance_lvl = float(self.df_4["High"].iloc[-1])

      if self.df_6 is not None and not self.df_6.empty:
        self.support_lvl = float(self.df_6["Low"].iloc[-1])
      elif self.df_4 is not None:
        self.support_lvl = float(self.df_4["Low"].iloc[-1])

      typical_price = (
          self.df_5["High"] + self.df_5["Low"] + self.df_5["Close"]
      ) / 3
      self.df_5["VWAP"] = (
          (typical_price * self.df_5["Volume"]).cumsum()
          / self.df_5["Volume"].cumsum()
      )
      self.df_5["Vol_SMA"] = self.df_5["Volume"].rolling(10).mean()

      delta = self.df_5["Close"].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / loss
      self.df_5["RSI_14"] = 100 - (100 / (1 + rs))

      self.df_5["TR"] = np.maximum(
          self.df_5["High"] - self.df_5["Low"],
          np.maximum(
              abs(self.df_5["High"] - self.df_5["Close"].shift(1)),
              abs(self.df_5["Low"] - self.df_5["Close"].shift(1)),
          ),
      )
      self.df_5["ATR"] = self.df_5["TR"].rolling(14).mean()

  def parse_and_scan(self):
    if self.df_5 is None or self.df_5.empty:
      return

    IST = timezone(timedelta(hours=5, minutes=30))

    for idx, row in enumerate(self.df_5.iterrows()):
      dt_utc = row[0]
      if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
      dt_ist = dt_utc.astimezone(IST)
      timestamp_str = dt_ist.strftime("%H:%M")

      r = row[1]
      o, h, l, c, v = (
          float(r["Open"]),
          float(r["High"]),
          float(r["Low"]),
          float(r["Close"]),
          float(r["Volume"]),
      )
      vwap = float(r["VWAP"]) if not pd.isna(r["VWAP"]) else c
      vol_avg = float(r["Vol_SMA"]) if not pd.isna(r["Vol_SMA"]) else v
      rsi = float(r["RSI_14"]) if not pd.isna(r["RSI_14"]) else 50.0
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
          "vwap": vwap,
          "rsi": rsi,
          "atr": atr,
      })

    for i in range(7, len(self.candles)):
      c = self.candles[i]
      prev1 = self.candles[i - 1]

      is_volume_strong = c["volume"] >= (c["vol_avg"] * 1.1)
      candle_body = abs(c["close"] - c["open"])
      total_range = c["high"] - c["low"]
      is_body_strong = (
          candle_body >= (total_range * 0.4) if total_range > 0 else True
      )
      atr = c["atr"]

      # Volatility / News Filter
      avg_atr = np.mean([x["atr"] for x in self.candles[-20:]])
      if atr > (2.5 * avg_atr):
        continue

      # Resistance Rejection Sell Check (Jab price 4H Resistance ke paas ho aur wahan se reject ho)
      is_near_resistance = (
          c["high"] >= self.resistance_lvl - (0.5 * atr)
          and c["high"] <= self.resistance_lvl + (0.8 * atr)
      )
      upper_wick_rejection = (
          c["high"] - max(c["open"], c["close"])
      ) >= candle_body * 1.2

      if (
          is_near_resistance
          and (upper_wick_rejection or c["close"] < c["open"])
          and c["close"] < self.resistance_lvl
          and c["close"] < c["vwap"]
          and rsi < 55.0
          and is_volume_strong
      ):
        sl = c["high"] + (1.2 * atr)
        tp1 = c["close"] - (1.0 * atr)
        tp2 = c["close"] - (2.0 * atr)
        tp3 = c["close"] - (3.5 * atr)

        if not any(
            s["Setup"] == "TWT Resistance Rejection Sell"
            and abs(s["Candle #"] - c["id"]) < 8
            for s in self.signals
        ):
          self.signals.append({
              "Time": c["time"],
              "Candle #": c["id"],
              "Setup": "TWT Resistance Rejection Sell",
              "Action": "Sell / Short (S)",
              "Entry Price": round(c["close"], 2),
              "Stop Loss (SL)": round(sl, 2),
              "TP 1": round(tp1, 2),
              "TP 2": round(tp2, 2),
              "TP 3": round(tp3, 2),
          })

      # Support Rejection Buy Check (Jab price 6H Support ke paas ho aur wahan se bounce kare)
      is_near_support = (
          c["low"] <= self.support_lvl + (0.5 * atr)
          and c["low"] >= self.support_lvl - (0.8 * atr)
      )
      lower_wick_rejection = (
          min(c["open"], c["close"]) - c["low"]
      ) >= candle_body * 1.2

      if (
          is_near_support
          and (lower_wick_rejection or c["close"] > c["open"])
          and c["close"] > self.support_lvl
          and c["close"] > c["vwap"]
          and rsi > 45.0
          and is_volume_strong
      ):
        sl = c["low"] - (1.2 * atr)
        tp1 = c["close"] + (1.0 * atr)
        tp2 = c["close"] + (2.0 * atr)
        tp3 = c["close"] + (3.5 * atr)

        if not any(
            s["Setup"] == "TWT Support Rejection Buy"
            and abs(s["Candle #"] - c["id"]) < 8
            for s in self.signals
        ):
          self.signals.append({
              "Time": c["time"],
              "Candle #": c["id"],
              "Setup": "TWT Support Rejection Buy",
              "Action": "Buy / Long (B)",
              "Entry Price": round(c["close"], 2),
              "Stop Loss (SL)": round(sl, 2),
              "TP 1": round(tp1, 2),
              "TP 2": round(tp2, 2),
              "TP 3": round(tp3, 2),
          })


with st.spinner(f"⚡ Loading TWT Delta Engine for {selected_symbol}..."):
  if df_5m is not None and not df_5m.empty:
    bot = TWTDeltaEngine(df_5m, df_4h, df_6h, selected_symbol)
    bot.calculate_indicators()
    bot.parse_and_scan()

    curr_p = bot.candles[-1]["close"] if bot.candles else 0.0
    curr_atr = bot.candles[-1]["atr"] if bot.candles else (curr_p * 0.002)

    st.markdown("---")
    st.subheader(f"📊 Delta Exchange Live Dashboard ({selected_symbol})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.metric(label="4H Resistance", value=f"{bot.resistance_lvl:.2f}")
    with col2:
      st.metric(label="6H Support", value=f"{bot.support_lvl:.2f}")
    with col3:
      st.metric(label="Current Market Price", value=f"{curr_p:.2f}")
    with col4:
      st.metric(
          label="Methodology",
          value="Resistance/Support Rejection",
          delta="Active",
      )

    st.markdown("---")
    st.subheader("🎯 Trade With Tuckyaa (TWT) Delta Rejection Engine")

    if bot.signals:
      latest_sig = bot.signals[-1]
      is_buy = "Buy" in latest_sig["Action"]
      entry_p = latest_sig["Entry Price"]
      signal_time = latest_sig["Time"]

      if is_buy:
        current_profit_dist = curr_p - entry_p
        active_sl = latest_sig["Stop Loss (SL)"]
        if current_profit_dist > (0.5 * curr_atr):
          active_sl = round(entry_p, 2)
          trail_status = "🔒 Zero Loss (Break-Even Locked)"
        else:
          trail_status = "🛡️ Initial Strict SL"

        st.success(
            f"🟢 **TWT DELTA SETUP: {latest_sig['Action']}**\n\n"
            f"🕒 **Candle Formation Time (IST):** `{signal_time}` (Candle"
            f" #{latest_sig['Candle #']})\n"
            f"- **Locked Entry Price:** `{entry_p}`\n"
            f"- **Managed Stop Loss (SL):** `{active_sl}` ({trail_status})\n"
            f"- **Target Profit 1 (TP1):** `{latest_sig['TP 1']}`\n"
            f"- **Target Profit 2 (TP2):** `{latest_sig['TP 2']}`\n"
            f"- **Target Profit 3 (TP3):** `{latest_sig['TP 3']}`\n\n"
            "📌 *Instructions:* Support bounce & rejection confirmed. Execute"
            " trade safely!"
        )
      else:
        current_profit_dist = entry_p - curr_p
        active_sl = latest_sig["Stop Loss (SL)"]
        if current_profit_dist > (0.5 * curr_atr):
          active_sl = round(entry_p, 2)
          trail_status = "🔒 Zero Loss (Break-Even Locked)"
        else:
          trail_status = "🛡️ Initial Strict SL"

        st.error(
            f"🔴 **TWT DELTA SETUP: {latest_sig['Action']}**\n\n"
            f"🕒 **Candle Formation Time (IST):** `{signal_time}` (Candle"
            f" #{latest_sig['Candle #']})\n"
            f"- **Locked Entry Price:** `{entry_p}`\n"
            f"- **Managed Stop Loss (SL):** `{active_sl}` ({trail_status})\n"
            f"- **Target Profit 1 (TP1):** `{latest_sig['TP 1']}`\n"
            f"- **Target Profit 2 (TP2):** `{latest_sig['TP 2']}`\n"
            f"- **Target Profit 3 (TP3):** `{latest_sig['TP 3']}`\n\n"
            "📌 *Instructions:* Resistance rejection confirmed. Execute short"
            " trade safely!"
        )
    else:
      st.warning(
          "🟡 **STATUS: HOLD / WAITING FOR KEY LEVEL REJECTION**\n\n"
          "📌 *Instructions:* Waiting for clean price reaction near 4H"
          " Resistance or 6H Support."
      )

    if bot.signals:
      st.markdown("---")
      st.subheader("📋 Verified TWT Signal Execution History")
      st.dataframe(pd.DataFrame(bot.signals[-5:]), width="stretch")

    st.markdown("---")
    st.subheader(f"📈 {selected_symbol} 5m Real-Time Chart")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    for c in bot.candles:
      col = "#26a69a" if c["color"] == "Green" else "#ef5350"
      body_size = abs(c["close"] - c["open"])
      min_body = curr_p * 0.0003
      plot_height = max(body_size, min_body)
      plot_bottom = (
          min(c["open"], c["close"]) - ((min_body - body_size) / 2)
          if body_size < min_body
          else min(c["open"], c["close"])
      )

      ax.bar(
          c["id"],
          plot_height,
          bottom=plot_bottom,
          color=col,
          width=0.7,
      )
      ax.vlines(c["id"], c["low"], c["high"], color="#b2b5be", linewidth=1.3)

    ax.axhline(
        bot.resistance_lvl,
        color="#ff7043",
        linestyle="--",
        linewidth=2,
        label="4H Resistance",
    )
    ax.text(
        2,
        bot.resistance_lvl,
        f"  4H Resistance ({bot.resistance_lvl:.1f})",
        color="#ff7043",
        weight="bold",
        fontsize=10,
    )

    ax.axhline(
        bot.support_lvl,
        color="#26a69a",
        linestyle="--",
        linewidth=2,
        label="6H Support",
    )
    ax.text(
        2,
        bot.support_lvl,
        f"  6H Support ({bot.support_lvl:.1f})",
        color="#26a69a",
        weight="bold",
        fontsize=10,
    )

    lows = [c["low"] for c in bot.candles]
    highs = [c["high"] for c in bot.candles]
    if bot.resistance_lvl > 0:
      highs.append(bot.resistance_lvl)
    if bot.support_lvl > 0:
      lows.append(bot.support_lvl)

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
        f"{selected_symbol} 5m — TRADE WITH TUCKYAA (TUCKYAA) Delta Engine",
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
