from datetime import datetime, time, timedelta, timezone
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
    "⚡ TRADE WITH TUCKYAA (TUCKYAA) — Delta Exchange Multi-Retest Level Engine"
)
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | 01:30 4H Resistance & 05:30"
    " 6H Support + Smart Multi-Touch Retest Engine + Multi-TP**"
)

# Delta Exchange 20 Crypto & Metal Assets
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
server_time_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
st.sidebar.text(f"Server time          ———  {server_time_ist.strftime('%H:%M')}")
st.sidebar.text("Support Ref (6H)   ———  05:30 IST Candle Low")
st.sidebar.text("Resistance Ref (4H) ———  01:30 IST Candle High")
st.sidebar.text("No-Trade Zone      ———  01:00 ➔ 06:00 IST")
st.sidebar.markdown("---")
st.sidebar.text("Exchange Platform  — Delta Exchange (20 Assets)")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Trading Settings")
st.sidebar.text("Entry method     : Smart Multi-Touch Retest + Dump Protection")
st.sidebar.text("Take profit        : Multi-TP (TP1, TP2, TP3)")
st.sidebar.text("Stop loss          : ATR Managed")

# Asset Selection Dropdown
selected_symbol = st.selectbox(
    "Select Delta Exchange Asset", options=ASSET_LIST, index=5
)


def get_decimals(price):
  if price < 1:
    return 4
  elif price < 10:
    return 3
  else:
    return 2


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
        ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=30)
        ohlcv_6h = exchange.fetch_ohlcv(sym, timeframe="6h", limit=30)

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
    IST = timezone(timedelta(hours=5, minutes=30))
    if self.df_5 is not None and len(self.df_5) >= 30:
      if self.df_4 is not None and not self.df_4.empty:
        idx_utc = (
            self.df_4.index.tz_localize("UTC")
            if self.df_4.index.tz is None
            else self.df_4.index
        )
        df_4_ist = idx_utc.tz_convert(IST)
        mask_4h = df_4_ist.hour == 1
        filtered_4h = self.df_4[mask_4h]
        if not filtered_4h.empty:
          self.resistance_lvl = float(filtered_4h["High"].iloc[-1])
        else:
          self.resistance_lvl = float(self.df_4["High"].iloc[-1])

      if self.df_6 is not None and not self.df_6.empty:
        idx_utc_6 = (
            self.df_6.index.tz_localize("UTC")
            if self.df_6.index.tz is None
            else self.df_6.index
        )
        df_6_ist = idx_utc_6.tz_convert(IST)
        mask_6h = df_6_ist.hour == 5
        filtered_6h = self.df_6[mask_6h]
        if not filtered_6h.empty:
          self.support_lvl = float(filtered_6h["Low"].iloc[-1])
        else:
          self.support_lvl = float(filtered_6h["Low"].iloc[-1])
      elif self.df_4 is not None:
        self.support_lvl = float(self.df_4["Low"].iloc[-1])

      # EMAs
      self.df_5["EMA20"] = self.df_5["Close"].ewm(span=20, adjust=False).mean()
      self.df_5["EMA50"] = self.df_5["Close"].ewm(span=50, adjust=False).mean()
      self.df_5["EMA200"] = self.df_5["Close"].ewm(span=200, adjust=False).mean()

      # ATR 14 & Volume SMA
      self.df_5["TR"] = np.maximum(
          self.df_5["High"] - self.df_5["Low"],
          np.maximum(
              abs(self.df_5["High"] - self.df_5["Close"].shift(1)),
              abs(self.df_5["Low"] - self.df_5["Close"].shift(1)),
          ),
      )
      self.df_5["ATR"] = self.df_5["TR"].rolling(14).mean()
      self.df_5["Vol_SMA"] = self.df_5["Volume"].rolling(14).mean()

      # RSI 14
      delta = self.df_5["Close"].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
      rs = gain / (loss + 1e-9)
      self.df_5["RSI"] = 100 - (100 / (1 + rs))

      # ADX 14 approximation
      plus_dm = self.df_5["High"].diff()
      minus_dm = self.df_5["Low"].diff()
      plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
      minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
      tr14 = self.df_5["TR"].rolling(14).sum()
      plus_di = (
          100
          * pd.Series(plus_dm, index=self.df_5.index).rolling(14).sum()
          / (tr14 + 1e-9)
      )
      minus_di = (
          100
          * pd.Series(minus_dm, index=self.df_5.index).rolling(14).sum()
          / (tr14 + 1e-9)
      )
      dx = (
          100
          * abs(plus_di - minus_di)
          / (plus_di + minus_di + 1e-9)
      )
      self.df_5["ADX"] = dx.rolling(14).mean()

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
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)
      vol_sma = float(r["Vol_SMA"]) if not pd.isna(r["Vol_SMA"]) else v
      ema20 = float(r["EMA20"]) if not pd.isna(r["EMA20"]) else c
      ema50 = float(r["EMA50"]) if not pd.isna(r["EMA50"]) else c
      ema200 = float(r["EMA200"]) if not pd.isna(r["EMA200"]) else c
      rsi = float(r["RSI"]) if not pd.isna(r["RSI"]) else 50.0
      adx = float(r["ADX"]) if not pd.isna(r["ADX"]) else 20.0

      self.candles.append({
          "id": idx + 1,
          "time": timestamp_str,
          "dt_ist": dt_ist,
          "color": "Green" if c >= o else "Red",
          "open": o,
          "high": h,
          "low": l,
          "close": c,
          "volume": v,
          "vol_sma": vol_sma,
          "atr": atr,
          "ema20": ema20,
          "ema50": ema50,
          "ema200": ema200,
          "rsi": rsi,
          "adx": adx,
      })

    res_touch_count = 0
    sup_touch_count = 0
    last_res_candle_id = -99
    last_sup_candle_id = -99

    for i in range(5, len(self.candles)):
      c = self.candles[i]

      c_time = c["dt_ist"].time()
      if time(1, 0) <= c_time <= time(6, 0):
        continue

      atr = c["atr"]
      dec = get_decimals(c["close"])

      # --- PANIC DUMP & SUPPORT BREAKDOWN FILTER ---
      # Blocks buy signals if price crashes below support with high volume (> 1.5x average volume)
      is_support_breakdown = (
          c["close"] < self.support_lvl
          and c["close"] < c["open"]
          and c["volume"] > (c["vol_sma"] * 1.5)
      )

      # --- RESISTANCE BREAKOUT TRAP FILTER ---
      # Blocks sell signals if price pumps above resistance with high volume
      is_resistance_breakout = (
          c["close"] > self.resistance_lvl
          and c["close"] > c["open"]
          and c["volume"] > (c["vol_sma"] * 1.5)
      )

      if is_support_breakdown:
        sup_touch_count = 0
      if is_resistance_breakout:
        res_touch_count = 0

      # --- 4H RESISTANCE TOUCH TRACKING WITH STRICT REJECTION ---
      is_near_res = (
          c["high"] >= self.resistance_lvl - (0.8 * atr)
          and c["high"] <= self.resistance_lvl + (0.8 * atr)
      )
      if is_near_res and not is_resistance_breakout and (i - last_res_candle_id > 5):
        res_touch_count += 1
        last_res_candle_id = i

        # 2nd or 3rd Touch Sell Trigger with bearish rejection candle check
        if res_touch_count >= 2 and c["close"] < c["open"]:
          sl = round(c["high"] + (1.2 * atr), dec)
          tp1 = round(c["close"] - (1.0 * atr), dec)
          tp2 = round(c["close"] - (2.2 * atr), dec)
          tp3 = round(c["close"] - (3.8 * atr), dec)

          if not any(
              s["Setup"] == "TWT Protected Resistance Sell"
              and abs(s["Candle #"] - c["id"]) < 6
              for s in self.signals
          ):
            self.signals.append({
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "TWT Protected Resistance Sell",
                "Action": f"Sell / Short (S) [Retest #{res_touch_count}] 🔴",
                "Entry Price": round(c["close"], dec),
                "Stop Loss (SL)": sl,
                "TP 1": tp1,
                "TP 2": tp2,
                "TP 3": tp3,
            })

      # --- 6H SUPPORT TOUCH TRACKING WITH STRICT REJECTION ---
      is_near_sup = (
          c["low"] <= self.support_lvl + (0.8 * atr)
          and c["low"] >= self.support_lvl - (0.8 * atr)
      )
      if is_near_sup and not is_support_breakdown and (i - last_sup_candle_id > 5):
        sup_touch_count += 1
        last_sup_candle_id = i

        # 2nd or 3rd Touch Buy Trigger with bullish rejection candle check
        if sup_touch_count >= 2 and c["close"] > c["open"]:
          sl = round(c["low"] - (1.2 * atr), dec)
          tp1 = round(c["close"] + (1.0 * atr), dec)
          tp2 = round(c["close"] + (2.2 * atr), dec)
          tp3 = round(c["close"] + (3.8 * atr), dec)

          if not any(
              s["Setup"] == "TWT Protected Support Buy"
              and abs(s["Candle #"] - c["id"]) < 6
              for s in self.signals
          ):
            self.signals.append({
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "TWT Protected Support Buy",
                "Action": f"Buy / Long (B) [Retest #{sup_touch_count}] 🟢",
                "Entry Price": round(c["close"], dec),
                "Stop Loss (SL)": sl,
                "TP 1": tp1,
                "TP 2": tp2,
                "TP 3": tp3,
            })


with st.spinner(f"⚡ Loading TWT Delta Engine for {selected_symbol}..."):
  if df_5m is not None and not df_5m.empty:
    bot = TWTDeltaEngine(df_5m, df_4h, df_6h, selected_symbol)
    bot.calculate_indicators()
    bot.parse_and_scan()

    curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

    st.markdown("---")
    st.subheader(f"📊 Delta Exchange Live Dashboard ({selected_symbol})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.metric(
          label="4H Resistance (01:30 High)", value=f"{bot.resistance_lvl:.2f}"
      )
    with col2:
      st.metric(label="6H Support (05:30 Low)", value=f"{bot.support_lvl:.2f}")
    with col3:
      st.metric(label="Current Market Price", value=f"{curr_p:.2f}")
    with col4:
      st.metric(
          label="Methodology",
          value="Smart Multi-Touch Retest + Dump Protection",
          delta="Active",
      )

    st.markdown("---")
    st.subheader("🎯 Trade With Tuckyaa (TUCKYAA) Multi-Touch Engine")

    if bot.signals:
      latest_sig = bot.signals[-1]
      is_buy = "Buy" in latest_sig["Action"]
      entry_p = latest_sig["Entry Price"]
      signal_time = latest_sig["Time"]

      if is_buy:
        st.success(
            f"🟢 **TWT DELTA SETUP: {latest_sig['Action']}**\n\n"
            f"🕒 **Candle Time (IST):** `{signal_time}` (Candle"
            f" #{latest_sig['Candle #']})\n"
            f"- **Entry Price:** `{entry_p}`\n"
            f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
            f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:**"
            f" `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`\n\n"
            "📌 *Instructions:* Protected Support multi-touch retest confirmed!"
        )
      else:
        st.error(
            f"🔴 **TWT DELTA SETUP: {latest_sig['Action']}**\n\n"
            f"🕒 **Candle Time (IST):** `{signal_time}` (Candle"
            f" #{latest_sig['Candle #']})\n"
            f"- **Entry Price:** `{entry_p}`\n"
            f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
            f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:**"
            f" `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`\n\n"
            "📌 *Instructions:* Protected Resistance multi-touch retest confirmed!"
        )
    else:
      st.warning(
          "🟡 **STATUS: SCANNING WITH PANIC DUMP PROTECTION**\n\n"
          "📌 *Instructions:* Monitoring price action near 4H Resistance and 6H Support while filtering high-volume breakdown traps."
      )

    if bot.signals:
      st.markdown("---")
      st.subheader("📋 Verified TWT Signal History")
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
        color="#ffeb3b",
        linestyle="--",
        linewidth=2,
        label="4H Resistance",
    )
    ax.text(
        2,
        bot.resistance_lvl,
        f"  4H Resistance ({bot.resistance_lvl:.1f})",
        color="#ffeb3b",
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
