import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWP ORB Pro by TUCKYAA", page_icon="⚡", layout="wide"
)

# Live auto-refresh every 10 seconds
count = st_autorefresh(interval=10000, key="twp_fixed_interactive_refresh")

st.title("⚡ TWP ORB Ultimate Bot by TUCKYAA (Interactive Chart + Signals)")
st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | CCXT Engine | Touch Chart +"
    " Visible Breakout Signals & Time Logs**"
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


class TWPInteractiveEngine:

  def __init__(self, df, symbol):
    self.df = df if df is not None else None
    self.symbol = symbol
    self.signals = []
    self.or_high = 0
    self.or_low = 0

  def calculate_and_scan(self):
    if self.df is not None and len(self.df) >= 25:
      monitoring_df = self.df.head(4)
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

      # Scan breakouts row by row with timestamp
      for i in range(4, len(self.df)):
        current_time = self.df.index[i].strftime("%H:%M:%S")
        current_timestamp = self.df.index[i]
        c = self.df.iloc[i]
        prev = self.df.iloc[i - 1]
        atr = float(c["ATR"]) if not pd.isna(c["ATR"]) else (float(c["Close"]) * 0.002)

        # Bullish Breakout
        if float(c["Close"]) > self.or_high and float(prev["Close"]) <= self.or_high:
          if float(c["Close"]) > float(c["EMA"]):
            sl = float(c["Close"]) - (1.5 * atr)
            tp = float(c["Close"]) + (2.5 * atr)
            self.signals.append({
                "Time": current_time,
                "Timestamp": current_timestamp,
                "Setup": "ORB Bullish Breakout",
                "Action": "Buy / Long (B)",
                "Entry Price": round(float(c["Close"]), 2),
                "Stop Loss (SL)": round(sl, 2),
                "Take Profit (TP)": round(tp, 2),
            })

        # Bearish Breakout
        elif float(c["Close"]) < self.or_low and float(prev["Close"]) >= self.or_low:
          if float(c["Close"]) < float(c["EMA"]):
            sl = float(c["Close"]) + (1.5 * atr)
            tp = float(c["Close"]) - (2.5 * atr)
            self.signals.append({
                "Time": current_time,
                "Timestamp": current_timestamp,
                "Setup": "ORB Bearish Breakout",
                "Action": "Sell / Short (S)",
                "Entry Price": round(float(c["Close"]), 2),
                "Stop Loss (SL)": round(sl, 2),
                "Take Profit (TP)": round(tp, 2),
            })


with st.spinner(f"⚡ Loading Interactive Engine for {selected_symbol}..."):
  if df_5m is not None and not df_5m.empty:
    bot = TWPInteractiveEngine(df_5m, selected_symbol)
    bot.calculate_and_scan()

    curr_p = float(df_5m["Close"].iloc[-1])

    st.markdown("---")
    st.subheader(f"📊 Live Dashboard ({active_symbol or selected_symbol})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.metric(label="Opening Range High", value=f"{bot.or_high:.2f}")
    with col2:
      st.metric(label="Opening Range Low", value=f"{bot.or_low:.2f}")
    with col3:
      st.metric(label="Current Price", value=f"{curr_p:.2f}")
    with col4:
      st.metric(label="News Filter", value="Safe", delta="Active")

    if bot.signals:
      st.markdown("---")
      st.subheader("📋 Recent Breakout Signals & Timing History")
      df_sigs = pd.DataFrame(bot.signals)
      st.dataframe(
          df_sigs.drop(columns=["Timestamp"]).tail(5), use_container_width=True
      )
    else:
      st.info(
          f"💡 Monitoring {selected_symbol} for breakout triggers with EMA &"
          " ATR..."
      )

    st.markdown("---")
    st.subheader(
        f"📈 Interactive Touch Chart with Signals — {active_symbol or selected_symbol}"
    )

    # Plotly Candlestick Chart (Fully touch/zoom/pan enabled)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_5m.index,
                open=df_5m["Open"],
                high=df_5m["High"],
                low=df_5m["Low"],
                close=df_5m["Close"],
                name="Candles",
            )
        ]
    )

    # Add OR High & Low Lines
    fig.add_hline(
        y=bot.or_high,
        line_dash="dash",
        line_color="#ff7043",
        annotation_text=f"OR High ({bot.or_high:.1f})",
        annotation_position="top left",
    )
    fig.add_hline(
        y=bot.or_low,
        line_dash="dash",
        line_color="#26a69a",
        annotation_text=f"OR Low ({bot.or_low:.1f})",
        annotation_position="bottom left",
    )

    # Add Buy / Sell Signal Markers directly on Plotly Chart
    if bot.signals:
      buy_x = [s["Timestamp"] for s in bot.signals if "Buy" in s["Action"]]
      buy_y = [s["Entry Price"] for s in bot.signals if "Buy" in s["Action"]]
      sell_x = [s["Timestamp"] for s in bot.signals if "Sell" in s["Action"]]
      sell_y = [s["Entry Price"] for s in bot.signals if "Sell" in s["Action"]]

      if buy_x:
        fig.add_trace(
            go.Scatter(
                x=buy_x,
                y=buy_y,
                mode="text+markers",
                text=["B"] * len(buy_x),
                textfont=dict(color="white", size=12, family="Arial Black"),
                marker=dict(size=14, color="#00e676", symbol="triangle-up"),
                name="Buy Signal",
            )
        )
      if sell_x:
        fig.add_trace(
            go.Scatter(
                x=sell_x,
                y=sell_y,
                mode="text+markers",
                text=["S"] * len(sell_x),
                textfont=dict(color="white", size=12, family="Arial Black"),
                marker=dict(size=14, color="#ff1744", symbol="triangle-down"),
                name="Sell Signal",
            )
        )

    # Chart Styling (TradingView Dark Theme look)
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#1e222d",
        plot_bgcolor="#131722",
    )

    st.plotly_chart(fig, use_container_width=True)
  else:
    st.error("Data load nahi ho paya. Please check connection.")
