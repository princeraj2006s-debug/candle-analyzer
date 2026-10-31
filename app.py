import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="CANDLESENSE Live Pro", page_icon="📈", layout="wide"
)

st.title("🕯️ CANDLESENSE Pro: Real-Time Gold & Forex Signal Bot")
st.markdown(
    "**Live Market Candle Analysis, Wick Scoring, Entry, Stop-Loss & Target"
    " Calculator**"
)

# Sidebar / Top Controls for Asset & Timeframe
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
  asset_choice = st.selectbox(
      "Select Asset / Pair",
      options=[
          "Gold (XAUUSD)",
          "EUR/USD",
          "GBP/USD",
          "USD/JPY",
          "Bitcoin (BTCUSD)",
      ],
      index=0,
  )
with col_s2:
  tf_choice = st.selectbox(
      "Select Timeframe", options=["1m", "5m", "15m", "1h", "1d"], index=1
  )
with col_s3:
  mode_choice = st.selectbox(
      "Analysis Mode", options=["Live Market Feed", "Upload Screenshot"]
  )

# Ticker symbol mapping
symbol_map = {
    "Gold (XAUUSD)": "GC=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "Bitcoin (BTCUSD)": "BTC-USD",
}
selected_ticker = symbol_map[asset_choice]

# Interval mapping
interval_map = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "60m",
    "1d": "1d",
}
selected_interval = interval_map[tf_choice]


@st.cache_data(ttl=30)
def fetch_live_candles(ticker, interval):
  try:
    period = (
        "1d"
        if interval in ["1m", "5m", "15m"]
        else ("5d" if interval == "60m" else "1mo")
    )
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
      return None
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    return df.tail(35)  # Last 35 candles for accurate pattern scanning
  except Exception as e:
    return None


class LiveCandleSenseAnalyzer:

  def __init__(self, df):
    self.df = df
    self.candles = []
    self.analysis_report = []
    self.buyer_score = 0.0
    self.seller_score = 0.0
    self.full_points = 0
    self.half_points = 0
    self.untouched = 0
    self.confirmed_breaks = 0

  def parse_candles(self):
    for idx, row in enumerate(self.df.iterrows()):
      r = row[1]
      o, h, l, c = (
          float(r["Open"]),
          float(r["High"]),
          float(r["Low"]),
          float(r["Close"]),
      )
      color = "Green" if c >= o else "Red"
      body_top = max(o, c)
      body_bottom = min(o, c)

      self.candles.append({
          "id": idx + 1,
          "color": color,
          "open": o,
          "high": h,
          "low": l,
          "close": c,
          "body_top": body_top,
          "body_bottom": body_bottom,
          "upper_wick_tip": h,
          "lower_wick_tip": l,
          "upper_wick_detected": h > body_top,
          "lower_wick_detected": l < body_bottom,
      })

  def calculate_rules(self):
    for i in range(len(self.candles)):
      curr = self.candles[i]
      if i == len(self.candles) - 1:
        self.analysis_report.append({
            "Candle #": curr["id"],
            "Colour": curr["color"],
            "Close": f"{curr['close']:.2f}",
            "Next Candle Result": "Latest Live Candle (Current)",
            "Points": "0",
        })
        continue

      nxt = self.candles[i + 1]
      result_text = ""
      points_awarded = "0"

      # 1. Green followed by Green (Buyer scoring)
      if curr["color"] == "Green" and nxt["color"] == "Green":
        ref_level = curr["upper_wick_tip"]
        if nxt["body_top"] < ref_level:
          self.buyer_score += 1.0
          self.full_points += 1
          self.confirmed_breaks += 1
          result_text = "Upper wick crossed (Proper Break)"
          points_awarded = "Buyer +1"
        elif abs(nxt["body_top"] - ref_level) <= (ref_level * 0.0005):
          self.buyer_score += 0.5
          self.half_points += 1
          result_text = "Upper wick touched (No break)"
          points_awarded = "Buyer +½"
        else:
          self.untouched += 1
          result_text = "Level untouched"
          points_awarded = "0"

      # 2. Red followed by Red (Seller scoring)
      elif curr["color"] == "Red" and nxt["color"] == "Red":
        ref_level = curr["lower_wick_tip"]
        if nxt["body_bottom"] > ref_level:
          self.seller_score += 1.0
          self.full_points += 1
          self.confirmed_breaks += 1
          result_text = "Lower wick crossed (Proper Break)"
          points_awarded = "Seller +1"
        elif abs(nxt["body_bottom"] - ref_level) <= (ref_level * 0.0005):
          self.seller_score += 0.5
          self.half_points += 1
          result_text = "Lower wick touched (No break)"
          points_awarded = "Seller +½"
        else:
          self.untouched += 1
          result_text = "Level untouched"
          points_awarded = "0"

      # 3. Green followed by Red
      elif curr["color"] == "Green" and nxt["color"] == "Red":
        ref_level = curr["lower_wick_tip"]
        if nxt["body_bottom"] > ref_level:
          self.seller_score += 1.0
          self.full_points += 1
          self.confirmed_breaks += 1
          result_text = "Lower wick crossed (Downward break)"
          points_awarded = "Seller +1"
        elif abs(nxt["body_bottom"] - ref_level) <= (ref_level * 0.0005):
          self.buyer_score += 0.5
          self.half_points += 1
          result_text = "Lower wick touched"
          points_awarded = "Buyer +½"
        else:
          self.untouched += 1
          result_text = "Level untouched"
          points_awarded = "0"

      # 4. Red followed by Green
      elif curr["color"] == "Red" and nxt["color"] == "Green":
        ref_level = curr["upper_wick_tip"]
        if nxt["body_top"] < ref_level:
          self.buyer_score += 1.0
          self.full_points += 1
          self.confirmed_breaks += 1
          result_text = "Upper wick crossed (Upward break)"
          points_awarded = "Buyer +1"
        elif abs(nxt["body_top"] - ref_level) <= (ref_level * 0.0005):
          self.seller_score += 0.5
          self.half_points += 1
          result_text = "Upper wick touched"
          points_awarded = "Seller +½"
        else:
          self.untouched += 1
          result_text = "Level untouched"
          points_awarded = "0"

      self.analysis_report.append({
          "Candle #": curr["id"],
          "Colour": curr["color"],
          "Close": f"{curr['close']:.2f}",
          "Next Candle Result": result_text,
          "Points": points_awarded,
      })


if mode_choice == "Live Market Feed":
  st.subheader(
      f"📡 Real-Time Data Feed: {asset_choice} ({tf_choice} Timeframe)"
  )

  if st.button(
      "Fetch Live Market & Generate Signal",
      type="primary",
      use_container_width=True,
  ):
    with st.spinner(
        f"Connecting to live feed for {asset_choice} and running CANDLESENSE"
        " rules..."
    ):
      df_live = fetch_live_candles(selected_ticker, selected_interval)

      if df_live is not None and not df_live.empty:
        analyzer = LiveCandleSenseAnalyzer(df_live)
        analyzer.parse_candles()
        analyzer.calculate_rules()

        st.success(
            f"Successfully analyzed {len(analyzer.candles)} live candles!"
        )

        # Scorecard
        st.markdown("---")
        st.subheader("📊 Live Scorecard & Market Bias")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Total Buyer Score", f"{analyzer.buyer_score:.1f} Points"
        )
        c2.metric(
            "Total Seller Score", f"{analyzer.seller_score:.1f} Points"
        )
        diff = analyzer.buyer_score - analyzer.seller_score
        bias = (
            "🚀 Bullish Momentum"
            if diff > 0
            else ("🔻 Bearish Momentum" if diff < 0 else "⚖️ Neutral / Mixed")
        )
        c3.metric("Point Balance & Bias", f"{diff:+.1f}", delta=bias)

        # Safe Entry, SL, TP Calculation
        last_c = analyzer.candles[-1]
        is_bullish = analyzer.buyer_score >= analyzer.seller_score

        if is_bullish:
          entry_price = last_c["close"]
          stop_loss = last_c["low"] - (
              (last_c["high"] - last_c["low"]) * 0.2
          )  # Safe buffer below lower wick
          risk = entry_price - stop_loss
          target_tp = entry_price + (risk * 2.0)  # 1:2 Risk-Reward ratio
          signal_type = "BUY / LONG"
        else:
          entry_price = last_c["close"]
          stop_loss = last_c["high"] + (
              (last_c["high"] - last_c["low"]) * 0.2
          )  # Safe buffer above upper wick
          risk = stop_loss - entry_price
          target_tp = entry_price - (risk * 2.0)  # 1:2 Risk-Reward ratio
          signal_type = "SELL / SHORT"

        st.markdown("---")
        st.subheader("🎯 Safe Trade Setup (Entry, Stop-Loss & Target)")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Signal Recommendation", signal_type)
        t2.metric("Suggested Entry", f"{entry_price:.2f}")
        t3.metric("Stop-Loss (SL)", f"{stop_loss:.2f}")
        t4.metric("Take-Profit (TP)", f"{target_tp:.2f}")

        # Detailed Report Table
        st.markdown("---")
        st.subheader("📋 Candle-by-Candle Live Report")
        st.dataframe(
            pd.DataFrame(analyzer.analysis_report), use_container_width=True
        )

        # Visual Chart Plot
        st.markdown("---")
        st.subheader("📈 Live Chart Price & Wick Structure")
        fig, ax = plt.subplots(figsize=(10, 4))
        for c in analyzer.candles:
          col = "green" if c["color"] == "Green" else "red"
          ax.bar(
              c["id"],
              abs(c["close"] - c["open"])
              if abs(c["close"] - c["open"]) > 0
              else 0.001,
              bottom=min(c["open"], c["close"]),
              color=col,
              width=0.6,
          )
          ax.vlines(c["id"], c["low"], c["high"], color="black", alpha=0.6)
        ax.set_title(f"Live Market Chart for {asset_choice}")
        st.pyplot(fig)

      else:
        st.error(
            "Could not fetch live market data right now. Please check your"
            " internet connection or try again."
        )

else:
  # Fallback to Screenshot Upload Mode if preferred
  uploaded_file = st.file_uploader(
      "Upload Chart Screenshot", type=["png", "jpg", "jpeg"]
  )
  if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Screenshot Analysis Ready",
        use_container_width=True,
    )
    st.info(
        "Switch to 'Live Market Feed' mode above for automated real-time Gold &"
        " Forex analysis!"
    )
