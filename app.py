from datetime import datetime, time, timedelta, timezone
import sqlite3
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWT Delta Pro by TUCKYAA", page_icon="⚡", layout="wide"
)

# Hide Streamlit Header, Footer, Menu & GitHub Icon via CSS
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------------------------------
# DATABASE INITIALIZATION (SQLite)
# ---------------------------------------------------------
conn = sqlite3.connect("twt_users.db", check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users 
             (username TEXT PRIMARY KEY, password TEXT, status TEXT, role TEXT)""")
# Default Admin Account (Username: tuckyaa, Password: 8002771200)
c.execute(
    "INSERT OR IGNORE INTO users VALUES ('tuckyaa', '8002771200', 'approved',"
    " 'admin')"
)
conn.commit()

# ---------------------------------------------------------
# AUTHENTICATION & ADMIN APPROVAL GATE
# ---------------------------------------------------------
if "user" not in st.session_state:
  st.session_state.user = None
if "role" not in st.session_state:
  st.session_state.role = None


def login_register_page():
  st.title("🔒 TRADE WITH TUCKYAA — VIP Portal")
  st.markdown("Login or Request Access to the Delta Trading Engine.")

  tab1, tab2 = st.tabs(["🔑 Login", "📝 Request Access"])

  with tab1:
    st.subheader("User Login")
    l_user = st.text_input("Username / Mobile Number", key="l_user")
    l_pass = st.text_input("Password", type="password", key="l_pass")
    if st.button("Login to Dashboard"):
      c.execute(
          "SELECT password, status, role FROM users WHERE username=?", (l_user,)
      )
      row = c.fetchone()
      if row:
        db_pass, db_status, db_role = row
        if db_pass == l_pass:
          if db_status == "approved":
            st.session_state.user = l_user
            st.session_state.role = db_role
            st.success("Login successful!")
            st.rerun()
          else:
            st.warning(
                "⏳ Your account is pending Admin approval. Please wait for"
                " Tuckyaa to approve."
            )
        else:
          st.error("❌ Incorrect Password!")
      else:
        st.error("❌ Username not found! Please register first.")

  with tab2:
    st.subheader("New User Registration")
    r_user = st.text_input("Choose Username / Mobile Number", key="r_user")
    r_pass = st.text_input("Choose Password", type="password", key="r_pass")
    if st.button("Submit Access Request"):
      if r_user and r_pass:
        try:
          c.execute(
              "INSERT INTO users VALUES (?, ?, 'pending', 'user')",
              (r_user, r_pass),
          )
          conn.commit()
          st.success(
              "✅ Request submitted successfully! Admin (Tuckyaa) will approve"
              " your account soon."
          )
        except sqlite3.IntegrityError:
          st.error(
              "❌ Username already exists! Try logging in or use another name."
          )
      else:
        st.warning("Please fill in all fields.")


if st.session_state.user is None:
  login_register_page()
  st.stop()

# ---------------------------------------------------------
# ADMIN PANEL (Visible ONLY to Admin 'tuckyaa')
# ---------------------------------------------------------
if st.session_state.role == "admin":
  st.sidebar.markdown("---")
  st.sidebar.markdown("### 👑 Admin Control Panel")
  if st.sidebar.button("Logout Admin"):
    st.session_state.user = None
    st.session_state.role = None
    st.rerun()

  with st.expander("🛠️ Manage User Approvals (Admin Only)"):
    st.subheader("Pending User Requests")
    c.execute("SELECT username FROM users WHERE status='pending'")
    pending_users = c.fetchall()

    if pending_users:
      for p_user in pending_users:
        u_name = p_user[0]
        col_a, col_b = st.columns([2, 1])
        col_a.write(f"User: **{u_name}**")
        if col_b.button("Approve", key=f"app_{u_name}"):
          c.execute(
              "UPDATE users SET status='approved' WHERE username=?", (u_name,)
          )
          conn.commit()
          st.success(f"Approved {u_name}!")
          st.rerun()
    else:
      st.info("No pending user requests.")

    st.markdown("---")
    st.subheader("Approved VIP Users")
    c.execute("SELECT username FROM users WHERE status='approved'")
    approved_users = c.fetchall()
    for a_user in approved_users:
      st.text(f"✅ {a_user[0]}")
else:
  if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.role = None
    st.rerun()

# ---------------------------------------------------------
# TRADING TIMEZONES DICTIONARY (TradingView Standard)
# ---------------------------------------------------------
TRADING_TIMEZONES = {
    "(UTC+5:30) Kolkata / India (IST)": timezone(
        timedelta(hours=5, minutes=30)
    ),
    "(UTC+0:00) London / UTC / GMT": timezone.utc,
    "(UTC-5:00) New York / US East (EST)": timezone(timedelta(hours=-5)),
    "(UTC-8:00) Los Angeles / US West (PST)": timezone(timedelta(hours=-8)),
    "(UTC+8:00) Singapore / Hong Kong (SGT)": timezone(timedelta(hours=8)),
    "(UTC+9:00) Tokyo / Japan (JST)": timezone(timedelta(hours=9)),
    "(UTC+4:00) Dubai / Gulf (GST)": timezone(timedelta(hours=4)),
    "(UTC+11:00) Sydney / Australia (AEDT)": timezone(timedelta(hours=11)),
    "(UTC+1:00) Frankfurt / Europe (CET)": timezone(timedelta(hours=1)),
}

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

# ---------------------------------------------------------
# MAIN TRADING ENGINE & DASHBOARD CONTROL
# ---------------------------------------------------------
count = st_autorefresh(interval=10000, key="twt_delta_refresh")

st.title(
    "⚡ TRADE WITH TUCKYAA (TUCKYAA) — Delta Exchange Multi-Retest Level Engine"
)

# TOP SELECTION BAR FOR TRADING CONTROLS
col_sym, col_tf, col_tz = st.columns([2, 1, 2])

with col_sym:
  selected_symbol = st.selectbox(
      "💎 Select Asset", options=ASSET_LIST, index=5
  )

with col_tf:
  selected_tf = st.selectbox(
      "⏱️ Timeframe", options=["5m", "15m", "30m", "1h"], index=0
  )

with col_tz:
  selected_tz_name = st.selectbox(
      "🌍 Trading Timezone", options=list(TRADING_TIMEZONES.keys()), index=0
  )
  active_tz = TRADING_TIMEZONES[selected_tz_name]

# Sidebar Panel Info
st.sidebar.markdown("### 🤖 TWT Expert Panel")
st.sidebar.markdown("---")
tz_short = selected_tz_name.split(")")[0] + ")"
server_time_target = datetime.now(active_tz)
st.sidebar.text(
    f"Selected Zone Time — {server_time_target.strftime('%H:%M:%S')}"
)
st.sidebar.text("Support Ref (6H)   ———  05:30 IST Candle Low")
st.sidebar.text("Resistance Ref (4H) ———  01:30 IST Candle High")
st.sidebar.text("No-Trade Zone      ———  01:00 ➔ 06:00 IST")
st.sidebar.markdown("---")
st.sidebar.text("Exchange Platform  — Delta Exchange (20 Assets)")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Trading Settings")
st.sidebar.text(f"Selected Zone    : {tz_short}")
st.sidebar.text(f"Scanning TF      : {selected_tf}")
st.sidebar.text("Entry method     : Smart Multi-Touch Retest")
st.sidebar.text("Take profit        : Multi-TP (TP1, TP2, TP3)")
st.sidebar.text("Stop loss          : ATR Managed")

st.markdown(
    f"**Live Feed Active (Refresh Count: {count}) | Asset: {selected_symbol} |"
    f" TF: {selected_tf} | Zone: {tz_short}**"
)


def get_decimals(price):
  if price < 1:
    return 4
  elif price < 10:
    return 3
  else:
    return 2


@st.cache_data(ttl=10)
def fetch_twt_delta_data(symbol, tf):
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
        ohlcv_tf = exchange.fetch_ohlcv(sym, timeframe=tf, limit=100)
        ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=30)
        ohlcv_6h = exchange.fetch_ohlcv(sym, timeframe="6h", limit=30)

        if ohlcv_tf and ohlcv_4h:
          df_chart = pd.DataFrame(
              ohlcv_tf,
              columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
          )
          df_chart["timestamp"] = pd.to_datetime(
              df_chart["timestamp"], unit="ms"
          )
          df_chart = df_chart.set_index("timestamp")

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

          return df_chart, df_4, df_6, symbol
    except Exception:
      continue
  return None, None, None, None


df_chart, df_4h, df_6h, active_symbol = fetch_twt_delta_data(
    selected_symbol, selected_tf
)


class TWTDeltaEngine:

  def __init__(self, df_main, df_4, df_6, symbol, user_tz):
    self.df_main = df_main if df_main is not None else None
    self.df_4 = df_4 if df_4 is not None else None
    self.df_6 = df_6 if df_6 is not None else None
    self.symbol = symbol
    self.user_tz = user_tz
    self.candles = []
    self.signals = []
    self.resistance_lvl = 0
    self.support_lvl = 0

  def calculate_indicators(self):
    IST = timezone(timedelta(hours=5, minutes=30))
    if self.df_main is not None and len(self.df_main) >= 30:
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
          self.support_lvl = float(self.df_6["Low"].iloc[-1])
      elif self.df_4 is not None:
        self.support_lvl = float(self.df_4["Low"].iloc[-1])

      self.df_main["TR"] = np.maximum(
          self.df_main["High"] - self.df_main["Low"],
          np.maximum(
              abs(self.df_main["High"] - self.df_main["Close"].shift(1)),
              abs(self.df_main["Low"] - self.df_main["Close"].shift(1)),
          ),
      )
      self.df_main["ATR"] = self.df_main["TR"].rolling(14).mean()

  def parse_and_scan(self):
    if self.df_main is None or self.df_main.empty:
      return

    IST = timezone(timedelta(hours=5, minutes=30))

    for idx, row in enumerate(self.df_main.iterrows()):
      dt_utc = row[0]
      if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

      # Convert timestamp to selected trading timezone
      dt_user_tz = dt_utc.astimezone(self.user_tz)
      timestamp_str = dt_user_tz.strftime("%H:%M")

      dt_ist = dt_utc.astimezone(IST)

      r = row[1]
      o, h, l, c, v = (
          float(r["Open"]),
          float(r["High"]),
          float(r["Low"]),
          float(r["Close"]),
          float(r["Volume"]),
      )
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)

      self.candles.append({
          "id": idx + 1,
          "time": timestamp_str,
          "dt_ist": dt_ist,
          "dt_user": dt_user_tz,
          "color": "Green" if c >= o else "Red",
          "open": o,
          "high": h,
          "low": l,
          "close": c,
          "volume": v,
          "atr": atr,
      })

    res_touch_count = 0
    sup_touch_count = 0
    last_res_candle_id = -99
    last_sup_candle_id = -99

    for i in range(5, len(self.candles)):
      c = self.candles[i]

      c_time_ist = c["dt_ist"].time()
      if time(1, 0) <= c_time_ist <= time(6, 0):
        continue

      atr = c["atr"]
      dec = get_decimals(c["close"])

      # --- 4H RESISTANCE TOUCH TRACKING ---
      is_near_res = (
          c["high"] >= self.resistance_lvl - (0.8 * atr)
          and c["high"] <= self.resistance_lvl + (0.8 * atr)
      )
      if is_near_res and (i - last_res_candle_id > 5):
        res_touch_count += 1
        last_res_candle_id = i

        if res_touch_count >= 2:
          sl = round(c["high"] + (1.2 * atr), dec)
          tp1 = round(c["close"] - (1.0 * atr), dec)
          tp2 = round(c["close"] - (2.2 * atr), dec)
          tp3 = round(c["close"] - (3.8 * atr), dec)

          if not any(
              s["Setup"] == "TWT Multi-Touch Resistance Sell"
              and abs(s["Candle #"] - c["id"]) < 6
              for s in self.signals
          ):
            self.signals.append({
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "TWT Multi-Touch Resistance Sell",
                "Action": f"Sell / Short (S) [Retest #{res_touch_count}] 🔴",
                "Entry Price": round(c["close"], dec),
                "Stop Loss (SL)": sl,
                "TP 1": tp1,
                "TP 2": tp2,
                "TP 3": tp3,
            })

      # --- 6H SUPPORT TOUCH TRACKING ---
      is_near_sup = (
          c["low"] <= self.support_lvl + (0.8 * atr)
          and c["low"] >= self.support_lvl - (0.8 * atr)
      )
      if is_near_sup and (i - last_sup_candle_id > 5):
        sup_touch_count += 1
        last_sup_candle_id = i

        if sup_touch_count >= 2:
          sl = round(c["low"] - (1.2 * atr), dec)
          tp1 = round(c["close"] + (1.0 * atr), dec)
          tp2 = round(c["close"] + (2.2 * atr), dec)
          tp3 = round(c["close"] + (3.8 * atr), dec)

          if not any(
              s["Setup"] == "TWT Multi-Touch Support Buy"
              and abs(s["Candle #"] - c["id"]) < 6
              for s in self.signals
          ):
            self.signals.append({
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "TWT Multi-Touch Support Buy",
                "Action": f"Buy / Long (B) [Retest #{sup_touch_count}] 🟢",
                "Entry Price": round(c["close"], dec),
                "Stop Loss (SL)": sl,
                "TP 1": tp1,
                "TP 2": tp2,
                "TP 3": tp3,
            })


with st.spinner(
    f"⚡ Loading TWT Delta Engine for {selected_symbol} ({selected_tf})..."
):
  if df_chart is not None and not df_chart.empty:
    bot = TWTDeltaEngine(df_chart, df_4h, df_6h, selected_symbol, active_tz)
    bot.calculate_indicators()
    bot.parse_and_scan()

    curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

    st.markdown("---")
    st.subheader(
        f"📊 Delta Exchange Live Dashboard ({selected_symbol} — {selected_tf})"
    )

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
          label="Trading Zone",
          value=tz_short,
          delta=f"Timeframe: {selected_tf}",
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
            f"🕒 **Candle Time ({tz_short}):** `{signal_time}` (Candle"
            f" #{latest_sig['Candle #']})\n"
            f"- **Entry Price:** `{entry_p}`\n"
            f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
            f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:**"
            f" `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`\n\n"
            "📌 *Instructions:* Support multi-touch retest confirmed!"
        )
      else:
        st.error(
            f"🔴 **TWT DELTA SETUP: {latest_sig['Action']}**\n\n"
            f"🕒 **Candle Time ({tz_short}):** `{signal_time}` (Candle"
            f" #{latest_sig['Candle #']})\n"
            f"- **Entry Price:** `{entry_p}`\n"
            f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
            f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:**"
            f" `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`\n\n"
            "📌 *Instructions:* Resistance multi-touch retest confirmed!"
        )
    else:
      st.warning(
          "🟡 **STATUS: WAITING FOR 2ND / 3RD TOUCH RETEST**\n\n"
          "📌 *Instructions:* Monitoring price action near 4H Resistance and 6H"
          " Support."
      )

    if bot.signals:
      st.markdown("---")
      st.subheader(f"📋 Verified TWT Signal History ({tz_short} Time)")
      st.dataframe(pd.DataFrame(bot.signals[-5:]), width="stretch")

    st.markdown("---")
    st.subheader(
        f"📈 {selected_symbol} {selected_tf} Real-Time Chart ({tz_short})"
    )
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
        f"{selected_symbol} {selected_tf} — TWT Delta Engine ({tz_short})",
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
