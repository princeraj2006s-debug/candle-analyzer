from concurrent.futures import ThreadPoolExecutor, as_completed
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
# DATABASE INITIALIZATION (SQLite Authentication)
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
# AUTHENTICATION & ACCESS GATE
# ---------------------------------------------------------
if "user" not in st.session_state:
  st.session_state.user = None
if "role" not in st.session_state:
  st.session_state.role = None


def login_register_page():
  st.title("🔒 TRADE WITH TUCKYAA — VIP Portal")
  st.markdown("Login or Request Access to the Professional Delta Trading Engine.")

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
          if db_status == "blocked":
            st.error(
                "❌ Your service has been blocked by Admin (Tuckyaa). Contact"
                " support."
            )
            return

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

  with st.expander("🛠️ Manage User Approvals & Services (Admin Only)"):
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
    st.subheader("Manage Active Users (Block/Unblock Service)")
    c.execute("SELECT username, status FROM users WHERE role!='admin'")
    all_users = c.fetchall()

    if all_users:
      for u_name, u_status in all_users:
        col1, col2 = st.columns([2, 1])
        col1.write(f"**{u_name}** (Status: `{u_status}`)")
        if u_status == "approved":
          if col2.button("Block Service", key=f"block_{u_name}"):
            c.execute(
                "UPDATE users SET status='blocked' WHERE username=?", (u_name,)
            )
            conn.commit()
            st.warning(f"Blocked {u_name}!")
            st.rerun()
        else:
          if col2.button("Unblock", key=f"unblock_{u_name}"):
            c.execute(
                "UPDATE users SET status='approved' WHERE username=?", (u_name,)
            )
            conn.commit()
            st.success(f"Unblocked {u_name}!")
            st.rerun()
    else:
      st.info("No regular users registered yet.")
else:
  if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.role = None
    st.rerun()

# ---------------------------------------------------------
# GLOBAL TRADING TIMEZONES & COMPLETE 20+ ASSETS LIST
# ---------------------------------------------------------
TRADING_TIMEZONES = {
    "(UTC+5:30) Kolkata / India (IST)": timezone(
        timedelta(hours=5, minutes=30)
    ),
    "(UTC+1:00) Berlin / Paris / CET": timezone(timedelta(hours=1)),
    "(UTC+0:00) London / UTC / GMT": timezone.utc,
    "(UTC-5:00) New York / US East (EST)": timezone(timedelta(hours=-5)),
    "(UTC+8:00) Singapore / Hong Kong (SGT)": timezone(timedelta(hours=8)),
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

# Safe Auto-refresh every 30 seconds
count = st_autorefresh(interval=30000, key="twt_delta_refresh")

st.title(
    "⚡ TRADE WITH TUCKYAA (TUCKYAA) — Sequential Exhaustion Engine"
)

col_tf, col_tz = st.columns([1, 2])
with col_tf:
  selected_tf = st.selectbox(
      "⏱️ Timeframe", options=["5m", "15m", "30m", "1h"], index=0
  )
with col_tz:
  selected_tz_name = st.selectbox(
      "🌍 Trading Timezone", options=list(TRADING_TIMEZONES.keys()), index=0
  )
  active_tz = TRADING_TIMEZONES[selected_tz_name]

tz_short = selected_tz_name.split(")")[0] + ")"

# Sidebar Panel
st.sidebar.markdown("### 🤖 TWT Ultimate Price Action Panel")
st.sidebar.markdown("---")
server_time_target = datetime.now(active_tz)
st.sidebar.text(
    f"Server time          ———  {server_time_target.strftime('%H:%M')}"
)
st.sidebar.text("Support Ref (6H)   ———  05:30 IST Candle Low")
st.sidebar.text("Resistance Ref (4H) ———  01:30 IST Candle High")
st.sidebar.text("No-Trade Zone      ———  01:00 ➔ 06:00 IST")
st.sidebar.markdown("---")
st.sidebar.text("Exchange Platform  — Delta Exchange (20 Assets)")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Trading Strategy")
st.sidebar.text("Logic            : 5-6 Red Breakdown + Strong Green Rebound")
st.sidebar.text("Take profit      : Multi-TP (TP1, TP2, TP3)")
st.sidebar.text("Stop loss        : ATR Managed")


def get_decimals(price):
  if price < 1:
    return 4
  elif price < 10:
    return 3
  else:
    return 2


@st.cache_data(ttl=30)
def fetch_twt_delta_data(symbol, tf):
  exchanges_to_try = [
      (ccxt.binance, symbol),
      (ccxt.bybit, symbol),
      (ccxt.kucoin, symbol),
  ]

  for ex_class, sym in exchanges_to_try:
    try:
      exchange = ex_class({"enableRateLimit": True, "timeout": 15000})
      exchange.load_markets()
      if sym in exchange.markets:
        ohlcv_tf = exchange.fetch_ohlcv(sym, timeframe=tf, limit=100)
        ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=25)
        ohlcv_6h = exchange.fetch_ohlcv(sym, timeframe="6h", limit=25)

        if ohlcv_tf and ohlcv_4h:
          df_5 = pd.DataFrame(
              ohlcv_tf,
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
          return df_5, df_4, df_6
    except Exception:
      continue
  return None, None, None


class TWTSequentialEngine:

  def __init__(self, df_5, df_4, df_6, symbol, user_tz):
    self.df_5 = df_5
    self.df_4 = df_4
    self.df_6 = df_6
    self.symbol = symbol
    self.user_tz = user_tz
    self.candles = []
    self.signals = []
    self.resistance_lvl = 0
    self.support_lvl = 0

  def calculate_and_scan(self):
    if self.df_5 is None or self.df_5.empty:
      return

    IST = timezone(timedelta(hours=5, minutes=30))
    current_utc_time = datetime.now(timezone.utc)

    # 1. Calculate Key Levels
    if self.df_4 is not None and not self.df_4.empty:
      idx_utc = (
          self.df_4.index.tz_localize("UTC")
          if self.df_4.index.tz is None
          else self.df_4.index
      )
      df_4_ist = idx_utc.tz_convert(IST)
      filtered_4h = self.df_4[df_4_ist.hour == 1]
      self.resistance_lvl = (
          float(filtered_4h["High"].iloc[-1])
          if not filtered_4h.empty
          else float(self.df_4["High"].iloc[-1])
      )

    if self.df_6 is not None and not self.df_6.empty:
      idx_utc_6 = (
          self.df_6.index.tz_localize("UTC")
          if self.df_6.index.tz is None
          else self.df_6.index
      )
      df_6_ist = idx_utc_6.tz_convert(IST)
      filtered_6h = self.df_6[df_6_ist.hour == 5]
      self.support_lvl = (
          float(filtered_6h["Low"].iloc[-1])
          if not filtered_6h.empty
          else float(self.df_6["Low"].iloc[-1])
      )
    elif self.df_4 is not None:
      self.support_lvl = float(self.df_4["Low"].iloc[-1])

    # 2. Indicators Calculation
    self.df_5["TR"] = np.maximum(
        self.df_5["High"] - self.df_5["Low"],
        np.maximum(
            abs(self.df_5["High"] - self.df_5["Close"].shift(1)),
            abs(self.df_5["Low"] - self.df_5["Close"].shift(1)),
        ),
    )
    self.df_5["ATR"] = self.df_5["TR"].rolling(14).mean()

    for idx, row in enumerate(self.df_5.iterrows()):
      dt_utc = row[0]
      if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
      dt_ist = dt_utc.astimezone(IST)
      dt_user_tz = dt_utc.astimezone(self.user_tz)
      timestamp_str = dt_user_tz.strftime("%H:%M")

      r = row[1]
      o, h, l, c, v = (
          float(r["Open"]),
          float(r["High"]),
          float(r["Low"]),
          float(r["Close"]),
          float(r["Volume"]),
      )
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)

      age_minutes = (current_utc_time - dt_utc).total_seconds() / 60.0
      is_within_1hr = age_minutes <= 60.0

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
          "is_within_1hr": is_within_1hr,
      })

    cooldown = 0

    for i in range(7, len(self.candles)):
      if cooldown > 0:
        cooldown -= 1
        continue

      c = self.candles[i]
      if time(1, 0) <= c["dt_ist"].time() <= time(6, 0):
        continue

      atr = c["atr"]
      dec = get_decimals(c["close"])

      # --- BUY / LONG SETUP (5-6 Red Breakdown + Strong Green Rebound) ---
      # Check if price went below support and preceding 5 to 6 candles were predominantly Red (downward sequence)
      recent_slice = self.candles[i - 7 : i]
      below_support = any(x["low"] < self.support_lvl for x in recent_slice)
      red_count = sum(1 for x in recent_slice[:-1] if x["color"] == "Red")

      # If there were at least 5 red candles before this, and current candle is a strong Green reversal
      if below_support and red_count >= 5 and c["color"] == "Green" and c["close"] > c["open"]:
        sl = round(c["low"] - (1.2 * atr), dec)
        tp1 = round(c["close"] + (1.2 * atr), dec)
        tp2 = round(c["close"] + (2.5 * atr), dec)
        tp3 = round(c["close"] + (4.0 * atr), dec)

        self.signals.append({
            "Asset": self.symbol,
            "Time": c["time"],
            "Candle #": c["id"],
            "Setup": "5-6 Red Breakdown + Strong Green Rebound Long",
            "Action": "Buy / Long (B) [Exhaustion Rebound] 🟢",
            "Entry Price": round(c["close"], dec),
            "Stop Loss (SL)": sl,
            "TP 1": tp1,
            "TP 2": tp2,
            "TP 3": tp3,
            "Is_Within_1hr": c["is_within_1hr"],
        })
        cooldown = 15  # Cooldown to prevent multiple entries in the same zone
        continue

      # --- SELL / SHORT SETUP (Resistance Rejection) ---
      recent_slice_res = self.candles[i - 6 : i]
      above_resistance = any(x["high"] >= self.resistance_lvl - (0.4 * atr) for x in recent_slice_res)
      green_count = sum(1 for x in recent_slice_res[:-1] if x["color"] == "Green")

      if above_resistance and green_count >= 4 and c["color"] == "Red" and c["close"] < c["open"]:
        sl = round(self.resistance_lvl + (1.2 * atr), dec)
        tp1 = round(c["close"] - (1.2 * atr), dec)
        tp2 = round(c["close"] - (2.5 * atr), dec)
        tp3 = round(c["close"] - (3.8 * atr), dec)

        self.signals.append({
            "Asset": self.symbol,
            "Time": c["time"],
            "Candle #": c["id"],
            "Setup": "Resistance Exhaustion Rejection Short",
            "Action": "Sell / Short (S) [Exhaustion Rejection] 🔴",
            "Entry Price": round(c["close"], dec),
            "Stop Loss (SL)": sl,
            "TP 1": tp1,
            "TP 2": tp2,
            "TP 3": tp3,
            "Is_Within_1hr": c["is_within_1hr"],
        })
        cooldown = 15
        continue


# ---------------------------------------------------------
# MASTER SCANNER (Global Feed for Top Dashboard)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (Last 1 Hour)")

master_signals = []
asset_engines = {}


def scan_single_asset(sym):
  df_5, df_4, df_6 = fetch_twt_delta_data(sym, selected_tf)
  if df_5 is not None and not df_5.empty:
    engine = TWTSequentialEngine(df_5, df_4, df_6, sym, active_tz)
    engine.calculate_and_scan()
    recent_sigs = [s for s in engine.signals if s.get("Is_Within_1hr", False)]
    latest = recent_sigs[-1] if recent_sigs else None
    return sym, engine, latest
  return sym, None, None


with st.spinner("⚡ Scanning all 20 assets for sequential exhaustion setups..."):
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {
        executor.submit(scan_single_asset, sym): sym for sym in ASSET_LIST
    }
    for future in as_completed(futures):
      sym, engine, latest = future.result()
      if engine:
        asset_engines[sym] = engine
      if latest:
        master_signals.append(latest)

if master_signals:
  st.success(f"✅ Found {len(master_signals)} active sequential setup(s)!")
  df_display = pd.DataFrame(master_signals).drop(
      columns=["Is_Within_1hr"], errors="ignore"
  )
  st.dataframe(df_display, width="stretch")
else:
  st.info(
      "ℹ️ No active sequential signals in the last 1 hour. Waiting for 5-6 red"
      " candle breakdown sequence..."
  )

# ---------------------------------------------------------
# MANUAL ASSET SELECTOR & CHART VIEWER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Manual Asset Inspector & Chart")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart & Complete History",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df_5, df_4, df_6 = fetch_twt_delta_data(selected_symbol, selected_tf)
  bot = TWTSequentialEngine(df_5, df_4, df_6, selected_symbol, active_tz)
  bot.calculate_and_scan()

if bot.df_5 is not None and not bot.df_5.empty:
  curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

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
    st.metric(label="Timezone", value=tz_short, delta="Sequential Engine Active")

  if bot.signals:
    latest_sig = bot.signals[-1]
    is_buy = "Buy" in latest_sig["Action"]
    entry_p = latest_sig["Entry Price"]
    signal_time = latest_sig["Time"]

    st.warning("⚠️ **Don't use after 40 minutes!**")

    if is_buy:
      st.success(
          f"🟢 **TWT SEQUENTIAL SETUP: {latest_sig['Action']}**\n\n"
          f"🕒 **Candle Time ({tz_short}):** `{signal_time}` (Candle"
          f" #{latest_sig['Candle #']})\n"
          f"- **Entry Price:** `{entry_p}`\n"
          f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
          f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:**"
          f" `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`\n\n"
          "📌 *Instructions:* 5-6 Red candles breakdown -> Strong Green Rebound"
          " confirmed!"
      )
    else:
      st.error(
          f"🔴 **TWT SEQUENTIAL SETUP: {latest_sig['Action']}**\n\n"
          f"🕒 **Candle Time ({tz_short}):** `{signal_time}` (Candle"
          f" #{latest_sig['Candle #']})\n"
          f"- **Entry Price:** `{entry_p}`\n"
          f"- **Stop Loss (SL):** `{latest_sig['Stop Loss (SL)']}`\n"
          f"- **TP 1:** `{latest_sig['TP 1']}` | **TP 2:**"
          f" `{latest_sig['TP 2']}` | **TP 3:** `{latest_sig['TP 3']}`\n\n"
          "📌 *Instructions:* Exhaustion rejection near resistance confirmed!"
      )
  else:
    st.warning(
        "🟡 **STATUS: WAITING FOR SEQUENTIAL SETUP**\n\n"
        "📌 *Instructions:* Monitoring support breakdown exhaustion sequences."
    )

  # Complete Signal History
  if bot.signals:
    st.markdown("---")
    st.subheader(
        f"📋 Complete Signal History for {selected_symbol} (Exhaustion Engine)"
    )
    df_history = pd.DataFrame(bot.signals).drop(
        columns=["Is_Within_1hr"], errors="ignore"
    )
    st.dataframe(df_history, width="stretch")

  # Chart Rendering
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

    ax.bar(c["id"], plot_height, bottom=plot_bottom, color=col, width=0.7)
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

  for sig in bot.signals:
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
      f"{selected_symbol} 5m — TRADE WITH TUCKYAA Sequential Engine ({tz_short})",
      color="white",
      fontsize=14,
  )
  ax.set_facecolor("#131722")
  fig.patch.set_facecolor("#1e222d")
  ax.tick_params(colors="white")
  ax.xaxis.label.set_color("white")
  ax.yaxis.label.set_color("white")
  for spine in ax.spines.values():
    spine.set_color("#363a45")
  ax.grid(color="#2a2e39", linestyle="--", linewidth=0.5)

  st.pyplot(fig)
else:
  st.error("Market data load failed. Please check internet connection.")
