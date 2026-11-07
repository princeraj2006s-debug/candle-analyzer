from datetime import datetime, time, timedelta, timezone
import sqlite3
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TWT Delta Pro Master Scanner by TUCKYAA", page_icon="⚡", layout="wide"
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
# DATABASE INITIALIZATION (SQLite with Join Date & Status)
# ---------------------------------------------------------
conn = sqlite3.connect("twt_users.db", check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users 
             (username TEXT PRIMARY KEY, password TEXT, status TEXT, role TEXT, join_date TEXT)""")

# Default Admin Account (Username: tuckyaa, Password: 8002771200)
current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
c.execute(
    "INSERT OR IGNORE INTO users VALUES ('tuckyaa', '8002771200', 'approved',"
    f" 'admin', '{current_date_str}')"
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
  st.markdown(
      "Login or Request Access to the Delta Trading Engine (30-Day Trial)."
  )

  tab1, tab2 = st.tabs(["🔑 Login", "📝 Request Access"])

  with tab1:
    st.subheader("User Login")
    l_user = st.text_input("Username / Mobile Number", key="l_user")
    l_pass = st.text_input("Password", type="password", key="l_pass")
    if st.button("Login to Dashboard"):
      c.execute(
          "SELECT password, status, role, join_date FROM users WHERE"
          " username=?",
          (l_user,),
      )
      row = c.fetchone()
      if row:
        db_pass, db_status, db_role, db_join_date = row
        if db_pass == l_pass:
          if db_status == "blocked":
            st.error(
                "❌ Your service has been blocked by Admin (Tuckyaa). Contact"
                " support."
            )
            return

          if db_role != "admin" and db_join_date:
            try:
              join_dt = datetime.strptime(db_join_date, "%Y-%m-%d %H:%M:%S")
              days_passed = (datetime.now() - join_dt).days
              if days_passed > 30:
                st.error(
                    "⏳ Your 30-day access period has expired! Please renew"
                    " your subscription with Tuckyaa."
                )
                return
            except Exception:
              pass

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
    st.subheader("New User Registration (30-Day Access)")
    r_user = st.text_input("Choose Username / Mobile Number", key="r_user")
    r_pass = st.text_input("Choose Password", type="password", key="r_pass")
    if st.button("Submit Access Request"):
      if r_user and r_pass:
        try:
          reg_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
          c.execute(
              "INSERT INTO users VALUES (?, ?, 'pending', 'user', ?)",
              (r_user, r_pass, reg_date_str),
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
    st.subheader("Manage Active VIP Users (Block/Unblock Service)")
    c.execute(
        "SELECT username, status, join_date FROM users WHERE role!='admin'"
    )
    all_vips = c.fetchall()

    if all_vips:
      for v_user, v_status, v_join in all_vips:
        col1, col2, col3 = st.columns([2, 1, 1])
        col1.write(f"**{v_user}** (Status: `{v_status}`)")
        try:
          join_dt = datetime.strptime(v_join, "%Y-%m-%d %H:%M:%S")
          days_left = 30 - (datetime.now() - join_dt).days
          col2.text(
              f"📅 {max(0, days_left)} days left"
              if days_left >= 0
              else "Expired"
          )
        except Exception:
          col2.text("N/A")

        if v_status == "approved":
          if col3.button("Block Service", key=f"block_{v_user}"):
            c.execute(
                "UPDATE users SET status='blocked' WHERE username=?", (v_user,)
            )
            conn.commit()
            st.warning(f"Blocked {v_user}!")
            st.rerun()
        else:
          if col3.button("Unblock", key=f"unblock_{v_user}"):
            c.execute(
                "UPDATE users SET status='approved' WHERE username=?", (v_user,)
            )
            conn.commit()
            st.success(f"Unblocked {v_user}!")
            st.rerun()
    else:
      st.info("No regular users registered yet.")
else:
  if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.session_state.role = None
    st.rerun()

# ---------------------------------------------------------
# TRADING CONFIGURATIONS & ASSET LIST
# ---------------------------------------------------------
TRADING_TIMEZONES = {
    "(UTC+5:30) Kolkata / India (IST)": timezone(
        timedelta(hours=5, minutes=30)
    ),
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

count = st_autorefresh(interval=15000, key="twt_delta_refresh")

st.title(
    "⚡ TRADE WITH TUCKYAA (TUCKYAA) — Multi-Asset Master Scanner & Delta Engine"
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


def get_decimals(price):
  if price < 1:
    return 4
  elif price < 10:
    return 3
  else:
    return 2


@st.cache_data(ttl=15)
def fetch_single_asset_data(symbol, tf):
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
        ohlcv_tf = exchange.fetch_ohlcv(sym, timeframe=tf, limit=80)
        ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=25)
        ohlcv_6h = exchange.fetch_ohlcv(sym, timeframe="6h", limit=25)
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
          return df_chart, df_4, df_6
    except Exception:
      continue
  return None, None, None


# ---------------------------------------------------------
# TWT ENGINE LOGIC CLASS
# ---------------------------------------------------------
class TWTDeltaEngine:

  def __init__(self, df_main, df_4, df_6, symbol, user_tz):
    self.df_main = df_main
    self.df_4 = df_4
    self.df_6 = df_6
    self.symbol = symbol
    self.user_tz = user_tz
    self.candles = []
    self.signals = []
    self.resistance_lvl = 0
    self.support_lvl = 0

  def calculate_and_scan(self):
    if self.df_main is None or self.df_main.empty:
      return

    IST = timezone(timedelta(hours=5, minutes=30))
    current_utc_time = datetime.now(timezone.utc)

    # 4H Resistance (01:30 IST) & 6H Support (05:30 IST)
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

    # Indicators
    self.df_main["TR"] = np.maximum(
        self.df_main["High"] - self.df_main["Low"],
        np.maximum(
            abs(self.df_main["High"] - self.df_main["Close"].shift(1)),
            abs(self.df_main["Low"] - self.df_main["Close"].shift(1)),
        ),
    )
    self.df_main["ATR"] = self.df_main["TR"].rolling(14).mean()
    self.df_main["Vol_SMA"] = self.df_main["Volume"].rolling(14).mean()

    for idx, row in enumerate(self.df_main.iterrows()):
      dt_utc = row[0]
      if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
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
      vol_sma = float(r["Vol_SMA"]) if not pd.isna(r["Vol_SMA"]) else v
      is_expired = (current_utc_time - dt_utc).total_seconds() / 60.0 > 45

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
          "vol_sma": vol_sma,
          "atr": atr,
          "is_expired": is_expired,
      })

    res_touch_count = 0
    sup_touch_count = 0
    last_res_idx = -99
    last_sup_idx = -99

    for i in range(5, len(self.candles)):
      c = self.candles[i]
      if time(1, 0) <= c["dt_ist"].time() <= time(6, 0):
        continue

      atr = c["atr"]
      dec = get_decimals(c["close"])
      status_suffix = " ⌛ [EXPIRED]" if c["is_expired"] else ""

      is_support_breakdown = (
          c["close"] < self.support_lvl
          and c["close"] < c["open"]
          and c["volume"] > (c["vol_sma"] * 1.5)
      )
      is_resistance_breakout = (
          c["close"] > self.resistance_lvl
          and c["close"] > c["open"]
          and c["volume"] > (c["vol_sma"] * 1.5)
      )

      if is_support_breakdown:
        sup_touch_count = 0
      if is_resistance_breakout:
        res_touch_count = 0

      # Resistance Touch (Sell)
      is_near_res = (
          c["high"] >= self.resistance_lvl - (0.8 * atr)
          and c["high"] <= self.resistance_lvl + (0.8 * atr)
      )
      if is_near_res and not is_resistance_breakout and (i - last_res_idx > 5):
        res_touch_count += 1
        last_res_idx = i
        if res_touch_count >= 2 and c["close"] < c["open"]:
          sl = round(c["high"] + (1.2 * atr), dec)
          tp1 = round(c["close"] - (1.0 * atr), dec)
          tp2 = round(c["close"] - (2.2 * atr), dec)
          tp3 = round(c["close"] - (3.8 * atr), dec)
          self.signals.append({
              "Asset": self.symbol,
              "Time": c["time"],
              "Candle #": c["id"],
              "Setup": "Resistance Sell",
              "Action": (
                  f"🔴 Sell (Retest #{res_touch_count}){status_suffix}"
              ),
              "Entry Price": round(c["close"], dec),
              "Stop Loss (SL)": sl,
              "TP 1": tp1,
              "TP 2": tp2,
              "TP 3": tp3,
              "Expired": c["is_expired"],
          })

      # Support Touch (Buy)
      is_near_sup = (
          c["low"] <= self.support_lvl + (0.8 * atr)
          and c["low"] >= self.support_lvl - (0.8 * atr)
      )
      if is_near_sup and not is_support_breakdown and (i - last_sup_idx > 5):
        sup_touch_count += 1
        last_sup_idx = i
        if sup_touch_count >= 2 and c["close"] > c["open"]:
          sl = round(c["low"] - (1.2 * atr), dec)
          tp1 = round(c["close"] + (1.0 * atr), dec)
          tp2 = round(c["close"] + (2.2 * atr), dec)
          tp3 = round(c["close"] + (3.8 * atr), dec)
          self.signals.append({
              "Asset": self.symbol,
              "Time": c["time"],
              "Candle #": c["id"],
              "Setup": "Support Buy",
              "Action": f"🟢 Buy (Retest #{sup_touch_count}){status_suffix}",
              "Entry Price": round(c["close"], dec),
              "Stop Loss (SL)": sl,
              "TP 1": tp1,
              "TP 2": tp2,
              "TP 3": tp3,
              "Expired": c["is_expired"],
          })


# ---------------------------------------------------------
# BACKGROUND SCANNER FOR ALL ASSETS (MASTER DASHBOARD)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Live Master Signal Scanner (All 20+ Assets)")
st.markdown(
    "Scanning all Delta assets simultaneously for 4H Resistance & 6H Support"
    " retests..."
)

master_signals = []
asset_engines = {}

with st.spinner("⚡ Scanning all 20+ assets in real-time... Please wait."):
  for symbol in ASSET_LIST:
    df_m, df_4, df_6 = fetch_single_asset_data(symbol, selected_tf)
    if df_m is not None:
      engine = TWTDeltaEngine(df_m, df_4, df_6, symbol, active_tz)
      engine.calculate_and_scan()
      asset_engines[symbol] = engine
      if engine.signals:
        # Get the latest signal for this asset
        latest = engine.signals[-1]
        master_signals.append(latest)

if master_signals:
  st.success(f"✅ Found {len(master_signals)} active setups across assets!")
  df_master = pd.DataFrame(master_signals)
  st.dataframe(df_master, width="stretch")
else:
  st.warning(
      "🟡 No immediate retest signals right now across assets. Monitoring price"
      " action..."
  )

# ---------------------------------------------------------
# INDIVIDUAL ASSET DEEP DIVE & CHART VIEWER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Detailed Asset Inspector & Chart View")

selected_symbol = st.selectbox(
    "🔍 Select Asset for Deep Dive & Chart", options=ASSET_LIST, index=0
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df_m, df_4, df_6 = fetch_single_asset_data(selected_symbol, selected_tf)
  bot = TWTDeltaEngine(df_m, df_4, df_6, selected_symbol, active_tz)
  bot.calculate_and_scan()

if bot.df_main is not None and not bot.df_main.empty:
  curr_p = bot.candles[-1]["close"] if bot.candles else 0.0

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric(
        label="4H Resistance (01:30 High)", value=f"{bot.resistance_lvl:.2f}"
    )
  with col2:
    st.metric(label="6H Support (05:30 Low)", value=f"{bot.support_lvl:.2f}")
  with col3:
    st.metric(label="Current Price", value=f"{curr_p:.2f}")
  with col4:
    st.metric(
        label="Active Timeframe", value=selected_tf, delta=f"Zone: {tz_short}"
    )

  if bot.signals:
    latest_sig = bot.signals[-1]
    is_buy = "Buy" in latest_sig["Action"]
    if is_buy:
      st.success(
          f"🟢 **LATEST SETUP FOR {selected_symbol}: {latest_sig['Action']}**\n\n"
          f"- **Entry:** `{latest_sig['Entry Price']}` | **SL:**"
          f" `{latest_sig['Stop Loss (SL)']}`\n- **TPs:** `{latest_sig['TP 1']}`"
          f" | `{latest_sig['TP 2']}` | `{latest_sig['TP 3']}`"
      )
    else:
      st.error(
          f"🔴 **LATEST SETUP FOR {selected_symbol}: {latest_sig['Action']}**\n\n"
          f"- **Entry:** `{latest_sig['Entry Price']}` | **SL:**"
          f" `{latest_sig['Stop Loss (SL)']}`\n- **TPs:** `{latest_sig['TP 1']}`"
          f" | `{latest_sig['TP 2']}` | `{latest_sig['TP 3']}`"
      )

  # Chart Generation
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
  ax.axhline(
      bot.support_lvl,
      color="#26a69a",
      linestyle="--",
      linewidth=2,
      label="6H Support",
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
    tag = "EX" if sig["Expired"] else ("B" if "Buy" in sig["Action"] else "S")
    tag_color = (
        "#ffa726" if sig["Expired"] else ("#00e676" if tag == "B" else "#ff1744")
    )
    ax.text(
        sig["Candle #"],
        sig["Entry Price"],
        f" {tag} ",
        color="white",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=tag_color, alpha=0.9),
    )

  ax.set_title(
      f"{selected_symbol} {selected_tf} — TRADE WITH TUCKYAA Engine"
      f" ({tz_short})",
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
  st.error("Failed to fetch market data for selected asset.")
