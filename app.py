from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="TWT Ultimate Multi-Confluence Bot",
    page_icon="⚡",
    layout="wide",
)

# Hide Streamlit Default Elements
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Safe Auto-refresh every 30 seconds
count = st_autorefresh(interval=30000, key="twt_multi_refresh")

st.title("⚡ TRADE WITH TUCKYAA — Ultimate Multi-Confluence Engine")

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Multi-Asset Configurations")
st.sidebar.markdown("---")

IANA_TIMEZONES = [
    "Asia/Kolkata",
    "UTC",
    "America/New_York",
    "Europe/London",
    "Asia/Singapore",
]
selected_tz_name = st.sidebar.selectbox(
    "🌍 Trading Timezone (IANA)", options=IANA_TIMEZONES, index=0
)
active_tz = ZoneInfo(selected_tz_name)

selected_tf = "5m"
st.sidebar.text("⏱️ Entry Timeframe : 5m (Fixed)")
st.sidebar.text("📊 Confluence : 01:30, 05:30 & 21:30 4H + Prev Day 1D")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Master Confluence Rules")
st.sidebar.text("Demand Zones : 01:30, 05:30 & 21:30 Lows + Prev Day Low")
st.sidebar.text("Supply Zones : 01:30, 05:30 & 21:30 Highs + Prev Day High")
st.sidebar.text("BUY  : HH+HL Structure + Zone Bounce + Vol/Body + EMA50")
st.sidebar.text("SELL : LH+LL Structure + Zone Rejection + Vol/Body + EMA50")
st.sidebar.text("Filter : No Sideways / Confused Market | Max 1B/1S per Day")

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


def get_decimals(price):
  if price < 1:
    return 4
  elif price < 10:
    return 3
  else:
    return 2


# ---------------------------------------------------------
# DATA FETCHING MODULE (1d + 4h + 5m)
# ---------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_data(symbol):
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
        ohlcv_1d = exchange.fetch_ohlcv(sym, timeframe="1d", limit=5)
        ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=50)
        ohlcv_5m = exchange.fetch_ohlcv(sym, timeframe="5m", limit=300)
        if ohlcv_1d and ohlcv_4h and ohlcv_5m:
          df_1d = pd.DataFrame(ohlcv_1d, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
          df_1d["timestamp"] = pd.to_datetime(df_1d["timestamp"], unit="ms", utc=True)
          df_1d = df_1d.set_index("timestamp")

          df_4h = pd.DataFrame(ohlcv_4h, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
          df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], unit="ms", utc=True)
          df_4h = df_4h.set_index("timestamp")

          df_5m = pd.DataFrame(ohlcv_5m, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
          df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], unit="ms", utc=True)
          df_5m = df_5m.set_index("timestamp")
          return df_1d, df_4h, df_5m
    except Exception:
      continue
  return None, None, None


# ---------------------------------------------------------
# ULTIMATE MULTI-CONFLUENCE ENGINE
# ---------------------------------------------------------
class TWTUltimateEngine:

  def __init__(self, df_1d, df_4h, df_5m, symbol, user_tz):
    self.df_1d = df_1d
    self.df_4h = df_4h
    self.df_5m = df_5m
    self.symbol = symbol
    self.user_tz = user_tz
    self.candles_5m = []
    self.history = []
    self.signals = []

    # Levels & Zones
    self.prev_high = None
    self.prev_low = None
    self.demand_box_1 = None
    self.supply_box_1 = None
    self.demand_box_2 = None
    self.supply_box_2 = None
    self.demand_box_3 = None
    self.supply_box_3 = None

  def _build_zones(self, anchor_candle):
    high_val = anchor_candle["high"]
    low_val = anchor_candle["low"]
    rng = high_val - low_val
    zone_thickness = (rng * 0.25) if rng > 0 else (high_val * 0.002)
    return (low_val, low_val + zone_thickness), (high_val - zone_thickness, high_val)

  def _get_2130_anchor_for_date(self, candles_4h, target_date):
    evening_candidates = [
        c for c in candles_4h if c["date"] == target_date and c["time_obj"] >= time(20, 0)
    ]
    if evening_candidates:
      return min(
          evening_candidates,
          key=lambda x: abs(
              datetime.combine(target_date, x["time_obj"])
              - datetime.combine(target_date, time(21, 30))
          ),
      )
    return None

  def _touches_zone(self, low_val, high_val, zone_box, is_demand=True):
    if not zone_box:
      return False
    z_low, z_high = zone_box
    if is_demand:
      return (low_val <= z_high) and (high_val >= z_low)
    else:
      return (high_val >= z_low) and (low_val <= z_high)

  def _compute_power(self, strong_vol, strong_body, structure_aligned):
    score = 55
    if structure_aligned:
      score += 18
    if strong_vol:
      score += 12
    if strong_body:
      score += 10
    return min(score, 98)

  def calculate_and_scan(self):
    if self.df_1d is None or self.df_1d.empty or self.df_4h is None or self.df_4h.empty or self.df_5m is None or self.df_5m.empty:
      return

    current_utc_time = datetime.now(timezone.utc)

    # 1. Previous Day High & Low
    if len(self.df_1d) >= 2:
      prev_candle = self.df_1d.iloc[-2]
    else:
      prev_candle = self.df_1d.iloc[-1]

    self.prev_high = float(prev_candle["High"])
    self.prev_low = float(prev_candle["Low"])

    # 2. 4H Anchor Zones (01:30, 05:30, and rolling 21:30 PM)
    candles_4h = []
    for row in self.df_4h.iterrows():
      dt_utc = row[0]
      dt_user = dt_utc.astimezone(self.user_tz)
      r = row[1]
      candles_4h.append({
          "date": dt_user.date(),
          "time_obj": dt_user.time(),
          "dt_user": dt_user,
          "high": float(r["High"]),
          "low": float(r["Low"])
      })

    latest_date = candles_4h[-1]["date"] if candles_4h else datetime.now(self.user_tz).date()
    today_4h = [c for c in candles_4h if c["date"] == latest_date]
    if not today_4h:
      today_4h = candles_4h

    # Anchor 1 (~01:30 AM)
    anchor_1 = min(
        today_4h,
        key=lambda x: abs(datetime.combine(latest_date, x["time_obj"]) - datetime.combine(latest_date, time(1, 30))),
        default=today_4h[0]
    )
    anchor_dt_1 = anchor_1["dt_user"]
    self.demand_box_1, self.supply_box_1 = self._build_zones(anchor_1)

    # Anchor 2 (~05:30 AM)
    anchor_2_candidates = [c for c in today_4h if c["time_obj"] >= time(5, 0)]
    anchor_2 = None
    anchor_dt_2 = None
    if anchor_2_candidates:
      anchor_2 = min(
          anchor_2_candidates,
          key=lambda x: abs(datetime.combine(latest_date, x["time_obj"]) - datetime.combine(latest_date, time(5, 30)))
      )
      anchor_dt_2 = anchor_2["dt_user"]
      self.demand_box_2, self.supply_box_2 = self._build_zones(anchor_2)

    # Anchor 3 (~21:30 PM 4H candle, rolling across days until next 21:30 forms)
    unique_dates = sorted({c["date"] for c in candles_4h}, reverse=True)
    anchor_3 = None
    anchor_dt_3 = None
    for d in unique_dates:
      candidate_2130 = self._get_2130_anchor_for_date(candles_4h, d)
      if candidate_2130 is not None:
        anchor_3 = candidate_2130
        anchor_dt_3 = anchor_3["dt_user"]
        self.demand_box_3, self.supply_box_3 = self._build_zones(anchor_3)
        break

    # 3. Process 5M Candles & Indicators (Structure, Volume, Body, ATR, EMA50)
    df5 = self.df_5m.copy()
    df5["TR"] = np.maximum(
        df5["High"] - df5["Low"],
        np.maximum(
            abs(df5["High"] - df5["Close"].shift(1)),
            abs(df5["Low"] - df5["Close"].shift(1)),
        ),
    )
    df5["ATR"] = df5["TR"].rolling(14).mean()
    df5["EMA50"] = df5["Close"].ewm(span=50, adjust=False).mean()
    df5["VOL_SMA20"] = df5["Volume"].rolling(20).mean()
    df5["BODY"] = abs(df5["Close"] - df5["Open"])
    df5["BODY_SMA14"] = df5["BODY"].rolling(14).mean()
    df5["SWING_HIGH_10"] = df5["High"].shift(1).rolling(10).max()
    df5["SWING_LOW_10"] = df5["Low"].shift(1).rolling(10).min()

    for idx, row in enumerate(df5.iterrows()):
      dt_utc = row[0]
      dt_user = dt_utc.astimezone(self.user_tz)
      r = row[1]
      o, h, l, c, v = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]), float(r["Volume"])
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)
      ema50 = float(r["EMA50"]) if not pd.isna(r["EMA50"]) else c
      vol_sma = float(r["VOL_SMA20"]) if not pd.isna(r["VOL_SMA20"]) and r["VOL_SMA20"] > 0 else v
      body = float(r["BODY"])
      body_sma = float(r["BODY_SMA14"]) if not pd.isna(r["BODY_SMA14"]) and r["BODY_SMA14"] > 0 else body
      sh10 = float(r["SWING_HIGH_10"]) if not pd.isna(r["SWING_HIGH_10"]) else h
      sl10 = float(r["SWING_LOW_10"]) if not pd.isna(r["SWING_LOW_10"]) else l
      age_min = (current_utc_time - dt_utc).total_seconds() / 60.0

      self.candles_5m.append({
          "id": idx + 1,
          "time": dt_user.strftime("%H:%M"),
          "date": dt_user.date(),
          "dt_user": dt_user,
          "color": "Green" if c >= o else "Red",
          "open": o, "high": h, "low": l, "close": c, "volume": v,
          "atr": atr, "ema50": ema50,
          "vol_sma": vol_sma, "body": body, "body_sma": body_sma,
          "sh10": sh10, "sl10": sl10,
          "is_within_1hr": age_min <= 60.0
      })

    today_5m = [c for c in self.candles_5m if c["date"] == latest_date]
    scan_candles = today_5m if len(today_5m) >= 12 else self.candles_5m[-144:]
    if not scan_candles:
      scan_candles = self.candles_5m

    first_buy = None
    first_sell = None
    broke_above_pdh = False
    broke_below_pdl = False

    # Track recent swing lows/highs inside scan loop for HH+HL / LH+LL structure
    last_swing_low = None
    prev_swing_low = None
    last_swing_high = None
    prev_swing_high = None

    for c in scan_candles:
      atr = c["atr"]
      ema = c["ema50"]
      dec = get_decimals(c["close"])
      high_val = c["high"]
      low_val = c["low"]
      close_val = c["close"]

      # Update rolling swing highs/lows
      if low_val <= c["sl10"]:
        prev_swing_low = last_swing_low
        last_swing_low = low_val
      if high_val >= c["sh10"]:
        prev_swing_high = last_swing_high
        last_swing_high = high_val

      # Structure & Sideways/Confused filter
      hh_hl_ok = (
          (last_swing_high is not None and prev_swing_high is not None and last_swing_high > prev_swing_high)
          or (last_swing_low is not None and prev_swing_low is not None and last_swing_low > prev_swing_low)
          or (close_val > c["sh10"])
      )
      lh_ll_ok = (
          (last_swing_low is not None and prev_swing_low is not None and last_swing_low < prev_swing_low)
          or (last_swing_high is not None and prev_swing_high is not None and last_swing_high < prev_swing_high)
          or (close_val < c["sl10"])
      )

      # Avoid flat/confused EMA50 chop (price sitting tightly on EMA50 with tiny ATR)
      not_confused = atr > (close_val * 0.0003) and abs(close_val - ema) > (atr * 0.1)

      # Strong candle body & volume confirmation
      strong_body = c["body"] >= (c["body_sma"] * 0.75)
      strong_vol = c["volume"] >= (c["vol_sma"] * 1.05)
      quality_ok = strong_body and strong_vol and not_confused

      bull_trend_ok = close_val > ema and hh_hl_ok and quality_ok
      bear_trend_ok = close_val < ema and lh_ll_ok and quality_ok

      touch_demand_1 = self._touches_zone(low_val, high_val, self.demand_box_1, is_demand=True)
      touch_demand_2 = self._touches_zone(low_val, high_val, self.demand_box_2, is_demand=True)
      touch_demand_3 = self._touches_zone(low_val, high_val, self.demand_box_3, is_demand=True)
      near_prev_low = low_val <= self.prev_low

      touch_supply_1 = self._touches_zone(low_val, high_val, self.supply_box_1, is_demand=False)
      touch_supply_2 = self._touches_zone(low_val, high_val, self.supply_box_2, is_demand=False)
      touch_supply_3 = self._touches_zone(low_val, high_val, self.supply_box_3, is_demand=False)

      power_val = self._compute_power(strong_vol, strong_body, True)

      # --- A. PREVIOUS DAY HIGH / LOW REJECTION STRATEGY ---
      if high_val > self.prev_high:
        broke_above_pdh = True
      if not first_sell and broke_above_pdh and close_val < self.prev_high and c["color"] == "Red" and bear_trend_ok:
        sl = round(max(self.prev_high, high_val) + (1.5 * atr), dec)
        first_sell = {
            "Asset": self.symbol,
            "Date": str(c["date"]),
            "Time": c["time"],
            "Candle #": c["id"],
            "Type": "SELL 🔴",
            "Setup": "Prev Day High Rejection",
            "Power": f"{power_val}%",
            "Entry Price": round(close_val, dec),
            "Stop Loss (SL)": sl,
            "TP 1": round(close_val - (1.2 * atr), dec),
            "TP 2": round(close_val - (2.5 * atr), dec),
            "TP 3": round(close_val - (4.0 * atr), dec),
            "Is_Within_1hr": c["is_within_1hr"],
        }
        broke_above_pdh = False

      if low_val < self.prev_low:
        broke_below_pdl = True
      if not first_buy and (broke_below_pdl or near_prev_low) and close_val >= self.prev_low and c["color"] == "Green" and bull_trend_ok:
        sl = round(min(self.prev_low, low_val) - (1.5 * atr), dec)
        first_buy = {
            "Asset": self.symbol,
            "Date": str(c["date"]),
            "Time": c["time"],
            "Candle #": c["id"],
            "Type": "BUY 🟢",
            "Setup": "Prev Day Low Bounce",
            "Power": f"{power_val}%",
            "Entry Price": round(close_val, dec),
            "Stop Loss (SL)": sl,
            "TP 1": round(close_val + (1.2 * atr), dec),
            "TP 2": round(close_val + (2.5 * atr), dec),
            "TP 3": round(close_val + (4.0 * atr), dec),
            "Is_Within_1hr": c["is_within_1hr"],
        }
        broke_below_pdl = False

      # --- B. 4H ANCHOR ZONE 1 (01:30 AM) ---
      if anchor_dt_1 and c["dt_user"] >= anchor_dt_1 and self.demand_box_1 and self.supply_box_1:
        if not first_buy and c["color"] == "Green" and touch_demand_1 and bull_trend_ok:
          sl = round(min(self.demand_box_1[0], low_val) - (1.5 * atr), dec)
          first_buy = {
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Candle #": c["id"],
              "Type": "BUY 🟢",
              "Setup": "01:30 Zone Demand Bounce",
              "Power": f"{power_val}%",
              "Entry Price": round(close_val, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(close_val + (1.2 * atr), dec),
              "TP 2": round(close_val + (2.5 * atr), dec),
              "TP 3": round(close_val + (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
          }

        if not first_sell and c["color"] == "Red" and touch_supply_1 and bear_trend_ok:
          sl = round(max(self.supply_box_1[1], high_val) + (1.5 * atr), dec)
          first_sell = {
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Candle #": c["id"],
              "Type": "SELL 🔴",
              "Setup": "01:30 Zone Supply Rejection",
              "Power": f"{power_val}%",
              "Entry Price": round(close_val, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(close_val - (1.2 * atr), dec),
              "TP 2": round(close_val - (2.5 * atr), dec),
              "TP 3": round(close_val - (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
          }

      # --- C. 4H ANCHOR ZONE 2 (05:30 AM) ---
      if anchor_dt_2 and c["dt_user"] >= anchor_dt_2 and self.demand_box_2 and self.supply_box_2:
        if not first_buy and c["color"] == "Green" and touch_demand_2 and bull_trend_ok:
          sl = round(min(self.demand_box_2[0], low_val) - (1.5 * atr), dec)
          first_buy = {
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Candle #": c["id"],
              "Type": "BUY 🟢",
              "Setup": "05:30 Zone Demand Bounce",
              "Power": f"{power_val}%",
              "Entry Price": round(close_val, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(close_val + (1.2 * atr), dec),
              "TP 2": round(close_val + (2.5 * atr), dec),
              "TP 3": round(close_val + (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
          }

        if not first_sell and c["color"] == "Red" and touch_supply_2 and bear_trend_ok:
          sl = round(max(self.supply_box_2[1], high_val) + (1.5 * atr), dec)
          first_sell = {
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Candle #": c["id"],
              "Type": "SELL 🔴",
              "Setup": "05:30 Zone Supply Rejection",
              "Power": f"{power_val}%",
              "Entry Price": round(close_val, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(close_val - (1.2 * atr), dec),
              "TP 2": round(close_val - (2.5 * atr), dec),
              "TP 3": round(close_val - (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
          }

      # --- D. 4H ANCHOR ZONE 3 (21:30 PM Rolling) ---
      if anchor_dt_3 and self.demand_box_3 and self.supply_box_3:
        if not first_buy and c["color"] == "Green" and touch_demand_3 and bull_trend_ok:
          sl = round(min(self.demand_box_3[0], low_val) - (1.5 * atr), dec)
          first_buy = {
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Candle #": c["id"],
              "Type": "BUY 🟢",
              "Setup": "21:30 Zone Demand Bounce",
              "Power": f"{power_val}%",
              "Entry Price": round(close_val, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(close_val + (1.2 * atr), dec),
              "TP 2": round(close_val + (2.5 * atr), dec),
              "TP 3": round(close_val + (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
          }

        if not first_sell and c["color"] == "Red" and touch_supply_3 and bear_trend_ok:
          sl = round(max(self.supply_box_3[1], high_val) + (1.5 * atr), dec)
          first_sell = {
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Candle #": c["id"],
              "Type": "SELL 🔴",
              "Setup": "21:30 Zone Supply Rejection",
              "Power": f"{power_val}%",
              "Entry Price": round(close_val, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(close_val - (1.2 * atr), dec),
              "TP 2": round(close_val - (2.5 * atr), dec),
              "TP 3": round(close_val - (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
          }

    collected = [sig for sig in [first_buy, first_sell] if sig is not None]
    collected.sort(key=lambda x: x["Candle #"])

    self.history = collected
    self.signals = [s for s in collected if s.get("Is_Within_1hr", False)]


# ---------------------------------------------------------
# MASTER SCANNER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (20+ Assets)")

master_signals = []
asset_engines = {}


def scan_asset(sym):
  df_1d, df_4h, df_5m = fetch_data(sym)
  if df_1d is not None and df_4h is not None and df_5m is not None:
    engine = TWTUltimateEngine(df_1d, df_4h, df_5m, sym, active_tz)
    engine.calculate_and_scan()
    return sym, engine, engine.signals
  return sym, None, []


with st.spinner("⚡ Scanning 20+ assets with Ultimate Confluence Engine..."):
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(scan_asset, sym): sym for sym in ASSET_LIST}
    for future in as_completed(futures):
      sym, engine, sigs = future.result()
      if engine:
        asset_engines[sym] = engine
      if sigs:
        master_signals.extend(sigs)

if master_signals:
  st.success(f"✅ Found {len(master_signals)} perfect confluence setup(s) across assets!")
  df_display = pd.DataFrame(master_signals).drop(columns=["Is_Within_1hr"], errors="ignore")
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No active perfect setups meeting strict rules. False entries avoided.")

# ---------------------------------------------------------
# MANUAL ASSET INSPECTOR & CHART VIEWER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector (All Zones & Prev Day Levels)")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df_1d, df_4h, df_5m = fetch_data(selected_symbol)
  bot = TWTUltimateEngine(df_1d, df_4h, df_5m, selected_symbol, active_tz)
  bot.calculate_and_scan()

if bot.candles_5m:
  curr_p = bot.candles_5m[-1]["close"]

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label=f"Current {selected_symbol} Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Locked Perfect Signals", value=len(bot.history))
  with col3:
    st.metric(label="Timezone", value=selected_tz_name)

  if bot.history:
    st.markdown(f"### 📋 Locked Entries Feed for {selected_symbol}")
    df_history = pd.DataFrame(bot.history).drop(columns=["Is_Within_1hr"], errors="ignore")
    st.dataframe(df_history, width="stretch")

  # --- PLOT CHART WITH ALL LEVELS ---
  fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
  for c in bot.candles_5m:
    col = "#26a69a" if c["close"] >= c["open"] else "#ef5350"
    lower_val = min(c["open"], c["close"])
    body_height = abs(c["close"] - c["open"])
    if body_height == 0:
      body_height = curr_p * 0.0005

    ax.bar(c["id"], body_height, bottom=lower_val, color=col, width=0.6, edgecolor=col)
    ax.vlines(c["id"], c["low"], c["high"], color=col, linewidth=1.2)

  # Plot 01:30 AM Zones
  if bot.demand_box_1:
    ax.axhspan(bot.demand_box_1[0], bot.demand_box_1[1], color="#00897b", alpha=0.30, label="Demand 1 (01:30)")
  if bot.supply_box_1:
    ax.axhspan(bot.supply_box_1[0], bot.supply_box_1[1], color="#e53935", alpha=0.30, label="Supply 1 (01:30)")

  # Plot 05:30 AM Zones
  if bot.demand_box_2:
    ax.axhspan(bot.demand_box_2[0], bot.demand_box_2[1], color="#26a69a", alpha=0.18, label="Demand 2 (05:30)")
  if bot.supply_box_2:
    ax.axhspan(bot.supply_box_2[0], bot.supply_box_2[1], color="#ff7043", alpha=0.18, label="Supply 2 (05:30)")

  # Plot 21:30 PM Zones
  if bot.demand_box_3:
    ax.axhspan(bot.demand_box_3[0], bot.demand_box_3[1], color="#29b6f6", alpha=0.20, label="Demand 3 (21:30)")
  if bot.supply_box_3:
    ax.axhspan(bot.supply_box_3[0], bot.supply_box_3[1], color="#ab47bc", alpha=0.20, label="Supply 3 (21:30)")

  # Plot Previous Day High & Low Lines
  if bot.prev_high is not None:
    ax.axhline(bot.prev_high, color="#ff9100", linestyle="--", linewidth=2, label=f"Prev Day High: {bot.prev_high}")
  if bot.prev_low is not None:
    ax.axhline(bot.prev_low, color="#00e676", linestyle="--", linewidth=2, label=f"Prev Day Low: {bot.prev_low}")

  lows = [c["low"] for c in bot.candles_5m]
  highs = [c["high"] for c in bot.candles_5m]
  if lows and highs:
    margin = (max(highs) - min(lows)) * 0.09
    if margin == 0:
      margin = curr_p * 0.005
    ax.set_ylim(min(lows) - margin, max(highs) + margin)

  for sig in bot.history:
    is_b = "BUY" in sig["Type"]
    bg_col = "#00e676" if is_b else "#ff5252"
    txt = f" BUY ({sig['Power']}) " if is_b else f" SELL ({sig['Power']}) "
    ax.text(
        sig["Candle #"],
        sig["Entry Price"],
        txt,
        color="white",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_col, alpha=0.9),
        ha="center",
        va="center",
    )

  ax.set_title(
      f"{selected_symbol} 5m Chart — Ultimate Multi-Confluence Strategy",
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
  ax.legend(loc="upper left", facecolor="#1e222d", edgecolor="none", labelcolor="white")

  st.pyplot(fig)
else:
  st.error("Market data fetch failed or empty.")
