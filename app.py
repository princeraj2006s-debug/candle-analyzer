from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta, timezone
import json
from pathlib import Path
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
    page_title="TWT 4H POI & Confluence Engine",
    page_icon="⚡",
    layout="wide",
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

count = st_autorefresh(interval=30000, key="twt_multi_refresh")

st.title("⚡ TRADE WITH TUCKYAA — 4H POI Engine (Strict Retest + Wick + Strength)")

LEDGER_FILE = Path("twt_signals_ledger_v6.json")
LOCKED_ENTRIES_FILE = Path("twt_locked_entries_v2.json")
LIVE_WINDOW_MINUTES = 60.0
MIN_ZONE_STRENGTH = 65.0
MIN_WICK_RATIO = 0.25


# ---------------------------------------------------------
# SESSION & LOCK HELPERS (STRICTLY FIRST CONFIRMED SIGNAL LOCKED)
# ---------------------------------------------------------
def get_poi_session_key(c0530_close):
  if c0530_close is None:
    return "NO_SESSION"
  return c0530_close.strftime("%Y-%m-%d_%H:%M")


def load_locked_entries():
  if LOCKED_ENTRIES_FILE.exists():
    try:
      data = json.loads(LOCKED_ENTRIES_FILE.read_text(encoding="utf-8"))
      if isinstance(data, dict):
        return data
    except Exception:
      pass
  return {}


def save_locked_entries(locks):
  try:
    LOCKED_ENTRIES_FILE.write_text(json.dumps(locks, indent=2), encoding="utf-8")
  except Exception:
    pass


# ---------------------------------------------------------
# PERSISTENT LEDGER HELPERS (STRICTLY PER-ASSET ISOLATED)
# ---------------------------------------------------------
def load_ledger():
  if LEDGER_FILE.exists():
    try:
      data = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
      if isinstance(data, list):
        return data
    except Exception:
      pass
  return []


def save_ledger(records):
  try:
    LEDGER_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
  except Exception:
    pass


def update_ledger_with_engine(records, engine):
  if not engine.candles_5m:
    return records

  by_id = {r["id"]: r for r in records}
  changed = False

  for sig in engine.all_detected_signals:
    sig_id = f"{sig['Asset']}|{sig.get('Session', '')}|{sig['Direction']}"
    if sig_id not in by_id:
      by_id[sig_id] = {
          "id": sig_id,
          "Asset": sig["Asset"],
          "Date": sig["Date"],
          "Time": sig["Time"],
          "Window": sig.get("Window", ""),
          "Direction": sig["Direction"],
          "Type": sig["Type"],
          "Grade": sig.get("Grade", "CONFIRMED ⚡"),
          "Zone Buyer %": sig.get("Zone Buyer %", "0.0%"),
          "Zone Seller %": sig.get("Zone Seller %", "0.0%"),
          "Setup": sig["Setup"],
          "Entry Price": float(sig["Entry Price"]),
          "Stop Loss (SL)": float(sig["Stop Loss (SL)"]),
          "TP 1": float(sig["TP 1"]),
          "TP 2": float(sig["TP 2"]),
          "TP 3": float(sig["TP 3"]),
          "Signal_DT": sig.get("Signal_DT", ""),
          "TP1_Hit": False,
          "TP2_Hit": False,
          "TP3_Hit": False,
          "SL_Hit": False,
          "Status": "OPEN ⏳",
      }
      changed = True

  for rec in by_id.values():
    if rec["Asset"] != engine.symbol:
      continue
    sig_dt_str = rec.get("Signal_DT", "")
    is_buy = rec["Direction"] == "BUY"
    tp1, tp2, tp3, sl = rec["TP 1"], rec["TP 2"], rec["TP 3"], rec["Stop Loss (SL)"]

    for c in engine.candles_5m:
      c_iso = c["dt_user"].isoformat()
      if sig_dt_str and c_iso <= sig_dt_str:
        continue

      if is_buy:
        if c["low"] <= sl and not rec["TP1_Hit"]:
          rec["SL_Hit"] = True
          changed = True
          break
        if c["high"] >= tp1 and not rec["TP1_Hit"]:
          rec["TP1_Hit"] = True
          changed = True
        if c["high"] >= tp2 and not rec["TP2_Hit"]:
          rec["TP2_Hit"] = True
          changed = True
        if c["high"] >= tp3 and not rec["TP3_Hit"]:
          rec["TP3_Hit"] = True
          changed = True
        if c["low"] <= sl:
          rec["SL_Hit"] = True
          changed = True
          break
      else:
        if c["high"] >= sl and not rec["TP1_Hit"]:
          rec["SL_Hit"] = True
          changed = True
          break
        if c["low"] <= tp1 and not rec["TP1_Hit"]:
          rec["TP1_Hit"] = True
          changed = True
        if c["low"] <= tp2 and not rec["TP2_Hit"]:
          rec["TP2_Hit"] = True
          changed = True
        if c["low"] <= tp3 and not rec["TP3_Hit"]:
          rec["TP3_Hit"] = True
          changed = True
        if c["high"] >= sl:
          rec["SL_Hit"] = True
          changed = True
          break

    if rec["TP3_Hit"]:
      rec["Status"] = "TP3 HIT ✅"
    elif rec["TP2_Hit"]:
      rec["Status"] = "TP2 HIT ✅"
    elif rec["TP1_Hit"]:
      rec["Status"] = "TP1 HIT ✅"
    elif rec["SL_Hit"]:
      rec["Status"] = "SL HIT ❌"
    else:
      rec["Status"] = "OPEN ⏳"

  return list(by_id.values()) if changed else records


def _pct(num, den):
  return f"{round((num / den) * 100.0, 1)}%" if den > 0 else "0.0%"


def summarize_asset_ledger(records, symbol, start_date_str="2026-11-01"):
  filtered = [
      r for r in records
      if r.get("Asset") == symbol and str(r.get("Date", "")) >= start_date_str
  ]

  def stats_for(direction):
    sub = [r for r in filtered if r.get("Direction") == direction]
    total = len(sub)
    tp1 = sum(1 for r in sub if r.get("TP1_Hit"))
    tp2 = sum(1 for r in sub if r.get("TP2_Hit"))
    tp3 = sum(1 for r in sub if r.get("TP3_Hit"))
    return {
        "Total Signals": total,
        "TP1 Win %": _pct(tp1, total),
        "TP2 Win %": _pct(tp2, total),
        "TP3 Win %": _pct(tp3, total),
    }

  buy_s = stats_for("BUY")
  sell_s = stats_for("SELL")
  tot_sig = buy_s["Total Signals"] + sell_s["Total Signals"]
  tot_tp1 = sum(1 for r in filtered if r.get("TP1_Hit"))
  tot_tp2 = sum(1 for r in filtered if r.get("TP2_Hit"))
  tot_tp3 = sum(1 for r in filtered if r.get("TP3_Hit"))

  combined = {
      "Total Signals": tot_sig,
      "TP1 Win %": _pct(tot_tp1, tot_sig),
      "TP2 Win %": _pct(tot_tp2, tot_sig),
      "TP3 Win %": _pct(tot_tp3, tot_sig),
  }
  return pd.DataFrame([
      {"Side": "BUY 🟢", **buy_s},
      {"Side": "SELL 🔴", **sell_s},
      {"Side": "TOTAL ⚡", **combined},
  ])


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

use_chop_filter = st.sidebar.checkbox(
    "🛡️ Enable Chop / Sideways Filter", value=True
)

require_full_confluence = st.sidebar.checkbox(
    "🎯 Extra Strict Wick & Trend Filter", value=False
)

ledger_start_date = st.sidebar.text_input(
    "📅 Ledger Start Date (YYYY-MM-DD)", value="2026-11-01"
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Strict POI + Lock Rules")
st.sidebar.text("Anchor   : 01:30→05:30 4H Candle")
st.sidebar.text("Entry Cap: Max 1 BUY + 1 SELL")
st.sidebar.text("Strength : Buyer>=65% | Seller>=65%")
st.sidebar.text("Wick/Ret : >=25% Wick + Zone Retest")

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


def _df_from_ohlcv(ohlcv):
  df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
  df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
  return df.set_index("timestamp")


def _compute_zero_lag_series(df, length=21, band_mult=1.2):
  if df is None or df.empty or len(df) < 5:
    return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=str)

  eff_len = max(3, min(length, len(df)))
  lag = max(1, (eff_len - 1) // 2)
  close_s = df["Close"].astype(float)
  zlema_input = close_s + (close_s - close_s.shift(lag)).fillna(0.0)
  zl_line = zlema_input.ewm(span=eff_len, adjust=False).mean()

  tr = np.maximum(
      df["High"] - df["Low"],
      np.maximum(
          abs(df["High"] - close_s.shift(1)),
          abs(df["Low"] - close_s.shift(1)),
      ),
  )
  atr = pd.Series(tr, index=df.index).rolling(min(14, len(df))).mean().bfill()
  band = atr * band_mult
  upper = zl_line + band
  lower = zl_line - band

  states = []
  curr_trend = "BULL" if float(close_s.iloc[0]) >= float(zl_line.iloc[0]) else "BEAR"
  for i in range(len(df)):
    c_val = float(close_s.iloc[i])
    u_val = float(upper.iloc[i])
    l_val = float(lower.iloc[i])
    if c_val > u_val:
      curr_trend = "BULL"
    elif c_val < l_val:
      curr_trend = "BEAR"
    states.append(curr_trend)

  return zl_line, band, pd.Series(states, index=df.index)


def _wick_ratios(c):
  o = float(c["open"])
  h = float(c["high"])
  l = float(c["low"])
  cl = float(c["close"])
  rng = max(h - l, 1e-9)
  body_top = max(o, cl)
  body_bot = min(o, cl)
  upper_wick = max(0.0, h - body_top)
  lower_wick = max(0.0, body_bot - l)
  return (lower_wick / rng), (upper_wick / rng)


def _candle_zone_strength(c, zone_type):
  """
  Calculates Buyer Strength % in Demand Zone and Seller Strength % in Supply Zone,
  incorporating wick rejection, close position, and candle direction.
  """
  o = float(c["open"])
  h = float(c["high"])
  l = float(c["low"])
  cl = float(c["close"])
  rng = max(h - l, 1e-9)

  lower_ratio, upper_ratio = _wick_ratios(c)
  body_ratio = min(1.0, abs(cl - o) / rng)
  close_pos = max(0.0, min(1.0, (cl - l) / rng))

  if zone_type == "DEMAND":
    if cl <= o:
      return 25.0, 75.0
    buyer_pct = round(
        max(5.0, min(99.0, 35.0 + (lower_ratio * 35.0) + (close_pos * 20.0) + (body_ratio * 10.0))),
        1,
    )
    seller_pct = round(max(1.0, 100.0 - buyer_pct), 1)
    return buyer_pct, seller_pct
  else:
    if cl >= o:
      return 75.0, 25.0
    seller_pct = round(
        max(5.0, min(99.0, 35.0 + (upper_ratio * 35.0) + ((1.0 - close_pos) * 20.0) + (body_ratio * 10.0))),
        1,
    )
    buyer_pct = round(max(1.0, 100.0 - seller_pct), 1)
    return buyer_pct, seller_pct


# ---------------------------------------------------------
# DATA FETCHING MODULE (1d + 4h + 1h + 15m + 5m)
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
        ohlcv_1d = exchange.fetch_ohlcv(sym, timeframe="1d", limit=35)
        ohlcv_4h = exchange.fetch_ohlcv(sym, timeframe="4h", limit=50)
        ohlcv_1h = exchange.fetch_ohlcv(sym, timeframe="1h", limit=35)
        ohlcv_15m = exchange.fetch_ohlcv(sym, timeframe="15m", limit=35)
        ohlcv_5m = exchange.fetch_ohlcv(sym, timeframe="5m", limit=400)
        if ohlcv_1d and ohlcv_4h and ohlcv_5m:
          dfs_by_tf = {
              "5m": _df_from_ohlcv(ohlcv_5m),
              "15m": _df_from_ohlcv(ohlcv_15m) if ohlcv_15m else None,
              "1h": _df_from_ohlcv(ohlcv_1h) if ohlcv_1h else None,
              "4h": _df_from_ohlcv(ohlcv_4h),
              "1d": _df_from_ohlcv(ohlcv_1d),
          }
          return dfs_by_tf["1d"], dfs_by_tf["4h"], dfs_by_tf["5m"], dfs_by_tf
    except Exception:
      continue
  return None, None, None, {}


# ---------------------------------------------------------
# 01:30 -> 05:30 4H POI ENGINE (STRICT RETEST + WICK + STRENGTH)
# ---------------------------------------------------------
class TWTUltimateEngine:

  def __init__(
      self,
      df_1d,
      df_4h,
      df_5m,
      symbol,
      user_tz,
      chop_filter=True,
      strict_confluence=False,
      locked_entries=None,
      live_window_min=LIVE_WINDOW_MINUTES,
  ):
    self.df_1d = df_1d
    self.df_4h = df_4h
    self.df_5m = df_5m
    self.symbol = symbol
    self.user_tz = user_tz
    self.chop_filter = chop_filter
    self.strict_confluence = strict_confluence
    self.locked_entries = locked_entries if locked_entries is not None else {}
    self.locks_mutated = False
    self.live_window_min = live_window_min

    self.candles_5m = []
    self.all_detected_signals = []
    self.live_1hr_signals = []
    self.signals = []
    self.diy_checklist = []

    self.c0130_closed = False
    self.c0130_start = None
    self.c0530_close = None
    self.c0130_high = None
    self.c0130_low = None
    self.supply_poi_zone = None
    self.demand_poi_zone = None

  def _is_choppy(self, c):
    if not self.chop_filter:
      return False
    atr = c["atr"]
    atr_sma = c["atr_sma"]
    ema = c["ema50"]
    close_val = c["close"]
    low_atr = (atr_sma is not None and atr_sma > 0 and atr < (0.60 * atr_sma))
    near_ema_chop = (atr > 0 and abs(close_val - ema) < (0.10 * atr))
    return low_atr or near_ema_chop

  def _has_proper_retest_and_wick(self, prev_c, c, side, zone_low, zone_high):
    """
    Requires:
    1) Prior candle/move approached/tested the POI zone (proper retest)
    2) Current candle touches/retests inside POI zone and closes back on confirmed side
    3) Strong wick rejection (>= MIN_WICK_RATIO)
    """
    lower_ratio, upper_ratio = _wick_ratios(c)
    o, h, l, cl = c["open"], c["high"], c["low"], c["close"]

    if side == "BUY":
      retested_zone = (l <= zone_high) and (l >= zone_low * 0.992)
      prior_approached = (prev_c is None) or (prev_c["low"] <= zone_high * 1.004)
      bull_close = (cl > o) and (cl > zone_low)
      wick_ok = lower_ratio >= MIN_WICK_RATIO
      if self.strict_confluence:
        wick_ok = lower_ratio >= 0.35
      return retested_zone and prior_approached and bull_close and wick_ok
    else:
      retested_zone = (h >= zone_low) and (h <= zone_high * 1.008)
      prior_approached = (prev_c is None) or (prev_c["high"] >= zone_low * 0.996)
      bear_close = (cl < o) and (cl < zone_high)
      wick_ok = upper_ratio >= MIN_WICK_RATIO
      if self.strict_confluence:
        wick_ok = upper_ratio >= 0.35
      return retested_zone and prior_approached and bear_close and wick_ok

  def _evaluate_strict_confirmation(self, prev_c, c, side, buyer_pct, seller_pct, zone_low, zone_high):
    retest_wick_ok = self._has_proper_retest_and_wick(prev_c, c, side, zone_low, zone_high)
    if not retest_wick_ok:
      return False, "REJECTED", ""

    if side == "BUY":
      strength_ok = buyer_pct >= MIN_ZONE_STRENGTH
      zl_ok = c.get("zl_trend") == "BULL"
      above_ema = c["close"] >= c["ema50"]
    else:
      strength_ok = seller_pct >= MIN_ZONE_STRENGTH
      zl_ok = c.get("zl_trend") == "BEAR"
      above_ema = c["close"] <= c["ema50"]

    if not strength_ok or not zl_ok:
      return False, "REJECTED", ""

    if self.strict_confluence and not above_ema:
      return False, "REJECTED", ""

    grade = "A+ GRADE 🔥" if above_ema else "CONFIRMED ⚡"
    note = f"Retest+Wick:OK | ZL:{'BULL' if side == 'BUY' else 'BEAR'} | Str>={MIN_ZONE_STRENGTH:.0f}%"
    return True, grade, note

  def _find_latest_completed_0130_4h(self, now_user):
    today_0130 = datetime.combine(now_user.date(), time(1, 30), tzinfo=self.user_tz)
    today_0530 = today_0130 + timedelta(hours=4)

    if now_user >= today_0530:
      c_start, c_end = today_0130, today_0530
    else:
      c_start = today_0130 - timedelta(days=1)
      c_end = c_start + timedelta(hours=4)

    sub_4h = [c for c in self.candles_5m if c_start <= c["dt_user"] < c_end]
    if len(sub_4h) >= 4:
      h_val = max(c["high"] for c in sub_4h)
      l_val = min(c["low"] for c in sub_4h)
      return True, c_start, c_end, h_val, l_val

    if len(self.df_4h) >= 2:
      prev_4h = self.df_4h.iloc[-2]
      h_val = float(prev_4h["High"])
      l_val = float(prev_4h["Low"])
      dt_4h_start = prev_4h.name.tz_convert(self.user_tz)
      dt_4h_end = dt_4h_start + timedelta(hours=4)
      return True, dt_4h_start, dt_4h_end, h_val, l_val

    return False, None, None, None, None

  def _scan_and_lock_first_signals(self, now_user):
    """
    Locks the FIRST confirmed SELL near Supply POI (with high seller strength, retest & wick)
    and the FIRST confirmed BUY near Demand POI (with high buyer strength, retest & wick)
    for the current POI session. Once locked, it never shifts on re-hits.
    """
    if not self.c0130_closed or self.c0530_close is None:
      return []

    session_key = get_poi_session_key(self.c0530_close)
    sell_lock_key = f"{self.symbol}|{session_key}|SELL"
    buy_lock_key = f"{self.symbol}|{session_key}|BUY"

    locked_sell = self.locked_entries.get(sell_lock_key)
    locked_buy = self.locked_entries.get(buy_lock_key)

    sub = [c for c in self.candles_5m if self.c0530_close <= c["dt_user"] <= now_user]
    sup_low, sup_high = self.supply_poi_zone
    dem_low, dem_high = self.demand_poi_zone

    if not locked_sell or not locked_buy:
      for idx_c, c in enumerate(sub):
        if locked_sell and locked_buy:
          break
        if self._is_choppy(c):
          continue

        prev_c = sub[idx_c - 1] if idx_c > 0 else None
        atr = c["atr"]
        dec = get_decimals(c["close"])
        h, l, cl = c["high"], c["low"], c["close"]

        if not locked_sell:
          b_pct, s_pct = _candle_zone_strength(c, "SUPPLY")
          ok, grade, conf_note = self._evaluate_strict_confirmation(
              prev_c, c, "SELL", b_pct, s_pct, sup_low, sup_high
          )
          if ok:
            sl = round(max(self.c0130_high, h) + (1.2 * atr), dec)
            locked_sell = {
                "Asset": self.symbol,
                "Session": session_key,
                "Date": c["dt_user"].strftime("%Y-%m-%d"),
                "Time": c["dt_user"].strftime("%H:%M"),
                "Window": f"Post-{self.c0530_close.strftime('%H:%M')} Supply POI",
                "Candle #": c["id"],
                "Direction": "SELL",
                "Type": "SELL 🔴",
                "Grade": grade,
                "Zone Buyer %": f"{b_pct}%",
                "Zone Seller %": f"{s_pct}%",
                "Setup": f"Supply POI ({self.c0130_high:.{dec}f}) SELL | {conf_note}",
                "Entry Price": round(cl, dec),
                "Stop Loss (SL)": sl,
                "TP 1": round(cl - (1.2 * atr), dec),
                "TP 2": round(cl - (2.5 * atr), dec),
                "TP 3": round(cl - (4.0 * atr), dec),
                "Signal_DT": c["dt_user"].isoformat(),
            }
            self.locked_entries[sell_lock_key] = locked_sell
            self.locks_mutated = True

        if not locked_buy:
          b_pct, s_pct = _candle_zone_strength(c, "DEMAND")
          ok, grade, conf_note = self._evaluate_strict_confirmation(
              prev_c, c, "BUY", b_pct, s_pct, dem_low, dem_high
          )
          if ok:
            sl = round(min(self.c0130_low, l) - (1.2 * atr), dec)
            locked_buy = {
                "Asset": self.symbol,
                "Session": session_key,
                "Date": c["dt_user"].strftime("%Y-%m-%d"),
                "Time": c["dt_user"].strftime("%H:%M"),
                "Window": f"Post-{self.c0530_close.strftime('%H:%M')} Demand POI",
                "Candle #": c["id"],
                "Direction": "BUY",
                "Type": "BUY 🟢",
                "Grade": grade,
                "Zone Buyer %": f"{b_pct}%",
                "Zone Seller %": f"{s_pct}%",
                "Setup": f"Demand POI ({self.c0130_low:.{dec}f}) BUY | {conf_note}",
                "Entry Price": round(cl, dec),
                "Stop Loss (SL)": sl,
                "TP 1": round(cl + (1.2 * atr), dec),
                "TP 2": round(cl + (2.5 * atr), dec),
                "TP 3": round(cl + (4.0 * atr), dec),
                "Signal_DT": c["dt_user"].isoformat(),
            }
            self.locked_entries[buy_lock_key] = locked_buy
            self.locks_mutated = True

    result = []
    for item in (locked_sell, locked_buy):
      if item:
        rec = dict(item)
        matched_c = next((x for x in self.candles_5m if x["dt_user"].isoformat() == rec["Signal_DT"]), None)
        if matched_c:
          rec["Candle #"] = matched_c["id"]
        try:
          sig_dt = datetime.fromisoformat(rec["Signal_DT"])
          age_min = (now_user - sig_dt).total_seconds() / 60.0
        except Exception:
          age_min = 9999.0
        rec["Age_Min"] = round(age_min, 1)
        rec["Is_Live_1Hr"] = (0.0 <= age_min <= self.live_window_min)
        result.append(rec)

    result.sort(key=lambda x: x["Signal_DT"])
    return result

  def _build_diy_checklist(self):
    buy_cnt = sum(1 for s in self.signals if "BUY" in s["Type"])
    sell_cnt = sum(1 for s in self.signals if "SELL" in s["Type"])
    self.diy_checklist = [
        {
            "Rule (01:30–05:30 POI)": "1. 4H Closed High / Low",
            "BUY Side (Demand POI)": f"Low: {self.c0130_low:.2f}" if self.c0130_low else "WAIT ⏳",
            "SELL Side (Supply POI)": f"High: {self.c0130_high:.2f}" if self.c0130_high else "WAIT ⏳",
        },
        {
            "Rule (01:30–05:30 POI)": "2. POI Zone Range",
            "BUY Side (Demand POI)": f"{self.demand_poi_zone[0]:.2f} – {self.demand_poi_zone[1]:.2f}" if self.demand_poi_zone else "N/A",
            "SELL Side (Supply POI)": f"{self.supply_poi_zone[0]:.2f} – {self.supply_poi_zone[1]:.2f}" if self.supply_poi_zone else "N/A",
        },
        {
            "Rule (01:30–05:30 POI)": "3. Locked Entries (Max 1 BUY + 1 SELL)",
            "BUY Side (Demand POI)": f"{buy_cnt} / 1 BUY (Buyer>={MIN_ZONE_STRENGTH:.0f}%)",
            "SELL Side (Supply POI)": f"{sell_cnt} / 1 SELL (Seller>={MIN_ZONE_STRENGTH:.0f}%)",
        },
    ]

  def calculate_and_scan(self):
    if self.df_1d is None or self.df_1d.empty or self.df_4h is None or self.df_4h.empty or self.df_5m is None or self.df_5m.empty:
      return

    current_utc_time = datetime.now(timezone.utc)
    now_user = current_utc_time.astimezone(self.user_tz)

    self.df_5m["TR"] = np.maximum(
        self.df_5m["High"] - self.df_5m["Low"],
        np.maximum(
            abs(self.df_5m["High"] - self.df_5m["Close"].shift(1)),
            abs(self.df_5m["Low"] - self.df_5m["Close"].shift(1)),
        ),
    )
    self.df_5m["ATR"] = self.df_5m["TR"].rolling(14).mean()
    self.df_5m["ATR_SMA"] = self.df_5m["ATR"].rolling(20).mean()
    self.df_5m["EMA50"] = self.df_5m["Close"].ewm(span=50, adjust=False).mean()

    zl_line, zl_band, zl_trend_s = _compute_zero_lag_series(self.df_5m, length=21, band_mult=1.2)

    for idx, (dt_utc, r) in enumerate(self.df_5m.iterrows()):
      dt_user = dt_utc.tz_convert(self.user_tz)
      o, h, l, c = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)
      atr_sma = float(r["ATR_SMA"]) if not pd.isna(r["ATR_SMA"]) else atr
      ema50 = float(r["EMA50"]) if not pd.isna(r["EMA50"]) else c
      zl_val = float(zl_line.iloc[idx]) if not zl_line.empty and not pd.isna(zl_line.iloc[idx]) else c
      zl_b = float(zl_band.iloc[idx]) if not zl_band.empty and not pd.isna(zl_band.iloc[idx]) else (atr * 1.2)
      zl_tr = str(zl_trend_s.iloc[idx]) if not zl_trend_s.empty else ("BULL" if c >= o else "BEAR")

      self.candles_5m.append({
          "id": idx + 1,
          "time": dt_user.strftime("%H:%M"),
          "date": dt_user.date(),
          "dt_utc": dt_utc,
          "dt_user": dt_user,
          "color": "Green" if c >= o else "Red",
          "open": o, "high": h, "low": l, "close": c,
          "atr": atr, "atr_sma": atr_sma, "ema50": ema50,
          "zl_line": zl_val, "zl_upper": zl_val + zl_b, "zl_lower": zl_val - zl_b,
          "zl_trend": zl_tr,
      })

    closed, c_start, c_end, h_val, l_val = self._find_latest_completed_0130_4h(now_user)
    self.c0130_closed = closed
    self.c0130_start = c_start
    self.c0530_close = c_end
    self.c0130_high = h_val
    self.c0130_low = l_val

    if closed and h_val is not None and l_val is not None:
      rng = max(h_val - l_val, 1e-9)
      poi_thick = max(rng * 0.22, h_val * 0.0015)
      self.supply_poi_zone = (h_val - poi_thick, h_val + poi_thick * 0.4)
      self.demand_poi_zone = (l_val - poi_thick * 0.4, l_val + poi_thick)

    locked_sigs = self._scan_and_lock_first_signals(now_user)
    self.all_detected_signals = locked_sigs
    self.signals = locked_sigs
    self.live_1hr_signals = [s for s in locked_sigs if s.get("Is_Live_1Hr", False)]
    self.history = self.live_1hr_signals
    self._build_diy_checklist()


# ---------------------------------------------------------
# MASTER SCANNER (LAST 1 HOUR LOCKED SIGNALS ON LIVE DASHBOARD)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (Last 1 Hour | Locked)")

if "twt_locked_entries_v2" not in st.session_state:
  st.session_state.twt_locked_entries_v2 = load_locked_entries()

locked_entries = st.session_state.twt_locked_entries_v2
locks_updated = False

master_signals = []
asset_engines = {}


def scan_asset(sym):
  df_1d, df_4h, df_5m, _ = fetch_data(sym)
  return sym, df_1d, df_4h, df_5m


with st.spinner("⚡ Scanning 20+ assets (Strict Retest + Wick + Strength Lock)..."):
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(scan_asset, sym): sym for sym in ASSET_LIST}
    for future in as_completed(futures):
      sym, df_1d, df_4h, df_5m = future.result()
      if df_1d is not None and df_4h is not None and df_5m is not None:
        engine = TWTUltimateEngine(
            df_1d,
            df_4h,
            df_5m,
            sym,
            active_tz,
            chop_filter=use_chop_filter,
            strict_confluence=require_full_confluence,
            locked_entries=locked_entries,
            live_window_min=LIVE_WINDOW_MINUTES,
        )
        engine.calculate_and_scan()
        if engine.locks_mutated:
          locks_updated = True
        asset_engines[sym] = engine
        if engine.live_1hr_signals:
          master_signals.extend(engine.live_1hr_signals)

if locks_updated:
  save_locked_entries(locked_entries)

ledger_records = load_ledger()
for eng in asset_engines.values():
  ledger_records = update_ledger_with_engine(ledger_records, eng)
save_ledger(ledger_records)

if master_signals:
  st.success(f"✅ {len(master_signals)} locked signal(s) from the last 1 hour!")
  df_display = pd.DataFrame(master_signals).drop(
      columns=["Is_Live_1Hr", "Age_Min", "Signal_DT", "Session", "Direction"], errors="ignore"
  )
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ Pichhle 1 hour mein strict retest + wick + strength wala koi naya signal nahi aaya hai. Neeche asset select karke locked entries aur ussi asset ki history dekho.")

# ---------------------------------------------------------
# INDIVIDUAL ASSET INSPECTOR & SELECTED ASSET HISTORY ONLY
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector & Locked BUY / SELL Entries")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart & History",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df_1d, df_4h, df_5m, _ = fetch_data(selected_symbol)
  bot = TWTUltimateEngine(
      df_1d,
      df_4h,
      df_5m,
      selected_symbol,
      active_tz,
      chop_filter=use_chop_filter,
      strict_confluence=require_full_confluence,
      locked_entries=locked_entries,
      live_window_min=LIVE_WINDOW_MINUTES,
  )
  bot.calculate_and_scan()
  if bot.locks_mutated:
    save_locked_entries(locked_entries)

if bot.diy_checklist:
  st.markdown(f"#### 🧩 {selected_symbol} 01:30–05:30 POI Status")
  st.dataframe(pd.DataFrame(bot.diy_checklist), width="stretch")

st.markdown(f"### 📈 {selected_symbol} Performance Summary (From {ledger_start_date})")
summary_df = summarize_asset_ledger(ledger_records, selected_symbol, start_date_str=ledger_start_date)
st.dataframe(summary_df, width="stretch")

selected_asset_history = [
    {
        "Asset": r.get("Asset"),
        "Date": r.get("Date"),
        "Time": r.get("Time"),
        "Window": r.get("Window", ""),
        "Type": r.get("Type"),
        "Grade": r.get("Grade"),
        "Zone Buyer %": r.get("Zone Buyer %", "0.0%"),
        "Zone Seller %": r.get("Zone Seller %", "0.0%"),
        "Entry Price": r.get("Entry Price"),
        "Stop Loss (SL)": r.get("Stop Loss (SL)"),
        "TP 1": r.get("TP 1"),
        "TP 2": r.get("TP 2"),
        "TP 3": r.get("TP 3"),
        "Status": r.get("Status"),
        "Setup": r.get("Setup"),
    }
    for r in ledger_records
    if r.get("Asset") == selected_symbol and str(r.get("Date", "")) >= ledger_start_date
]

if selected_asset_history:
  st.markdown(f"### 📜 Trade History for {selected_symbol} Only")
  st.dataframe(pd.DataFrame(selected_asset_history), width="stretch")

if bot.candles_5m:
  curr_p = bot.candles_5m[-1]["close"]

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label=f"Current {selected_symbol} Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Locked Entries (Max 2)", value=len(bot.signals))
  with col3:
    st.metric(label="4H POI Close Time", value=bot.c0530_close.strftime("%H:%M") if bot.c0530_close else "Waiting ⏳")

  if bot.c0130_high is not None and bot.c0130_low is not None:
    st.caption(
        f"📌 01:30–05:30 4H Candle — Supply POI High: {bot.c0130_high:.2f} | "
        f"Demand POI Low: {bot.c0130_low:.2f}"
    )

  if bot.signals:
    st.markdown(f"### 🔒 Locked BUY & SELL Entries for {selected_symbol}")
    df_sig_asset = pd.DataFrame(bot.signals).drop(
        columns=["Is_Live_1Hr", "Age_Min", "Signal_DT", "Session", "Direction"], errors="ignore"
    )
    st.dataframe(df_sig_asset, width="stretch")

  # --- PLOT CHART ---
  fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
  for c in bot.candles_5m:
    col = "#26a69a" if c["close"] >= c["open"] else "#ef5350"
    lower_val = min(c["open"], c["close"])
    body_height = abs(c["close"] - c["open"])
    if body_height == 0:
      body_height = curr_p * 0.0005

    ax.bar(c["id"], body_height, bottom=lower_val, color=col, width=0.6, edgecolor=col)
    ax.vlines(c["id"], c["low"], c["high"], color=col, linewidth=1.2)

  c_ids = [c["id"] for c in bot.candles_5m]
  zl_vals = [c["zl_line"] for c in bot.candles_5m]
  if c_ids and zl_vals:
    ax.plot(c_ids, zl_vals, color="#00e5ff", linewidth=1.5, label="Zero-Lag Trend Line (ZLEMA)")

  if bot.c0130_high is not None:
    ax.axhline(bot.c0130_high, color="#ff5252", linestyle="--", linewidth=1.8, label=f"01:30-05:30 High (Supply POI): {bot.c0130_high:.2f}")
  if bot.c0130_low is not None:
    ax.axhline(bot.c0130_low, color="#00e676", linestyle="--", linewidth=1.8, label=f"01:30-05:30 Low (Demand POI): {bot.c0130_low:.2f}")

  if bot.supply_poi_zone is not None:
    ax.axhspan(bot.supply_poi_zone[0], bot.supply_poi_zone[1], color="#ff5252", alpha=0.16, label="Supply POI Zone")
  if bot.demand_poi_zone is not None:
    ax.axhspan(bot.demand_poi_zone[0], bot.demand_poi_zone[1], color="#00e676", alpha=0.16, label="Demand POI Zone")

  if bot.c0530_close:
    post_candles = [c for c in bot.candles_5m if c["dt_user"] >= bot.c0530_close]
    if post_candles:
      ax.axvline(post_candles[0]["id"], color="#ffd54f", linestyle=":", linewidth=1.6, label=f"05:30 4H Close ({bot.c0530_close.strftime('%H:%M')})")

  lows = [c["low"] for c in bot.candles_5m]
  highs = [c["high"] for c in bot.candles_5m]
  if lows and highs:
    margin = (max(highs) - min(lows)) * 0.09
    if margin == 0:
      margin = curr_p * 0.005
    ax.set_ylim(min(lows) - margin, max(highs) + margin)

  for sig in bot.signals:
    is_b = "BUY" in sig["Type"]
    bg_col = "#00e676" if is_b else "#ff5252"
    txt = f" {'BUY' if is_b else 'SELL'} ({sig['Time']}) "
    ax.text(
        sig["Candle #"],
        sig["Entry Price"],
        txt,
        color="white",
        weight="bold",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_col, alpha=0.9),
        ha="center",
        va="center",
    )

  ax.set_title(
      f"{selected_symbol} 5m Chart — 01:30–05:30 POI (Strict Retest + Wick + Strength)",
      color="white",
      fontsize=13,
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
