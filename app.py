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

st.title("⚡ TRADE WITH TUCKYAA — 4H POI Engine (Last 1Hr Live | Max 1 BUY + 1 SELL)")

LEDGER_FILE = Path("twt_signals_ledger_v3.json")
STRENGTH_LOCK_FILE = Path("twt_strength_locks.json")
LIVE_WINDOW_MINUTES = 60.0  # Only signals within last 1 hour stay on Live Dashboard


# ---------------------------------------------------------
# 21:30 SESSION STRENGTH LOCK HELPERS
# ---------------------------------------------------------
def get_2130_session_key(now_user):
  anchor_2130_today = datetime.combine(now_user.date(), time(21, 30), tzinfo=now_user.tzinfo)
  if now_user >= anchor_2130_today:
    session_start = anchor_2130_today
  else:
    session_start = anchor_2130_today - timedelta(days=1)
  return session_start.strftime("%Y-%m-%d_%H:%M")


def load_strength_locks():
  if STRENGTH_LOCK_FILE.exists():
    try:
      data = json.loads(STRENGTH_LOCK_FILE.read_text(encoding="utf-8"))
      if isinstance(data, dict):
        return data
    except Exception:
      pass
  return {}


def save_strength_locks(locks):
  try:
    STRENGTH_LOCK_FILE.write_text(json.dumps(locks, indent=2), encoding="utf-8")
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

  # Save all detected signals (including >1hr expired live signals) into per-asset history
  for sig in engine.all_detected_signals:
    sig_id = f"{sig['Asset']}|{sig['Date']}|{sig['Time']}|{sig['Type']}|{sig['Entry Price']}"
    if sig_id not in by_id:
      by_id[sig_id] = {
          "id": sig_id,
          "Asset": sig["Asset"],
          "Date": sig["Date"],
          "Time": sig["Time"],
          "Window": sig.get("Window", ""),
          "Direction": "BUY" if "BUY" in sig["Type"] else "SELL",
          "Type": sig["Type"],
          "Grade": sig.get("Grade", "NORMAL ⚡"),
          "Locked Buyer %": sig.get("Locked Buyer %", "50.0%"),
          "Locked Seller %": sig.get("Locked Seller %", "50.0%"),
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
          "Status": "OPEN",
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
    "🎯 Strict All-Confluence Filter", value=False
)

ledger_start_date = st.sidebar.text_input(
    "📅 Ledger Start Date (YYYY-MM-DD)", value="2026-11-01"
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 4H POI + Confluence Rules")
st.sidebar.text("Anchor   : 01:30→05:30 4H Candle")
st.sidebar.text("Live Window: Last 1 Hour Only")
st.sidebar.text("POI High : Supply POI (Max 1 SELL)")
st.sidebar.text("POI Low  : Demand POI (Max 1 BUY)")
st.sidebar.text("History  : Selected Asset Only")

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


def _candle_strength_at_row(row):
  if row is None:
    return 50.0, 50.0, "Neutral ⚪"
  o = float(row["Open"])
  h = float(row["High"])
  l = float(row["Low"])
  c = float(row["Close"])
  rng = max(h - l, 1e-9)
  body_ratio = min(1.0, abs(c - o) / rng)

  if c >= o:
    close_pos = (c - l) / rng
    buyer = round(min(100.0, 50.0 + (body_ratio * 35.0) + ((close_pos - 0.5) * 30.0)), 1)
    seller = round(max(0.0, 100.0 - buyer), 1)
    label = "Buyer Green 🟢"
  else:
    close_pos = (c - l) / rng
    seller = round(min(100.0, 50.0 + (body_ratio * 35.0) + ((0.5 - close_pos) * 30.0)), 1)
    buyer = round(max(0.0, 100.0 - seller), 1)
    label = "Seller Red 🔴"

  return buyer, seller, label


def _pick_candle_at_or_before(df, dt_utc=None, use_closed_latest=True):
  if df is None or df.empty:
    return None, -1
  if dt_utc is not None:
    sub = df[df.index <= dt_utc]
    if not sub.empty:
      idx = len(sub) - 1
      return sub.iloc[-1], idx
  if use_closed_latest and len(df) >= 2:
    return df.iloc[-2], len(df) - 2
  return df.iloc[-1], len(df) - 1


def compute_mtf_candle_strength(dfs_by_tf, dt_utc=None, use_closed_latest=True):
  weights = {
      "1d": 0.30,
      "4h": 0.25,
      "1h": 0.20,
      "15m": 0.15,
      "5m": 0.10,
  }
  rows = []
  w_buyer = 0.0
  w_seller = 0.0
  tot_w = 0.0
  tf_scores = {}
  zl_mtf = {}

  for tf in ("5m", "15m", "1h", "4h", "1d"):
    df = dfs_by_tf.get(tf)
    row, r_idx = _pick_candle_at_or_before(df, dt_utc=dt_utc, use_closed_latest=use_closed_latest)
    b_val, s_val, c_label = _candle_strength_at_row(row)
    _, _, zl_series = _compute_zero_lag_series(df, length=21, band_mult=1.2)
    if zl_series is not None and not zl_series.empty and 0 <= r_idx < len(zl_series):
      zl_state = str(zl_series.iloc[r_idx])
    else:
      zl_state = "BULL" if b_val >= s_val else "BEAR"
    zl_mtf[tf] = zl_state

    tf_scores[tf] = {"buyer": b_val, "seller": s_val}
    w = weights[tf]
    w_buyer += b_val * w
    w_seller += s_val * w
    tot_w += w
    rows.append({
        "Timeframe": tf,
        "Candle": c_label,
        "Zero-Lag Trend": "Bullish 🟢" if zl_state == "BULL" else "Bearish 🔴",
        "Buyer Strength": f"{b_val}%",
        "Seller Strength": f"{s_val}%",
        "Dominant": "BUYER 🟢" if b_val >= s_val else "SELLER 🔴",
    })

  avg_buyer = round(w_buyer / tot_w, 1) if tot_w > 0 else 50.0
  avg_seller = round(w_seller / tot_w, 1) if tot_w > 0 else 50.0
  diff = round(avg_buyer - avg_seller, 1)

  h4_b = tf_scores.get("4h", {}).get("buyer", 50.0)
  h4_s = tf_scores.get("4h", {}).get("seller", 50.0)
  d1_b = tf_scores.get("1d", {}).get("buyer", 50.0)
  d1_s = tf_scores.get("1d", {}).get("seller", 50.0)

  htf_buy_aligned = (h4_b >= h4_s) and (d1_b >= d1_s)
  htf_sell_aligned = (h4_s > h4_b) and (d1_s > d1_b)

  zl_buy_aligned = (zl_mtf.get("4h") == "BULL") and (zl_mtf.get("1d") == "BULL")
  zl_sell_aligned = (zl_mtf.get("4h") == "BEAR") and (zl_mtf.get("1d") == "BEAR")

  if diff >= 8.0:
    verdict = f"BUY Zyada Faydemand 🟢 (+{diff}% Buyer Edge)"
  elif diff <= -8.0:
    verdict = f"SELL Zyada Faydemand 🔴 (+{abs(diff)}% Seller Edge)"
  else:
    verdict = f"Balanced / Mixed ⚖️ (Buyer {avg_buyer}% vs Seller {avg_seller}%)"

  return {
      "buyer_pct": avg_buyer,
      "seller_pct": avg_seller,
      "diff": diff,
      "verdict": verdict,
      "rows": rows,
      "htf_buy_aligned": htf_buy_aligned,
      "htf_sell_aligned": htf_sell_aligned,
      "zl_mtf": zl_mtf,
      "zl_buy_aligned": zl_buy_aligned,
      "zl_sell_aligned": zl_sell_aligned,
  }


def get_or_lock_2130_strength(symbol, user_tz, dfs_by_tf, locks_dict):
  now_user = datetime.now(timezone.utc).astimezone(user_tz)
  session_key = get_2130_session_key(now_user)
  lock_key = f"{symbol}|{session_key}"

  if lock_key in locks_dict and isinstance(locks_dict[lock_key], dict):
    cached = dict(locks_dict[lock_key])
    cached["session_key"] = session_key
    return cached, False

  computed = compute_mtf_candle_strength(dfs_by_tf, dt_utc=None, use_closed_latest=True)
  computed["session_key"] = session_key
  locks_dict[lock_key] = computed
  return dict(computed), True


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
# 01:30 -> 05:30 4H POI ENGINE (LAST 1 HR LIVE | MAX 1 BUY + 1 SELL)
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
      strength_info=None,
      dfs_by_tf=None,
      live_window_min=LIVE_WINDOW_MINUTES,
  ):
    self.df_1d = df_1d
    self.df_4h = df_4h
    self.df_5m = df_5m
    self.symbol = symbol
    self.user_tz = user_tz
    self.chop_filter = chop_filter
    self.strict_confluence = strict_confluence
    self.strength_info = strength_info or {}
    self.dfs_by_tf = dfs_by_tf or {}
    self.live_window_min = live_window_min

    self.candles_5m = []
    self.all_detected_signals = []  # Max 1 BUY + 1 SELL (saved to per-asset history)
    self.history = []              # Live signals within last 1 hour
    self.signals = []              # Live signals within last 1 hour (Max 1 BUY + 1 SELL)
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

  def _check_candle_wick_rejection(self, c, side):
    o = c["open"]
    h = c["high"]
    l = c["low"]
    cl = c["close"]
    rng = max(h - l, 1e-9)
    body_top = max(o, cl)
    body_bot = min(o, cl)

    if side == "SELL":
      upper_wick = max(0.0, h - body_top)
      return (cl < o) and ((upper_wick / rng) >= 0.05 or upper_wick > 0)
    else:
      lower_wick = max(0.0, body_bot - l)
      return (cl > o) and ((lower_wick / rng) >= 0.05 or lower_wick > 0)

  def _evaluate_all_confluences(self, c, side):
    locked_info = self.strength_info
    if side == "BUY":
      htf_ok = bool(locked_info.get("htf_buy_aligned", False))
      zl_5m_ok = c.get("zl_trend") == "BULL"
      zl_htf_ok = bool(locked_info.get("zl_buy_aligned", False))
      str_ok = float(locked_info.get("buyer_pct", 50.0)) >= 45.0
    else:
      htf_ok = bool(locked_info.get("htf_sell_aligned", False))
      zl_5m_ok = c.get("zl_trend") == "BEAR"
      zl_htf_ok = bool(locked_info.get("zl_sell_aligned", False))
      str_ok = float(locked_info.get("seller_pct", 50.0)) >= 45.0

    wick_ok = self._check_candle_wick_rejection(c, side)
    zl_ok = zl_5m_ok or zl_htf_ok
    score = sum([htf_ok, wick_ok, zl_ok, str_ok])

    if self.strict_confluence:
      passed = (score >= 3) and zl_ok and (wick_ok or htf_ok)
    else:
      passed = (score >= 2)

    grade = "A-GRADE 🔥" if score >= 3 else "NORMAL ⚡"
    note = f"POI + Conf {score}/4 | ZL:{'OK' if zl_ok else 'Mix'} | HTF:{'OK' if htf_ok else 'Mix'} | Wick:{'OK' if wick_ok else 'Lt'}"
    return passed, grade, note

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

  def _scan_poi_confluences(self, now_user):
    """
    Finds at most 1 SELL (near 01:30-05:30 High POI) and 1 BUY (near 01:30-05:30 Low POI).
    Uses exact candle open timestamp (`dt_user`) so entry time stays fixed across refreshes.
    """
    if not self.c0130_closed or self.c0530_close is None:
      return []

    sub = [c for c in self.candles_5m if self.c0530_close <= c["dt_user"] <= now_user]
    if not sub:
      return []

    sup_low, sup_high = self.supply_poi_zone
    dem_low, dem_high = self.demand_poi_zone
    b_lock = float(self.strength_info.get("buyer_pct", 50.0))
    s_lock = float(self.strength_info.get("seller_pct", 50.0))

    latest_by_side = {}

    for c in reversed(sub):
      if len(latest_by_side) >= 2:
        break

      if self._is_choppy(c):
        continue

      atr = c["atr"]
      dec = get_decimals(c["close"])
      o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
      age_min = (now_user - c["dt_user"]).total_seconds() / 60.0
      is_live_1hr = (0.0 <= age_min <= self.live_window_min)

      near_high_poi = (h >= sup_low) and (cl <= sup_high * 1.005)
      if "SELL" not in latest_by_side and near_high_poi and (cl < o):
        ok, grade, conf_note = self._evaluate_all_confluences(c, "SELL")
        if ok:
          sl = round(max(self.c0130_high, h) + (1.2 * atr), dec)
          latest_by_side["SELL"] = {
              "Asset": self.symbol,
              "Date": c["dt_user"].strftime("%Y-%m-%d"),
              "Time": c["dt_user"].strftime("%H:%M"),
              "Window": f"Post-{self.c0530_close.strftime('%H:%M')} POI",
              "Candle #": c["id"],
              "Type": "SELL 🔴",
              "Grade": grade,
              "Locked Buyer %": f"{b_lock}%",
              "Locked Seller %": f"{s_lock}%",
              "Setup": f"Near 01:30–05:30 High ({self.c0130_high:.{dec}f}) SELL | {conf_note}",
              "Entry Price": round(cl, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(cl - (1.2 * atr), dec),
              "TP 2": round(cl - (2.5 * atr), dec),
              "TP 3": round(cl - (4.0 * atr), dec),
              "Age_Min": round(age_min, 1),
              "Is_Live_1Hr": is_live_1hr,
              "Signal_DT": c["dt_user"].isoformat(),
          }
          continue

      near_low_poi = (l <= dem_high) and (cl >= dem_low * 0.995)
      if "BUY" not in latest_by_side and near_low_poi and (cl > o):
        ok, grade, conf_note = self._evaluate_all_confluences(c, "BUY")
        if ok:
          sl = round(min(self.c0130_low, l) - (1.2 * atr), dec)
          latest_by_side["BUY"] = {
              "Asset": self.symbol,
              "Date": c["dt_user"].strftime("%Y-%m-%d"),
              "Time": c["dt_user"].strftime("%H:%M"),
              "Window": f"Post-{self.c0530_close.strftime('%H:%M')} POI",
              "Candle #": c["id"],
              "Type": "BUY 🟢",
              "Grade": grade,
              "Locked Buyer %": f"{b_lock}%",
              "Locked Seller %": f"{s_lock}%",
              "Setup": f"Near 01:30–05:30 Low ({self.c0130_low:.{dec}f}) BUY | {conf_note}",
              "Entry Price": round(cl, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(cl + (1.2 * atr), dec),
              "TP 2": round(cl + (2.5 * atr), dec),
              "TP 3": round(cl + (4.0 * atr), dec),
              "Age_Min": round(age_min, 1),
              "Is_Live_1Hr": is_live_1hr,
              "Signal_DT": c["dt_user"].isoformat(),
          }

    found = list(latest_by_side.values())
    found.sort(key=lambda x: x["Signal_DT"])
    return found

  def _build_diy_checklist(self):
    buy_cnt = sum(1 for s in self.signals if "BUY" in s["Type"])
    sell_cnt = sum(1 for s in self.signals if "SELL" in s["Type"])
    self.diy_checklist = [
        {
            "Rule (01:30–05:30 POI)": "1. 4H Closed High / Low",
            "BUY Side (Near Low)": f"Low: {self.c0130_low:.2f}" if self.c0130_low else "WAIT ⏳",
            "SELL Side (Near High)": f"High: {self.c0130_high:.2f}" if self.c0130_high else "WAIT ⏳",
        },
        {
            "Rule (01:30–05:30 POI)": "2. POI Zone Range",
            "BUY Side (Near Low)": f"{self.demand_poi_zone[0]:.2f} – {self.demand_poi_zone[1]:.2f}" if self.demand_poi_zone else "N/A",
            "SELL Side (Near High)": f"{self.supply_poi_zone[0]:.2f} – {self.supply_poi_zone[1]:.2f}" if self.supply_poi_zone else "N/A",
        },
        {
            "Rule (01:30–05:30 POI)": "3. Live Signals (Last 1 Hr)",
            "BUY Side (Near Low)": f"{buy_cnt} / 1 BUY",
            "SELL Side (Near High)": f"{sell_cnt} / 1 SELL",
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

    sigs = self._scan_poi_confluences(now_user)
    self.all_detected_signals = sigs

    # Auto-remove signals older than 1 hour from live dashboard
    live_1hr_sigs = [s for s in sigs if s.get("Is_Live_1Hr", False)]
    self.signals = live_1hr_sigs
    self.history = live_1hr_sigs
    self._build_diy_checklist()


# ---------------------------------------------------------
# MASTER SCANNER (ONLY LAST 1 HOUR SIGNALS ON LIVE DASHBOARD)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (Last 1 Hour Only)")

if "twt_strength_locks" not in st.session_state:
  st.session_state.twt_strength_locks = load_strength_locks()

strength_locks = st.session_state.twt_strength_locks
locks_updated = False

master_signals = []
asset_engines = {}


def scan_asset(sym):
  df_1d, df_4h, df_5m, dfs_by_tf = fetch_data(sym)
  return sym, df_1d, df_4h, df_5m, dfs_by_tf


with st.spinner("⚡ Scanning 20+ assets (Showing only last 1 hour signals on Live Dashboard)..."):
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(scan_asset, sym): sym for sym in ASSET_LIST}
    for future in as_completed(futures):
      sym, df_1d, df_4h, df_5m, dfs_by_tf = future.result()
      if df_1d is not None and df_4h is not None and df_5m is not None:
        locked_strength, newly_computed = get_or_lock_2130_strength(
            sym, active_tz, dfs_by_tf, strength_locks
        )
        if newly_computed:
          locks_updated = True
        engine = TWTUltimateEngine(
            df_1d,
            df_4h,
            df_5m,
            sym,
            active_tz,
            chop_filter=use_chop_filter,
            strict_confluence=require_full_confluence,
            strength_info=locked_strength,
            dfs_by_tf=dfs_by_tf,
            live_window_min=LIVE_WINDOW_MINUTES,
        )
        engine.calculate_and_scan()
        asset_engines[sym] = engine
        if engine.signals:
          master_signals.extend(engine.signals)

if locks_updated:
  save_strength_locks(strength_locks)

ledger_records = load_ledger()
for eng in asset_engines.values():
  ledger_records = update_ledger_with_engine(ledger_records, eng)
save_ledger(ledger_records)

if master_signals:
  st.success(f"✅ {len(master_signals)} active signal(s) from the last 1 hour!")
  df_display = pd.DataFrame(master_signals).drop(
      columns=["Is_Live_1Hr", "Age_Min", "Signal_DT"], errors="ignore"
  )
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ Pichhle 1 hour mein koi naya POI signal nahi aaya hai. Purane signals neeche Selected Asset ki History mein hain.")

# ---------------------------------------------------------
# INDIVIDUAL ASSET INSPECTOR & SELECTED ASSET HISTORY ONLY
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector & Selected Asset History")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart & History",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df_1d, df_4h, df_5m, dfs_by_tf = fetch_data(selected_symbol)
  locked_strength, newly_computed = get_or_lock_2130_strength(
      selected_symbol, active_tz, dfs_by_tf, strength_locks
  )
  if newly_computed:
    save_strength_locks(strength_locks)
  bot = TWTUltimateEngine(
      df_1d,
      df_4h,
      df_5m,
      selected_symbol,
      active_tz,
      chop_filter=use_chop_filter,
      strict_confluence=require_full_confluence,
      strength_info=locked_strength,
      dfs_by_tf=dfs_by_tf,
      live_window_min=LIVE_WINDOW_MINUTES,
  )
  bot.calculate_and_scan()

if bot.diy_checklist:
  st.markdown(f"#### 🧩 {selected_symbol} 01:30–05:30 POI Status")
  st.dataframe(pd.DataFrame(bot.diy_checklist), width="stretch")

st.markdown(f"### 📈 {selected_symbol} Performance Summary (From {ledger_start_date})")
summary_df = summarize_asset_ledger(ledger_records, selected_symbol, start_date_str=ledger_start_date)
st.dataframe(summary_df, width="stretch")

# Strictly filtered to ONLY `selected_symbol`
selected_asset_history = [
    {
        "Asset": r.get("Asset"),
        "Date": r.get("Date"),
        "Time": r.get("Time"),
        "Window": r.get("Window", ""),
        "Type": r.get("Type"),
        "Grade": r.get("Grade"),
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
    st.metric(label="Live Signals (Last 1 Hr)", value=len(bot.signals))
  with col3:
    st.metric(label="4H POI Close Time", value=bot.c0530_close.strftime("%H:%M") if bot.c0530_close else "Waiting ⏳")

  if bot.c0130_high is not None and bot.c0130_low is not None:
    st.caption(
        f"📌 01:30–05:30 4H Candle — Supply POI High: {bot.c0130_high:.2f} | "
        f"Demand POI Low: {bot.c0130_low:.2f}"
    )

  if bot.signals:
    st.markdown(f"### 📋 Live Signals (Last 1 Hr) for {selected_symbol}")
    df_live_asset = pd.DataFrame(bot.signals).drop(
        columns=["Is_Live_1Hr", "Age_Min", "Signal_DT"], errors="ignore"
    )
    st.dataframe(df_live_asset, width="stretch")

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
      f"{selected_symbol} 5m Chart — 01:30–05:30 POI (Last 1Hr Live Signals)",
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
