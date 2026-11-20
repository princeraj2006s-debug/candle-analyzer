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
    page_title="TWT Multi-Signal 4H Cycle Engine",
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

st.title("⚡ TRADE WITH TUCKYAA — 4H Candle High/Low Multi-Signal Engine")

LEDGER_FILE = Path("twt_signals_ledger.json")
STRENGTH_LOCK_FILE = Path("twt_strength_locks.json")


# ---------------------------------------------------------
# 21:30 SESSION STRENGTH LOCK HELPERS
# ---------------------------------------------------------
def get_2130_session_key(now_user):
  """Returns session key locking strength until the next 21:30 (9:30 PM) 4H candle window."""
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
# PERSISTENT LEDGER HELPERS (PER-ASSET ISOLATED)
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

  for sig in engine.history:
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

ledger_start_date = st.sidebar.text_input(
    "📅 Ledger Start Date (YYYY-MM-DD)", value="2026-11-01"
)

st.sidebar.text("⏱️ Entry Timeframe : 5m")
st.sidebar.text("🔄 Cycle : 4H Candle Close → Next 4H Trade")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 4H Cycle Rules")
st.sidebar.text("01:30→05:30 4H : Trade 05:30–09:30")
st.sidebar.text("05:30→09:30 4H : Trade 09:30–13:30")
st.sidebar.text("09:30→13:30 4H : Trade 13:30–17:30")
st.sidebar.text("Rule : Only At Marked High/Low Zones")
st.sidebar.text("Mode : Multiple Signals Allowed")

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
# 4H HIGH/LOW MULTI-SIGNAL ENGINE
# ---------------------------------------------------------
class TWTUltimateEngine:
  """
    4H Candle Cycle Rule:
    - A 4H candle builds over 4 hours (e.g. 01:30 -> 05:30, 05:30 -> 09:30, 09:30 -> 13:30).
    - Once completed at candle_end (e.g. 05:30), its High & Low are marked as active levels.
    - Entries are searched during the next 4h window [candle_end, candle_end + 4h) ONLY near
      that completed 4H candle's High (SELL rejection/cross) and Low (BUY rejection/cross) —
      NOT in the middle of the range.
    - When the next 4H candle completes (e.g. at 09:30), the old High/Low is replaced by the
      newly completed 4H candle's High/Low.
    - Multiple signals are allowed per window.
    """

  def __init__(
      self,
      df_1d,
      df_4h,
      df_5m,
      symbol,
      user_tz,
      chop_filter=True,
      strength_info=None,
      dfs_by_tf=None,
  ):
    self.df_1d = df_1d
    self.df_4h = df_4h
    self.df_5m = df_5m
    self.symbol = symbol
    self.user_tz = user_tz
    self.chop_filter = chop_filter
    self.strength_info = strength_info or {}
    self.dfs_by_tf = dfs_by_tf or {}
    self.candles_5m = []
    self.history = []
    self.signals = []
    self.diy_checklist = []

    self.active_4h_high = None
    self.active_4h_low = None
    self.active_4h_high_mid = None
    self.active_4h_low_mid = None
    self.active_4h_high_zone = None
    self.active_4h_low_zone = None
    self.active_4h_source_label = ""
    self.active_trade_window_label = ""
    self.active_window_start = None
    self.active_window_end = None

  def _is_choppy(self, c):
    if not self.chop_filter:
      return False
    atr = c["atr"]
    atr_sma = c["atr_sma"]
    ema = c["ema50"]
    close_val = c["close"]
    low_atr = (atr_sma is not None and atr_sma > 0 and atr < (0.75 * atr_sma))
    near_ema_chop = (atr > 0 and abs(close_val - ema) < (0.18 * atr))
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
      return (cl < o) and ((upper_wick / rng) >= 0.15)
    else:
      lower_wick = max(0.0, body_bot - l)
      return (cl > o) and ((lower_wick / rng) >= 0.15)

  def _grade_signal(self, c, side):
    locked_info = self.strength_info
    if side == "BUY":
      htf_ok = bool(locked_info.get("htf_buy_aligned", False))
      zl_5m_ok = c.get("zl_trend") == "BULL"
      zl_htf_ok = bool(locked_info.get("zl_buy_aligned", False))
      str_ok = float(locked_info.get("buyer_pct", 50.0)) >= 52.0
    else:
      htf_ok = bool(locked_info.get("htf_sell_aligned", False))
      zl_5m_ok = c.get("zl_trend") == "BEAR"
      zl_htf_ok = bool(locked_info.get("zl_sell_aligned", False))
      str_ok = float(locked_info.get("seller_pct", 50.0)) >= 52.0

    wick_ok = self._check_candle_wick_rejection(c, side)
    conf_score = sum([htf_ok, wick_ok, zl_5m_ok or zl_htf_ok, str_ok])

    if conf_score >= 3:
      return "A-GRADE 🔥", f"Score {conf_score}/4 | Multi-TF Confirmed"
    return "NORMAL ⚡", f"Score {conf_score}/4 | 4H High/Low Reaction"

  def _build_completed_4h_windows(self, now_user):
    """
    Builds completed 4H candle windows and their corresponding 4-hour trading windows.
    Uses 5m candles to accurately capture custom start times like 01:30 -> 05:30,
    05:30 -> 09:30, 09:30 -> 13:30, 13:30 -> 17:30, 17:30 -> 21:30, 21:30 -> 01:30.
    """
    if not self.candles_5m:
      return []

    first_dt = self.candles_5m[0]["dt_user"]
    # Anchor at 01:30 on first_dt.date() - 1 day
    anchor_base = datetime.combine(
        first_dt.date() - timedelta(days=1), time(1, 30), tzinfo=self.user_tz
    )
    while anchor_base + timedelta(hours=4) < first_dt:
      anchor_base += timedelta(hours=4)

    windows = []
    cur_start = anchor_base
    while cur_start <= now_user:
      cur_end = cur_start + timedelta(hours=4)
      trade_start = cur_end
      trade_end = cur_end + timedelta(hours=4)

      # Only mark High & Low once the 4H candle itself is completed (now_user >= cur_end)
      if now_user >= cur_end:
        sub_4h = [c for c in self.candles_5m if cur_start <= c["dt_user"] < cur_end]
        if len(sub_4h) >= 6:
          h_val = max(c["high"] for c in sub_4h)
          l_val = min(c["low"] for c in sub_4h)
          rng = max(h_val - l_val, 1e-9)
          zone_t = max(rng * 0.12, h_val * 0.0008)
          high_zone = (h_val - zone_t, h_val + zone_t * 0.35)
          low_zone = (l_val - zone_t * 0.35, l_val + zone_t)
          high_mid = (high_zone[0] + high_zone[1]) / 2.0
          low_mid = (low_zone[0] + low_zone[1]) / 2.0

          windows.append({
              "src_start": cur_start,
              "src_end": cur_end,
              "trade_start": trade_start,
              "trade_end": trade_end,
              "src_label": f"{cur_start.strftime('%H:%M')}–{cur_end.strftime('%H:%M')} 4H",
              "trade_label": f"{trade_start.strftime('%H:%M')}–{trade_end.strftime('%H:%M')}",
              "high": h_val,
              "low": l_val,
              "high_zone": high_zone,
              "low_zone": low_zone,
              "high_mid": high_mid,
              "low_mid": low_mid,
          })

      cur_start = cur_end

    return windows

  def _scan_4h_high_low_multiple_signals(self, windows, now_user):
    """
    Scans each completed 4H candle's High & Low during [trade_start, min(trade_end, now_user)].
    Allows MULTIPLE signals (BUY at Low, SELL at High, retests/confirmations on distinct candles).
    Avoids middle-of-range noise by requiring price interaction with High/Low zone or mid.
    """
    all_signals = []
    b_lock = float(self.strength_info.get("buyer_pct", 50.0))
    s_lock = float(self.strength_info.get("seller_pct", 50.0))

    for w in windows:
      w_start = w["trade_start"]
      w_end = min(w["trade_end"], now_user)
      if w_end < w_start:
        continue

      sub = [c for c in self.candles_5m if w_start <= c["dt_user"] <= w_end]
      if not sub:
        continue

      h_val = w["high"]
      l_val = w["low"]
      h_mid = w["high_mid"]
      l_mid = w["low_mid"]
      h_low_edge, h_high_edge = w["high_zone"]
      l_low_edge, l_high_edge = w["low_zone"]

      prev_c = None
      last_sig_candle_id = -999

      for c in sub:
        # Prevent back-to-back spam on immediate consecutive candle unless spaced by >= 2 candles
        if (c["id"] - last_sig_candle_id) < 2:
          prev_c = c
          continue

        if self._is_choppy(c):
          prev_c = c
          continue

        atr = c["atr"]
        dec = get_decimals(c["close"])
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        prev_cl = prev_c["close"] if prev_c is not None else cl
        prev_h = prev_c["high"] if prev_c is not None else h
        prev_l = prev_c["low"] if prev_c is not None else l

        # SELL near 4H High (bearish candle reacting/rejecting/crossing below High / High Mid)
        touched_high = (h >= h_low_edge) or (prev_h >= h_low_edge)
        crossed_high_mid = (cl < h_mid) and (prev_cl >= h_mid or h >= h_mid)
        rejected_high = (cl < o) and touched_high and (cl <= h_high_edge)

        if cl < o and touched_high and (crossed_high_mid or rejected_high):
          grade, conf_note = self._grade_signal(c, "SELL")
          sl = round(max(h_val, h) + (1.2 * atr), dec)
          all_signals.append({
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Window": w["trade_label"],
              "Candle #": c["id"],
              "Type": "SELL 🔴",
              "Grade": grade,
              "Locked Buyer %": f"{b_lock}%",
              "Locked Seller %": f"{s_lock}%",
              "Setup": f"{w['src_label']} High ({h_val:.{dec}f}) SELL | {conf_note}",
              "Entry Price": round(cl, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(cl - (1.2 * atr), dec),
              "TP 2": round(cl - (2.5 * atr), dec),
              "TP 3": round(cl - (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
              "Signal_DT": c["dt_user"].isoformat(),
          })
          last_sig_candle_id = c["id"]

        # BUY near 4H Low (bullish candle reacting/bouncing/crossing above Low / Low Mid)
        touched_low = (l <= l_high_edge) or (prev_l <= l_high_edge)
        crossed_low_mid = (cl > l_mid) and (prev_cl <= l_mid or l <= l_mid)
        bounced_low = (cl > o) and touched_low and (cl >= l_low_edge)

        if cl > o and touched_low and (crossed_low_mid or bounced_low):
          grade, conf_note = self._grade_signal(c, "BUY")
          sl = round(min(l_val, l) - (1.2 * atr), dec)
          all_signals.append({
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Window": w["trade_label"],
              "Candle #": c["id"],
              "Type": "BUY 🟢",
              "Grade": grade,
              "Locked Buyer %": f"{b_lock}%",
              "Locked Seller %": f"{s_lock}%",
              "Setup": f"{w['src_label']} Low ({l_val:.{dec}f}) BUY | {conf_note}",
              "Entry Price": round(cl, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(cl + (1.2 * atr), dec),
              "TP 2": round(cl + (2.5 * atr), dec),
              "TP 3": round(cl + (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
              "Signal_DT": c["dt_user"].isoformat(),
          })
          last_sig_candle_id = c["id"]

        prev_c = c

    return all_signals

  def _build_diy_checklist(self):
    buy_count = sum(1 for s in self.history if "BUY" in s["Type"])
    sell_count = sum(1 for s in self.history if "SELL" in s["Type"])
    self.diy_checklist = [
        {
            "Rule (4H Cycle)": "1. Active Completed 4H Candle High/Low",
            "BUY Side": f"LOW: {self.active_4h_low:.2f}" if self.active_4h_low else "WAIT ⏳",
            "SELL Side": f"HIGH: {self.active_4h_high:.2f}" if self.active_4h_high else "WAIT ⏳",
        },
        {
            "Rule (4H Cycle)": "2. Active Trade Window",
            "BUY Side": self.active_trade_window_label or "WAIT ⏳",
            "SELL Side": self.active_trade_window_label or "WAIT ⏳",
        },
        {
            "Rule (4H Cycle)": "3. Signals Generated (Multiple Allowed)",
            "BUY Side": f"{buy_count} BUY Signal(s)",
            "SELL Side": f"{sell_count} SELL Signal(s)",
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

    for idx, row in enumerate(self.df_5m.iterrows()):
      dt_utc = row[0]
      dt_user = dt_utc.astimezone(self.user_tz)
      r = row[1]
      o, h, l, c, v = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]), float(r["Volume"])
      atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)
      atr_sma = float(r["ATR_SMA"]) if not pd.isna(r["ATR_SMA"]) else atr
      ema50 = float(r["EMA50"]) if not pd.isna(r["EMA50"]) else c
      zl_val = float(zl_line.iloc[idx]) if not zl_line.empty and not pd.isna(zl_line.iloc[idx]) else c
      zl_b = float(zl_band.iloc[idx]) if not zl_band.empty and not pd.isna(zl_band.iloc[idx]) else (atr * 1.2)
      zl_tr = str(zl_trend_s.iloc[idx]) if not zl_trend_s.empty else ("BULL" if c >= o else "BEAR")
      age_min = (current_utc_time - dt_utc).total_seconds() / 60.0

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
          "is_within_1hr": age_min <= 60.0
      })

    windows = self._build_completed_4h_windows(now_user)
    if windows:
      active_w = windows[-1]
      self.active_4h_high = active_w["high"]
      self.active_4h_low = active_w["low"]
      self.active_4h_high_mid = active_w["high_mid"]
      self.active_4h_low_mid = active_w["low_mid"]
      self.active_4h_high_zone = active_w["high_zone"]
      self.active_4h_low_zone = active_w["low_zone"]
      self.active_4h_source_label = active_w["src_label"]
      self.active_trade_window_label = active_w["trade_label"]
      self.active_window_start = active_w["trade_start"]
      self.active_window_end = active_w["trade_end"]

    collected = self._scan_4h_high_low_multiple_signals(windows, now_user)
    collected.sort(key=lambda x: x["Candle #"])

    self.history = collected
    self.signals = [s for s in collected if s.get("Is_Within_1hr", False)]
    self._build_diy_checklist()


# ---------------------------------------------------------
# MASTER SCANNER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (Multiple Signals Enabled)")

if "twt_strength_locks" not in st.session_state:
  st.session_state.twt_strength_locks = load_strength_locks()

strength_locks = st.session_state.twt_strength_locks
locks_updated = False

master_signals = []
asset_engines = {}


def scan_asset(sym):
  df_1d, df_4h, df_5m, dfs_by_tf = fetch_data(sym)
  return sym, df_1d, df_4h, df_5m, dfs_by_tf


with st.spinner("⚡ Scanning 20+ assets across 4H completed High/Low windows..."):
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
            strength_info=locked_strength,
            dfs_by_tf=dfs_by_tf,
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
  st.success(f"✅ Found {len(master_signals)} recent signal(s) across assets!")
  df_display = pd.DataFrame(master_signals).drop(columns=["Is_Within_1hr", "Signal_DT"], errors="ignore")
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No new signals within the last 60 minutes. Check individual asset history below for all generated trades.")

# ---------------------------------------------------------
# MANUAL ASSET INSPECTOR & FULL TRADE HISTORY
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector & Complete Trade History")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart",
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
      strength_info=locked_strength,
      dfs_by_tf=dfs_by_tf,
  )
  bot.calculate_and_scan()

if bot.diy_checklist:
  st.markdown(f"#### 🧩 {selected_symbol} 4H Cycle Status")
  st.dataframe(pd.DataFrame(bot.diy_checklist), width="stretch")

st.markdown(f"### 📈 {selected_symbol} Performance Summary (From {ledger_start_date})")
summary_df = summarize_asset_ledger(ledger_records, selected_symbol, start_date_str=ledger_start_date)
st.dataframe(summary_df, width="stretch")

# Full saved history table for selected asset
asset_history_rows = [
    {
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

if asset_history_rows:
  st.markdown(f"### 📜 Complete Trade History for {selected_symbol} (Time, Window & Result)")
  st.dataframe(pd.DataFrame(asset_history_rows), width="stretch")

if bot.candles_5m:
  curr_p = bot.candles_5m[-1]["close"]

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label=f"Current {selected_symbol} Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Total Signals in Current Window(s)", value=len(bot.history))
  with col3:
    st.metric(label="Active 4H Source", value=bot.active_4h_source_label or "Waiting ⏳")

  if bot.active_4h_high is not None and bot.active_4h_low is not None:
    st.caption(
        f"📌 Active Completed 4H ({bot.active_4h_source_label}) — "
        f"High: {bot.active_4h_high:.2f} | Low: {bot.active_4h_low:.2f} | "
        f"Trade Window: {bot.active_trade_window_label}"
    )

  if bot.history:
    st.markdown(f"### 📋 Generated Signals Feed for {selected_symbol}")
    df_history = pd.DataFrame(bot.history).drop(columns=["Is_Within_1hr", "Signal_DT"], errors="ignore")
    st.dataframe(df_history, width="stretch")

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

  if bot.active_4h_high is not None:
    ax.axhline(bot.active_4h_high, color="#ff5252", linestyle="--", linewidth=1.8, label=f"Active 4H High ({bot.active_4h_source_label}): {bot.active_4h_high:.2f}")
  if bot.active_4h_low is not None:
    ax.axhline(bot.active_4h_low, color="#00e676", linestyle="--", linewidth=1.8, label=f"Active 4H Low ({bot.active_4h_source_label}): {bot.active_4h_low:.2f}")

  if bot.active_4h_high_zone is not None:
    ax.axhspan(bot.active_4h_high_zone[0], bot.active_4h_high_zone[1], color="#ff5252", alpha=0.14, label="4H High Sell Zone")
  if bot.active_4h_low_zone is not None:
    ax.axhspan(bot.active_4h_low_zone[0], bot.active_4h_low_zone[1], color="#00e676", alpha=0.14, label="4H Low Buy Zone")

  if bot.active_window_start:
    w_candles = [c for c in bot.candles_5m if c["dt_user"] >= bot.active_window_start]
    if w_candles:
      ax.axvline(w_candles[0]["id"], color="#ffd54f", linestyle=":", linewidth=1.6, label=f"Trade Window Start ({bot.active_trade_window_label})")

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
      f"{selected_symbol} 5m Chart — Active 4H High/Low ({bot.active_4h_source_label}) Multi-Signal",
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
