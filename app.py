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

selected_tf = "5m"
st.sidebar.text("⏱️ Entry Timeframe : 5m (Fixed)")
st.sidebar.text("🕒 Session Window : Signals Only >= 21:30")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Master Confluence Rules")
st.sidebar.text("Anchors  : 01:30 (4H), 05:30 (6H), 17:30 (4H->21:30)")
st.sidebar.text("No-Entry : 17:30 to 21:30 Blocked")
st.sidebar.text("Signal   : Only After 21:30 PM")
st.sidebar.text("17:30 Zone: High-Low Zone + Mid Point")
st.sidebar.text("High Confl: 01:30/05:30 High Inside 17:30 -> Stronger Side")
st.sidebar.text("01:30/05:30 High: SELL Only | Low: BUY Only")
st.sidebar.text("Zero-Lag : ZLEMA + ATR Band Trend")

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
  """Zero-Lag Trend (AlgoAlpha MTF style): lag-compensated EMA + ATR volatility band."""
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
  """Returns (buyer_score_0_to_100, seller_score_0_to_100, label) from a candle row."""
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
  """Pick candle row at dt_utc if provided, else latest closed candle (iloc[-2]) if possible."""
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
  """Compute Buyer vs Seller candle strength + Zero-Lag MTF states across 5m, 15m, 1h, 4h, 1d."""
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
  """Calculate strength once per 21:30 4H cycle and lock until the next 21:30 candle."""
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
        ohlcv_5m = exchange.fetch_ohlcv(sym, timeframe="5m", limit=350)
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
# CUSTOM SESSION & CANDLE ANCHOR ENGINE
# - 01:30 AM (4H) High & Low (High -> SELL only, Low -> BUY only)
# - 05:30 AM (6H) High & Low (High -> SELL only, Low -> BUY only)
# - 17:30 PM (4H -> closes 21:30 PM) High-Low Zone & Mid Point
# - If 01:30 or 05:30 High lies inside 17:30 zone -> check stronger side at 17:30 high
# - No signals between 17:30 PM and 21:30 PM
# - Signals ONLY after 21:30 PM
# - No 21:30 4H demand/supply zone or mid point
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

    self.prev_high = None
    self.prev_low = None

    # 01:30 AM 4H candle High & Low
    self.c0130_high = None
    self.c0130_low = None

    # 05:30 AM 6H candle High & Low
    self.c0530_6h_high = None
    self.c0530_6h_low = None

    # 17:30 PM 4H candle (closes at 21:30 PM) -> Zone (High-Low) & Mid Point
    self.c1730_4h_start = None
    self.c1730_4h_close = None
    self.c1730_4h_closed = False
    self.c1730_4h_high = None
    self.c1730_4h_low = None
    self.c1730_4h_mid = None

    # High confluence inside 17:30 zone
    self.high_confluence_in_1730 = False
    self.confluence_highs_matched = []

    # Signal cutoff: only after 21:30 PM
    self.signal_start_2130 = None

  def _buyer_pct(self):
    return float(self.strength_info.get("buyer_pct", 50.0))

  def _seller_pct(self):
    return float(self.strength_info.get("seller_pct", 50.0))

  def _is_buyer_strong(self):
    b_pct = self._buyer_pct()
    s_pct = self._seller_pct()
    return (b_pct > s_pct) and (b_pct >= 52.0)

  def _is_seller_strong(self):
    b_pct = self._buyer_pct()
    s_pct = self._seller_pct()
    return (s_pct > b_pct) and (s_pct >= 52.0)

  def _stronger_side_at_1730_high(self):
    """Returns 'BUY' if Buyer is stronger, 'SELL' if Seller is stronger, else None."""
    b_pct = self._buyer_pct()
    s_pct = self._seller_pct()
    if b_pct > s_pct:
      return "BUY"
    if s_pct > b_pct:
      return "SELL"
    return None

  def _is_choppy(self, c):
    if not self.chop_filter:
      return False
    atr = c["atr"]
    atr_sma = c["atr_sma"]
    ema = c["ema50"]
    close_val = c["close"]
    low_atr = (atr_sma is not None and atr_sma > 0 and atr < (0.85 * atr_sma))
    near_ema_chop = (atr > 0 and abs(close_val - ema) < (0.25 * atr))
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
      str_ok = self._is_buyer_strong()
    else:
      htf_ok = bool(locked_info.get("htf_sell_aligned", False))
      zl_5m_ok = c.get("zl_trend") == "BEAR"
      zl_htf_ok = bool(locked_info.get("zl_sell_aligned", False))
      str_ok = self._is_seller_strong()

    wick_ok = self._check_candle_wick_rejection(c, side)
    conf_score = sum([htf_ok, wick_ok, zl_5m_ok or zl_htf_ok, str_ok])

    if conf_score >= 3 and (htf_ok or (zl_5m_ok and wick_ok)):
      return "A-GRADE 🔥", f"Score {conf_score}/4 | ZL+HTF+Wick"
    parts = [
        f"Score {conf_score}/4",
        "ZL OK" if (zl_5m_ok or zl_htf_ok) else "ZL Mix",
        "4H/1D OK" if htf_ok else "4H/1D Mix",
        "Wick OK" if wick_ok else "Wick Lt",
    ]
    return "NORMAL ⚡", " | ".join(parts)

  def _build_diy_checklist(self):
    has_buy = any("BUY" in s["Type"] for s in self.history)
    has_sell = any("SELL" in s["Type"] for s in self.history)
    ref_info = self.strength_info
    last_zl = self.candles_5m[-2]["zl_trend"] if len(self.candles_5m) >= 2 else (self.candles_5m[-1]["zl_trend"] if self.candles_5m else "BULL")
    zl_htf_buy = bool(ref_info.get("zl_buy_aligned", False))
    zl_htf_sell = bool(ref_info.get("zl_sell_aligned", False))
    htf_buy = bool(ref_info.get("htf_buy_aligned", False))
    htf_sell = bool(ref_info.get("htf_sell_aligned", False))

    self.diy_checklist = [
        {
            "Rule (DIY Builder)": "1. 17:30 4H Zone Closed at 21:30 PM",
            "BUY Side": "READY ✅" if self.c1730_4h_closed else "WAITING 21:30 ⏳",
            "SELL Side": "READY ✅" if self.c1730_4h_closed else "WAITING 21:30 ⏳",
        },
        {
            "Rule (DIY Builder)": "2. Post-21:30 Entry (17:30-21:30 Blocked)",
            "BUY Side": "ACTIVE ✅" if has_buy else "WAITING >=21:30 ⏳",
            "SELL Side": "ACTIVE ✅" if has_sell else "WAITING >=21:30 ⏳",
        },
        {
            "Rule (DIY Builder)": "3. High Inside 17:30 Zone Confluence",
            "BUY Side": "MATCHED ✅" if self.high_confluence_in_1730 else "NO MATCH ⚪",
            "SELL Side": "MATCHED ✅" if self.high_confluence_in_1730 else "NO MATCH ⚪",
        },
        {
            "Rule (DIY Builder)": "4. 01:30(4H) & 05:30(6H) High=SELL / Low=BUY Only",
            "BUY Side": "LOW-ONLY ✅" if self._is_buyer_strong() else "BUY WEAK ❌",
            "SELL Side": "HIGH-ONLY ✅" if self._is_seller_strong() else "SELL WEAK ❌",
        },
        {
            "Rule (DIY Builder)": "5. Zero-Lag Trend (ZLEMA + ATR Band)",
            "BUY Side": "BULLISH ✅" if last_zl == "BULL" else "BEARISH ⏳",
            "SELL Side": "BEARISH ✅" if last_zl == "BEAR" else "BULLISH ⏳",
        },
        {
            "Rule (DIY Builder)": "6. Zero-Lag MTF (4H + 1D Trend)",
            "BUY Side": "ALIGNED ✅" if zl_htf_buy else "NOT ALIGNED ⏳",
            "SELL Side": "ALIGNED ✅" if zl_htf_sell else "NOT ALIGNED ⏳",
        },
        {
            "Rule (DIY Builder)": "7. HTF Candle Strength (4H + 1D)",
            "BUY Side": "STRONG ✅" if htf_buy else "MIXED ⏳",
            "SELL Side": "STRONG ✅" if htf_sell else "MIXED ⏳",
        },
        {
            "Rule (DIY Builder)": "8. Chop / Sideways Filter",
            "BUY Side": "CLEAR ✅" if self.chop_filter else "OFF ⚪",
            "SELL Side": "CLEAR ✅" if self.chop_filter else "OFF ⚪",
        },
    ]

  def _range_high_low(self, start_dt, end_dt):
    """Computes high and low between start_dt (inclusive) and end_dt (exclusive)."""
    highs = []
    lows = []
    for c in self.candles_5m:
      if start_dt <= c["dt_user"] < end_dt:
        highs.append(c["high"])
        lows.append(c["low"])

    if not highs and self.dfs_by_tf.get("1h") is not None:
      df_1h = self.dfs_by_tf["1h"]
      for dt_utc, r in df_1h.iterrows():
        dt_u = dt_utc.astimezone(self.user_tz)
        if start_dt <= dt_u < end_dt:
          highs.append(float(r["High"]))
          lows.append(float(r["Low"]))

    if highs and lows:
      return max(highs), min(lows)
    return None, None

  def _detect_1730_high_confluence(self):
    """Checks if 01:30 (4H) High or 05:30 (6H) High lies inside the closed 17:30->21:30 4H zone."""
    matched = []
    if self.c1730_4h_high is None or self.c1730_4h_low is None:
      return False, matched

    z_low = min(self.c1730_4h_low, self.c1730_4h_high)
    z_high = max(self.c1730_4h_low, self.c1730_4h_high)

    if self.c0130_high is not None and (z_low <= self.c0130_high <= z_high):
      matched.append(f"01:30 4H High ({self.c0130_high:.2f})")
    if self.c0530_6h_high is not None and (z_low <= self.c0530_6h_high <= z_high):
      matched.append(f"05:30 6H High ({self.c0530_6h_high:.2f})")

    return len(matched) > 0, matched

  def _scan_post_2130_signals(self, candles, after_2130):
    """
    Strictly:
    - No signals between 17:30 and 21:30.
    - Signals ONLY on candles AT OR AFTER 21:30 PM (`c["dt_user"] >= after_2130`).
    - If 01:30 (4H) High or 05:30 (6H) High is inside the 17:30->21:30 4H zone:
        * At 17:30 High/Zone, check stronger side (Buyer vs Seller) and give entry on the stronger side.
    - Otherwise:
        * 01:30 (4H) & 05:30 (6H) High -> SELL only (never BUY at High)
        * 01:30 (4H) & 05:30 (6H) Low  -> BUY only (never SELL at Low)
        * 17:30 (4H -> closes 21:30) Mid Point cross logic after 21:30.
    """
    if after_2130 is None or not candles:
      return []

    sub = [c for c in candles if c["dt_user"] >= after_2130]
    if not sub:
      return []

    buyer_strong = self._is_buyer_strong()
    seller_strong = self._is_seller_strong()
    stronger_side = self._stronger_side_at_1730_high()

    buy_sig = None
    sell_sig = None
    prev_c = None
    seen_above_1730_mid = False
    seen_below_1730_mid = False

    b_lock = self._buyer_pct()
    s_lock = self._seller_pct()

    high_sell_levels = [
        ("01:30 4H High", self.c0130_high),
        ("05:30 6H High", self.c0530_6h_high),
    ]
    low_buy_levels = [
        ("01:30 4H Low", self.c0130_low),
        ("05:30 6H Low", self.c0530_6h_low),
    ]

    for c in sub:
      atr = c["atr"]
      dec = get_decimals(c["close"])
      high_val = c["high"]
      low_val = c["low"]
      close_val = c["close"]
      is_green = c["color"] == "Green"
      is_red = c["color"] == "Red"
      choppy = self._is_choppy(c)
      prev_close = prev_c["close"] if prev_c is not None else close_val

      # A) 17:30 High Confluence Entry (when 01:30 or 05:30 High is inside 17:30 zone)
      if self.high_confluence_in_1730 and self.c1730_4h_high is not None and (not choppy):
        at_1730_high = (high_val >= self.c1730_4h_high) or (prev_c is not None and prev_c["high"] >= self.c1730_4h_high)
        if at_1730_high:
          matched_str = ", ".join(self.confluence_highs_matched)
          if stronger_side == "SELL" and not sell_sig and is_red:
            grade, conf_note = self._grade_signal(c, "SELL")
            sl = round(max(self.c1730_4h_high, high_val) + (1.5 * atr), dec)
            sell_sig = {
                "Asset": self.symbol,
                "Date": str(c["date"]),
                "Time": c["time"],
                "Candle #": c["id"],
                "Type": "SELL 🔴",
                "Grade": grade,
                "Locked Buyer %": f"{b_lock}%",
                "Locked Seller %": f"{s_lock}%",
                "Setup": f"17:30 High Confluence [{matched_str}] -> Seller Stronger SELL ({conf_note})",
                "Entry Price": round(close_val, dec),
                "Stop Loss (SL)": sl,
                "TP 1": round(close_val - (1.2 * atr), dec),
                "TP 2": round(close_val - (2.5 * atr), dec),
                "TP 3": round(close_val - (4.0 * atr), dec),
                "Is_Within_1hr": c["is_within_1hr"],
                "Signal_DT": c["dt_user"].isoformat(),
            }
          elif stronger_side == "BUY" and not buy_sig and is_green:
            grade, conf_note = self._grade_signal(c, "BUY")
            sl = round(min(self.c1730_4h_low if self.c1730_4h_low is not None else low_val, low_val) - (1.5 * atr), dec)
            buy_sig = {
                "Asset": self.symbol,
                "Date": str(c["date"]),
                "Time": c["time"],
                "Candle #": c["id"],
                "Type": "BUY 🟢",
                "Grade": grade,
                "Locked Buyer %": f"{b_lock}%",
                "Locked Seller %": f"{s_lock}%",
                "Setup": f"17:30 High Confluence [{matched_str}] -> Buyer Stronger BUY ({conf_note})",
                "Entry Price": round(close_val, dec),
                "Stop Loss (SL)": sl,
                "TP 1": round(close_val + (1.2 * atr), dec),
                "TP 2": round(close_val + (2.5 * atr), dec),
                "TP 3": round(close_val + (4.0 * atr), dec),
                "Is_Within_1hr": c["is_within_1hr"],
                "Signal_DT": c["dt_user"].isoformat(),
            }

      # B) 01:30 (4H) & 05:30 (6H) HIGH -> SELL ONLY (No BUY on High)
      if not sell_sig and seller_strong and (not choppy) and is_red:
        for lbl, h_lvl in high_sell_levels:
          if h_lvl is not None and high_val >= h_lvl:
            grade, conf_note = self._grade_signal(c, "SELL")
            sl = round(max(h_lvl, high_val) + (1.5 * atr), dec)
            sell_sig = {
                "Asset": self.symbol,
                "Date": str(c["date"]),
                "Time": c["time"],
                "Candle #": c["id"],
                "Type": "SELL 🔴",
                "Grade": grade,
                "Locked Buyer %": f"{b_lock}%",
                "Locked Seller %": f"{s_lock}%",
                "Setup": f"{lbl} SELL Only ({conf_note})",
                "Entry Price": round(close_val, dec),
                "Stop Loss (SL)": sl,
                "TP 1": round(close_val - (1.2 * atr), dec),
                "TP 2": round(close_val - (2.5 * atr), dec),
                "TP 3": round(close_val - (4.0 * atr), dec),
                "Is_Within_1hr": c["is_within_1hr"],
                "Signal_DT": c["dt_user"].isoformat(),
            }
            break

      # C) 01:30 (4H) & 05:30 (6H) LOW -> BUY ONLY (No SELL on Low)
      if not buy_sig and buyer_strong and (not choppy) and is_green:
        for lbl, l_lvl in low_buy_levels:
          if l_lvl is not None and low_val <= l_lvl:
            grade, conf_note = self._grade_signal(c, "BUY")
            sl = round(min(l_lvl, low_val) - (1.5 * atr), dec)
            buy_sig = {
                "Asset": self.symbol,
                "Date": str(c["date"]),
                "Time": c["time"],
                "Candle #": c["id"],
                "Type": "BUY 🟢",
                "Grade": grade,
                "Locked Buyer %": f"{b_lock}%",
                "Locked Seller %": f"{s_lock}%",
                "Setup": f"{lbl} BUY Only ({conf_note})",
                "Entry Price": round(close_val, dec),
                "Stop Loss (SL)": sl,
                "TP 1": round(close_val + (1.2 * atr), dec),
                "TP 2": round(close_val + (2.5 * atr), dec),
                "TP 3": round(close_val + (4.0 * atr), dec),
                "Is_Within_1hr": c["is_within_1hr"],
                "Signal_DT": c["dt_user"].isoformat(),
            }
            break

      # D) 17:30 PM (4H -> closes 21:30 PM) Mid Point logic after 21:30 PM
      mid_1730 = self.c1730_4h_mid
      if mid_1730 is not None and (not choppy):
        if high_val >= mid_1730 or prev_close >= mid_1730:
          seen_above_1730_mid = True
        if low_val <= mid_1730 or prev_close <= mid_1730:
          seen_below_1730_mid = True

        red_mid_cross = (
            seller_strong
            and is_red
            and (close_val < mid_1730)
            and (prev_close >= mid_1730 or high_val >= mid_1730)
            and seen_above_1730_mid
        )
        if not sell_sig and red_mid_cross:
          grade, conf_note = self._grade_signal(c, "SELL")
          sl = round(max(mid_1730, high_val) + (1.5 * atr), dec)
          sell_sig = {
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Candle #": c["id"],
              "Type": "SELL 🔴",
              "Grade": grade,
              "Locked Buyer %": f"{b_lock}%",
              "Locked Seller %": f"{s_lock}%",
              "Setup": f"17:30 4H Mid Cross SELL ({conf_note})",
              "Entry Price": round(close_val, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(close_val - (1.2 * atr), dec),
              "TP 2": round(close_val - (2.5 * atr), dec),
              "TP 3": round(close_val - (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
              "Signal_DT": c["dt_user"].isoformat(),
          }

        green_mid_cross = (
            buyer_strong
            and is_green
            and (close_val > mid_1730)
            and (prev_close <= mid_1730 or low_val <= mid_1730)
            and seen_below_1730_mid
        )
        if not buy_sig and green_mid_cross:
          grade, conf_note = self._grade_signal(c, "BUY")
          sl = round(min(mid_1730, low_val) - (1.5 * atr), dec)
          buy_sig = {
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Candle #": c["id"],
              "Type": "BUY 🟢",
              "Grade": grade,
              "Locked Buyer %": f"{b_lock}%",
              "Locked Seller %": f"{s_lock}%",
              "Setup": f"17:30 4H Mid Cross BUY ({conf_note})",
              "Entry Price": round(close_val, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(close_val + (1.2 * atr), dec),
              "TP 2": round(close_val + (2.5 * atr), dec),
              "TP 3": round(close_val + (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
              "Signal_DT": c["dt_user"].isoformat(),
          }

      prev_c = c

    out = [s for s in [buy_sig, sell_sig] if s is not None]
    out.sort(key=lambda x: x["Candle #"])
    return out

  def calculate_and_scan(self):
    if self.df_1d is None or self.df_1d.empty or self.df_4h is None or self.df_4h.empty or self.df_5m is None or self.df_5m.empty:
      return

    current_utc_time = datetime.now(timezone.utc)
    now_user = current_utc_time.astimezone(self.user_tz)

    if len(self.df_1d) >= 2:
      prev_candle = self.df_1d.iloc[-2]
    else:
      prev_candle = self.df_1d.iloc[-1]

    self.prev_high = float(prev_candle["High"])
    self.prev_low = float(prev_candle["Low"])

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

    # Determine active day anchor for 01:30 (4H) and 05:30 (6H)
    base_date = now_user.date() if now_user.time() >= time(1, 30) else (now_user.date() - timedelta(days=1))
    d0130_start = datetime.combine(base_date, time(1, 30), tzinfo=self.user_tz)
    d0130_end = d0130_start + timedelta(hours=4)
    self.c0130_high, self.c0130_low = self._range_high_low(d0130_start, d0130_end)

    d0530_start = datetime.combine(base_date, time(5, 30), tzinfo=self.user_tz)
    d0530_end = d0530_start + timedelta(hours=6)
    self.c0530_6h_high, self.c0530_6h_low = self._range_high_low(d0530_start, d0530_end)

    # 17:30 PM 4H candle takes 4h to close -> completes at 21:30 PM
    t1730_today = datetime.combine(now_user.date(), time(17, 30), tzinfo=self.user_tz)
    t2130_today = datetime.combine(now_user.date(), time(21, 30), tzinfo=self.user_tz)

    if now_user >= t2130_today:
      self.c1730_4h_start = t1730_today
      self.c1730_4h_close = t2130_today
      self.c1730_4h_closed = True
      self.signal_start_2130 = t2130_today
      h_1730, l_1730 = self._range_high_low(self.c1730_4h_start, self.c1730_4h_close)
      self.c1730_4h_high = h_1730
      self.c1730_4h_low = l_1730
      self.c1730_4h_mid = ((h_1730 + l_1730) / 2.0) if (h_1730 is not None and l_1730 is not None) else None
    elif now_user >= t1730_today:
      # Between 17:30 and 21:30: candle is forming, NO signals between 17:30 and 21:30
      self.c1730_4h_start = t1730_today
      self.c1730_4h_close = t2130_today
      self.c1730_4h_closed = False
      self.signal_start_2130 = t2130_today
    else:
      # Before 17:30 today -> previous day's 17:30->21:30 4H candle is closed
      prev_d = now_user.date() - timedelta(days=1)
      self.c1730_4h_start = datetime.combine(prev_d, time(17, 30), tzinfo=self.user_tz)
      self.c1730_4h_close = datetime.combine(prev_d, time(21, 30), tzinfo=self.user_tz)
      self.c1730_4h_closed = True
      self.signal_start_2130 = self.c1730_4h_close
      h_1730, l_1730 = self._range_high_low(self.c1730_4h_start, self.c1730_4h_close)
      self.c1730_4h_high = h_1730
      self.c1730_4h_low = l_1730
      self.c1730_4h_mid = ((h_1730 + l_1730) / 2.0) if (h_1730 is not None and l_1730 is not None) else None

    if self.c1730_4h_closed:
      self.high_confluence_in_1730, self.confluence_highs_matched = self._detect_1730_high_confluence()

    # Strictly no signal between 17:30 and 21:30; signals only >= 21:30
    in_1730_to_2130_window = (t1730_today <= now_user < t2130_today)
    if in_1730_to_2130_window or not self.c1730_4h_closed:
      self.history = []
      self.signals = []
    else:
      collected = self._scan_post_2130_signals(self.candles_5m, self.signal_start_2130)
      self.history = collected
      self.signals = [s for s in collected if s.get("Is_Within_1hr", False)]

    self._build_diy_checklist()


# ---------------------------------------------------------
# MASTER SCANNER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (20+ Assets)")

if "twt_strength_locks" not in st.session_state:
  st.session_state.twt_strength_locks = load_strength_locks()

strength_locks = st.session_state.twt_strength_locks
locks_updated = False

master_signals = []
asset_engines = {}


def scan_asset(sym):
  df_1d, df_4h, df_5m, dfs_by_tf = fetch_data(sym)
  return sym, df_1d, df_4h, df_5m, dfs_by_tf


with st.spinner("⚡ Scanning 20+ assets (01:30 4H + 05:30 6H + 17:30-21:30 4H Zone & Mid)..."):
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
  st.success(f"✅ Found {len(master_signals)} post-21:30 PM setup(s) across assets!")
  df_display = pd.DataFrame(master_signals).drop(columns=["Is_Within_1hr", "Signal_DT"], errors="ignore")
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No active post-21:30 PM setups right now.")

# ---------------------------------------------------------
# MANUAL ASSET INSPECTOR & PER-ASSET LEDGER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector (01:30 4H + 05:30 6H + 17:30-21:30 4H Zone & Mid)")

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

display_strength = bot.strength_info
session_tag = display_strength.get("session_key", "") if display_strength else ""
display_title_tag = f"(Cycle: {session_tag})" if session_tag else ""

if display_strength and display_strength.get("rows"):
  st.markdown(f"#### 💪 {selected_symbol} Buyer vs Seller Strength + Zero-Lag MTF {display_title_tag}")
  s_col1, s_col2, s_col3 = st.columns(3)
  with s_col1:
    st.metric("🟢 Buyer Strength", f"{display_strength['buyer_pct']}%")
  with s_col2:
    st.metric("🔴 Seller Strength", f"{display_strength['seller_pct']}%")
  with s_col3:
    st.metric("⚡ Zyada Faydemand Side", display_strength["verdict"])

  st.dataframe(pd.DataFrame(display_strength["rows"]), width="stretch")

if bot.diy_checklist:
  st.markdown(f"#### 🧩 {selected_symbol} DIY Custom Strategy Builder Status")
  st.dataframe(pd.DataFrame(bot.diy_checklist), width="stretch")

st.markdown(f"### 📈 {selected_symbol} Performance Ledger (From {ledger_start_date})")
summary_df = summarize_asset_ledger(ledger_records, selected_symbol, start_date_str=ledger_start_date)
st.dataframe(summary_df, width="stretch")

if bot.candles_5m:
  curr_p = bot.candles_5m[-1]["close"]

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label=f"Current {selected_symbol} Price", value=f"{curr_p:.2f}")
  with col2:
    st.metric(label="Post-21:30 Signals", value=len(bot.history))
  with col3:
    st.metric(label="Timezone", value=selected_tz_name)

  info_parts = []
  if bot.c0130_high is not None and bot.c0130_low is not None:
    info_parts.append(f"01:30(4H) H:{bot.c0130_high:.2f} / L:{bot.c0130_low:.2f}")
  if bot.c0530_6h_high is not None and bot.c0530_6h_low is not None:
    info_parts.append(f"05:30(6H) H:{bot.c0530_6h_high:.2f} / L:{bot.c0530_6h_low:.2f}")
  if bot.c1730_4h_closed and bot.c1730_4h_high is not None and bot.c1730_4h_low is not None:
    info_parts.append(
        f"17:30->21:30(4H) Zone H:{bot.c1730_4h_high:.2f} / L:{bot.c1730_4h_low:.2f} / Mid:{bot.c1730_4h_mid:.2f}"
    )
  if bot.high_confluence_in_1730:
    info_parts.append("High Confluence Inside 17:30 Zone: " + ", ".join(bot.confluence_highs_matched))
  if info_parts:
    st.caption("🕒 Marked Levels & Zones — " + " | ".join(info_parts))

  if not bot.c1730_4h_closed:
    st.warning("⏳ 17:30 PM se 21:30 PM tak 4H candle close ka wait hai — is beech koi signal nahi aayega.")

  if bot.history:
    st.markdown(f"### 📋 Post-21:30 PM Entries Feed for {selected_symbol}")
    df_history = pd.DataFrame(bot.history).drop(columns=["Is_Within_1hr", "Signal_DT"], errors="ignore")
    st.dataframe(df_history, width="stretch")
  else:
    st.info("ℹ️ 21:30 PM ke baad abhi tak koi valid BUY/SELL entry nahi aayi hai.")

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
    ax.plot(c_ids, zl_vals, color="#00e5ff", linewidth=1.6, label="Zero-Lag Trend Line (ZLEMA)")

  # 01:30 (4H) High & Low
  if bot.c0130_high is not None:
    ax.axhline(bot.c0130_high, color="#ff5252", linestyle="-.", linewidth=1.4, label=f"01:30(4H) High [SELL]: {bot.c0130_high:.2f}")
  if bot.c0130_low is not None:
    ax.axhline(bot.c0130_low, color="#00e676", linestyle="-.", linewidth=1.4, label=f"01:30(4H) Low [BUY]: {bot.c0130_low:.2f}")

  # 05:30 (6H) High & Low
  if bot.c0530_6h_high is not None:
    ax.axhline(bot.c0530_6h_high, color="#ff8a80", linestyle=":", linewidth=1.5, label=f"05:30(6H) High [SELL]: {bot.c0530_6h_high:.2f}")
  if bot.c0530_6h_low is not None:
    ax.axhline(bot.c0530_6h_low, color="#69f0ae", linestyle=":", linewidth=1.5, label=f"05:30(6H) Low [BUY]: {bot.c0530_6h_low:.2f}")

  # 17:30 -> 21:30 (4H) Shaded Zone (High-Low) & Mid Point
  if bot.c1730_4h_closed and bot.c1730_4h_high is not None and bot.c1730_4h_low is not None:
    z_bot = min(bot.c1730_4h_low, bot.c1730_4h_high)
    z_top = max(bot.c1730_4h_low, bot.c1730_4h_high)
    zone_col = "#ab47bc" if bot.high_confluence_in_1730 else "#7e57c2"
    ax.axhspan(z_bot, z_top, color=zone_col, alpha=0.22, label=f"17:30-21:30(4H) Zone [{z_bot:.2f} - {z_top:.2f}]")
    ax.axhline(z_top, color="#ce93d8", linestyle="-.", linewidth=1.4, label=f"17:30-21:30(4H) High: {z_top:.2f}")
    ax.axhline(z_bot, color="#81d4fa", linestyle="-.", linewidth=1.4, label=f"17:30-21:30(4H) Low: {z_bot:.2f}")
    if bot.c1730_4h_mid is not None:
      ax.axhline(bot.c1730_4h_mid, color="#ffd54f", linestyle="--", linewidth=1.6, label=f"17:30-21:30(4H) Mid: {bot.c1730_4h_mid:.2f}")

  if bot.signal_start_2130:
    c_2130_list = [c for c in bot.candles_5m if c["dt_user"] >= bot.signal_start_2130]
    if c_2130_list:
      first_2130_id = c_2130_list[0]["id"]
      ax.axvline(first_2130_id, color="#ffd54f", linestyle=":", linewidth=1.5, label=f"21:30 Signal Start ({bot.signal_start_2130.strftime('%m-%d %H:%M')})")

  if bot.prev_high is not None:
    ax.axhline(bot.prev_high, color="#ff9100", linestyle="--", linewidth=1.3, label=f"Prev Day High: {bot.prev_high}")
  if bot.prev_low is not None:
    ax.axhline(bot.prev_low, color="#29b6f6", linestyle="--", linewidth=1.3, label=f"Prev Day Low: {bot.prev_low}")

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
    grade_prefix = "A+ " if "A-GRADE" in sig.get("Grade", "") else ""
    txt = f" {grade_prefix}{'BUY' if is_b else 'SELL'} "
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
      f"{selected_symbol} 5m Chart — 01:30(4H) + 05:30(6H) + 17:30-21:30(4H Zone & Mid)",
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
