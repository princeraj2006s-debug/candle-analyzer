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
# 17:30 SESSION STRENGTH LOCK HELPERS
# ---------------------------------------------------------
def get_1730_session_key(now_user):
  """Returns session key locking strength until the next 17:30 PM 4H candle close."""
  anchor_1730_today = datetime.combine(now_user.date(), time(17, 30), tzinfo=now_user.tzinfo)
  if now_user >= anchor_1730_today:
    session_start = anchor_1730_today
  else:
    session_start = anchor_1730_today - timedelta(days=1)
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
st.sidebar.text("🕒 4H Candle       : 13:30 PM -> Closes 17:30 PM")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Master Confluence Rules")
st.sidebar.text("4H Candle: Starts 13:30 PM, Closes 17:30 PM")
st.sidebar.text("High/Low : Marked After 17:30 PM Close")
st.sidebar.text("Entry    : Only After 17:30 PM")
st.sidebar.text("SELL     : At / Near 4H High (Seller Strong)")
st.sidebar.text("BUY      : At / Near 4H Low (Buyer Strong)")
st.sidebar.text("Zero-Lag : ZLEMA + ATR Band Trend")
st.sidebar.text("A-Grade  : HTF + Wick + Zero-Lag")

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


def get_or_lock_1730_strength(symbol, user_tz, dfs_by_tf, locks_dict):
  """Calculate strength once per 17:30 PM 4H close cycle and lock until next 17:30 PM."""
  now_user = datetime.now(timezone.utc).astimezone(user_tz)
  session_key = get_1730_session_key(now_user)
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
# 13:30 PM 4H CANDLE (CLOSES AT 17:30 PM) ENGINE
# HIGH -> SELL | LOW -> BUY | STRICT STRENGTH GATE
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

    # 13:30 PM 4H Candle -> Closes at 17:30 PM -> High & Low marked after 17:30 PM
    self.window_1330_start = None
    self.window_1730_close = None
    self.candle_4h_1330_high = None
    self.candle_4h_1330_low = None
    self.candle_4h_1330_closed = False

  def _is_buyer_dominant_and_strong(self):
    """True only if Buyer is genuinely stronger than Seller and >= 52%."""
    b_pct = float(self.strength_info.get("buyer_pct", 50.0))
    s_pct = float(self.strength_info.get("seller_pct", 50.0))
    return (b_pct > s_pct) and (b_pct >= 52.0)

  def _is_seller_dominant_and_strong(self):
    """True only if Seller is genuinely stronger than Buyer and >= 52%."""
    b_pct = float(self.strength_info.get("buyer_pct", 50.0))
    s_pct = float(self.strength_info.get("seller_pct", 50.0))
    return (s_pct > b_pct) and (s_pct >= 52.0)

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
      str_ok = self._is_buyer_dominant_and_strong()
    else:
      htf_ok = bool(locked_info.get("htf_sell_aligned", False))
      zl_5m_ok = c.get("zl_trend") == "BEAR"
      zl_htf_ok = bool(locked_info.get("zl_sell_aligned", False))
      str_ok = self._is_seller_dominant_and_strong()

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
    buyer_strong = self._is_buyer_dominant_and_strong()
    seller_strong = self._is_seller_dominant_and_strong()

    self.diy_checklist = [
        {
            "Rule (DIY Builder)": "1. 13:30 PM 4H Candle Closed at 17:30 PM",
            "BUY Side": "READY ✅" if self.candle_4h_1330_closed else "WAITING 17:30 ⏳",
            "SELL Side": "READY ✅" if self.candle_4h_1330_closed else "WAITING 17:30 ⏳",
        },
        {
            "Rule (DIY Builder)": "2. Strength Gate (Buyer/Seller Strong)",
            "BUY Side": "STRONG ✅" if buyer_strong else "WEAK / BLOCKED ❌",
            "SELL Side": "STRONG ✅" if seller_strong else "WEAK / BLOCKED ❌",
        },
        {
            "Rule (DIY Builder)": "3. Post-17:30 Entry (High=SELL | Low=BUY)",
            "BUY Side": "ACTIVE ✅" if has_buy else "WAITING ⏳",
            "SELL Side": "ACTIVE ✅" if has_sell else "WAITING ⏳",
        },
        {
            "Rule (DIY Builder)": "4. Zero-Lag Trend (ZLEMA + ATR Band)",
            "BUY Side": "BULLISH ✅" if last_zl == "BULL" else "BEARISH ⏳",
            "SELL Side": "BEARISH ✅" if last_zl == "BEAR" else "BULLISH ⏳",
        },
        {
            "Rule (DIY Builder)": "5. Zero-Lag MTF (4H + 1D Trend)",
            "BUY Side": "ALIGNED ✅" if zl_htf_buy else "NOT ALIGNED ⏳",
            "SELL Side": "ALIGNED ✅" if zl_htf_sell else "NOT ALIGNED ⏳",
        },
        {
            "Rule (DIY Builder)": "6. HTF Candle Strength (4H + 1D)",
            "BUY Side": "STRONG ✅" if htf_buy else "MIXED ⏳",
            "SELL Side": "STRONG ✅" if htf_sell else "MIXED ⏳",
        },
        {
            "Rule (DIY Builder)": "7. Chop / Sideways Filter",
            "BUY Side": "CLEAR ✅" if self.chop_filter else "OFF ⚪",
            "SELL Side": "CLEAR ✅" if self.chop_filter else "OFF ⚪",
        },
    ]

  def _resolve_1330_to_1730_4h_window(self, now_user):
    """
    13:30 PM 4H candle starts at 13:30 and completes/closes at 17:30 (4 hours later).
    """
    today_1330 = datetime.combine(now_user.date(), time(13, 30), tzinfo=now_user.tzinfo)
    today_1730 = datetime.combine(now_user.date(), time(17, 30), tzinfo=now_user.tzinfo)

    if now_user >= today_1730:
      return today_1330, today_1730, True
    else:
      prev_day = now_user.date() - timedelta(days=1)
      prev_1330 = datetime.combine(prev_day, time(13, 30), tzinfo=now_user.tzinfo)
      prev_1730 = datetime.combine(prev_day, time(17, 30), tzinfo=now_user.tzinfo)
      return prev_1330, prev_1730, True

  def _mark_1330_4h_high_low(self, start_1330, close_1730):
    """Marks exact High and Low of the 13:30 PM 4H candle once closed at 17:30 PM."""
    if self.df_4h is not None and not self.df_4h.empty:
      now_u = datetime.now(timezone.utc).astimezone(self.user_tz)
      for dt_utc, r in self.df_4h.iterrows():
        dt_u = dt_utc.astimezone(self.user_tz)
        if abs((dt_u - start_1330).total_seconds()) <= 5400 and (dt_u + timedelta(hours=4)) <= now_u:
          return float(r["High"]), float(r["Low"])

    highs = []
    lows = []
    for c in self.candles_5m:
      if start_1330 <= c["dt_user"] < close_1730:
        highs.append(c["high"])
        lows.append(c["low"])

    if highs and lows:
      return max(highs), min(lows)
    return None, None

  def _scan_post_1730_signals(self, candles, after_1730, h_4h, l_4h):
    """
    Strictly:
    - Only candles AT OR AFTER 17:30 PM
    - SELL at / near 4H High (only if Seller is stronger than Buyer and >= 52%)
    - BUY at / near 4H Low (only if Buyer is stronger than Seller and >= 52%)
    """
    if after_1730 is None or h_4h is None or l_4h is None or not candles:
      return None, None

    sub = [c for c in candles if c["dt_user"] >= after_1730]
    if not sub:
      return None, None

    buy_allowed = self._is_buyer_dominant_and_strong()
    sell_allowed = self._is_seller_dominant_and_strong()

    buy_sig = None
    sell_sig = None

    b_lock = float(self.strength_info.get("buyer_pct", 50.0))
    s_lock = float(self.strength_info.get("seller_pct", 50.0))

    for c in sub:
      atr = c["atr"]
      dec = get_decimals(c["close"])
      high_val = c["high"]
      low_val = c["low"]
      close_val = c["close"]
      is_green = c["color"] == "Green"
      is_red = c["color"] == "Red"
      choppy = self._is_choppy(c)

      # SELL at 4H High: price reaches/touches 4H High, red rejection, Seller strong
      high_sell_cond = (
          sell_allowed
          and (high_val >= h_4h)
          and is_red
          and (not choppy)
      )
      if not sell_sig and high_sell_cond:
        grade, conf_note = self._grade_signal(c, "SELL")
        sl = round(max(h_4h, high_val) + (1.5 * atr), dec)
        sell_sig = {
            "Asset": self.symbol,
            "Date": str(c["date"]),
            "Time": c["time"],
            "Candle #": c["id"],
            "Type": "SELL 🔴",
            "Grade": grade,
            "Locked Buyer %": f"{b_lock}%",
            "Locked Seller %": f"{s_lock}%",
            "Setup": f"4H High SELL ({conf_note})",
            "Entry Price": round(close_val, dec),
            "Stop Loss (SL)": sl,
            "TP 1": round(close_val - (1.2 * atr), dec),
            "TP 2": round(close_val - (2.5 * atr), dec),
            "TP 3": round(close_val - (4.0 * atr), dec),
            "Is_Within_1hr": c["is_within_1hr"],
            "Signal_DT": c["dt_user"].isoformat(),
        }

      # BUY at 4H Low: price reaches/touches 4H Low, green bounce, Buyer strong
      low_buy_cond = (
          buy_allowed
          and (low_val <= l_4h)
          and is_green
          and (not choppy)
      )
      if not buy_sig and low_buy_cond:
        grade, conf_note = self._grade_signal(c, "BUY")
        sl = round(min(l_4h, low_val) - (1.5 * atr), dec)
        buy_sig = {
            "Asset": self.symbol,
            "Date": str(c["date"]),
            "Time": c["time"],
            "Candle #": c["id"],
            "Type": "BUY 🟢",
            "Grade": grade,
            "Locked Buyer %": f"{b_lock}%",
            "Locked Seller %": f"{s_lock}%",
            "Setup": f"4H Low BUY ({conf_note})",
            "Entry Price": round(close_val, dec),
            "Stop Loss (SL)": sl,
            "TP 1": round(close_val + (1.2 * atr), dec),
            "TP 2": round(close_val + (2.5 * atr), dec),
            "TP 3": round(close_val + (4.0 * atr), dec),
            "Is_Within_1hr": c["is_within_1hr"],
            "Signal_DT": c["dt_user"].isoformat(),
        }

    return buy_sig, sell_sig

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

    start_1330, close_1730, is_closed = self._resolve_1330_to_1730_4h_window(now_user)
    self.window_1330_start = start_1330
    self.window_1730_close = close_1730
    self.candle_4h_1330_closed = is_closed

    if now_user >= close_1730:
      h_4h, l_4h = self._mark_1330_4h_high_low(start_1330, close_1730)
      self.candle_4h_1330_high = h_4h
      self.candle_4h_1330_low = l_4h
      buy_sig, sell_sig = self._scan_post_1730_signals(self.candles_5m, close_1730, h_4h, l_4h)
      collected = [sig for sig in [buy_sig, sell_sig] if sig is not None]
      collected.sort(key=lambda x: x["Candle #"])
      self.history = collected
      self.signals = [s for s in collected if s.get("Is_Within_1hr", False)]
    else:
      self.candle_4h_1330_high = None
      self.candle_4h_1330_low = None
      self.candle_4h_1330_closed = False
      self.history = []
      self.signals = []

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


with st.spinner("⚡ Scanning 20+ assets (13:30-17:30 4H Candle + Strength Gate)..."):
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(scan_asset, sym): sym for sym in ASSET_LIST}
    for future in as_completed(futures):
      sym, df_1d, df_4h, df_5m, dfs_by_tf = future.result()
      if df_1d is not None and df_4h is not None and df_5m is not None:
        locked_strength, newly_computed = get_or_lock_1730_strength(
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
  st.success(f"✅ Found {len(master_signals)} post-17:30 PM 4H setup(s) across assets!")
  df_display = pd.DataFrame(master_signals).drop(columns=["Is_Within_1hr", "Signal_DT"], errors="ignore")
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No active post-17:30 PM 4H setups right now.")

# ---------------------------------------------------------
# MANUAL ASSET INSPECTOR & PER-ASSET LEDGER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector (13:30-17:30 4H Candle + Strength Gate)")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df_1d, df_4h, df_5m, dfs_by_tf = fetch_data(selected_symbol)
  locked_strength, newly_computed = get_or_lock_1730_strength(
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
    st.metric(label="Post-17:30 Signals", value=len(bot.history))
  with col3:
    st.metric(label="Timezone", value=selected_tz_name)

  if bot.candle_4h_1330_closed and bot.candle_4h_1330_high is not None and bot.candle_4h_1330_low is not None:
    st.caption(
        f"🕒 13:30 PM 4H Candle Closed at 17:30 PM ({bot.window_1730_close.strftime('%Y-%m-%d %H:%M')}) — "
        f"4H High (SELL): {bot.candle_4h_1330_high:.4f} | 4H Low (BUY): {bot.candle_4h_1330_low:.4f}"
    )
  elif bot.window_1330_start:
    st.info(f"⏳ Waiting for 13:30 PM 4H candle to close at 17:30 PM ({bot.window_1730_close.strftime('%Y-%m-%d %H:%M')})...")

  if bot.history:
    st.markdown(f"### 📋 Post-17:30 PM Entries Feed for {selected_symbol}")
    df_history = pd.DataFrame(bot.history).drop(columns=["Is_Within_1hr", "Signal_DT"], errors="ignore")
    st.dataframe(df_history, width="stretch")
  else:
    st.info("ℹ️ 17:30 PM ke baad abhi tak koi valid BUY/SELL entry nahi aayi hai.")

  # --- PLOT CHART WITH 13:30-17:30 4H HIGH/LOW + ZERO-LAG LINE ---
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

  if bot.window_1730_close:
    c_1730_list = [c for c in bot.candles_5m if c["dt_user"] >= bot.window_1730_close]
    if c_1730_list:
      first_1730_id = c_1730_list[0]["id"]
      ax.axvline(first_1730_id, color="#ffd54f", linestyle=":", linewidth=1.5, label=f"17:30 PM Close ({bot.window_1730_close.strftime('%m-%d %H:%M')})")

  if bot.candle_4h_1330_high is not None:
    ax.axhline(bot.candle_4h_1330_high, color="#ff5252", linestyle="-.", linewidth=1.6, label=f"4H High (SELL): {bot.candle_4h_1330_high:.2f}")
  if bot.candle_4h_1330_low is not None:
    ax.axhline(bot.candle_4h_1330_low, color="#00e676", linestyle="-.", linewidth=1.6, label=f"4H Low (BUY): {bot.candle_4h_1330_low:.2f}")

  if bot.prev_high is not None:
    ax.axhline(bot.prev_high, color="#ff9100", linestyle="--", linewidth=1.5, label=f"Prev Day High: {bot.prev_high}")
  if bot.prev_low is not None:
    ax.axhline(bot.prev_low, color="#29b6f6", linestyle="--", linewidth=1.5, label=f"Prev Day Low: {bot.prev_low}")

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
      f"{selected_symbol} 5m Chart — 4H High=SELL | 4H Low=BUY (Strength Gated)",
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
