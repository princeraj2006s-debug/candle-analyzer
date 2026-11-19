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
st.sidebar.text("📊 Priority : 21:30 Mid -> Fallback 01:30 Mid")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Master Confluence Rules")
st.sidebar.text("Primary  : 21:30 4H Mid Point Cross")
st.sidebar.text("Fallback : 01:30 4H Mid Point Cross")
st.sidebar.text("SELL     : Red Candle Crosses Below Mid")
st.sidebar.text("BUY      : Green Candle Crosses Above Mid")
st.sidebar.text("Strength : 5m, 15m, 1h, 4h, 1d Candle Power")
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


def _candle_strength_score(df):
  """Returns (buyer_score_0_to_100, seller_score_0_to_100, label) from latest candle."""
  if df is None or df.empty:
    return 50.0, 50.0, "Neutral ⚪"
  last = df.iloc[-1]
  o = float(last["Open"])
  h = float(last["High"])
  l = float(last["Low"])
  c = float(last["Close"])
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


def compute_mtf_candle_strength(dfs_by_tf):
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
    b_val, s_val, c_label = _candle_strength_score(df)
    _, _, zl_series = _compute_zero_lag_series(df, length=21, band_mult=1.2)
    zl_state = str(zl_series.iloc[-1]) if (zl_series is not None and not zl_series.empty) else ("BULL" if b_val >= s_val else "BEAR")
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
        ohlcv_5m = exchange.fetch_ohlcv(sym, timeframe="5m", limit=300)
        if ohlcv_1d and ohlcv_4h and ohlcv_5m:
          df_1d = _df_from_ohlcv(ohlcv_1d)
          df_4h = _df_from_ohlcv(ohlcv_4h)
          df_1h = _df_from_ohlcv(ohlcv_1h) if ohlcv_1h else None
          df_15m = _df_from_ohlcv(ohlcv_15m) if ohlcv_15m else None
          df_5m = _df_from_ohlcv(ohlcv_5m)

          strength_info = compute_mtf_candle_strength({
              "5m": df_5m,
              "15m": df_15m,
              "1h": df_1h,
              "4h": df_4h,
              "1d": df_1d,
          })
          return df_1d, df_4h, df_5m, strength_info
    except Exception:
      continue
  return None, None, None, {}


# ---------------------------------------------------------
# MULTI-CONFLUENCE MID-CROSS ENGINE (ZERO-LAG + DIY BUILDER)
# ---------------------------------------------------------
class TWTUltimateEngine:

  def __init__(self, df_1d, df_4h, df_5m, symbol, user_tz, chop_filter=True, strength_info=None):
    self.df_1d = df_1d
    self.df_4h = df_4h
    self.df_5m = df_5m
    self.symbol = symbol
    self.user_tz = user_tz
    self.chop_filter = chop_filter
    self.strength_info = strength_info or {}
    self.candles_5m = []
    self.history = []
    self.signals = []
    self.diy_checklist = []

    # Levels & Zones
    self.prev_high = None
    self.prev_low = None
    self.demand_box_1 = None
    self.supply_box_1 = None
    self.demand_1_mid = None
    self.supply_1_mid = None

    self.demand_box_2 = None
    self.supply_box_2 = None

    self.demand_box_3 = None
    self.supply_box_3 = None
    self.demand_3_mid = None
    self.supply_3_mid = None

    self.anchor_2130_candle = None
    self.anchor_2130_start = None
    self.anchor_0130_start = None

  def _build_zones(self, anchor_candle):
    high_val = anchor_candle["high"]
    low_val = anchor_candle["low"]
    rng = high_val - low_val
    zone_thickness = (rng * 0.25) if rng > 0 else (high_val * 0.002)
    dem_box = (low_val, low_val + zone_thickness)
    sup_box = (high_val - zone_thickness, high_val)
    dem_mid = (dem_box[0] + dem_box[1]) / 2.0
    sup_mid = (sup_box[0] + sup_box[1]) / 2.0
    return dem_box, sup_box, dem_mid, sup_mid

  def _get_latest_2130_anchor(self, candles_4h):
    candidates_2130 = [
        c for c in candles_4h
        if abs(
            (datetime.combine(c["date"], c["time_obj"]) - datetime.combine(c["date"], time(21, 30))).total_seconds()
        ) <= 3600
    ]
    if candidates_2130:
      latest_2130 = max(candidates_2130, key=lambda x: x["dt_user"])
      return latest_2130, latest_2130["dt_user"]
    return None, None

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
    if side == "BUY":
      htf_ok = bool(self.strength_info.get("htf_buy_aligned", False))
      zl_5m_ok = c.get("zl_trend") == "BULL"
      zl_htf_ok = bool(self.strength_info.get("zl_buy_aligned", False))
      str_ok = float(self.strength_info.get("buyer_pct", 50.0)) >= 52.0
    else:
      htf_ok = bool(self.strength_info.get("htf_sell_aligned", False))
      zl_5m_ok = c.get("zl_trend") == "BEAR"
      zl_htf_ok = bool(self.strength_info.get("zl_sell_aligned", False))
      str_ok = float(self.strength_info.get("seller_pct", 50.0)) >= 52.0

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
    last_zl = self.candles_5m[-1]["zl_trend"] if self.candles_5m else "BULL"
    zl_htf_buy = bool(self.strength_info.get("zl_buy_aligned", False))
    zl_htf_sell = bool(self.strength_info.get("zl_sell_aligned", False))
    htf_buy = bool(self.strength_info.get("htf_buy_aligned", False))
    htf_sell = bool(self.strength_info.get("htf_sell_aligned", False))

    self.diy_checklist = [
        {
            "Rule (DIY Builder)": "1. Leading Trigger (21:30 / 01:30 Mid Cross)",
            "BUY Side": "ACTIVE ✅" if has_buy else "WAITING ⏳",
            "SELL Side": "ACTIVE ✅" if has_sell else "WAITING ⏳",
        },
        {
            "Rule (DIY Builder)": "2. Zero-Lag Trend (5m ZLEMA + ATR Band)",
            "BUY Side": "BULLISH ✅" if last_zl == "BULL" else "BEARISH ⏳",
            "SELL Side": "BEARISH ✅" if last_zl == "BEAR" else "BULLISH ⏳",
        },
        {
            "Rule (DIY Builder)": "3. Zero-Lag MTF (4H + 1D Trend)",
            "BUY Side": "ALIGNED ✅" if zl_htf_buy else "NOT ALIGNED ⏳",
            "SELL Side": "ALIGNED ✅" if zl_htf_sell else "NOT ALIGNED ⏳",
        },
        {
            "Rule (DIY Builder)": "4. HTF Candle Strength (4H + 1D)",
            "BUY Side": "STRONG ✅" if htf_buy else "MIXED ⏳",
            "SELL Side": "STRONG ✅" if htf_sell else "MIXED ⏳",
        },
        {
            "Rule (DIY Builder)": "5. Chop / Sideways Filter",
            "BUY Side": "CLEAR ✅" if self.chop_filter else "OFF ⚪",
            "SELL Side": "CLEAR ✅" if self.chop_filter else "OFF ⚪",
        },
    ]

  def _scan_mid_cross(self, candles, start_dt, sup_mid, dem_mid, label):
    if start_dt is None or not candles:
      return None, None

    sub = [c for c in candles if c["dt_user"] > start_dt]
    if not sub:
      return None, None

    buy_sig = None
    sell_sig = None
    seen_above_sup = False
    seen_below_dem = False
    prev_c = None

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

      if sup_mid is not None:
        if high_val >= sup_mid or prev_close >= sup_mid:
          seen_above_sup = True
        red_cross_down = (
            is_red
            and (close_val < sup_mid)
            and (prev_close >= sup_mid or high_val >= sup_mid)
            and seen_above_sup
            and (not choppy)
        )
        if not sell_sig and red_cross_down:
          grade, conf_note = self._grade_signal(c, "SELL")
          sl = round(max(sup_mid, high_val) + (1.5 * atr), dec)
          sell_sig = {
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Candle #": c["id"],
              "Type": "SELL 🔴",
              "Grade": grade,
              "Setup": f"{label} Supply Mid Cross SELL ({conf_note})",
              "Entry Price": round(close_val, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(close_val - (1.2 * atr), dec),
              "TP 2": round(close_val - (2.5 * atr), dec),
              "TP 3": round(close_val - (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
              "Signal_DT": c["dt_user"].isoformat(),
          }

      if dem_mid is not None:
        if low_val <= dem_mid or prev_close <= dem_mid:
          seen_below_dem = True
        green_cross_up = (
            is_green
            and (close_val > dem_mid)
            and (prev_close <= dem_mid or low_val <= dem_mid)
            and seen_below_dem
            and (not choppy)
        )
        if not buy_sig and green_cross_up:
          grade, conf_note = self._grade_signal(c, "BUY")
          sl = round(min(dem_mid, low_val) - (1.5 * atr), dec)
          buy_sig = {
              "Asset": self.symbol,
              "Date": str(c["date"]),
              "Time": c["time"],
              "Candle #": c["id"],
              "Type": "BUY 🟢",
              "Grade": grade,
              "Setup": f"{label} Demand Mid Cross BUY ({conf_note})",
              "Entry Price": round(close_val, dec),
              "Stop Loss (SL)": sl,
              "TP 1": round(close_val + (1.2 * atr), dec),
              "TP 2": round(close_val + (2.5 * atr), dec),
              "TP 3": round(close_val + (4.0 * atr), dec),
              "Is_Within_1hr": c["is_within_1hr"],
              "Signal_DT": c["dt_user"].isoformat(),
          }

      prev_c = c

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

    latest_date = candles_4h[-1]["date"] if candles_4h else now_user.date()
    today_4h = [c for c in candles_4h if c["date"] == latest_date]
    if not today_4h:
      today_4h = candles_4h

    anchor_1 = min(
        today_4h,
        key=lambda x: abs(datetime.combine(latest_date, x["time_obj"]) - datetime.combine(latest_date, time(1, 30))),
        default=today_4h[0]
    )
    self.anchor_0130_start = anchor_1["dt_user"]
    self.demand_box_1, self.supply_box_1, self.demand_1_mid, self.supply_1_mid = self._build_zones(anchor_1)

    anchor_2_candidates = [c for c in today_4h if c["time_obj"] >= time(5, 0)]
    if anchor_2_candidates:
      anchor_2 = min(
          anchor_2_candidates,
          key=lambda x: abs(datetime.combine(latest_date, x["time_obj"]) - datetime.combine(latest_date, time(5, 30)))
      )
      self.demand_box_2, self.supply_box_2, _, _ = self._build_zones(anchor_2)

    anchor_3, anchor_dt_3 = self._get_latest_2130_anchor(candles_4h)
    if anchor_3:
      self.anchor_2130_candle = anchor_3
      self.anchor_2130_start = anchor_dt_3
      self.demand_box_3, self.supply_box_3, self.demand_3_mid, self.supply_3_mid = self._build_zones(anchor_3)

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
          "dt_user": dt_user,
          "color": "Green" if c >= o else "Red",
          "open": o, "high": h, "low": l, "close": c,
          "atr": atr, "atr_sma": atr_sma, "ema50": ema50,
          "zl_line": zl_val, "zl_upper": zl_val + zl_b, "zl_lower": zl_val - zl_b,
          "zl_trend": zl_tr,
          "is_within_1hr": age_min <= 60.0
      })

    buy_2130, sell_2130 = self._scan_mid_cross(
        self.candles_5m,
        self.anchor_2130_start,
        self.supply_3_mid,
        self.demand_3_mid,
        "21:30",
    )

    buy_0130, sell_0130 = self._scan_mid_cross(
        self.candles_5m,
        self.anchor_0130_start,
        self.supply_1_mid,
        self.demand_1_mid,
        "01:30",
    )

    final_buy = buy_2130 if buy_2130 is not None else buy_0130
    final_sell = sell_2130 if sell_2130 is not None else sell_0130

    collected = [sig for sig in [final_buy, final_sell] if sig is not None]
    collected.sort(key=lambda x: x["Candle #"])

    self.history = collected
    self.signals = [s for s in collected if s.get("Is_Within_1hr", False)]
    self._build_diy_checklist()


# ---------------------------------------------------------
# MASTER SCANNER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔥 TWT Master Live Signal Dashboard (20+ Assets)")

master_signals = []
asset_engines = {}


def scan_asset(sym):
  df_1d, df_4h, df_5m, strength_info = fetch_data(sym)
  if df_1d is not None and df_4h is not None and df_5m is not None:
    engine = TWTUltimateEngine(
        df_1d,
        df_4h,
        df_5m,
        sym,
        active_tz,
        chop_filter=use_chop_filter,
        strength_info=strength_info,
    )
    engine.calculate_and_scan()
    return sym, engine, engine.signals
  return sym, None, []


with st.spinner("⚡ Scanning 20+ assets (Zero-Lag MTF + DIY Builder)..."):
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(scan_asset, sym): sym for sym in ASSET_LIST}
    for future in as_completed(futures):
      sym, engine, sigs = future.result()
      if engine:
        asset_engines[sym] = engine
      if sigs:
        master_signals.extend(sigs)

ledger_records = load_ledger()
for eng in asset_engines.values():
  ledger_records = update_ledger_with_engine(ledger_records, eng)
save_ledger(ledger_records)

if master_signals:
  st.success(f"✅ Found {len(master_signals)} mid-cross confluence setup(s) across assets!")
  df_display = pd.DataFrame(master_signals).drop(columns=["Is_Within_1hr", "Signal_DT"], errors="ignore")
  st.dataframe(df_display, width="stretch")
else:
  st.info("ℹ️ No active mid-cross setups right now.")

# ---------------------------------------------------------
# MANUAL ASSET INSPECTOR & PER-ASSET LEDGER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Individual Asset Inspector (Zero-Lag MTF + DIY Builder)")

selected_symbol = st.selectbox(
    "🔍 Select Asset to Load Chart",
    options=ASSET_LIST,
    index=0,
)

if selected_symbol in asset_engines:
  bot = asset_engines[selected_symbol]
else:
  df_1d, df_4h, df_5m, strength_info = fetch_data(selected_symbol)
  bot = TWTUltimateEngine(
      df_1d,
      df_4h,
      df_5m,
      selected_symbol,
      active_tz,
      chop_filter=use_chop_filter,
      strength_info=strength_info,
  )
  bot.calculate_and_scan()

if bot.strength_info and bot.strength_info.get("rows"):
  st.markdown(f"#### 💪 {selected_symbol} Buyer vs Seller Strength + Zero-Lag MTF (5m, 15m, 1h, 4h, 1d)")
  s_col1, s_col2, s_col3 = st.columns(3)
  with s_col1:
    st.metric("🟢 Buyer Strength (MTF)", f"{bot.strength_info['buyer_pct']}%")
  with s_col2:
    st.metric("🔴 Seller Strength (MTF)", f"{bot.strength_info['seller_pct']}%")
  with s_col3:
    st.metric("⚡ Zyada Faydemand Side", bot.strength_info["verdict"])

  st.dataframe(pd.DataFrame(bot.strength_info["rows"]), width="stretch")

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
    st.metric(label="Locked Signals", value=len(bot.history))
  with col3:
    st.metric(label="Timezone", value=selected_tz_name)

  if bot.anchor_2130_candle:
    st.caption(
        f"🕒 21:30 4H Candle ({bot.anchor_2130_start.strftime('%Y-%m-%d %H:%M')}) — "
        f"Supply 3 Mid: {bot.supply_3_mid:.4f} | Demand 3 Mid: {bot.demand_3_mid:.4f} "
        f"(Fallback -> 01:30 Supply Mid: {bot.supply_1_mid:.4f} | Demand Mid: {bot.demand_1_mid:.4f})"
    )

  if bot.history:
    st.markdown(f"### 📋 Locked Entries Feed for {selected_symbol}")
    df_history = pd.DataFrame(bot.history).drop(columns=["Is_Within_1hr", "Signal_DT"], errors="ignore")
    st.dataframe(df_history, width="stretch")

  # --- PLOT CHART WITH ALL LEVELS + ZERO-LAG LINE ---
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

  if bot.anchor_2130_start:
    c_2130_list = [c for c in bot.candles_5m if c["dt_user"] >= bot.anchor_2130_start]
    if c_2130_list:
      first_2130_id = c_2130_list[0]["id"]
      ax.axvline(first_2130_id, color="#ffd54f", linestyle=":", linewidth=1.5, label=f"21:30 Candle Start ({bot.anchor_2130_start.strftime('%m-%d %H:%M')})")

  if bot.demand_box_1:
    ax.axhspan(bot.demand_box_1[0], bot.demand_box_1[1], color="#00897b", alpha=0.30, label="Demand 1 (01:30)")
  if bot.demand_1_mid is not None:
    ax.axhline(bot.demand_1_mid, color="#80cbc4", linestyle="-.", linewidth=1.1, label=f"Demand 1 Mid: {bot.demand_1_mid:.2f}")
  if bot.supply_box_1:
    ax.axhspan(bot.supply_box_1[0], bot.supply_box_1[1], color="#e53935", alpha=0.30, label="Supply 1 (01:30)")
  if bot.supply_1_mid is not None:
    ax.axhline(bot.supply_1_mid, color="#ef9a9a", linestyle="-.", linewidth=1.1, label=f"Supply 1 Mid: {bot.supply_1_mid:.2f}")

  if bot.demand_box_2:
    ax.axhspan(bot.demand_box_2[0], bot.demand_box_2[1], color="#26a69a", alpha=0.18, label="Demand 2 (05:30)")
  if bot.supply_box_2:
    ax.axhspan(bot.supply_box_2[0], bot.supply_box_2[1], color="#ff7043", alpha=0.18, label="Supply 2 (05:30)")

  if bot.demand_box_3:
    ax.axhspan(bot.demand_box_3[0], bot.demand_box_3[1], color="#29b6f6", alpha=0.20, label="Demand 3 (21:30)")
  if bot.demand_3_mid is not None:
    ax.axhline(bot.demand_3_mid, color="#81d4fa", linestyle="-.", linewidth=1.1, label=f"Demand 3 Mid: {bot.demand_3_mid:.2f}")
  if bot.supply_box_3:
    ax.axhspan(bot.supply_box_3[0], bot.supply_box_3[1], color="#ab47bc", alpha=0.20, label="Supply 3 (21:30)")
  if bot.supply_3_mid is not None:
    ax.axhline(bot.supply_3_mid, color="#ce93d8", linestyle="-.", linewidth=1.1, label=f"Supply 3 Mid: {bot.supply_3_mid:.2f}")

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
