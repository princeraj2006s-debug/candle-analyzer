from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ==========================================================
# PAGE CONFIGURATION & STYLING
# ==========================================================
st.set_page_config(
    page_title="TWT Advanced Pullback Bot Pro", page_icon="⚡", layout="wide"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Using IANA Timezones with ZoneInfo for proper Daylight Saving Time (DST) handling
TRADING_TIMEZONES = {
    "(UTC+5:30) Kolkata / India (IST)": ZoneInfo("Asia/Kolkata"),
    "(UTC+0:00) London / UTC / GMT": ZoneInfo("Europe/London"),
    "(UTC-5:00) New York / US East (EST)": ZoneInfo("America/New_York"),
}

ASSET_LIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", 
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "NEAR/USDT"
]

# Automatic Refresh every 30 seconds
st_autorefresh(interval=30000, key="twt_advanced_refresh")

st.title("⚡ TWT ADVANCED BUYER & SELLER PULLBACK ENTRY SYSTEM (PRO)")

# ==========================================================
# HEADER CONTROLS (TIMEFRAME & TIMEZONE)
# ==========================================================
col_tf, col_tz = st.columns([1, 2])
with col_tf:
    selected_tf = st.selectbox("⏱️ Timeframe", options=["5m", "15m", "30m", "1h"], index=0)
with col_tz:
    selected_tz_name = st.selectbox("🌍 Timezone", options=list(TRADING_TIMEZONES.keys()), index=0)
    active_tz = TRADING_TIMEZONES[selected_tz_name]

tz_short = selected_tz_name.split(")")[0] + ")"

# ==========================================================
# ROBUST EXCHANGE DATA FETCHING WITH FRESHNESS & FALLBACK FIX
# ==========================================================
@st.cache_data(ttl=30)
def fetch_data(symbol, tf):
    exchanges = [ccxt.binance, ccxt.bybit]
    for ex_class in exchanges:
        try:
            ex = ex_class({"enableRateLimit": True, "timeout": 30000})
            ex.load_markets()
            if symbol in ex.markets:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=250)
                if ohlcv and len(ohlcv) > 60:
                    df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).set_index("timestamp")
                    
                    # Ensure timezone-aware UTC index
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    else:
                        df.index = df.index.tz_convert("UTC")
                        
                    # Data Freshness Validation Check
                    latest_candle_time = df.index[-1]
                    now_utc = datetime.now(timezone.utc)
                    
                    tf_minutes = 5
                    if tf == "15m": tf_minutes = 15
                    elif tf == "30m": tf_minutes = 30
                    elif tf == "1h": tf_minutes = 60
                    
                    max_stale_delta = timedelta(minutes=tf_minutes * 10) 
                    if (now_utc - latest_candle_time) > max_stale_delta:
                        continue
                        
                    return df, ex.id.upper()
        except Exception:
            continue
            
    # Fallback simulation feed if live API connections fail/restricted
    try:
        freq_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h'}
        freq_val = freq_map.get(tf, '5min')
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=120, freq=freq_val)
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(120) * 0.4)
        df_fallback = pd.DataFrame({
            "Open": close_prices - np.random.rand(120)*0.2,
            "High": close_prices + np.abs(np.random.randn(120)*0.4),
            "Low": close_prices - np.abs(np.random.randn(120)*0.4),
            "Close": close_prices,
            "Volume": np.random.randint(1000, 5000, 120)
        }, index=dates)
        return df_fallback, "MOCK/FALLBACK (NETWORK RESTRICTED)"
    except Exception:
        return None, None

# ==========================================================
# TWT ADVANCED PULLBACK BOT ENGINE (STATE MACHINE & STRATEGY)
# ==========================================================
class TWTAdvancedPullbackBot:
    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol

    def classify_candles(self):
        """Classify candles using completed historical candles only (no look-ahead bias)."""
        df = self.df.copy()
        if len(df) < 30:
            df["SizeClass"] = "UNAVAILABLE"
            return df

        df["Body"] = abs(df["Close"] - df["Open"])
        df["UpperWick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["LowerWick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["Range"] = df["High"] - df["Low"]

        completed_body = df["Body"].shift(1)
        completed_range = df["Range"].shift(1)

        df["AvgBody20"] = completed_body.rolling(window=20).mean()
        df["AvgRange20"] = completed_range.rolling(window=20).mean()

        df["BodyPct"] = np.where(df["Range"] > 0, (df["Body"] / df["Range"]) * 100, 0)
        df["UpperWickPct"] = np.where(df["Range"] > 0, (df["UpperWick"] / df["Range"]) * 100, 0)
        df["LowerWickPct"] = np.where(df["Range"] > 0, (df["LowerWick"] / df["Range"]) * 100, 0)
        df["Color"] = np.where(df["Close"] >= df["Open"], "Green", "Red")

        def size_label(row):
            b, ab, r, ar = row["Body"], row["AvgBody20"], row["Range"], row["AvgRange20"]
            if pd.isna(ab) or pd.isna(ar) or ab == 0 or ar == 0:
                return "Unavailable"
            if r > ar * 2.5 or b > ab * 2.5:
                return "Abnormally Large"
            elif b > ab * 1.4:
                return "Long"
            elif b < ab * 0.6:
                return "Small"
            else:
                return "Medium"

        df["SizeClass"] = df.apply(size_label, axis=1)
        return df

    def get_validated_swing_points(self, completed_df):
        """Generate strictly alternating, chronological pivot swing points ('H' and 'L') without duplication."""
        highs = completed_df["High"].values
        lows = completed_df["Low"].values
        swings = []
        lookback = 3
        
        for i in range(lookback, len(completed_df) - lookback):
            is_high = highs[i] == max(highs[i-lookback:i+lookback+1])
            is_low = lows[i] == min(lows[i-lookback:i+lookback+1])
            
            if is_high and not is_low:
                if not swings or swings[-1][0] != 'H':
                    swings.append(('H', i, highs[i]))
                elif swings[-1][0] == 'H' and highs[i] > swings[-1][2]:
                    swings[-1] = ('H', i, highs[i])
            elif is_low and not is_high:
                if not swings or swings[-1][0] != 'L':
                    swings.append(('L', i, lows[i]))
                elif swings[-1][0] == 'L' and lows[i] < swings[-1][2]:
                    swings[-1] = ('L', i, lows[i])
        return swings

    def analyze_market_structure(self, df):
        """Confirmed pivot-based structural trend identification using historical bars only."""
        if len(df) < 40:
            return "RANGING — NO ENTRY", None, None

        completed_df = df.iloc[:-1]
        swings = self.get_validated_swing_points(completed_df)
        
        ema20 = completed_df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = completed_df["Close"].ewm(span=50, adjust=False).mean().iloc[-1]

        shs = [s[2] for s in swings if s[0] == 'H']
        sls = [s[2] for s in swings if s[0] == 'L']

        if len(shs) >= 2 and len(sls) >= 2:
            hh = shs[-1] > shs[-2]
            hl = sls[-1] > sls[-2]
            lh = shs[-1] < shs[-2]
            ll = sls[-1] < sls[-2]

            if hh and hl and completed_df["Close"].iloc[-1] > ema20:
                return "Uptrend", shs[-1], sls[-1]
            elif lh and ll and completed_df["Close"].iloc[-1] < ema20:
                return "Downtrend", shs[-1], sls[-1]
            elif (hh and ll) or (lh and hl):
                return "POTENTIAL TREND REVERSAL — WAIT", None, None

        closes = completed_df["Close"].iloc[-15:]
        if completed_df["Close"].iloc[-1] > ema20 > ema50 and closes.is_monotonic_increasing:
            return "Uptrend", completed_df["High"].max(), completed_df["Low"].min()
        elif completed_df["Close"].iloc[-1] < ema20 < ema50 and closes.is_monotonic_decreasing:
            return "Downtrend", completed_df["High"].max(), completed_df["Low"].min()

        return "RANGING — NO ENTRY", None, None

    def detect_structural_pullbacks(self, df, trend):
        """State Machine Pullback Detection tracking the exact current structural retracement sequence."""
        completed_df = df.iloc[:-1]
        if len(completed_df) < 40:
            return 0, "PULLBACK NOT CONFIRMED — WAIT", "WAITING_FOR_TREND"

        swings = self.get_validated_swing_points(completed_df)
        if len(swings) < 4:
            return 0, "PULLBACK NOT CONFIRMED — WAIT", "WAITING_FOR_TREND"

        pullback_count = 0
        state = "TREND_MOVEMENT"

        if trend == "Downtrend":
            upward_swings = 0
            for idx in range(1, len(swings)):
                p_type, _, p_price = swings[idx-1]
                c_type, _, c_price = swings[idx]
                if p_type == 'L' and c_type == 'H' and c_price > p_price:
                    upward_swings += 1
            pullback_count = min(max(upward_swings, 1), 5)
            state = "RETRACEMENT" if pullback_count >= 3 else "TREND_MOVEMENT"
        elif trend == "Uptrend":
            downward_swings = 0
            for idx in range(1, len(swings)):
                p_type, _, p_price = swings[idx-1]
                c_type, _, c_price = swings[idx]
                if p_type == 'H' and c_type == 'L' and c_price < p_price:
                    downward_swings += 1
            pullback_count = min(max(downward_swings, 1), 5)
            state = "RETRACEMENT" if pullback_count >= 3 else "TREND_MOVEMENT"
        else:
            return 0, "PULLBACK NOT CONFIRMED — WAIT", "WAITING_FOR_TREND"

        return pullback_count, f"Pullback {pullback_count}", state

    def evaluate_buyer_weakness(self, df):
        """Verify seller weakness specifically within the active pullback sequence candles."""
        completed_df = df.iloc[:-1]
        recent_bars = completed_df.iloc[-5:]
        reds = recent_bars[recent_bars["Color"] == "Red"]
        if len(reds) == 0:
            return False, 0

        red_bodies = reds["Body"].values
        recent_body = red_bodies[-1]
        avg_body = red_bodies.mean() if len(red_bodies) > 0 else recent_body

        lower_wick_ok = reds.iloc[-1]["LowerWickPct"] > 25 or reds.iloc[-1]["LowerWick"] > reds.iloc[-1]["Body"]
        body_shrinking = recent_body <= avg_body * 1.3

        weakness_confirmed = body_shrinking and lower_wick_ok
        return weakness_confirmed, len(reds)

    def evaluate_seller_weakness(self, df):
        """Verify buyer weakness specifically within the active pullback sequence candles."""
        completed_df = df.iloc[:-1]
        recent_bars = completed_df.iloc[-5:]
        greens = recent_bars[recent_bars["Color"] == "Green"]
        if len(greens) == 0:
            return False, 0

        green_bodies = greens["Body"].values
        recent_body = green_bodies[-1]
        avg_body = green_bodies.mean() if len(green_bodies) > 0 else recent_body

        upper_wick_ok = greens.iloc[-1]["UpperWickPct"] > 25 or greens.iloc[-1]["UpperWick"] > greens.iloc[-1]["Body"]
        body_shrinking = recent_body <= avg_body * 1.3

        weakness_confirmed = body_shrinking and upper_wick_ok
        return weakness_confirmed, len(greens)

    def evaluate_buyer_pullback_3(self, df, trend):
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]
        weakness_ok, count = self.evaluate_buyer_weakness(df)
        valid_wick = prev2["LowerWickPct"] > 25 or prev2["LowerWick"] > prev2["Body"] * 0.5
        first_med_green = prev2["Color"] == "Green" and prev2["SizeClass"] in ["Medium", "Long"]
        second_conf_green = prev1["Color"] == "Green" and prev1["Close"] > prev2["High"]
        return weakness_ok and valid_wick and first_med_green and second_conf_green, count

    def evaluate_buyer_pullback_4(self, df, trend):
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]
        weakness_ok, count = self.evaluate_buyer_weakness(df)
        valid_wick = prev2["LowerWickPct"] > 30
        first_med_green = prev2["Color"] == "Green" and prev2["SizeClass"] in ["Medium", "Long", "Abnormally Large"]
        second_conf_green = prev1["Color"] == "Green" and prev1["Close"] > prev2["High"] and prev1["Body"] >= prev2["Body"] * 0.7
        return weakness_ok and valid_wick and first_med_green and second_conf_green, count

    def evaluate_buyer_pullback_5(self, df, trend):
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]
        weakness_ok, count = self.evaluate_buyer_weakness(df)
        valid_wick = prev2["LowerWickPct"] > 35
        first_med_green = prev2["Color"] == "Green" and prev2["SizeClass"] in ["Medium", "Long"]
        second_conf_green = prev1["Color"] == "Green" and prev1["Close"] > prev2["High"]
        return weakness_ok and valid_wick and first_med_green and second_conf_green, count

    def evaluate_seller_pullback_3(self, df, trend):
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]
        weakness_ok, count = self.evaluate_seller_weakness(df)
        valid_wick = prev2["UpperWickPct"] > 25 or prev2["UpperWick"] > prev2["Body"] * 0.5
        first_med_red = prev2["Color"] == "Red" and prev2["SizeClass"] in ["Medium", "Long"]
        second_conf_red = prev1["Color"] == "Red" and prev1["Close"] < prev2["Low"]
        return weakness_ok and valid_wick and first_med_red and second_conf_red, count

    def evaluate_seller_pullback_4(self, df, trend):
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]
        weakness_ok, count = self.evaluate_seller_weakness(df)
        valid_wick = prev2["UpperWickPct"] > 30
        first_med_red = prev2["Color"] == "Red" and prev2["SizeClass"] in ["Medium", "Long", "Abnormally Large"]
        second_conf_red = prev1["Color"] == "Red" and prev1["Close"] < prev2["Low"] and prev1["Body"] >= prev2["Body"] * 0.7
        return weakness_ok and valid_wick and first_med_red and second_conf_red, count

    def evaluate_seller_pullback_5(self, df, trend):
        prev1 = df.iloc[-2]
        prev2 = df.iloc[-3]
        weakness_ok, count = self.evaluate_seller_weakness(df)
        valid_wick = prev2["UpperWickPct"] > 35
        first_med_red = prev2["Color"] == "Red" and prev2["SizeClass"] in ["Medium", "Long"]
        second_conf_red = prev1["Color"] == "Red" and prev1["Close"] < prev2["Low"]
        return weakness_ok and valid_wick and first_med_red and second_conf_red, count

    def calculate_stop_loss_and_target(self, df, is_buyer, entry_price):
        dec = 4 if entry_price < 1 else (2 if entry_price > 10 else 3)
        completed_df = df.iloc[:-1]
        
        if is_buyer:
            swing_low = completed_df.iloc[-10:-1]["Low"].min()
            sl = round(swing_low - (completed_df.iloc[-1]["Range"] * 0.2), dec)
            if sl >= entry_price:
                sl = round(entry_price - (completed_df.iloc[-1]["Range"] * 1.0), dec)
            resistance = completed_df.iloc[-20:-1]["High"].max()
            tp = round(resistance if resistance > entry_price else entry_price + ((entry_price - sl) * 2.0), dec)
            risk = entry_price - sl
            reward = tp - entry_price
        else:
            swing_high = completed_df.iloc[-10:-1]["High"].max()
            sl = round(swing_high + (completed_df.iloc[-1]["Range"] * 0.2), dec)
            if sl <= entry_price:
                sl = round(entry_price + (completed_df.iloc[-1]["Range"] * 1.0), dec)
            support = completed_df.iloc[-20:-1]["Low"].min()
            tp = round(support if support < entry_price else entry_price - ((sl - entry_price) * 2.0), dec)
            risk = sl - entry_price
            reward = entry_price - tp

        rr_ratio = reward / risk if risk > 0 else 0
        return sl, tp, rr_ratio

    def calculate_signal_score(self, trend, pb_num, weakness_ok, wick_ok, conf_ok, rr_ratio):
        score = 0
        if "Uptrend" in trend or "Downtrend" in trend: score += 20
        if pb_num in [3, 4, 5]: score += 20
        if weakness_ok: score += 20
        if wick_ok: score += 15
        if conf_ok: score += 15
        if rr_ratio >= 1.5: score += 10
        return score

    def run_system(self):
        df = self.classify_candles()
        if df is None or len(df) < 50:
            return {"Status": "DATA UNAVAILABLE — NO ENTRY", "Trend": "Insufficient Data", "State": "WAITING_FOR_TREND"}

        trend, major_high, major_low = self.analyze_market_structure(df)
        if "RANGING" in trend or "REVERSAL" in trend:
            return {"Status": f"{trend} — Trend not confirmed", "Trend": trend, "Pullback": "N/A", "State": "WAITING_FOR_TREND"}

        pb_num, pb_label, state = self.detect_structural_pullbacks(df, trend)
        if pb_num < 3:
            return {"Status": "PULLBACK NOT CONFIRMED — WAIT — Pullback not confirmed", "Trend": trend, "Pullback": pb_label, "State": state}

        prev1 = df.iloc[-2]
        dec = 4 if prev1["Close"] < 1 else (2 if prev1["Close"] > 10 else 3)
        entry_price = round(prev1["Close"], dec)

        if trend == "Downtrend":
            if pb_num == 3: valid, count = self.evaluate_buyer_pullback_3(df, trend)
            elif pb_num == 4: valid, count = self.evaluate_buyer_pullback_4(df, trend)
            elif pb_num == 5: valid, count = self.evaluate_buyer_pullback_5(df, trend)
            else: valid, count = False, 0

            weakness_ok, _ = self.evaluate_buyer_weakness(df)
            if not weakness_ok:
                return {"Status": "WEAKNESS NOT CONFIRMED — WAIT — Weakness not confirmed", "Trend": trend, "Pullback": pb_label, "State": "WEAKNESS_CONFIRMATION"}
            if not valid:
                return {"Status": "ENTRY CONFIRMATION NOT COMPLETED — Entry confirmation missing", "Trend": trend, "Pullback": pb_label, "State": "ENTRY_CONFIRMATION"}

            sl, tp, rr_ratio = self.calculate_stop_loss_and_target(df, True, entry_price)
            if sl >= entry_price or tp <= entry_price:
                return {"Status": "INVALID SL/TP — Invalid SL/TP", "Trend": trend, "Pullback": pb_label, "State": "SIGNAL_INVALID"}
            if rr_ratio < 1.5:
                return {"Status": "RISK-TO-REWARD FAILED — Risk-reward failed", "Trend": trend, "Pullback": pb_label, "State": "SIGNAL_INVALID"}

            wick_ok = df.iloc[-3]["LowerWickPct"] > 25
            score = self.calculate_signal_score(trend, pb_num, weakness_ok, wick_ok, True, rr_ratio)

            return {
                "Type": "BUYER", "Trend": trend, "Pullback": pb_label, "State": "SIGNAL_VALID",
                "BuyerStrength": "Confirmed", "SellerWeakness": f"Confirmed ({count} mild pullbacks)",
                "WickRejection": f"Lower Wick {df.iloc[-3]['LowerWickPct']:.1f}%",
                "Confirmation": f"Two green candles (Pullback {pb_num} Rules)",
                "Entry": entry_price, "StopLoss": sl, "Target": tp, "RR": f"1:{rr_ratio:.2f}",
                "Status": "VALID", "Score": score
            }

        elif trend == "Uptrend":
            if pb_num == 3: valid, count = self.evaluate_seller_pullback_3(df, trend)
            elif pb_num == 4: valid, count = self.evaluate_seller_pullback_4(df, trend)
            elif pb_num == 5: valid, count = self.evaluate_seller_pullback_5(df, trend)
            else: valid, count = False, 0

            weakness_ok, _ = self.evaluate_seller_weakness(df)
            if not weakness_ok:
                return {"Status": "WEAKNESS NOT CONFIRMED — WAIT — Weakness not confirmed", "Trend": trend, "Pullback": pb_label, "State": "WEAKNESS_CONFIRMATION"}
            if not valid:
                return {"Status": "ENTRY CONFIRMATION NOT COMPLETED — Entry confirmation missing", "Trend": trend, "Pullback": pb_label, "State": "ENTRY_CONFIRMATION"}

            sl, tp, rr_ratio = self.calculate_stop_loss_and_target(df, False, entry_price)
            if sl <= entry_price or tp >= entry_price:
                return {"Status": "INVALID SL/TP — Invalid SL/TP", "Trend": trend, "Pullback": pb_label, "State": "SIGNAL_INVALID"}
            if rr_ratio < 1.5:
                return {"Status": "RISK-TO-REWARD FAILED — Risk-reward failed", "Trend": trend, "Pullback": pb_label, "State": "SIGNAL_INVALID"}

            wick_ok = df.iloc[-3]["UpperWickPct"] > 25
            score = self.calculate_signal_score(trend, pb_num, weakness_ok, wick_ok, True, rr_ratio)

            return {
                "Type": "SELLER", "Trend": trend, "Pullback": pb_label, "State": "SIGNAL_VALID",
                "SellerStrength": "Confirmed", "BuyerWeakness": f"Confirmed ({count} mild pullbacks)",
                "WickRejection": f"Upper Wick {df.iloc[-3]['UpperWickPct']:.1f}%",
                "Confirmation": f"Two red candles (Pullback {pb_num} Rules)",
                "Entry": entry_price, "StopLoss": sl, "Target": tp, "RR": f"1:{rr_ratio:.2f}",
                "Status": "VALID", "Score": score
            }

        return {"Status": "NO ENTRY — WAIT FOR A VALID PULLBACK", "Trend": trend, "Pullback": "N/A", "State": "WAITING_FOR_TREND"}

    def run_backtest(self):
        df = self.classify_candles()
        if len(df) < 60: return None
        trades = []
        fee_pct = 0.0004 
        last_trade_exit_bar = -1

        for i in range(50, len(df) - 6):
            if i <= last_trade_exit_bar: continue
            sub_df = df.iloc[:i+1].copy()
            temp_bot = TWTAdvancedPullbackBot(sub_df, self.symbol)
            sub_df = temp_bot.classify_candles()
            temp_bot.df = sub_df
            
            trend, _, _ = temp_bot.analyze_market_structure(sub_df)
            if "RANGING" in trend or "REVERSAL" in trend: continue
            pb_num, _, _ = temp_bot.detect_structural_pullbacks(sub_df, trend)
            if pb_num < 3: continue

            prev1 = sub_df.iloc[-2]
            if trend == "Downtrend":
                valid = False
                if pb_num == 3: valid, _ = temp_bot.evaluate_buyer_pullback_3(sub_df, trend)
                elif pb_num == 4: valid, _ = temp_bot.evaluate_buyer_pullback_4(sub_df, trend)
                elif pb_num == 5: valid, _ = temp_bot.evaluate_buyer_pullback_5(sub_df, trend)

                if valid:
                    entry = prev1["Close"] * (1 + fee_pct)
                    sl, tp, rr = temp_bot.calculate_stop_loss_and_target(sub_df, True, entry)
                    if rr >= 1.5 and sl < entry:
                        outcome = "Ambiguous"
                        net_pnl = 0.0
                        exit_idx = i
                        future_bars = df.iloc[i+1:i+15]
                        for idx, (_, bar) in enumerate(future_bars.iterrows()):
                            current_bar_idx = i + 1 + idx
                            if bar["Low"] <= sl and bar["High"] >= tp:
                                outcome = "Loss"; net_pnl = - (entry - sl) - (entry * fee_pct * 2); exit_idx = current_bar_idx; break
                            elif bar["Low"] <= sl:
                                outcome = "Loss"; net_pnl = - (entry - sl) - (entry * fee_pct * 2); exit_idx = current_bar_idx; break
                            elif bar["High"] >= tp:
                                outcome = "Win"; net_pnl = (tp - entry) - (entry * fee_pct * 2); exit_idx = current_bar_idx; break
                        trades.append({"Type": "BUYER", "Outcome": outcome, "NetPnL": net_pnl})
                        last_trade_exit_bar = exit_idx

            elif trend == "Uptrend":
                valid = False
                if pb_num == 3: valid, _ = temp_bot.evaluate_seller_pullback_3(sub_df, trend)
                elif pb_num == 4: valid, _ = temp_bot.evaluate_seller_pullback_4(sub_df, trend)
                elif pb_num == 5: valid, _ = temp_bot.evaluate_seller_pullback_5(sub_df, trend)

                if valid:
                    entry = prev1["Close"] * (1 - fee_pct)
                    sl, tp, rr = temp_bot.calculate_stop_loss_and_target(sub_df, False, entry)
                    if rr >= 1.5 and sl > entry:
                        outcome = "Ambiguous"
                        net_pnl = 0.0
                        exit_idx = i
                        future_bars = df.iloc[i+1:i+15]
                        for idx, (_, bar) in enumerate(future_bars.iterrows()):
                            current_bar_idx = i + 1 + idx
                            if bar["High"] >= sl and bar["Low"] <= tp:
                                outcome = "Loss"; net_pnl = - (sl - entry) - (entry * fee_pct * 2); exit_idx = current_bar_idx; break
                            elif bar["High"] >= sl:
                                outcome = "Loss"; net_pnl = - (sl - entry) - (entry * fee_pct * 2); exit_idx = current_bar_idx; break
                            elif bar["Low"] <= tp:
                                outcome = "Win"; net_pnl = (entry - tp) - (entry * fee_pct * 2); exit_idx = current_bar_idx; break
                        trades.append({"Type": "SELLER", "Outcome": outcome, "NetPnL": net_pnl})
                        last_trade_exit_bar = exit_idx

        if not trades: return None
        total = len(trades)
        wins = sum(1 for t in trades if t["Outcome"] == "Win")
        losses = sum(1 for t in trades if t["Outcome"] == "Loss")
        ambiguous = sum(1 for t in trades if t["Outcome"] == "Ambiguous")
        win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
        
        gross_profit = sum(t["NetPnL"] for t in trades if t["NetPnL"] > 0)
        gross_loss = abs(sum(t["NetPnL"] for t in trades if t["NetPnL"] < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

        return {"Total": total, "Wins": wins, "Losses": losses, "Ambiguous": ambiguous, "WinRate": win_rate, "ProfitFactor": profit_factor}

# ==========================================================
# STREAMLIT UI EXECUTION & SESSION STATE
# ==========================================================
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []
if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []

selected_symbol = st.selectbox("🔍 Select Asset", options=ASSET_LIST, index=0)

df_asset, active_exchange = fetch_data(selected_symbol, selected_tf)

if df_asset is not None and not df_asset.empty:
    df_display = df_asset.copy()
    df_display.index = df_display.index.tz_convert(active_tz)

    # ---------------------------------------------------------
    # LIVE PLOTLY CANDLESTICK CHART DISPLAY
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader(f"📈 Live Market Price Chart — {selected_symbol} ({selected_tf})")
    
    chart_df = df_display.tail(100)
    fig = go.Figure(data=[go.Candlestick(
        x=chart_df.index,
        open=chart_df['Open'],
        high=chart_df['High'],
        low=chart_df['Low'],
        close=chart_df['Close'],
        name="Candles"
    )])
    fig.update_layout(
        title=f"{selected_symbol} Live Chart",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    bot = TWTAdvancedPullbackBot(df_asset, selected_symbol)
    result = bot.run_system()
    backtest_metrics = bot.run_backtest()

    st.markdown("---")
    st.subheader(f"📊 Analysis Result for {selected_symbol}")

    if result and result.get("Status") == "VALID":
        completed_candle_ts = df_asset.index[-2]
        current_sig_id = f"{selected_symbol}_{selected_tf}_{result['Type']}_{completed_candle_ts}"
        
        if not any(s.get("id") == current_sig_id for s in st.session_state.signal_history):
            result["id"] = current_sig_id
            result["Timestamp"] = df_display.index[-2].strftime('%Y-%m-%d %H:%M:%S')
            result["Asset"] = selected_symbol
            result["Exchange"] = active_exchange
            st.session_state.signal_history.insert(0, result)

        if result["Type"] == "BUYER":
            st.success("🟢 **BUYER SETUP DETECTED**")
            st.code(f"""BUYER SETUP DETECTED
Trend: {result['Trend']}
Pullback: {result['Pullback']}
State: {result['State']}
Buyer Strength: {result['BuyerStrength']}
Seller Weakness: {result['SellerWeakness']}
Wick Rejection: {result.get('WickRejection', 'N/A')}
Confirmation: {result['Confirmation']}
Entry: {result['Entry']}
Stop Loss: {result['StopLoss']}
Target: {result['Target']}
Risk-to-Reward: {result['RR']}
Signal Status: VALID (Score: {result['Score']}/100)""", language="text")
        else:
            st.error("🔴 **SELLER SETUP DETECTED**")
            st.code(f"""SELLER SETUP DETECTED
Trend: {result['Trend']}
Pullback: {result['Pullback']}
State: {result['State']}
Seller Strength: {result['SellerStrength']}
Buyer Weakness: {result['BuyerWeakness']}
Wick Rejection: {result.get('WickRejection', 'N/A')}
Confirmation: {result['Confirmation']}
Entry: {result['Entry']}
Stop Loss: {result['StopLoss']}
Target: {result['Target']}
Risk-to-Reward: {result['RR']}
Signal Status: VALID (Score: {result['Score']}/100)""", language="text")
    else:
        status_msg = result.get('Status', 'NO ENTRY — WAIT FOR A VALID PULLBACK')
        st.warning(f"🟡 **{status_msg}**")
        st.write(f"Current detected market structure trend: **{result.get('Trend', 'Unknown')}** | Pullback Count: **{result.get('Pullback', 'N/A')}** | State: **{result.get('State', 'WAITING')}**. Diagnostic reason: *{status_msg}*.")

    # ==========================================================
    # INTERACTIVE PAPER TRADING DESK
    # ==========================================================
    st.markdown("---")
    st.subheader("⚡ Interactive Paper Trading Desk")
    
    current_live_price = float(df_asset['Close'].iloc[-1])
    suggested_entry = result.get('Entry', current_live_price) if result.get("Status") == "VALID" else current_live_price
    suggested_sl = result.get('StopLoss', current_live_price * 0.99) if result.get("Status") == "VALID" else current_live_price * 0.99
    suggested_tp = result.get('Target', current_live_price * 1.02) if result.get("Status") == "VALID" else current_live_price * 1.02
    suggested_type = result.get('Type', 'BUYER') if result.get("Status") == "VALID" else 'BUYER'

    with st.form("paper_trading_form"):
        c1, c2, c3, c4, c5 = st.columns(5)
        trade_type = c1.selectbox("Order Type", options=["BUYER", "SELLER"], index=0 if suggested_type=="BUYER" else 1)
        entry_input = c2.number_input("Entry Price", value=float(suggested_entry), format="%.4f")
        sl_input = c3.number_input("Stop Loss (SL)", value=float(suggested_sl), format="%.4f")
        tp_input = c4.number_input("Target (TP)", value=float(suggested_tp), format="%.4f")
        amount_input = c5.number_input("Position Size (USDT)", value=100.0, min_value=10.0, step=10.0)
        
        submitted = st.form_submit_button("🚀 Place Paper Trade")
        if submitted:
            new_trade = {
                "id": f"{selected_symbol}_{datetime.now(timezone.utc).timestamp()}",
                "Asset": selected_symbol,
                "Type": trade_type,
                "Entry": entry_input,
                "SL": sl_input,
                "TP": tp_input,
                "Amount": amount_input,
                "Status": "OPEN",
                "PnL": 0.0,
                "Time": datetime.now(active_tz).strftime('%Y-%m-%d %H:%M:%S')
            }
            st.session_state.paper_trades.insert(0, new_trade)
            st.success(f"Paper Trade Executed Successfully for {selected_symbol} at {entry_input}!")

    # Evaluate open paper trades against current live price
    if st.session_state.paper_trades:
        st.write("### Active & Closed Paper Trades")
        updated_trades = []
        for t in st.session_state.paper_trades:
            if t["Status"] == "OPEN":
                # Calculate floating PnL
                if t["Type"] == "BUYER":
                    if current_live_price <= t["SL"]:
                        t["Status"] = "CLOSED (SL HIT)"
                        t["PnL"] = - (t["Entry"] - t["SL"]) * (t["Amount"] / t["Entry"])
                    elif current_live_price >= t["TP"]:
                        t["Status"] = "CLOSED (TP HIT)"
                        t["PnL"] = (t["TP"] - t["Entry"]) * (t["Amount"] / t["Entry"])
                    else:
                        t["PnL"] = (current_live_price - t["Entry"]) * (t["Amount"] / t["Entry"])
                else: # SELLER
                    if current_live_price >= t["SL"]:
                        t["Status"] = "CLOSED (SL HIT)"
                        t["PnL"] = - (t["SL"] - t["Entry"]) * (t["Amount"] / t["Entry"])
                    elif current_live_price <= t["TP"]:
                        t["Status"] = "CLOSED (TP HIT)"
                        t["PnL"] = (t["Entry"] - t["TP"]) * (t["Amount"] / t["Entry"])
                    else:
                        t["PnL"] = (t["Entry"] - current_live_price) * (t["Amount"] / t["Entry"])
            updated_trades.append(t)
        st.session_state.paper_trades = updated_trades
        
        paper_df = pd.DataFrame(st.session_state.paper_trades)
        st.dataframe(paper_df[["Time", "Asset", "Type", "Entry", "SL", "TP", "Amount", "Status", "PnL"]], use_container_width=True)
        
        if st.button("🗑️ Clear Paper Trade History"):
            st.session_state.paper_trades = []
            st.rerun()
    else:
        st.info("No paper trades placed yet. Use the form above to enter simulated positions.")

    # ==========================================================
    # DASHBOARD DIAGNOSTICS METRICS
    # ==========================================================
    st.markdown("---")
    st.subheader("📈 System Diagnostic Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Asset", selected_symbol)
    m2.metric("Exchange Source", active_exchange if active_exchange else "Binance / Bybit")
    m3.metric("Timeframe / Zone", f"{selected_tf} ({tz_short})")
    m4.metric("Market Structure Trend", result.get('Trend', 'N/A'))

    # ==========================================================
    # PAPER TRADING & SIGNAL HISTORY LOG
    # ==========================================================
    st.markdown("---")
    st.subheader("📋 Bot Signal History Log")
    if st.session_state.signal_history:
        history_df = pd.DataFrame(st.session_state.signal_history)
        st.dataframe(history_df[["Timestamp", "Asset", "Exchange", "Type", "Pullback", "State", "Entry", "StopLoss", "Target", "RR", "Score", "Status"]], use_container_width=True)
    else:
        st.info("No confirmed signals logged in the bot signal history buffer yet for this session.")

    # ==========================================================
    # HISTORICAL BACKTESTING PERFORMANCE SUMMARY
    # ==========================================================
    st.markdown("---")
    st.subheader("📊 Backtesting Performance Summary (Actual Historical Simulation)")
    if backtest_metrics and backtest_metrics["Total"] > 0:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Simulated Signals", backtest_metrics["Total"])
        b2.metric("Calculated Win Rate", f"{backtest_metrics['WinRate']:.1f}%")
        b3.metric("Ambiguous / Timeout", backtest_metrics["Ambiguous"])
        b4.metric("Calculated Profit Factor", f"{backtest_metrics['ProfitFactor']:.2f}" if backtest_metrics['ProfitFactor'] != float('inf') else "∞")
    else:
        st.info("BACKTEST NOT AVAILABLE — Insufficient historical setups for statistical backtesting on current timeframe.")

else:
    st.error("⚠️ DATA UNAVAILABLE / STALE — Failed to load valid live market data from Binance or Bybit. Please verify network connection or select another asset.")
