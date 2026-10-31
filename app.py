import streamlit as st
import cv2
import numpy as np

class PriceActionAnalyzer:
    def __init__(self):
        self.buyer_score = 0
        self.seller_score = 0
        
    def analyze_candles(self, candles_data):
        self.buyer_score = 0
        self.seller_score = 0
        analysis_log = []
        
        total_candles = len(candles_data)
        if total_candles == 0:
            return "No candles detected.", 0, 0
            
        red_count = sum(1 for c in candles_data if c['color'] == 'red')
        red_upper_wick = sum(1 for c in candles_data if c['color'] == 'red' and c['upper_wick'] > 0)
        
        if total_candles >= 4 and red_count >= 4 and red_upper_wick >= 3:
            self.buyer_score += 1.5
            analysis_log.append("Bullish Signal: Red candles with persistent upper wicks (Buying Pressure).")
            
        if total_candles >= 3:
            c1, c2, c3 = candles_data[-3:]
            if c1['color'] == 'red' and c2['color'] == 'red' and c3['color'] == 'red':
                if c1['body_size'] > c2['body_size'] > c3['body_size']:
                    self.buyer_score += 1.0
                    analysis_log.append("Bullish Signal: Bearish momentum weakening (Red candle sizes decreasing).")
                    
        red_highs = [c['high'] for c in candles_data if c['color'] == 'red']
        if len(red_highs) >= 3:
            recent_highs = red_highs[-3:]
            if max(recent_highs) - min(recent_highs) < 0.05 * recent_highs[0]:
                self.seller_score += 1.5
                analysis_log.append("Bearish Signal: Strong resistance rejection at same level.")
                
        green_candles = [c for c in candles_data if c['color'] == 'green']
        if len(green_candles) >= 2:
            if green_candles[-2]['body_size'] > green_candles[-1]['body_size'] and green_candles[-1]['lower_wick'] > 0:
                self.seller_score += 1.0
                analysis_log.append("Bearish Signal: Green candle shrinking with lower wick (Buyer exhaustion).")
                
        if total_candles >= 4:
            c1, c2, c3, c4 = candles_data[-4:]
            if c1['color'] == 'green' and c2['color'] == 'green' and c3['color'] == 'green' and c4['color'] == 'red':
                avg_green_size = (c1['body_size'] + c2['body_size'] + c3['body_size']) / 3
                if c4['body_size'] > (1.5 * avg_green_size):
                    self.seller_score += 2.0
                    analysis_log.append("Bearish Signal: Engulfing Red candle after 3 medium green candles.")
                    
        if total_candles >= 6:
            window = candles_data[-6:]
            greens = [c for c in window if c['color'] == 'green']
            reds = [c for c in window if c['color'] == 'red']
            if len(greens) >= 4:
                avg_red_body = (sum(c['body_size'] for c in reds) / len(reds)) if reds else 0
                large_greens = sum(1 for g in greens if g['body_size'] > avg_red_body)
                if large_greens >= 3:
                    self.buyer_score += 1.5
                    analysis_log.append("Bullish Signal: Strong Green candle ratio & size dominance (6-candle window).")
                    
        if total_candles >= 6:
            window = candles_data[-6:]
            reds = [c for c in window if c['color'] == 'red']
            greens = [c for c in window if c['color'] == 'green']
            if len(reds) >= 4:
                avg_green_body = (sum(c['body_size'] for c in greens) / len(greens)) if greens else 0
                large_reds = sum(1 for r in reds if r['body_size'] > avg_green_body)
                if large_reds >= 3:
                    self.seller_score += 1.5
                    analysis_log.append("Bearish Signal: Strong Red candle ratio & size dominance (6-candle window).")
                    
        if total_candles >= 5:
            c = candles_data[-5:]
            if c[0]['color'] == 'red' and c[1]['color'] == 'red' and c[2]['color'] == 'red' and c[3]['color'] == 'green' and c[4]['color'] == 'red':
                if c[3]['body_size'] < (c[2]['body_size'] * 0.5):
                    self.seller_score += 1.5
                    analysis_log.append("Bearish Signal: Fake Green pullback in strong Red downtrend.")
                    
        if total_candles >= 2:
            prev = candles_data[-2]
            curr = candles_data[-1]
            if prev['color'] == 'green' and curr['color'] == 'green' and curr['close'] > prev['high']:
                self.buyer_score += 1.0
                analysis_log.append("Buyer Point (+1): Green candle crossed previous upper wick.")
            if prev['color'] == 'green' and curr['color'] == 'red' and curr['close'] < prev['low']:
                self.seller_score += 1.0
                analysis_log.append("Seller Point (+1): Red candle crossed green lower wick.")
            if prev['color'] == 'red' and curr['color'] == 'green' and curr['close'] > prev['high']:
                self.buyer_score += 1.0
                analysis_log.append("Buyer Point (+1): Green candle crossed red upper wick.")
                
        return analysis_log, self.buyer_score, self.seller_score

st.title("Price Action & Candlestick Analysis App")
st.write("Screenshot Upload karke Candlestick Patterns and Points Analyze Karein.")

uploaded_file = st.file_uploader("Upload Chart Screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Chart", use_column_width=True)
    
    mock_candles = [
        {'color': 'red', 'open': 100, 'close': 90, 'high': 105, 'low': 89, 'body_size': 10, 'upper_wick': 5, 'lower_wick': 1},
        {'color': 'red', 'open': 90, 'close': 83, 'high': 92, 'low': 82, 'body_size': 7, 'upper_wick': 2, 'lower_wick': 1},
        {'color': 'red', 'open': 83, 'close': 79, 'high': 86, 'low': 78, 'body_size': 4, 'upper_wick': 3, 'lower_wick': 1},
        {'color': 'green', 'open': 79, 'close': 88, 'high': 90, 'low': 78, 'body_size': 9, 'upper_wick': 2, 'lower_wick': 1}
    ]
    
    analyzer = PriceActionAnalyzer()
    logs, b_score, s_score = analyzer.analyze_candles(mock_candles)
    
    st.subheader("Analysis Summary")
    for log in logs:
        st.write(f"- {log}")
        
    st.markdown(f"### **Buyer Points:** `{b_score}`")
    st.markdown(f"### **Seller Points:** `{s_score}`")
    
    if b_score > s_score:
        st.success("Overall Bias: **BULLISH MOVE EXPECTED**")
    elif s_score > b_score:
        st.error("Overall Bias: **BEARISH MOVE EXPECTED**")
    else:
        st.warning("Overall Bias: **NEUTRAL / CONFUSION**")
