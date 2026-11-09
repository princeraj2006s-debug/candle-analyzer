class TWTPrecisionEngine:
    """
    BUY sequence (support):
      1. GREEN candle, lower wick touches support, closes above support  -> Retest 1
      2. Seller pushes down: 2 RED candles, both stay ABOVE support
      3. 2 GREEN candles, last one closes above the high of those 2 reds
      4. RED pullback candle (low still above support)
      5. GREEN confirmation candle -> BUY
      If a candle CLOSES below support -> find nearest demand zone below
      (swing low), run the same sequence on that zone.

    SELL sequence (resistance) = exact mirror:
      1. RED candle, upper wick touches resistance, closes below it -> Retest 1
      2. Buyer pushes up: 2 GREEN candles, both stay BELOW resistance
      3. 2 RED candles, last one closes below the low of those 2 greens
      4. GREEN pullback candle, weak / rejection (small body), below resistance
      5. RED confirmation candle -> SELL
      If a candle CLOSES above resistance -> nearest supply zone (swing high).
    """

    # --- tunable settings ---
    TOUCH_TOL_ATR = 0.3      # how close wick must come to level
    MAX_PIERCE_ATR = 0.5     # wick may pierce level by this much (but close must hold)
    MAX_AGE_PUSH = 12        # candles allowed for step 2
    MAX_AGE_CROSS = 10       # candles allowed for step 3
    MAX_AGE_PULLBACK = 8     # candles allowed for step 4
    WEAK_BODY_ATR = 0.6      # sell pullback body must be <= this * ATR

    def __init__(self, df_5, df_4, df_6, symbol, user_tz):
        self.df_5 = df_5
        self.df_4 = df_4
        self.df_6 = df_6
        self.symbol = symbol
        self.user_tz = user_tz
        self.candles = []
        self.signals = []
        self.resistance_lvl = 0
        self.support_lvl = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _fresh():
        return {"state": 0, "age": 0, "ref": 0.0, "pb_open": 0.0}

    def _long_machine(self, cs, support, require_weak=False):
        """Runs the BUY sequence on candles `cs` (long orientation).
        Returns list of hits: {"idx", "mode", "lvl", "lvl_top"}."""
        hits = []
        st = self._fresh()
        mode = "support"
        lvl = support
        lvl_top = support
        brk = None

        for i in range(2, len(cs)):
            c, p = cs[i], cs[i - 1]
            atr = c["atr"]
            tol = self.TOUCH_TOL_ATR * atr
            pierce = self.MAX_PIERCE_ATR * atr
            is_green = c["color"] == "Green"
            is_red = not is_green

            # ---- support broken -> go find demand zone ----
            if mode == "support" and c["close"] < support:
                mode = "demand"
                lvl = None
                brk = i
                st = self._fresh()
                continue

            if mode == "demand":
                # price reclaimed support -> back to normal support logic
                if c["close"] > support:
                    mode = "support"
                    lvl = lvl_top = support
                    st = self._fresh()
                    continue
                # zone failed (closed below it) -> look for new one
                if lvl is not None and c["close"] < lvl - pierce:
                    lvl = None
                    st = self._fresh()
                # find swing-low (fractal) = demand zone
                if lvl is None and i >= 4:
                    j = i - 2
                    if j >= brk:
                        lj = cs[j]["low"]
                        if (
                            lj < support
                            and lj < cs[j - 1]["low"]
                            and lj < cs[j - 2]["low"]
                            and lj < cs[j + 1]["low"]
                            and lj < cs[j + 2]["low"]
                        ):
                            lvl = lj
                            lvl_top = max(cs[j]["open"], cs[j]["close"])
                            st = self._fresh()
                            continue
                if lvl is None:
                    continue

            # ---- state machine ----
            s = st["state"]

            if s == 0:
                # Retest 1: GREEN candle, lower wick touches, close above level
                if (
                    is_green
                    and c["low"] <= lvl_top + tol
                    and c["low"] >= lvl - pierce
                    and c["close"] > lvl
                ):
                    st = self._fresh()
                    st["state"] = 1
                continue

            st["age"] += 1

            if c["close"] <= lvl:  # level lost -> restart
                st = self._fresh()
                continue

            if s == 1:
                if st["age"] > self.MAX_AGE_PUSH:
                    st = self._fresh()
                    continue
                # 2 red candles pushing down, both ABOVE level
                if is_red and p["color"] == "Red" and c["low"] > lvl and p["low"] > lvl:
                    st["ref"] = max(c["high"], p["high"])
                    st["state"] = 2
                    st["age"] = 0

            elif s == 2:
                if st["age"] > self.MAX_AGE_CROSS:
                    st = self._fresh()
                    continue
                # 2 green candles, last closes above the red pair's high
                if is_green and p["color"] == "Green" and c["close"] > st["ref"]:
                    st["state"] = 3
                    st["age"] = 0

            elif s == 3:
                if st["age"] > self.MAX_AGE_PULLBACK:
                    st = self._fresh()
                    continue
                # red pullback candle, still above level
                if is_red and c["low"] > lvl:
                    body = abs(c["close"] - c["open"])
                    if (not require_weak) or body <= self.WEAK_BODY_ATR * atr:
                        st["pb_open"] = c["open"]
                        st["state"] = 4
                        st["age"] = 0

            elif s == 4:
                if st["age"] > 3:
                    st = self._fresh()
                    continue
                if is_green and c["close"] > st["pb_open"]:
                    hits.append({"idx": i, "mode": mode, "lvl": lvl, "lvl_top": lvl_top})
                    st = self._fresh()
                elif is_red and c["low"] > lvl:
                    st["pb_open"] = c["open"]  # pullback continues

        return hits

    def _mirror(self):
        m = []
        for c in self.candles:
            m.append({
                "id": c["id"],
                "open": -c["open"],
                "high": -c["low"],
                "low": -c["high"],
                "close": -c["close"],
                "atr": c["atr"],
                "color": "Red" if c["color"] == "Green" else "Green",
            })
        return m

    # ------------------------------------------------------------------
    def calculate_and_scan(self):
        if self.df_5 is None or self.df_5.empty:
            return

        IST = timezone(timedelta(hours=5, minutes=30))
        current_utc_time = datetime.now(timezone.utc)

        if self.df_4 is not None and not self.df_4.empty:
            idx_utc = self.df_4.index.tz_localize("UTC") if self.df_4.index.tz is None else self.df_4.index
            df_4_ist = idx_utc.tz_convert(IST)
            filtered_4h = self.df_4[df_4_ist.hour == 1]
            self.resistance_lvl = float(filtered_4h["High"].iloc[-1]) if not filtered_4h.empty else float(self.df_4["High"].iloc[-1])

        if self.df_6 is not None and not self.df_6.empty:
            idx_utc_6 = self.df_6.index.tz_localize("UTC") if self.df_6.index.tz is None else self.df_6.index
            df_6_ist = idx_utc_6.tz_convert(IST)
            filtered_6h = self.df_6[df_6_ist.hour == 5]
            self.support_lvl = float(filtered_6h["Low"].iloc[-1]) if not filtered_6h.empty else float(self.df_6["Low"].iloc[-1])
        elif self.df_4 is not None:
            self.support_lvl = float(self.df_4["Low"].iloc[-1])

        # ATR
        self.df_5["TR"] = np.maximum(
            self.df_5["High"] - self.df_5["Low"],
            np.maximum(
                abs(self.df_5["High"] - self.df_5["Close"].shift(1)),
                abs(self.df_5["Low"] - self.df_5["Close"].shift(1)),
            ),
        )
        self.df_5["ATR"] = self.df_5["TR"].rolling(14).mean()

        for idx, row in enumerate(self.df_5.iterrows()):
            dt_utc = row[0]
            if dt_utc.tzinfo is None:
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            dt_ist = dt_utc.astimezone(IST)
            dt_user_tz = dt_utc.astimezone(self.user_tz)

            r = row[1]
            o, h, l, c, v = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]), float(r["Volume"])
            atr = float(r["ATR"]) if not pd.isna(r["ATR"]) else (c * 0.002)
            age_minutes = (current_utc_time - dt_utc).total_seconds() / 60.0

            self.candles.append({
                "id": idx + 1,
                "time": dt_user_tz.strftime("%H:%M"),
                "dt_ist": dt_ist,
                "dt_user": dt_user_tz,
                "color": "Green" if c >= o else "Red",
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "atr": atr,
                "is_within_1hr": age_minutes <= 60.0,
            })

        if len(self.candles) < 20:
            return

        buy_signals = []
        sell_signals = []

        # ---------------- BUY ----------------
        for hit in self._long_machine(self.candles, self.support_lvl, require_weak=False):
            c = self.candles[hit["idx"]]
            if time(1, 0) <= c["dt_ist"].time() <= time(6, 0):
                continue
            atr = c["atr"]
            dec = get_decimals(c["close"])
            is_zone = hit["mode"] == "demand"
            sl_base = hit["lvl"]
            sl = round(sl_base - ((1.0 if is_zone else 1.2) * atr), dec)
            buy_signals.append({
                "Asset": self.symbol,
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "Demand Zone Retest + Pullback Long" if is_zone else "Support Double Retest + Pullback Long",
                "Action": "Buy / Long (B) 🟢",
                "Entry Price": round(c["close"], dec),
                "Stop Loss (SL)": sl,
                "TP 1": round(c["close"] + 1.2 * atr, dec),
                "TP 2": round(c["close"] + 2.5 * atr, dec),
                "TP 3": round(c["close"] + 4.0 * atr, dec),
                "Is_Within_1hr": c["is_within_1hr"],
            })

        # ---------------- SELL (mirror) ----------------
        mirrored = self._mirror()
        for hit in self._long_machine(mirrored, -self.resistance_lvl, require_weak=True):
            c = self.candles[hit["idx"]]
            if time(1, 0) <= c["dt_ist"].time() <= time(6, 0):
                continue
            atr = c["atr"]
            dec = get_decimals(c["close"])
            is_zone = hit["mode"] == "demand"  # = supply zone in real prices
            real_lvl = -hit["lvl"]
            sl = round(real_lvl + ((1.0 if is_zone else 1.2) * atr), dec)
            sell_signals.append({
                "Asset": self.symbol,
                "Time": c["time"],
                "Candle #": c["id"],
                "Setup": "Supply Zone Retest + Pullback Short" if is_zone else "Resistance Double Retest + Pullback Short",
                "Action": "Sell / Short (S) 🔴",
                "Entry Price": round(c["close"], dec),
                "Stop Loss (SL)": sl,
                "TP 1": round(c["close"] - 1.2 * atr, dec),
                "TP 2": round(c["close"] - 2.5 * atr, dec),
                "TP 3": round(c["close"] - 3.8 * atr, dec),
                "Is_Within_1hr": c["is_within_1hr"],
            })

        if buy_signals:
            self.signals.extend(buy_signals[-2:])
        if sell_signals:
            self.signals.extend(sell_signals[-2:])
