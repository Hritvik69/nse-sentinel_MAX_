"""
app_stock_aura_section.py  v2.0
──────────────────────────────────────────────────────────────────────────
🧠 STOCK AURA — Enhanced Single-Stock Decision Engine for NSE Sentinel

WHAT'S NEW vs v1
────────────────
• Score-based verdict (0–100 Aura Score) — no more all-or-nothing binary pass/fail.
  One weak factor no longer kills an otherwise strong setup.
• 8 weighted factor components → transparent, balanced scoring.
• ATR(14)-based stop-loss — far more realistic than fixed EMA20 distance.
• Trend Strength % — how many of the last 10 sessions price closed above EMA20.
• EMA20 slope % — measures how steeply the mean is rising (not just direction).
• Consecutive green/red day detection — early reversal / exhaustion signal.
• 4 clean verdicts: BUY TODAY / BUY TOMORROW / WATCH / DON'T BUY.
• 6 granular timing sub-labels (e.g. "WAIT FOR INTRADAY DIP", "CONFIRM VOLUME").
• Trade Plan card: Entry zone, ATR-based SL, T1 & T2 targets, Risk %.
• Aura Score gauge displayed prominently in the verdict header.
• NOT too strict — designed to tolerate 1–2 weak factors when dominant ones shine.

HOW TO INTEGRATE INTO app.py  (unchanged from v1)
──────────────────────────────────────────────────
1. Near the top of app.py, add:
       from app_stock_aura_section import render_stock_aura_panel

2. Inside `with st.sidebar:`, after the csv_scan_clicked button, add:
       aura_clicked = st.button("🔮 Stock Aura", key="stock_aura_btn")
       if aura_clicked:
           st.session_state["aura_show_panel"] = True

3. In the main body (anywhere after the battle/csv sections), add:
       render_stock_aura_panel()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# ── Internal pipeline helpers ─────────────────────────────────────────
try:
    from strategy_engines._engine_utils import ema as _ema, rsi_vec as _rsi_vec
    _UTILS_OK = True
except ImportError:
    _UTILS_OK = False

try:
    from strategy_engines import get_df_for_ticker as _get_df
    _GETDF_OK = True
except ImportError:
    _GETDF_OK = False

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False


# ══════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════

def _sf(v: object, default: float = 0.0) -> float:
    """safe float — never raises."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _ema_s(s: pd.Series, n: int) -> pd.Series:
    if _UTILS_OK:
        return _ema(s, n)
    return s.ewm(span=n, adjust=False).mean()


def _rsi_last(close: pd.Series, period: int = 14) -> float:
    try:
        if _UTILS_OK:
            return float(_rsi_vec(close).iloc[-1])
        d = close.diff()
        g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        return float((100 - 100 / (1 + g / l.replace(0, np.nan))).iloc[-1])
    except Exception:
        return 50.0


def _atr_last(df: pd.DataFrame, period: int = 14) -> float:
    """ATR(14) — average true range.  Falls back to close-based estimate."""
    try:
        hi = df["High"].dropna()
        lo = df["Low"].dropna()
        cl = df["Close"].dropna()
        pc = cl.shift(1)
        tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
    except Exception:
        try:
            cl = df["Close"].dropna()
            return float((cl.pct_change().abs().rolling(14).mean() * cl).iloc[-1])
        except Exception:
            return 0.0


def _fetch_data(symbol: str) -> pd.DataFrame | None:
    """Fetch OHLCV; respects Time-Travel cutoff when active."""
    ticker_ns = symbol.upper().strip()
    if not ticker_ns.endswith(".NS"):
        ticker_ns += ".NS"

    cutoff_date = None
    try:
        from datetime import date as _date_cls
        raw = st.session_state.get("aura_tt_date")
        if isinstance(raw, _date_cls):
            cutoff_date = raw
    except Exception:
        pass

    if cutoff_date is None:
        try:
            from time_travel_engine import is_active as _tt_active, get_reference_date as _tt_date
            if _tt_active():
                cutoff_date = _tt_date()
        except Exception:
            pass

    def _trunc(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty or cutoff_date is None:
            return df
        try:
            mask = pd.to_datetime(df.index).date <= cutoff_date
            t = df.loc[mask]
            return t if len(t) >= 20 else None
        except Exception:
            return df

    if _GETDF_OK:
        try:
            df = _get_df(ticker_ns)
            if df is not None and len(df) >= 20:
                return _trunc(df)
        except Exception:
            pass

    if _YF_OK:
        try:
            df = yf.download(
                ticker_ns, period="6mo", interval="1d",
                auto_adjust=True, progress=False, timeout=15, threads=False,
            )
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.strip().title() for c in df.columns]
            df = df.dropna(subset=["Close", "Volume"])
            return _trunc(df) if len(df) >= 20 else None
        except Exception:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════
# AURA RESULT CONTAINER
# ══════════════════════════════════════════════════════════════════════

class AuraResult:
    """All computed data + final verdict for one symbol."""

    # ── Verdict constants ─────────────────────────────────────────────
    BUY_TODAY    = "🔥 BUY TODAY"
    BUY_TOMORROW = "📅 BUY TOMORROW"
    WATCH        = "👀 WATCH"
    DONT_BUY     = "❌ DON'T BUY"

    # ── Timing label constants ────────────────────────────────────────
    T_OPEN_ENTRY      = "OPEN ENTRY"
    T_INTRADAY_DIP    = "WAIT FOR INTRADAY DIP"
    T_BUY_TOMORROW    = "BUY TOMORROW OPEN"
    T_CONFIRM_VOL     = "WAIT FOR VOLUME CONFIRMATION"
    T_ENTRY_FORMING   = "ENTRY FORMING — 2 TO 3 DAYS"
    T_WAIT_PULLBACK   = "WAIT FOR PULLBACK"
    T_AVOID           = "AVOID"

    def __init__(self) -> None:
        # ── Market data ───────────────────────────────────────────────
        self.symbol        = ""
        self.price         = 0.0
        self.rsi           = 50.0
        self.ema20         = 0.0
        self.ema50         = 0.0
        self.atr           = 0.0
        self.vol_ratio     = 1.0
        self.delta_ema20   = 0.0
        self.delta_20h     = 0.0
        self.ret_5d        = 0.0
        self.ret_20d       = 0.0
        self.ema_slope_pct = 0.0   # % change in EMA20 over last 3 days
        self.trend_str_pct = 0.0   # % of last 10 sessions above EMA20
        self.consec_green  = 0     # consecutive green candles
        self.consec_red    = 0     # consecutive red candles

        # ── Trade plan ────────────────────────────────────────────────
        self.entry_low   = 0.0
        self.entry_high  = 0.0
        self.sl_price    = 0.0
        self.sl_pct      = 0.0
        self.target1     = 0.0
        self.target1_pct = 0.0
        self.target2     = 0.0
        self.target2_pct = 0.0
        self.rr_ratio    = 0.0

        # ── Component scores (each capped at its max) ─────────────────
        self.score_trend    = 0.0   # /25
        self.score_setup    = 0.0   # /20
        self.score_volume   = 0.0   # /15
        self.score_momentum = 0.0   # /20
        self.score_entry    = 0.0   # /10
        self.score_rr       = 0.0   # /10
        self.score_pattern  = 0.0   # / 5  (candle + trend-str bonus)
        self.score_penalty  = 0.0   # negative deductions

        self.aura_score    = 0.0    # sum of all components (0-100)

        # ── Verdict ───────────────────────────────────────────────────
        self.verdict       = self.DONT_BUY
        self.timing        = self.T_AVOID
        self.verdict_color = "#ff4d6d"
        self.timing_reason = ""

        # ── Setup label ───────────────────────────────────────────────
        self.setup_type    = "None"  # "Breakout" | "Pullback-EMA20" | "Pullback-EMA50" | "None"

        # ── Reason bullets ────────────────────────────────────────────
        self.reasons_positive: list[str] = []
        self.reasons_warning:  list[str] = []
        self.reasons_reject:   list[str] = []

        self.market_note   = ""


# ══════════════════════════════════════════════════════════════════════
# CORE ENGINE
# ══════════════════════════════════════════════════════════════════════

def _run_aura_engine(df: pd.DataFrame, symbol: str, market_bias: dict | None) -> AuraResult:
    """
    Score-based decision engine.

    Eight weighted components → Aura Score (0–100):
        Trend Quality   0–25 pts
        Setup Type      0–20 pts
        Volume          0–15 pts
        Momentum/RSI    0–20 pts
        Entry Quality   0–10 pts
        Risk-Reward     0–10 pts
        Pattern Bonus   0– 5 pts
        Penalties       negative

    Verdict thresholds:
        ≥ 75 → BUY TODAY
        60–74 → BUY TOMORROW
        45–59 → WATCH
        < 45  → DON'T BUY
    """
    r = AuraResult()
    r.symbol = symbol.upper().replace(".NS", "")

    try:
        close  = df["Close"].dropna()
        volume = df["Volume"].dropna()
        high_s = df["High"].dropna()  if "High" in df.columns else close.copy()
        low_s  = df["Low"].dropna()   if "Low"  in df.columns else close.copy()
        open_s = df["Open"].dropna()  if "Open" in df.columns else close.copy()

        if len(close) < 30:
            r.reasons_reject.append("Insufficient price history (< 30 sessions)")
            return r

        # ── Raw market data ──────────────────────────────────────────
        lc       = _sf(close.iloc[-1])
        e20_ser  = _ema_s(close, 20)
        e50_ser  = _ema_s(close, 50)
        e20      = _sf(e20_ser.iloc[-1])
        e50      = _sf(e50_ser.iloc[-1])
        # EMA20 slope over last 3 bars (annualised pct)
        e20_3    = _sf(e20_ser.iloc[-4]) if len(e20_ser) >= 4 else e20
        ema_slope_pct = ((e20 / e20_3) - 1.0) * 100 if e20_3 > 0 else 0.0

        avg_vol  = _sf(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else _sf(volume.mean())
        lv       = _sf(volume.iloc[-1])
        vol_r    = round(lv / avg_vol, 2) if avg_vol > 0 else 1.0

        # Rolling highs/lows
        h20 = _sf(close.iloc[-21:-1].max()) if len(close) >= 21 else _sf(close.max())
        h10 = _sf(close.iloc[-11:-1].max()) if len(close) >= 11 else h20
        l20 = _sf(low_s.iloc[-21:-1].min()) if len(low_s) >= 21 else _sf(low_s.min())

        # Returns
        ret_5d  = (lc / _sf(close.iloc[-6])  - 1.0) * 100 if len(close) >= 6  else 0.0
        ret_20d = (lc / _sf(close.iloc[-21]) - 1.0) * 100 if len(close) >= 21 else 0.0

        # EMA distances
        delta_ema20 = (lc / e20 - 1.0) * 100 if e20 > 0 else 0.0
        delta_20h   = (lc / h20 - 1.0) * 100  if h20 > 0 else 0.0
        delta_e50   = (lc / e50 - 1.0) * 100  if e50 > 0 else 0.0

        rsi_val = _rsi_last(close)
        atr_val = _atr_last(df)

        # Trend strength: % of last 10 sessions price closed > EMA20
        _n = min(10, len(close))
        _tail_close = close.iloc[-_n:]
        _tail_e20   = e20_ser.iloc[-_n:]
        trend_str_pct = float((_tail_close.values > _tail_e20.values).mean() * 100)

        # Consecutive green / red candles
        consec_green = 0
        consec_red   = 0
        if len(open_s) >= 2 and len(close) >= 2:
            for i in range(-1, -min(6, len(close) + 1), -1):
                try:
                    is_green = close.iloc[i] > open_s.iloc[i]
                    if consec_green == 0 and is_green:
                        consec_green += 1
                    elif consec_green > 0 and is_green:
                        consec_green += 1
                    elif consec_red == 0 and not is_green:
                        consec_red += 1
                    elif consec_red > 0 and not is_green:
                        consec_red += 1
                    else:
                        break
                except Exception:
                    break

        # Store for display
        r.price        = round(lc, 2)
        r.rsi          = round(rsi_val, 1)
        r.ema20        = round(e20, 2)
        r.ema50        = round(e50, 2)
        r.atr          = round(atr_val, 2)
        r.vol_ratio    = round(vol_r, 2)
        r.delta_ema20  = round(delta_ema20, 2)
        r.delta_20h    = round(delta_20h, 2)
        r.ret_5d       = round(ret_5d, 2)
        r.ret_20d      = round(ret_20d, 2)
        r.ema_slope_pct = round(ema_slope_pct, 3)
        r.trend_str_pct = round(trend_str_pct, 1)
        r.consec_green  = consec_green
        r.consec_red    = consec_red

        price_above_e20 = lc > e20 > 0
        price_near_e20  = e20 > 0 and -1.5 <= delta_ema20 <= 2.0  # testing EMA20 ± 1.5%
        ema_stack       = e20 > e50 > 0
        ema_slope_up    = ema_slope_pct > 0

        # ══ COMPONENT 1 — TREND QUALITY (0–25 pts) ═══════════════════
        tr = 0.0
        if price_above_e20 and ema_stack:
            tr += 15
            r.reasons_positive.append("Strong uptrend — Price > EMA20 > EMA50")
        elif price_above_e20 and not ema_stack:
            tr += 6
            r.reasons_warning.append("Price above EMA20 but EMA20 < EMA50 — weak structure")
        elif lc > e50 > 0:
            tr += 3
            r.reasons_warning.append("Price between EMA20 and EMA50 — watch for reclaim")
        else:
            r.reasons_reject.append("Downtrend — price below both EMAs")

        if ema_slope_up:
            tr += 5
            if ema_slope_pct > 0.3:
                r.reasons_positive.append(f"EMA20 rising strongly (+{ema_slope_pct:.2f}% in 3 days)")
            else:
                r.reasons_positive.append("EMA20 slope is rising — trend intact")
        else:
            r.reasons_warning.append("EMA20 slope flat or declining — momentum slowing")

        # Trend strength bonus (0–5)
        if trend_str_pct >= 80:
            tr += 5
            r.reasons_positive.append(f"Trend strength {trend_str_pct:.0f}% — price consistently above EMA20")
        elif trend_str_pct >= 60:
            tr += 3
        elif trend_str_pct >= 40:
            tr += 1
        else:
            r.reasons_warning.append(f"Trend strength only {trend_str_pct:.0f}% — choppy above/below EMA20")

        r.score_trend = float(np.clip(tr, 0, 25))

        # ══ COMPONENT 2 — SETUP TYPE (0–20 pts) ══════════════════════
        su = 0.0
        breakout_vol_ok = vol_r >= 1.4
        at_high_zone    = -2.0 <= delta_20h <= 0.5      # within 2% of 20D high
        near_high_zone  = -5.0 <= delta_20h < -2.0      # 2–5% below high
        pullback_e20    = (-5.0 <= delta_ema20 <= 2.5) and (price_above_e20 or price_near_e20)
        pullback_e50    = (lc > e50 > 0) and (delta_e50 <= 3.0) and not price_above_e20
        vol_calming     = 0.7 <= vol_r <= 1.4

        if at_high_zone and breakout_vol_ok:
            su = 20
            r.setup_type = "Breakout"
            r.reasons_positive.append("Breakout setup — at 20D high with volume surge ✦")
        elif at_high_zone and not breakout_vol_ok:
            su = 12
            r.setup_type = "Breakout"
            r.reasons_warning.append("Near 20D high but volume not yet confirming — partial setup")
        elif pullback_e20 and ema_slope_up and vol_calming:
            su = 16
            r.setup_type = "Pullback-EMA20"
            r.reasons_positive.append("Classic EMA20 pullback in uptrend — high probability re-entry")
        elif pullback_e20 and ema_slope_up:
            su = 12
            r.setup_type = "Pullback-EMA20"
            r.reasons_positive.append("Pullback to EMA20 support in uptrend")
        elif near_high_zone and ema_stack:
            su = 12
            r.setup_type = "Pullback-EMA20"
            r.reasons_positive.append(f"Healthy pullback ({delta_20h:.1f}% from high) — watch for re-entry")
        elif pullback_e50 and ema_stack:
            su = 10
            r.setup_type = "Pullback-EMA50"
            r.reasons_warning.append("Pullback to EMA50 — deeper correction; confirm before entry")
        elif delta_20h < -5.0 and delta_20h >= -10.0 and ema_stack:
            su = 6
            r.setup_type = "None"
            r.reasons_warning.append(f"Setup not formed — {delta_20h:.1f}% from 20D high; wait for base")
        else:
            su = 0
            r.setup_type = "None"
            r.reasons_reject.append(f"No valid setup — {delta_20h:.1f}% from 20D high, no base forming")

        r.score_setup = float(np.clip(su, 0, 20))

        # ══ COMPONENT 3 — VOLUME CONVICTION (0–15 pts) ═══════════════
        if   vol_r > 2.5:   vp = 15; vl = f"Explosive ({vol_r:.1f}×) — strong institutional signal"
        elif vol_r > 2.0:   vp = 13; vl = f"Strong ({vol_r:.1f}×) — confirmed participation"
        elif vol_r > 1.5:   vp = 10; vl = f"Good ({vol_r:.1f}×) — valid confirmation"
        elif vol_r > 1.2:   vp =  7; vl = f"Building ({vol_r:.1f}×) — acceptable"
        elif vol_r > 1.0:   vp =  3; vl = f"Neutral ({vol_r:.1f}×) — no conviction signal"
        else:               vp =  0; vl = f"Weak ({vol_r:.1f}×) — below average participation"

        r.score_volume = float(np.clip(vp, 0, 15))
        if vp >= 10:
            r.reasons_positive.append(f"Volume: {vl}")
        elif vp >= 3:
            r.reasons_warning.append(f"Volume: {vl}")
        else:
            r.reasons_reject.append(f"Volume: {vl}")

        # ══ COMPONENT 4 — MOMENTUM / RSI (0–20 pts) ══════════════════
        # Base RSI score
        if   52.0 <= rsi_val <= 65.0:  rsi_pts = 18; rsi_lbl = f"RSI {rsi_val:.0f} — ideal zone"
        elif 48.0 <= rsi_val <  52.0:  rsi_pts = 13; rsi_lbl = f"RSI {rsi_val:.0f} — early accumulation"
        elif 65.0 <  rsi_val <= 70.0:  rsi_pts = 13; rsi_lbl = f"RSI {rsi_val:.0f} — upper acceptable zone"
        elif 45.0 <= rsi_val <  48.0:  rsi_pts =  6; rsi_lbl = f"RSI {rsi_val:.0f} — weak momentum"
        elif 70.0 <  rsi_val <= 75.0:  rsi_pts =  6; rsi_lbl = f"RSI {rsi_val:.0f} — elevated, caution"
        else:                          rsi_pts =  0; rsi_lbl = f"RSI {rsi_val:.0f} — extreme zone"

        # 5D return bonus / penalty
        ret_bonus = 0.0
        if   1.0 <= ret_5d <= 6.0:  ret_bonus =  2.0
        elif 6.0 <  ret_5d <= 9.0:  ret_bonus =  0.5
        elif ret_5d > 9.0:          ret_bonus = -2.0

        # 20D return bonus
        if ret_20d > 5.0:  ret_bonus += 1.0

        mp = float(np.clip(rsi_pts + ret_bonus, 0, 20))
        r.score_momentum = mp

        if rsi_pts >= 13:
            r.reasons_positive.append(f"{rsi_lbl} | 5D Return {ret_5d:+.1f}%")
        elif rsi_pts >= 6:
            r.reasons_warning.append(f"{rsi_lbl} | 5D Return {ret_5d:+.1f}%")
        else:
            r.reasons_reject.append(f"{rsi_lbl} | 5D Return {ret_5d:+.1f}%")

        # ══ COMPONENT 5 — ENTRY QUALITY (0–10 pts) ═══════════════════
        if   delta_ema20 <= 2.0:  ep = 10; el = f"Tight ({delta_ema20:+.1f}% from EMA20) — ideal entry"
        elif delta_ema20 <= 4.0:  ep =  8; el = f"Good ({delta_ema20:+.1f}% from EMA20) — acceptable entry"
        elif delta_ema20 <= 6.0:  ep =  5; el = f"Extended ({delta_ema20:+.1f}% from EMA20) — manageable"
        elif delta_ema20 <= 8.0:  ep =  2; el = f"Stretched ({delta_ema20:+.1f}% from EMA20) — risk higher"
        else:                     ep =  0; el = f"Overextended ({delta_ema20:+.1f}% from EMA20) — late entry"

        r.score_entry = float(np.clip(ep, 0, 10))
        if ep >= 8:
            r.reasons_positive.append(f"Entry quality: {el}")
        elif ep >= 2:
            r.reasons_warning.append(f"Entry quality: {el}")
        else:
            r.reasons_reject.append(f"Entry quality: {el}")

        # ══ COMPONENT 6 — RISK-REWARD (0–10 pts) ═════════════════════
        # Stop: ATR-based (1.5× ATR below entry, floor at EMA20)
        atr_sl   = lc - max(1.5 * atr_val, lc - e20) if atr_val > 0 else e20
        sl_price = float(np.clip(atr_sl, lc * 0.80, lc * 0.98))  # sanity-capped 2–20% below
        downside = lc - sl_price
        if downside < 0.001:
            downside = lc * 0.05

        # Target: at 20D high for pullbacks, or +6% extension for breakouts
        if r.setup_type == "Breakout":
            tgt_main = lc * 1.065          # breakout → extend 6.5%
        elif h20 > lc:
            tgt_main = h20                  # pullback → prior high
        else:
            tgt_main = lc * 1.06

        upside = max(tgt_main - lc, 0.0)
        rr     = upside / downside if downside > 0 else 0.0

        if   rr >= 3.0:  rrp = 10; rrl = f"RR {rr:.1f}:1 — excellent"
        elif rr >= 2.5:  rrp =  9; rrl = f"RR {rr:.1f}:1 — very good"
        elif rr >= 2.0:  rrp =  7; rrl = f"RR {rr:.1f}:1 — good"
        elif rr >= 1.5:  rrp =  5; rrl = f"RR {rr:.1f}:1 — acceptable"
        elif rr >= 1.0:  rrp =  2; rrl = f"RR {rr:.1f}:1 — marginal"
        else:            rrp =  0; rrl = f"RR {rr:.1f}:1 — unfavorable"

        r.score_rr  = float(np.clip(rrp, 0, 10))
        r.rr_ratio  = round(rr, 2)
        if rrp >= 7:
            r.reasons_positive.append(rrl)
        elif rrp >= 2:
            r.reasons_warning.append(rrl)
        else:
            r.reasons_reject.append(rrl)

        # Trade plan numbers
        r.sl_price    = round(sl_price, 2)
        r.sl_pct      = round(((lc - sl_price) / lc) * 100, 2)
        r.entry_low   = round(lc * 0.995, 2)   # minor tolerance for live entry
        r.entry_high  = round(lc * 1.005, 2)
        r.target1     = round(lc + downside * 1.5, 2)   # 1.5:1
        r.target1_pct = round(((r.target1 - lc) / lc) * 100, 2)
        r.target2     = round(tgt_main, 2)
        r.target2_pct = round(((tgt_main - lc) / lc) * 100, 2)

        # ══ COMPONENT 7 — PATTERN BONUS (0–5 pts) ════════════════════
        pat = 0.0
        if consec_green >= 3:
            pat += 2.5
            r.reasons_positive.append(f"{consec_green} consecutive green candles — bullish momentum")
        elif consec_green == 2:
            pat += 1.5
        elif consec_red >= 3:
            pat -= 1.0
            r.reasons_warning.append(f"{consec_red} consecutive red candles — short-term weakness")

        # Trend strength bonus
        if trend_str_pct >= 80:
            pat += 2.5
        elif trend_str_pct >= 60:
            pat += 1.5

        r.score_pattern = float(np.clip(pat, 0, 5))

        # ══ COMPONENT 8 — MARKET CONTEXT & PENALTIES ═════════════════
        penalty = 0.0

        # RSI extreme
        if rsi_val > 77:
            penalty -= 8
            r.reasons_reject.append(f"RSI overbought ({rsi_val:.0f}) — significant reversal risk")
        elif rsi_val < 40:
            penalty -= 4
            r.reasons_warning.append(f"RSI weak ({rsi_val:.0f}) — trend may not be ready")

        # EMA breakdown
        if not ema_stack:
            penalty -= 10
            # (reason already added above)

        # Exhaustion
        if ret_5d > 12.0:
            penalty -= 6
            r.reasons_reject.append(f"5D return {ret_5d:.1f}% — short-term exhaustion risk")

        # Low volume
        if vol_r < 0.8:
            penalty -= 5
            # (reason already added above)

        # Very overextended
        if delta_ema20 > 9.0:
            penalty -= 5

        # Consecutive red streak
        if consec_red >= 3:
            penalty -= 4

        # Market bias
        mb = market_bias if isinstance(market_bias, dict) else {}
        bias_raw = str(mb.get("bias", "")).strip()
        bias_low = bias_raw.lower()
        bias_conf = _sf(mb.get("confidence", 50), 50)

        if any(w in bias_low for w in ("bearish", "weak", "caution")):
            adj = -5 if bias_conf >= 70 else -2
            penalty += adj
            r.market_note = f"⚠️ Market: {bias_raw} — reduces conviction"
            r.reasons_warning.append(f"Market backdrop: {bias_raw}")
        elif any(w in bias_low for w in ("bullish", "trending up", "strong")):
            r.market_note = f"✅ Market: {bias_raw}"
            r.reasons_positive.append(f"Favorable market ({bias_raw})")
        else:
            r.market_note = f"Market: {bias_raw or 'Unknown — run Market Bias first'}"

        r.score_penalty = float(penalty)

        # ══ FINAL AURA SCORE ══════════════════════════════════════════
        raw = (
            r.score_trend    +
            r.score_setup    +
            r.score_volume   +
            r.score_momentum +
            r.score_entry    +
            r.score_rr       +
            r.score_pattern  +
            r.score_penalty
        )
        r.aura_score = float(np.clip(raw, 0, 100))

        # ══ VERDICT + TIMING ══════════════════════════════════════════
        s  = r.aura_score
        overextended = delta_ema20 > 4.5
        vol_confirmed = vol_r >= 1.25

        if s >= 75:
            r.verdict       = AuraResult.BUY_TODAY
            r.verdict_color = "#00d4a8"
            if not overextended:
                r.timing        = AuraResult.T_OPEN_ENTRY
                r.timing_reason = "All key factors aligned — enter at market or on open"
            else:
                r.timing        = AuraResult.T_INTRADAY_DIP
                r.timing_reason = (
                    f"Strong setup but price is {delta_ema20:.1f}% above EMA20 — "
                    "wait for an intraday pullback toward EMA20 before entering"
                )

        elif s >= 57:
            r.verdict       = AuraResult.BUY_TOMORROW
            r.verdict_color = "#0094ff"
            if vol_confirmed:
                r.timing        = AuraResult.T_BUY_TOMORROW
                r.timing_reason = "Good setup — let today's candle close as confirmation, buy on tomorrow's open"
            else:
                r.timing        = AuraResult.T_CONFIRM_VOL
                r.timing_reason = (
                    "Setup forming but volume not yet confirming — "
                    "wait for a session with vol > 1.3× avg before entering"
                )

        elif s >= 40:
            r.verdict       = AuraResult.WATCH
            r.verdict_color = "#f0b429"
            if overextended:
                r.timing        = AuraResult.T_WAIT_PULLBACK
                r.timing_reason = (
                    f"Trend is fine but price is {delta_ema20:.1f}% extended — "
                    "wait for a natural pullback toward EMA20 before evaluating entry"
                )
            else:
                r.timing        = AuraResult.T_ENTRY_FORMING
                r.timing_reason = "Setup is developing — check again in 2 to 3 trading sessions"

        else:
            r.verdict       = AuraResult.DONT_BUY
            r.verdict_color = "#ff4d6d"
            r.timing        = AuraResult.T_AVOID
            # Surface the top rejection reason
            top_reason = r.reasons_reject[0] if r.reasons_reject else "Multiple factors failing"
            r.timing_reason = f"Primary issue: {top_reason}"

        return r

    except Exception as exc:
        r.reasons_reject.append(f"Engine error: {exc}")
        return r


# ══════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════

def _score_bar(label: str, score: float, max_score: float, color: str) -> str:
    """Mini progress bar for the factor scorecard."""
    pct   = min(100.0, (score / max_score) * 100) if max_score > 0 else 0
    s_fmt = f"{score:.0f}/{max_score:.0f}"
    bar_w = f"{pct:.0f}%"
    return (
        f'<div style="margin:4px 0;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:2px;">'
        f'<span style="font-size:10px;color:#8ab4d8;">{label}</span>'
        f'<span style="font-size:10px;font-weight:700;color:{color};">{s_fmt}</span></div>'
        f'<div style="background:#0b1017;border-radius:3px;height:5px;">'
        f'<div style="background:{color};width:{bar_w};height:5px;border-radius:3px;'
        f'transition:width 0.3s;"></div></div></div>'
    )


def _timing_pill(timing: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;border:1.5px solid {color};border-radius:20px;'
        f'padding:4px 14px;font-size:11px;font-weight:800;color:{color};'
        f'letter-spacing:0.5px;">{timing}</span>'
    )


def _render_aura_card(r: AuraResult) -> None:
    """Render the full Aura verdict card."""

    score_color = r.verdict_color

    # ── Main verdict header with Aura Score gauge ─────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0b1017 60%,{score_color}12);'
        f'border:2px solid {score_color};border-radius:16px;padding:22px 26px;margin:12px 0 18px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">'
        # Left: symbol + stats
        f'<div>'
        f'<div style="font-size:11px;color:#4a6480;letter-spacing:1.5px;text-transform:uppercase;'
        f'margin-bottom:3px;">🔮 STOCK AURA</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:28px;font-weight:900;'
        f'color:#ccd9e8;margin-bottom:4px;">{r.symbol}</div>'
        f'<div style="font-size:11px;color:#4a6480;margin-bottom:14px;">'
        f'₹{r.price:.2f} &nbsp;·&nbsp; RSI {r.rsi:.0f} &nbsp;·&nbsp; '
        f'Vol {r.vol_ratio:.1f}× &nbsp;·&nbsp; EMA20 {r.delta_ema20:+.1f}% &nbsp;·&nbsp; '
        f'5D {r.ret_5d:+.1f}% &nbsp;·&nbsp; ATR ₹{r.atr:.1f}</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:20px;font-weight:900;'
        f'color:{score_color};margin-bottom:10px;">{r.verdict}</div>'
        f'<div>{_timing_pill(r.timing, score_color)}</div>'
        f'</div>'
        # Right: Aura Score gauge
        f'<div style="text-align:center;min-width:90px;">'
        f'<div style="font-size:10px;color:#4a6480;letter-spacing:1px;margin-bottom:2px;">AURA SCORE</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:46px;font-weight:900;'
        f'color:{score_color};line-height:1;">{r.aura_score:.0f}</div>'
        f'<div style="font-size:10px;color:#4a6480;">/ 100</div>'
        f'</div>'
        f'</div>'
        # Timing reason (explanatory line)
        + (
            f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid {score_color}33;'
            f'font-size:11px;color:#8ab4d8;font-style:italic;">'
            f'💡 {r.timing_reason}</div>'
            if r.timing_reason else ""
        )
        + f'</div>',
        unsafe_allow_html=True,
    )

    # ── Three-column layout ───────────────────────────────────────────
    col_why, col_scores, col_plan = st.columns([3, 2, 2])

    # ── WHY / WARNINGS ────────────────────────────────────────────────
    with col_why:
        if r.reasons_positive:
            pos_html = "".join(
                f'<div style="padding:4px 0;font-size:11px;color:#ccd9e8;">'
                f'<span style="color:#00d4a8;font-weight:800;">✔</span>&nbsp; {x}</div>'
                for x in r.reasons_positive
            )
            st.markdown(
                f'<div style="background:#0a1a14;border:1px solid #1a4030;border-radius:10px;'
                f'padding:12px 14px;margin-bottom:10px;">'
                f'<div style="font-size:10px;font-weight:800;color:#00d4a8;'
                f'letter-spacing:0.5px;margin-bottom:6px;">✔ FACTORS IN FAVOUR</div>'
                f'{pos_html}</div>',
                unsafe_allow_html=True,
            )

        warn_items = [(w, "#f0b429") for w in r.reasons_warning] + \
                     [(e, "#ff4d6d") for e in r.reasons_reject]
        if warn_items:
            warn_html = "".join(
                f'<div style="padding:4px 0;font-size:11px;color:#ccd9e8;">'
                f'<span style="color:{c};font-weight:800;">✖</span>&nbsp; {t}</div>'
                for t, c in warn_items
            )
            st.markdown(
                f'<div style="background:#150a0a;border:1px solid #3a1a1a;border-radius:10px;'
                f'padding:12px 14px;margin-bottom:10px;">'
                f'<div style="font-size:10px;font-weight:800;color:#f0b429;'
                f'letter-spacing:0.5px;margin-bottom:6px;">⚠ WARNINGS & CAUTIONS</div>'
                f'{warn_html}</div>',
                unsafe_allow_html=True,
            )

        # Market note
        if r.market_note:
            note_c = "#f0b429" if "⚠" in r.market_note else "#4a6480"
            st.markdown(
                f'<div style="font-size:10px;color:{note_c};padding:4px 0;">'
                f'🌐 {r.market_note}</div>',
                unsafe_allow_html=True,
            )

    # ── FACTOR SCORE BREAKDOWN ────────────────────────────────────────
    with col_scores:
        # Color per factor based on fill %
        def _fc(s, mx):
            p = s / mx if mx else 0
            if p >= 0.70: return "#00d4a8"
            if p >= 0.45: return "#f0b429"
            return "#ff4d6d"

        bars = (
            _score_bar("Trend Quality",    r.score_trend,    25, _fc(r.score_trend,    25)) +
            _score_bar("Setup Type",       r.score_setup,    20, _fc(r.score_setup,    20)) +
            _score_bar("Volume",           r.score_volume,   15, _fc(r.score_volume,   15)) +
            _score_bar("Momentum/RSI",     r.score_momentum, 20, _fc(r.score_momentum, 20)) +
            _score_bar("Entry Quality",    r.score_entry,    10, _fc(r.score_entry,    10)) +
            _score_bar("Risk-Reward",      r.score_rr,       10, _fc(r.score_rr,       10)) +
            _score_bar("Pattern Bonus",    r.score_pattern,   5, _fc(r.score_pattern,   5))
        )

        # Penalty row
        pen_c = "#ff4d6d" if r.score_penalty < -4 else "#f0b429" if r.score_penalty < 0 else "#4a6480"
        pen_row = (
            f'<div style="margin:4px 0;display:flex;justify-content:space-between;">'
            f'<span style="font-size:10px;color:#8ab4d8;">Penalties</span>'
            f'<span style="font-size:10px;font-weight:700;color:{pen_c};">'
            f'{r.score_penalty:.0f}</span></div>'
        )

        # Extra stats row
        extras = (
            f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid #1a2840;">'
            f'<div style="display:flex;justify-content:space-between;font-size:10px;margin:2px 0;">'
            f'<span style="color:#4a6480;">Trend Strength</span>'
            f'<span style="color:#8ab4d8;font-weight:700;">{r.trend_str_pct:.0f}% above EMA20</span></div>'
            f'<div style="display:flex;justify-content:space-between;font-size:10px;margin:2px 0;">'
            f'<span style="color:#4a6480;">EMA20 Slope</span>'
            f'<span style="color:{"#00d4a8" if r.ema_slope_pct > 0 else "#ff4d6d"};font-weight:700;">'
            f'{"+" if r.ema_slope_pct > 0 else ""}{r.ema_slope_pct:.2f}% / 3d</span></div>'
            f'<div style="display:flex;justify-content:space-between;font-size:10px;margin:2px 0;">'
            f'<span style="color:#4a6480;">Setup Type</span>'
            f'<span style="color:#8ab4d8;font-weight:700;">{r.setup_type}</span></div>'
            + (
                f'<div style="display:flex;justify-content:space-between;font-size:10px;margin:2px 0;">'
                f'<span style="color:#4a6480;">Candles</span>'
                f'<span style="color:#00d4a8;font-weight:700;">{r.consec_green} green</span></div>'
                if r.consec_green >= 2 else
                f'<div style="display:flex;justify-content:space-between;font-size:10px;margin:2px 0;">'
                f'<span style="color:#4a6480;">Candles</span>'
                f'<span style="color:#ff4d6d;font-weight:700;">{r.consec_red} red</span></div>'
                if r.consec_red >= 2 else ""
            )
            + f'</div>'
        )

        st.markdown(
            f'<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:10px;'
            f'padding:12px 14px;margin-bottom:10px;">'
            f'<div style="font-size:10px;font-weight:800;color:#8ab4d8;'
            f'letter-spacing:0.5px;margin-bottom:8px;">📊 SCORE BREAKDOWN</div>'
            f'{bars}{pen_row}{extras}</div>',
            unsafe_allow_html=True,
        )

    # ── TRADE PLAN ────────────────────────────────────────────────────
    with col_plan:
        rr_c = "#00d4a8" if r.rr_ratio >= 2.0 else "#f0b429" if r.rr_ratio >= 1.5 else "#ff4d6d"
        sl_c = "#ff4d6d"
        t1_c = "#f0b429"
        t2_c = "#00d4a8"

        plan_rows = (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:6px 0;border-bottom:1px solid #1a2840;">'
            f'<span style="font-size:10px;color:#4a6480;">Entry Zone</span>'
            f'<span style="font-size:11px;font-weight:700;color:#ccd9e8;">'
            f'₹{r.entry_low:.1f} – ₹{r.entry_high:.1f}</span></div>'
            +
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:6px 0;border-bottom:1px solid #1a2840;">'
            f'<span style="font-size:10px;color:#4a6480;">Stop Loss</span>'
            f'<span style="font-size:11px;font-weight:700;color:{sl_c};">'
            f'₹{r.sl_price:.1f} &nbsp;<span style="font-size:10px;">(-{r.sl_pct:.1f}%)</span>'
            f'</span></div>'
            +
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:6px 0;border-bottom:1px solid #1a2840;">'
            f'<span style="font-size:10px;color:#4a6480;">Target 1</span>'
            f'<span style="font-size:11px;font-weight:700;color:{t1_c};">'
            f'₹{r.target1:.1f} &nbsp;<span style="font-size:10px;">(+{r.target1_pct:.1f}%)</span>'
            f'</span></div>'
            +
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:6px 0;border-bottom:1px solid #1a2840;">'
            f'<span style="font-size:10px;color:#4a6480;">Target 2</span>'
            f'<span style="font-size:11px;font-weight:700;color:{t2_c};">'
            f'₹{r.target2:.1f} &nbsp;<span style="font-size:10px;">(+{r.target2_pct:.1f}%)</span>'
            f'</span></div>'
            +
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:6px 0;">'
            f'<span style="font-size:10px;color:#4a6480;">Risk : Reward</span>'
            f'<span style="font-size:12px;font-weight:900;color:{rr_c};">'
            f'{r.rr_ratio:.1f} : 1</span></div>'
        )

        # ATR note
        atr_note = (
            f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #1a2840;'
            f'font-size:10px;color:#4a6480;">'
            f'SL uses ATR(14) = ₹{r.atr:.1f} · 1.5× ATR method</div>'
        ) if r.atr > 0 else ""

        st.markdown(
            f'<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:10px;'
            f'padding:12px 14px;margin-bottom:10px;">'
            f'<div style="font-size:10px;font-weight:800;color:#8ab4d8;'
            f'letter-spacing:0.5px;margin-bottom:6px;">📋 TRADE PLAN</div>'
            f'{plan_rows}{atr_note}</div>',
            unsafe_allow_html=True,
        )

        # How-to-read tip
        st.markdown(
            f'<div style="font-size:10px;color:#2a4060;padding:4px 0;line-height:1.5;">'
            f'Entry Zone = current price ± 0.5%<br>'
            f'T1 = 1.5:1 RR &nbsp;·&nbsp; T2 = prior high / projection</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT — called from app.py
# ══════════════════════════════════════════════════════════════════════

def render_stock_aura_panel() -> None:
    """
    Render the full Stock Aura panel.
    Call unconditionally — guards itself via session_state.
    """
    if not st.session_state.get("aura_show_panel", False):
        return

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:\'Syne\',sans-serif;font-weight:900;font-size:22px;'
        'color:#ccd9e8;margin-bottom:4px;">🔮 Stock Aura</h2>'
        '<div style="font-size:12px;color:#4a6480;margin-bottom:18px;">'
        'Score-based single stock decision engine · ATR stop loss · '
        'Precise buy/watch/avoid timing &nbsp;·&nbsp; v2.0</div>',
        unsafe_allow_html=True,
    )

    # ── Time-travel banner ────────────────────────────────────────────
    _tt = st.session_state.get("aura_tt_date")
    _tt_str = ""
    if _tt is not None:
        try:
            _tt_str = _tt.strftime("%d %b %Y")
        except Exception:
            _tt_str = str(_tt)
        st.markdown(
            f'<div style="background:#1a0a00;border:1.5px solid #f0b429;border-radius:8px;'
            f'padding:8px 14px;margin-bottom:14px;font-size:12px;color:#f0b429;">'
            f'🕰️ <b>TIME TRAVEL ACTIVE</b> — Simulating {_tt_str} post-market close · '
            f'No future data used</div>',
            unsafe_allow_html=True,
        )

    # ── Input row ─────────────────────────────────────────────────────
    col_inp, col_btn, col_close = st.columns([3, 1, 1])
    with col_inp:
        ticker_raw = st.text_input(
            "Enter Stock Symbol",
            placeholder="e.g.  RELIANCE  or  TCS  or  HDFCBANK",
            key="aura_ticker_input",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_clicked = st.button("🧠 Analyze Aura", key="aura_analyze_btn")
    with col_close:
        if st.button("✕ Close", key="aura_close_btn"):
            st.session_state["aura_show_panel"] = False
            st.rerun()

    if analyze_clicked and ticker_raw.strip():
        sym = ticker_raw.strip().upper().replace(".NS", "")
        spinner_msg = (
            f"🕰️ Reading historical aura for {sym} ({_tt_str})…"
            if _tt is not None else f"🔮 Analyzing aura for {sym}…"
        )
        with st.spinner(spinner_msg):
            df = _fetch_data(sym)

        if df is None or df.empty:
            st.error(
                f"❌ No data found for **{sym}**.  "
                "Check the symbol (e.g. RELIANCE not RELIANCE.NS) and try again."
            )
            return

        # Market bias
        mb = st.session_state.get("market_bias_result", None)
        if _tt is not None:
            mb = dict(mb) if isinstance(mb, dict) else {}
            if not mb.get("bias"):
                mb["bias"] = f"Historical ({_tt_str}) — run Market Bias for context"

        result = _run_aura_engine(df, sym, mb)
        _render_aura_card(result)

        st.markdown(
            '<div style="font-size:10px;color:#1e3050;margin-top:12px;text-align:center;">'
            '⚠️ Stock Aura is for educational purposes only. Not financial advice. '
            'Always do your own research before trading.</div>',
            unsafe_allow_html=True,
        )

    elif analyze_clicked and not ticker_raw.strip():
        st.warning("Enter a stock symbol first (e.g. RELIANCE, TCS, INFY)")
