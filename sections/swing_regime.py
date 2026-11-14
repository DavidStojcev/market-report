"""Section: Swing-Trade Market Gate & Exposure Posture (Top of Report)."""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from sections.base import SectionResult

logger = logging.getLogger(__name__)


@dataclass
class RiskRangeItem:
    symbol: str
    name: str
    last_price: float
    range_low: float
    pivot_20ema: float
    range_high: float
    position_pct: float
    atr_pct: float
    trade_dir: str       # 1-3W duration
    trend_dir: str       # 3M duration
    tail_dir: str        # 1Y+ duration
    zone_note: str


@dataclass
class PostureTriggers:
    target_beta_band: str
    upgrade_trigger: str
    downgrade_trigger: str


@dataclass
class SwingGateData:
    as_of_date: str
    posture: str                      # 'FULL', 'REDUCED', 'SELECTIVE', 'DEFENSIVE'
    posture_badge_class: str          # 'badge-green', 'badge-primary', 'badge-yellow', 'badge-red'
    triggered_rule: str
    spy_price: float
    spy_20ema: float
    spy_20ema_dist_pct: float
    spy_20ema_pass: bool
    spy_50dma: float
    spy_50dma_dist_pct: float
    spy_50dma_pass: bool
    qqq_price: float
    qqq_20ema: float
    qqq_20ema_dist_pct: float
    qqq_20ema_pass: bool
    qqq_50dma: float
    qqq_50dma_dist_pct: float
    qqq_50dma_pass: bool
    vix_level: float
    vix_5d_trend: float
    vix_term_structure: str           # e.g., 'Contango (Normal)' or 'Backwardation / Inversion'
    breadth_pct_above_50dma: float
    breadth_ad_state: str
    net_new_highs: int
    execution_guidance: str
    risk_ranges: List[RiskRangeItem] = field(default_factory=list)
    posture_triggers: Optional[PostureTriggers] = None
    catalyst_playbook: Optional[Any] = None
    range_adherence: Optional[Any] = None


def classify_event_impact(title: str, declared_importance: str = "") -> str:
    """Classify calendar event into strict HIGH, MEDIUM, LOW impact tiers per Stage 1: 1.6."""
    t = title.lower()
    # HIGH = FOMC decision, Nonfarm Payrolls, CPI, Chair speech/press conference, Treasury refunding announcement, elections.
    if any(k in t for k in [
        "fomc", "rate decision", "nonfarm payrolls", "employment situation",
        "consumer price index", "cpi", "chair keynote", "chair speech",
        "chair press conference", "treasury quarterly refunding",
        "quarterly refunding announcement", "general election", "midterm election", "presidential election"
    ]):
        return "HIGH"

    # MEDIUM = ISM, ADP, PPI, retail sales, GDP, claims, major auctions, other Fed speakers.
    if any(k in t for k in [
        "ism manufacturing", "ism services", "adp employment", "producer price index",
        "ppi", "retail sales", "gross domestic product", "gdp", "jobless claims",
        "treasury 10-year note auction", "treasury 30-year bond auction",
        "fed governor", "fed vice chair", "fed president"
    ]):
        return "MEDIUM"

    if declared_importance.upper() == "HIGH":
        return "HIGH"
    elif declared_importance.upper() == "MEDIUM":
        return "MEDIUM"
    return "LOW"


def _compute_risk_range_item(
    symbol: str,
    name: str,
    df: Optional[pd.DataFrame],
    as_of_dt: pd.Timestamp,
) -> Optional[RiskRangeItem]:
    """Compute Hedgeye-style volatility-adjusted Risk Range and Trade/Trend/Tail regime from OHLCV bars."""
    if df is None or df.empty or "Close" not in df.columns:
        return None
    sub = df[df.index <= as_of_dt].dropna(subset=["Close"])
    if len(sub) < 20:
        return None

    close_s = sub["Close"].astype(float)
    last_close = float(close_s.iloc[-1])
    if last_close <= 0:
        return None

    ema20_s = close_s.ewm(span=20, adjust=False).mean()
    ema20 = float(ema20_s.iloc[-1])
    ema20_prev5 = float(ema20_s.iloc[-6]) if len(ema20_s) >= 6 else ema20

    dma50_s = close_s.rolling(50).mean() if len(close_s) >= 50 else close_s.expanding().mean()
    dma50 = float(dma50_s.iloc[-1])
    dma50_prev10 = float(dma50_s.iloc[-11]) if len(dma50_s) >= 11 and pd.notna(dma50_s.iloc[-11]) else dma50

    has_200 = len(close_s) >= 200
    dma200 = float(close_s.rolling(200).mean().iloc[-1]) if has_200 else dma50

    # 14D ATR (fallback to daily close return vol if High/Low missing)
    if "High" in sub.columns and "Low" in sub.columns:
        high_s = sub["High"].astype(float)
        low_s = sub["Low"].astype(float)
        prev_c = close_s.shift(1)
        tr = pd.concat([
            (high_s - low_s),
            (high_s - prev_c).abs(),
            (low_s - prev_c).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14, min_periods=5).mean().iloc[-1])
        high_20d = float(high_s.iloc[-20:].max())
        low_20d = float(low_s.iloc[-20:].min())
    else:
        atr14 = float(close_s.diff().abs().rolling(14, min_periods=5).mean().iloc[-1])
        high_20d = float(close_s.iloc[-20:].max())
        low_20d = float(close_s.iloc[-20:].min())

    if pd.isna(atr14) or atr14 <= 0:
        atr14 = last_close * 0.01

    atr_pct = (atr14 / last_close) * 100.0
    # Blend 20D price structure with 1.5x ATR envelope for smooth, actionable boundaries
    range_low = round(0.5 * low_20d + 0.5 * max(0.01, last_close - 1.5 * atr14), 2)
    range_high = round(0.5 * high_20d + 0.5 * (last_close + 1.5 * atr14), 2)
    if range_high <= range_low:
        range_high = round(range_low + max(0.02, atr14), 2)

    pos_pct = ((last_close - range_low) / (range_high - range_low)) * 100.0
    pos_pct = max(0.0, min(100.0, pos_pct))

    # Multi-duration Trade / Trend / Tail classification
    if last_close > ema20 * 1.002 and ema20 >= ema20_prev5:
        trade_dir = "BULLISH"
    elif last_close < ema20 * 0.998 and ema20 <= ema20_prev5:
        trade_dir = "BEARISH"
    elif last_close < ema20 * 0.985:
        trade_dir = "BEARISH"
    else:
        trade_dir = "NEUTRAL"

    if len(close_s) >= 50:
        if last_close > dma50 and dma50 >= dma50_prev10:
            trend_dir = "BULLISH"
        elif last_close < dma50 * 0.995 and dma50 <= dma50_prev10:
            trend_dir = "BEARISH"
        elif last_close < dma50 * 0.98:
            trend_dir = "BEARISH"
        else:
            trend_dir = "NEUTRAL"
    else:
        trend_dir = trade_dir

    if has_200:
        if last_close > dma200 and dma50 >= dma200:
            tail_dir = "BULLISH"
        elif last_close < dma200 and dma50 < dma200:
            tail_dir = "BEARISH"
        else:
            tail_dir = "NEUTRAL"
    else:
        tail_dir = trend_dir

    if pos_pct <= 25.0:
        zone_note = "Near Support (Lower Range Bound)"
    elif pos_pct >= 75.0:
        zone_note = "Near Resistance (Upper Range Bound)"
    elif last_close < ema20:
        zone_note = "Below 20-EMA Pivot (Overhead Supply)"
    else:
        zone_note = "Above 20-EMA Pivot (Mid-Range)"

    return RiskRangeItem(
        symbol=symbol,
        name=name,
        last_price=round(last_close, 2),
        range_low=range_low,
        pivot_20ema=round(ema20, 2),
        range_high=range_high,
        position_pct=round(pos_pct, 1),
        atr_pct=round(atr_pct, 2),
        trade_dir=trade_dir,
        trend_dir=trend_dir,
        tail_dir=tail_dir,
        zone_note=zone_note,
    )


def _build_posture_triggers(
    posture: str,
    spy_price: float,
    spy_20ema: float,
    spy_50dma: float,
    qqq_price: float,
    qqq_20ema: float,
    qqq_50dma: float,
    spy_range: Optional[RiskRangeItem],
) -> PostureTriggers:
    """Build objective Model Regime Tier and Price Flip Conditions."""
    beta_map = {
        "FULL": "Tier 1 — Full Trend Expansion Regime",
        "REDUCED": "Tier 2 — Measured Consolidation Regime",
        "SELECTIVE": "Tier 3 — Selective Divergence Regime",
        "DEFENSIVE": "Tier 4 — Defensive Risk-Off Regime",
    }
    target_beta_band = beta_map.get(posture, "Tier 3 — Selective Divergence Regime")

    def _dist(target: float, px: float) -> str:
        if px <= 0 or target <= 0:
            return "n/a"
        return f"{((target - px) / px) * 100.0:+.2f}%"

    spy_support = spy_range.range_low if spy_range else round(spy_price * 0.985, 2)
    if posture == "DEFENSIVE":
        upgrade_trigger = (
            f"(to SELECTIVE) Daily close above SPY ${spy_50dma:,.2f} ({_dist(spy_50dma, spy_price)}, 50-DMA) "
            f"OR QQQ ${qqq_50dma:,.2f} ({_dist(qqq_50dma, qqq_price)}, 50-DMA); REDUCED requires both > 50-DMA."
        )
        downgrade_trigger = (
            f"Daily close below SPY range support ${spy_support:,.2f} "
            f"({_dist(spy_support, spy_price)}), VIX spike >= 26.0, or S&P 500 > 50-DMA breadth breaking below 40%."
        )
    elif posture == "SELECTIVE":
        upgrade_trigger = (
            f"(to REDUCED) Reclaim both SPY ${spy_50dma:,.2f} ({_dist(spy_50dma, spy_price)}, 50-DMA) "
            f"and QQQ ${qqq_50dma:,.2f} ({_dist(qqq_50dma, qqq_price)}, 50-DMA) with VIX < 22.0."
        )
        downgrade_trigger = (
            f"(to DEFENSIVE) Dual break below SPY ${spy_50dma:,.2f} and QQQ ${qqq_50dma:,.2f}, "
            f"SPY close < ${spy_support:,.2f}, or VIX >= 26.0."
        )
    elif posture == "REDUCED":
        upgrade_trigger = (
            f"(to FULL) Reclaim SPY ${spy_20ema:,.2f} ({_dist(spy_20ema, spy_price)}, 20-EMA) "
            f"and QQQ ${qqq_20ema:,.2f} ({_dist(qqq_20ema, qqq_price)}, 20-EMA) with VIX < 19.5 and Breadth >= 50%."
        )
        downgrade_trigger = (
            f"(to SELECTIVE) Daily close below SPY ${spy_50dma:,.2f} (50-DMA) or QQQ ${qqq_50dma:,.2f} (50-DMA), or VIX >= 22.0."
        )
    else:  # FULL
        upgrade_trigger = "At maximum expansion tier (Tier 1); momentum structure holds while price remains above 20-EMA."
        downgrade_trigger = (
            f"(to REDUCED) Daily close below SPY ${spy_20ema:,.2f} (20-EMA) or QQQ ${qqq_20ema:,.2f} (20-EMA), or VIX >= 19.5."
        )

    return PostureTriggers(
        target_beta_band=target_beta_band,
        upgrade_trigger=upgrade_trigger,
        downgrade_trigger=downgrade_trigger,
    )


def evaluate_swing_gate(
    spy_df: Optional[pd.DataFrame],
    qqq_df: Optional[pd.DataFrame],
    vix_current: float,
    vix_previous: float,
    vix_term_ratio: float,
    breadth_result: Optional[Any],
    as_of_date: str,
    cfg: Optional[Any] = None,
    upcoming_events: Optional[List[Any]] = None,
    event_risk_window_days: Optional[int] = None,
    asset_bars: Optional[Dict[str, pd.DataFrame]] = None,
) -> SectionResult:
    """Evaluate objective swing-trading exposure gate answering 'How aggressively can I be long?'."""
    try:
        as_of_dt = pd.to_datetime(as_of_date)

        # 1. SPY technical gates
        spy_sub = spy_df[spy_df.index <= as_of_dt] if spy_df is not None and not spy_df.empty else pd.DataFrame()
        if not spy_sub.empty and len(spy_sub) >= 50:
            spy_price = float(spy_sub.iloc[-1]["Close"])
            spy_20ema = float(spy_sub["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
            spy_50dma = float(spy_sub["Close"].rolling(window=50).mean().iloc[-1])
            spy_20_dist = ((spy_price - spy_20ema) / spy_20ema) * 100.0
            spy_50_dist = ((spy_price - spy_50dma) / spy_50dma) * 100.0
            spy_20_pass = spy_price >= spy_20ema
            spy_50_pass = spy_price >= spy_50dma
        else:
            spy_price, spy_20ema, spy_50dma = 0.0, 0.0, 0.0
            spy_20_dist, spy_50_dist = 0.0, 0.0
            spy_20_pass, spy_50_pass = False, False

        # 2. QQQ technical gates
        qqq_sub = qqq_df[qqq_df.index <= as_of_dt] if qqq_df is not None and not qqq_df.empty else pd.DataFrame()
        if not qqq_sub.empty and len(qqq_sub) >= 50:
            qqq_price = float(qqq_sub.iloc[-1]["Close"])
            qqq_20ema = float(qqq_sub["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
            qqq_50dma = float(qqq_sub["Close"].rolling(window=50).mean().iloc[-1])
            qqq_20_dist = ((qqq_price - qqq_20ema) / qqq_20ema) * 100.0
            qqq_50_dist = ((qqq_price - qqq_50dma) / qqq_50dma) * 100.0
            qqq_20_pass = qqq_price >= qqq_20ema
            qqq_50_pass = qqq_price >= qqq_50dma
        else:
            qqq_price, qqq_20ema, qqq_50dma = 0.0, 0.0, 0.0
            qqq_20_dist, qqq_50_dist = 0.0, 0.0
            qqq_20_pass, qqq_50_pass = False, False

        # 3. Volatility metrics
        vix_trend = vix_current - vix_previous
        term_str = "Contango (Normal)" if vix_term_ratio <= 1.00 else "Backwardation (Inversion)"

        # 4. Breadth metrics
        p50 = getattr(getattr(breadth_result, "pct_above_50_dma", None), "current", 50.0) if breadth_result else 50.0
        ad_state = getattr(breadth_result, "ad_confirmation", "Neutral") if breadth_result else "Neutral"
        net_highs = getattr(breadth_result, "net_new_highs", 0) if breadth_result else 0

        # 5. Strict Gate Evaluation Rules (Identical to Apex Scanner Market Gate)
        # DEFENSIVE: Dual 50-DMA break, extreme VIX (>26), or severe breadth breakdown (<40%)
        if (not spy_50_pass and not qqq_50_pass) or vix_current >= 26.0 or p50 < 40.0:
            posture = "DEFENSIVE"
            badge_class = "badge-red"
            triggered_rule = f"Regime Tier 4 (Defensive): Dual benchmark break of 50-DMA (SPY {spy_50_dist:+.2f}%, QQQ {qqq_50_dist:+.2f}%) or elevated risk regime."
            guidance = "Quantitative model criteria indicate capital preservation under elevated distribution risk. Benchmark averages trade below key moving average thresholds; broad equity risk appetite remains restrained."

        # SELECTIVE: Divergence with one index below 50-DMA, VIX 22-26, or breadth 40-50%
        elif (not spy_50_pass or not qqq_50_pass) or vix_current >= 22.0 or p50 < 50.0:
            posture = "SELECTIVE"
            badge_class = "badge-yellow"
            triggered_rule = f"Regime Tier 3 (Selective): Benchmark divergence or breakdown below 50-DMA (SPY {spy_50_dist:+.2f}%, QQQ {qqq_50_dist:+.2f}%)."
            guidance = "Quantitative model criteria reflect selective risk appetite. Technical parameters highlight relative-strength leadership showing independent accumulation amid mixed benchmark trends."

        # REDUCED: Both above 50-DMA, but one below 20-EMA or VIX 19.5-22.0
        elif (not spy_20_pass or not qqq_20_pass) or vix_current >= 19.5:
            posture = "REDUCED"
            badge_class = "badge-primary"
            triggered_rule = f"Regime Tier 2 (Measured): Benchmark pullback below 20-EMA with 50-DMA support intact (SPY {spy_20_dist:+.2f}%, QQQ {qqq_20_dist:+.2f}%)."
            guidance = "Quantitative model criteria reflect measured risk appetite. Short-term benchmark consolidation continues above primary 50-day moving average support."

        # FULL: Both above 20-EMA & 50-DMA, VIX < 19.5, Breadth >= 50%
        else:
            posture = "FULL"
            badge_class = "badge-green"
            triggered_rule = f"Regime Tier 1 (Expansion): SPY and QQQ holding above 20-EMA and 50-DMA with low volatility and supportive breadth."
            guidance = "Quantitative model criteria reflect broad market expansion. Favorable technical momentum with SPY and QQQ holding above primary moving averages and low implied volatility."

        # 6. Event-Risk Overlay (Stage 2: 2.6)
        # Within configurable window (default: 3 trading days) before HIGH-impact event,
        # cap posture one step more conservative and log reason.
        if event_risk_window_days is None:
            if cfg and hasattr(cfg, "swing_gate") and hasattr(cfg.swing_gate, "event_risk_window_days"):
                event_risk_window_days = getattr(cfg.swing_gate, "event_risk_window_days", 3)
            elif cfg and hasattr(cfg, "scoring") and hasattr(cfg.scoring, "thresholds"):
                event_risk_window_days = int(cfg.scoring.thresholds.get("event_risk_window_days", 3))
            else:
                event_risk_window_days = 3

        imminent_catalyst = None
        min_catalyst_days = 999
        if upcoming_events:
            for ev in upcoming_events:
                ev_imp = str(getattr(ev, "importance", ""))
                ev_title = str(getattr(ev, "title", ""))
                tier = classify_event_impact(ev_title, ev_imp)
                # Overlay considers HIGH only (Stage 1: 1.6)
                if tier != "HIGH":
                    continue

                ev_date_str = getattr(ev, "date", "")
                if not ev_date_str:
                    continue
                try:
                    ev_dt = pd.to_datetime(ev_date_str).date()
                    curr_dt = as_of_dt.date()
                    if ev_dt > curr_dt:
                        b_days = len(pd.bdate_range(start=curr_dt + pd.Timedelta(days=1), end=ev_dt))
                        if 1 <= b_days <= event_risk_window_days:
                            if b_days < min_catalyst_days:
                                min_catalyst_days = b_days
                                imminent_catalyst = (ev, b_days)
                except Exception:
                    continue

        if imminent_catalyst:
            ev_obj, b_days = imminent_catalyst
            if posture == "DEFENSIVE":
                # Apply T10: if already at the most conservative tier, state "no change (already Defensive)"
                overlay_rule = f"Event-Risk Overlay: no change (already Defensive; upcoming HIGH event '{ev_obj.title}' on {ev_obj.date})"
                triggered_rule = f"{triggered_rule} [{overlay_rule}]."
                guidance = f"Upcoming macro catalyst: '{ev_obj.title}' on {ev_obj.date}. " + guidance
            elif posture == "FULL":
                posture = "REDUCED"
                badge_class = "badge-primary"
                overlay_rule = f"Event-Risk Overlay: Capped 1-tier more conservative (FULL -> REDUCED) ahead of HIGH-impact catalyst '{ev_obj.title}' on {ev_obj.date} ({b_days} session{'s' if b_days > 1 else ''} away)"
                triggered_rule = f"{triggered_rule} [{overlay_rule}]."
                guidance = f"Upcoming macro catalyst: '{ev_obj.title}' on {ev_obj.date}. " + guidance
            elif posture == "REDUCED":
                posture = "SELECTIVE"
                badge_class = "badge-yellow"
                overlay_rule = f"Event-Risk Overlay: Capped 1-tier more conservative (REDUCED -> SELECTIVE) ahead of HIGH-impact catalyst '{ev_obj.title}' on {ev_obj.date} ({b_days} session{'s' if b_days > 1 else ''} away)"
                triggered_rule = f"{triggered_rule} [{overlay_rule}]."
                guidance = f"Upcoming macro catalyst: '{ev_obj.title}' on {ev_obj.date}. " + guidance
            elif posture == "SELECTIVE":
                posture = "DEFENSIVE"
                badge_class = "badge-red"
                overlay_rule = f"Event-Risk Overlay: Capped 1-tier more conservative (SELECTIVE -> DEFENSIVE) ahead of HIGH-impact catalyst '{ev_obj.title}' on {ev_obj.date} ({b_days} session{'s' if b_days > 1 else ''} away)"
                triggered_rule = f"{triggered_rule} [{overlay_rule}]."
                guidance = f"Upcoming macro catalyst: '{ev_obj.title}' on {ev_obj.date}. " + guidance

        # 7. Quantitative Risk Ranges & Posture Flip Triggers (Stage A2)
        bars_pool: Dict[str, Optional[pd.DataFrame]] = {
            "SPY": spy_df,
            "QQQ": qqq_df,
        }
        if asset_bars:
            bars_pool.update(asset_bars)

        target_assets = [
            ("SPY", "S&P 500 ETF"),
            ("QQQ", "Nasdaq-100 ETF"),
            ("IWM", "Russell 2000 ETF"),
            ("TLT", "20+ Year Treasury ETF"),
            ("CL=F", "WTI Crude Oil"),
            ("GC=F", "Gold Futures"),
        ]
        risk_ranges: List[RiskRangeItem] = []
        for sym, label in target_assets:
            item = _compute_risk_range_item(sym, label, bars_pool.get(sym), as_of_dt)
            if item is not None:
                risk_ranges.append(item)

        spy_range = next((r for r in risk_ranges if r.symbol == "SPY"), None)
        posture_triggers = _build_posture_triggers(
            posture=posture,
            spy_price=spy_price,
            spy_20ema=spy_20ema,
            spy_50dma=spy_50dma,
            qqq_price=qqq_price,
            qqq_20ema=qqq_20ema,
            qqq_50dma=qqq_50dma,
            spy_range=spy_range,
        )

        data = SwingGateData(
            as_of_date=as_of_date,
            posture=posture,
            posture_badge_class=badge_class,
            triggered_rule=triggered_rule,
            spy_price=spy_price,
            spy_20ema=spy_20ema,
            spy_20ema_dist_pct=spy_20_dist,
            spy_20ema_pass=spy_20_pass,
            spy_50dma=spy_50dma,
            spy_50dma_dist_pct=spy_50_dist,
            spy_50dma_pass=spy_50_pass,
            qqq_price=qqq_price,
            qqq_20ema=qqq_20ema,
            qqq_20ema_dist_pct=qqq_20_dist,
            qqq_20ema_pass=qqq_20_pass,
            qqq_50dma=qqq_50dma,
            qqq_50dma_dist_pct=qqq_50_dist,
            qqq_50dma_pass=qqq_50_pass,
            vix_level=vix_current,
            vix_5d_trend=vix_trend,
            vix_term_structure=term_str,
            breadth_pct_above_50dma=p50,
            breadth_ad_state=ad_state,
            net_new_highs=net_highs,
            execution_guidance=guidance,
            risk_ranges=risk_ranges,
            posture_triggers=posture_triggers,
        )

        return SectionResult(
            section_num=2,
            title="Market Regime & Risk Appetite",
            status="available",
            data=data,
        )

    except Exception as e:
        logger.error("Failed evaluating swing gate: %s", e)
        return SectionResult(
            section_num=2,
            title="Market Regime & Risk Appetite",
            status="unavailable",
            error_message=f"Market regime evaluation error: {e}",
        )
