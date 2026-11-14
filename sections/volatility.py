"""Section 4: Volatility and VIX analysis builder."""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from config.settings import SymbolItem
from data.calendar import PeriodInfo
from indicators.breadth import format_percentile_ordinal, trailing_percentile_rank
from sections.base import SectionResult


@dataclass
class VolatilityData:
    symbol: str
    name: str
    current_level: float
    prev_level: float
    change: float
    pct_change: float
    period_high: float
    period_low: float
    regime: str
    regime_description: str
    implied_daily_move_pct: float = 0.0
    # Phase 3B: Volatility Surface, Term Structure, Tail Risk & Realized Vol Cone
    vix_9d: Optional[float] = None
    vix_3m: Optional[float] = None
    vix_6m: Optional[float] = None
    ratio_9d_vix: Optional[float] = None
    ratio_vix_3m: Optional[float] = None
    term_slope_pts: Optional[float] = None
    term_structure_label: str = ""
    vvix_level: Optional[float] = None
    vvix_prev: Optional[float] = None
    vvix_change: Optional[float] = None
    vvix_regime: str = ""
    skew_level: Optional[float] = None
    skew_prev: Optional[float] = None
    skew_change: Optional[float] = None
    skew_regime: str = ""
    realized_vol_10d: Optional[float] = None
    realized_vol_20d: Optional[float] = None
    realized_vol_60d: Optional[float] = None
    vrp_20d: Optional[float] = None
    surface_synthesis: str = ""
    # Stage B1: Trailing 1-Year (252-session) Percentile Ranks
    vix_pctile_1y: Optional[float] = None
    vvix_pctile_1y: Optional[float] = None
    skew_pctile_1y: Optional[float] = None
    vrp_pctile_1y: Optional[float] = None


def _extract_level_and_prev(
    df: Optional[pd.DataFrame],
    curr_end_dt: pd.Timestamp,
    prev_dt: pd.Timestamp,
) -> tuple[Optional[float], Optional[float]]:
    """Return latest Close <= curr_end_dt and prior Close <= prev_dt (or prior bar)."""
    if df is None or df.empty or "Close" not in df.columns:
        return None, None
    sub = df[df.index <= curr_end_dt]
    if sub.empty:
        return None, None
    curr_val = float(sub.iloc[-1]["Close"])
    prev_slice = sub[sub.index <= prev_dt]
    if not prev_slice.empty:
        prev_val = float(prev_slice.iloc[-1]["Close"])
    elif len(sub) >= 2:
        prev_val = float(sub.iloc[-2]["Close"])
    else:
        prev_val = curr_val
    return round(curr_val, 2), round(prev_val, 2)


def _realized_vol(df: Optional[pd.DataFrame], curr_end_dt: pd.Timestamp, window: int) -> Optional[float]:
    """Compute annualized Close-to-Close realized volatility (%) over trailing `window` sessions."""
    if df is None or df.empty or "Close" not in df.columns:
        return None
    sub = df[df.index <= curr_end_dt]
    if len(sub) < window + 1:
        return None
    rets = sub["Close"].pct_change().dropna().tail(window)
    if len(rets) < window:
        return None
    return round(float(rets.std()) * float(np.sqrt(252.0)) * 100.0, 2)


def build_volatility(
    bars: Dict[str, pd.DataFrame],
    vix_symbol: Any,
    period: PeriodInfo,
    spx_df: Optional[pd.DataFrame] = None,
) -> SectionResult:
    """Build volatility section analyzing VIX levels and regime."""
    sym_str = vix_symbol.symbol if hasattr(vix_symbol, "symbol") else str(vix_symbol)
    df = bars.get(sym_str)
    if df is None or df.empty:
        return SectionResult(
            section_num=4,
            title="Volatility (VIX)",
            status="unavailable",
            error_message="VIX price series unavailable.",
        )

    curr_end_dt = pd.to_datetime(period.session_date)
    sub = df[df.index <= curr_end_dt]
    if sub.empty:
        return SectionResult(
            section_num=4,
            title="Volatility (VIX)",
            status="unavailable",
            error_message="No VIX bars found prior to target date.",
        )

    current_level = float(sub.iloc[-1]["Close"])

    # Baseline / previous period lookup
    prev_dt = pd.to_datetime(period.baseline_date)
    prev_slice = sub[sub.index <= prev_dt]
    if not prev_slice.empty:
        prev_level = float(prev_slice.iloc[-1]["Close"])
    elif len(sub) >= 2:
        prev_level = float(sub.iloc[-2]["Close"])
    else:
        prev_level = current_level

    change = current_level - prev_level
    pct_change = ((current_level - prev_level) / prev_level) * 100.0 if prev_level > 0 else 0.0

    # Period range
    start_dt = pd.to_datetime(period.current_start)
    period_slice = sub[(sub.index >= start_dt) & (sub.index <= curr_end_dt)]
    if not period_slice.empty:
        p_high = float(period_slice["High"].max()) if "High" in period_slice.columns else current_level
        p_low = float(period_slice["Low"].min()) if "Low" in period_slice.columns else current_level
    else:
        p_high = current_level
        p_low = current_level

    # Implied daily move via standard Rule of 16 (VIX / sqrt(252))
    implied_daily = current_level / 15.8745

    # Momentum / direction assessment
    if change > 0.5:
        dir_label = "Rising / Expanding Hedging"
    elif change < -0.5:
        dir_label = "Falling / Volatility Crush"
    else:
        dir_label = "Stable"

    # Multi-tier quant-grade regime classification aligned with config and reference table (Fix 1.6)
    if current_level < 16.5:
        base_regime = "Low / Complacent"
        base_desc = f"Subdued baseline implied volatility (±{implied_daily:.2f}% implied daily move). Low hedging demand; favorable for multi-day breakout setups."
    elif current_level < 20.0:
        base_regime = "Normal"
        base_desc = f"Standard historical baseline volatility (±{implied_daily:.2f}% implied daily move). Orderly price discovery and standard hedging distribution."
    elif current_level <= 26.0:
        base_regime = "Elevated Caution"
        base_desc = f"Elevated put protection demand (±{implied_daily:.2f}% implied daily move). Widening intraday swings and higher overnight gap hazard."
    else:
        base_regime = "High Fear / Tail Risk"
        base_desc = f"Acute volatility spike (±{implied_daily:.2f}% implied daily move). Severe market dislocation or panic liquidation."

    regime = f"{base_regime} ({dir_label})"
    desc = f"{base_desc} [Period &Delta;: {change:+.2f} pts]."

    name_str = vix_symbol.name if hasattr(vix_symbol, "name") else ("CBOE Volatility Index" if sym_str == "^VIX" else sym_str)

    # Phase 3B: Full CBOE Term Structure (^VIX9D / ^VIX / ^VIX3M / ^VIX6M), Tail Risk (^VVIX, ^SKEW), & RV Cone
    vix_9d, _ = _extract_level_and_prev(bars.get("^VIX9D"), curr_end_dt, prev_dt)
    vix_3m, _ = _extract_level_and_prev(bars.get("^VIX3M"), curr_end_dt, prev_dt)
    vix_6m, _ = _extract_level_and_prev(bars.get("^VIX6M"), curr_end_dt, prev_dt)

    ratio_9d_vix = round(vix_9d / current_level, 3) if (vix_9d is not None and current_level > 0) else None
    ratio_vix_3m = round(current_level / vix_3m, 3) if (vix_3m is not None and vix_3m > 0) else None
    term_slope_pts = round(vix_3m - current_level, 2) if vix_3m is not None else None

    if ratio_vix_3m is not None:
        if ratio_vix_3m < 0.95:
            term_structure_label = "Contango (Normal Carry)"
        elif ratio_vix_3m <= 1.02:
            term_structure_label = "Flat / Event Caution"
        else:
            term_structure_label = "Backwardation (Near-Term Stress)"
    elif ratio_9d_vix is not None:
        term_structure_label = "Steep Front Curve" if ratio_9d_vix < 0.95 else "Front-End Bid"
    else:
        term_structure_label = ""

    vvix_curr, vvix_prev = _extract_level_and_prev(bars.get("^VVIX"), curr_end_dt, prev_dt)
    vvix_chg = round(vvix_curr - vvix_prev, 2) if (vvix_curr is not None and vvix_prev is not None) else None
    vvix_regime = ""
    if vvix_curr is not None:
        if vvix_curr < 85.0:
            vvix_regime = "Calm Vol-of-Vol"
        elif vvix_curr <= 105.0:
            vvix_regime = "Normal Vol-of-Vol"
        elif vvix_curr <= 120.0:
            vvix_regime = "Elevated Convexity Demand"
        else:
            vvix_regime = "Acute Volatility Tail Bid"

    skew_curr, skew_prev = _extract_level_and_prev(bars.get("^SKEW"), curr_end_dt, prev_dt)
    skew_chg = round(skew_curr - skew_prev, 2) if (skew_curr is not None and skew_prev is not None) else None
    skew_regime = ""
    if skew_curr is not None:
        if skew_curr < 130.0:
            skew_regime = "Low OTM Tail Protection"
        elif skew_curr <= 145.0:
            skew_regime = "Moderate Downside Skew"
        else:
            skew_regime = "Heavy Left-Tail Hedging (>145)"

    bench_df = spx_df if (spx_df is not None and not spx_df.empty) else (bars.get("^GSPC") if "^GSPC" in bars else bars.get("SPY"))
    rv_10d = _realized_vol(bench_df, curr_end_dt, 10)
    rv_20d = _realized_vol(bench_df, curr_end_dt, 20)
    rv_60d = _realized_vol(bench_df, curr_end_dt, 60)
    vrp_20d = round(current_level - rv_20d, 2) if rv_20d is not None else None

    # Stage B1: Trailing 1Y (252-session) percentile ranks
    vix_pctile_1y = trailing_percentile_rank(sub["Close"])
    vvix_bar = bars.get("^VVIX")
    vvix_pctile_1y = (
        trailing_percentile_rank(vvix_bar[vvix_bar.index <= curr_end_dt]["Close"])
        if (vvix_bar is not None and not vvix_bar.empty and "Close" in vvix_bar.columns)
        else None
    )
    skew_bar = bars.get("^SKEW")
    skew_pctile_1y = (
        trailing_percentile_rank(skew_bar[skew_bar.index <= curr_end_dt]["Close"])
        if (skew_bar is not None and not skew_bar.empty and "Close" in skew_bar.columns)
        else None
    )
    vrp_pctile_1y: Optional[float] = None
    if bench_df is not None and not bench_df.empty and "Close" in bench_df.columns:
        vix_hist = pd.to_numeric(sub["Close"], errors="coerce").dropna().sort_index()
        b_close = pd.to_numeric(bench_df[bench_df.index <= curr_end_dt]["Close"], errors="coerce").dropna().sort_index()
        rv20_series = (b_close.pct_change().rolling(20, min_periods=20).std() * float(np.sqrt(252.0)) * 100.0).dropna()
        common_vol_idx = vix_hist.index.intersection(rv20_series.index)
        if len(common_vol_idx) >= 60:
            vrp_series = vix_hist.reindex(common_vol_idx) - rv20_series.reindex(common_vol_idx)
            vrp_pctile_1y = trailing_percentile_rank(vrp_series)

    def _ord_only(val: Optional[float]) -> str:
        return format_percentile_ordinal(val).replace(" %ile", "")

    synth_parts = []
    if vix_pctile_1y is not None:
        synth_parts.append(f"Spot VIX ({current_level:.2f}) ranks at the {_ord_only(vix_pctile_1y)} 1Y percentile.")
    if term_structure_label and ratio_vix_3m is not None:
        r9_note = f" (9D/1M: {ratio_9d_vix:.3f})" if ratio_9d_vix is not None else ""
        synth_parts.append(f"Implied curve sits in {term_structure_label} with VIX/VIX3M at {ratio_vix_3m:.3f}{r9_note}.")
    if vvix_curr is not None or skew_curr is not None:
        tail_bits = []
        if vvix_curr is not None:
            vv_pct_txt = f", {_ord_only(vvix_pctile_1y)} 1Y %ile" if vvix_pctile_1y is not None else ""
            tail_bits.append(f"VVIX at {vvix_curr:.1f} ({vvix_regime}{vv_pct_txt})")
        if skew_curr is not None:
            sk_pct_txt = f", {_ord_only(skew_pctile_1y)} 1Y %ile" if skew_pctile_1y is not None else ""
            tail_bits.append(f"CBOE SKEW at {skew_curr:.1f} ({skew_regime}{sk_pct_txt})")
        synth_parts.append("Tail pricing shows " + " alongside ".join(tail_bits) + ".")
    if rv_20d is not None and vrp_20d is not None:
        rv_span = f"10D: {rv_10d:.1f}% / 20D: {rv_20d:.1f}% / 60D: {rv_60d:.1f}%" if (rv_10d is not None and rv_60d is not None) else f"20D: {rv_20d:.1f}%"
        vrp_pct_txt = f" ({_ord_only(vrp_pctile_1y)} 1Y %ile)" if vrp_pctile_1y is not None else ""
        synth_parts.append(f"Realized volatility cone ({rv_span}) leaves 20D VRP at {vrp_20d:+.2f} pts{vrp_pct_txt}.")

    data = VolatilityData(
        symbol=sym_str,
        name=name_str,
        current_level=current_level,
        prev_level=prev_level,
        change=change,
        pct_change=pct_change,
        period_high=p_high,
        period_low=p_low,
        regime=regime,
        regime_description=desc,
        implied_daily_move_pct=implied_daily,
        vix_9d=vix_9d,
        vix_3m=vix_3m,
        vix_6m=vix_6m,
        ratio_9d_vix=ratio_9d_vix,
        ratio_vix_3m=ratio_vix_3m,
        term_slope_pts=term_slope_pts,
        term_structure_label=term_structure_label,
        vvix_level=vvix_curr,
        vvix_prev=vvix_prev,
        vvix_change=vvix_chg,
        vvix_regime=vvix_regime,
        skew_level=skew_curr,
        skew_prev=skew_prev,
        skew_change=skew_chg,
        skew_regime=skew_regime,
        realized_vol_10d=rv_10d,
        realized_vol_20d=rv_20d,
        realized_vol_60d=rv_60d,
        vrp_20d=vrp_20d,
        surface_synthesis=" ".join(synth_parts),
        vix_pctile_1y=vix_pctile_1y,
        vvix_pctile_1y=vvix_pctile_1y,
        skew_pctile_1y=skew_pctile_1y,
        vrp_pctile_1y=vrp_pctile_1y,
    )

    return SectionResult(
        section_num=4,
        title="Volatility (VIX)",
        status="available",
        data=data,
    )
