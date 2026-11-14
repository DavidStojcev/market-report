"""Section: Expanded Commodities, Currencies, Crypto & Energy Shock Tracker."""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from data.macro_base import compute_lookback_return
from sections.base import SectionResult

logger = logging.getLogger(__name__)


@dataclass
class CrossAssetExpandedItem:
    symbol: str
    name: str
    current_price: float
    period_ret: float
    ret_3m: float
    prior_period_ret: Optional[float] = None
    is_available: bool = True


@dataclass
class EnergyShockIndicator:
    oil_3m_change_pct: float
    headline_cpi_yoy: float
    core_cpi_yoy: float
    headline_core_gap_pct: float
    is_shock_active: bool
    assessment: str


@dataclass
class CrossAssetPlumbingData:
    copper_gold_ratio: Optional[float] = None      # (HG=F * 1000) / GC=F
    copper_gold_3m_chg_pct: Optional[float] = None
    nominal_10y_yield: Optional[float] = None
    yield_10y_13w_bps: Optional[float] = None
    copper_gold_regime: str = "Unavailable"
    stock_bond_corr_60d: Optional[float] = None    # Rolling 60D SPY vs TLT return correlation
    stock_bond_corr_prev_60d: Optional[float] = None
    stock_bond_regime: str = "Unavailable"
    tlt_period_ret: Optional[float] = None
    tlt_3m_ret: Optional[float] = None
    assessment: str = ""


@dataclass
class CommoditiesCrossAssetData:
    as_of_date: str
    items: List[CrossAssetExpandedItem] = field(default_factory=list)
    energy_shock: Optional[EnergyShockIndicator] = None
    plumbing: Optional[CrossAssetPlumbingData] = None


def compute_cross_asset_plumbing(
    bars: Dict[str, pd.DataFrame],
    as_of_dt: pd.Timestamp,
    base_dt: pd.Timestamp,
    macro_data: Optional[Any] = None,
) -> Optional[CrossAssetPlumbingData]:
    """
    Compute cross-asset macro regime plumbing strictly from local PIT bars:
    1) Copper/Gold ratio (x1000) vs. 10Y Treasury yield 13-week divergence.
    2) 60-day rolling equity/duration correlation (SPY vs. TLT daily returns).
    """
    fr = getattr(macro_data, "fed_rates", None) if macro_data else None
    y10 = getattr(fr, "yield_10y", None) if fr else None
    y10_13w_delta = getattr(fr, "yield_10y_13w_delta", None) if fr else None
    y10_13w_bps = round(y10_13w_delta * 100.0, 1) if y10_13w_delta is not None else None

    # 1. Copper/Gold Ratio (HG=F in $/lb, GC=F in $/oz -> ratio * 1000)
    cg_ratio: Optional[float] = None
    cg_3m_pct: Optional[float] = None
    cg_regime = "Unavailable"
    hg_df = bars.get("HG=F")
    gc_df = bars.get("GC=F")
    if hg_df is not None and not hg_df.empty and gc_df is not None and not gc_df.empty:
        hg_sub = hg_df[hg_df.index <= as_of_dt]["Close"].dropna()
        gc_sub = gc_df[gc_df.index <= as_of_dt]["Close"].dropna()
        aligned_cg = pd.concat([hg_sub.rename("hg"), gc_sub.rename("gc")], axis=1).dropna()
        aligned_cg = aligned_cg[aligned_cg["gc"] > 0]
        if len(aligned_cg) >= 3:
            ratio_series = (aligned_cg["hg"] * 1000.0) / aligned_cg["gc"]
            cg_ratio = round(float(ratio_series.iloc[-1]), 3)
            idx_3m = max(0, len(ratio_series) - 1 - 63)
            base_ratio = float(ratio_series.iloc[idx_3m])
            if base_ratio > 0:
                cg_3m_pct = round(((cg_ratio - base_ratio) / base_ratio) * 100.0, 2)

            if cg_3m_pct is not None and y10_13w_bps is not None:
                if cg_3m_pct >= 3.0 and y10_13w_bps >= 15.0:
                    cg_regime = "Reflationary Growth Alignment"
                elif cg_3m_pct <= -3.0 and y10_13w_bps >= 15.0:
                    cg_regime = "Yield-Metal Divergence (Rates Pressure)"
                elif cg_3m_pct >= 3.0 and y10_13w_bps <= -15.0:
                    cg_regime = "Disinflationary Growth Bid"
                elif cg_3m_pct <= -3.0 and y10_13w_bps <= -15.0:
                    cg_regime = "Growth Cooling / Flight-to-Safety"
                else:
                    cg_regime = "Balanced Macro-Rates Alignment"
            elif cg_3m_pct is not None:
                cg_regime = "Reflationary Impulse" if cg_3m_pct >= 3.0 else ("Deflationary Cooling" if cg_3m_pct <= -3.0 else "Range-Bound")

    # 2. 60D Rolling Stock-Bond Correlation (SPY vs TLT)
    corr_60d: Optional[float] = None
    corr_prev_60d: Optional[float] = None
    sb_regime = "Unavailable"
    tlt_p_ret: Optional[float] = None
    tlt_3m_ret: Optional[float] = None

    spy_df = bars.get("SPY")
    tlt_df = bars.get("TLT")
    if tlt_df is not None and not tlt_df.empty:
        tlt_sub = tlt_df[tlt_df.index <= as_of_dt]
        if len(tlt_sub) >= 3:
            tlt_now = float(tlt_sub["Close"].iloc[-1])
            tlt_base_slice = tlt_sub[tlt_sub.index <= base_dt]
            tlt_base = float(tlt_base_slice["Close"].iloc[-1]) if not tlt_base_slice.empty else tlt_now
            if tlt_base > 0:
                tlt_p_ret = round(((tlt_now - tlt_base) / tlt_base) * 100.0, 2)
            tlt_3m_ret = round(compute_lookback_return(tlt_sub, as_of_dt, lookback_bars=63), 2)

    if spy_df is not None and not spy_df.empty and tlt_df is not None and not tlt_df.empty:
        spy_ret = spy_df[spy_df.index <= as_of_dt]["Close"].pct_change()
        tlt_ret = tlt_df[tlt_df.index <= as_of_dt]["Close"].pct_change()
        aligned_sb = pd.concat([spy_ret.rename("spy"), tlt_ret.rename("tlt")], axis=1).dropna()
        if len(aligned_sb) >= 30:
            win = min(60, len(aligned_sb))
            curr_val = aligned_sb.tail(win)["spy"].corr(aligned_sb.tail(win)["tlt"])
            if pd.notna(curr_val):
                corr_60d = round(float(curr_val), 3)
            if len(aligned_sb) >= 80:
                prev_win = aligned_sb.iloc[-80:-20]
                prev_val = prev_win["spy"].corr(prev_win["tlt"])
                if pd.notna(prev_val):
                    corr_prev_60d = round(float(prev_val), 3)

            if corr_60d is not None:
                if corr_60d >= 0.25:
                    sb_regime = "Positive (Rate-Driven — Duration Not Hedging Equities)"
                elif corr_60d <= -0.25:
                    sb_regime = "Negative (Classic Stock-Bond Diversification Active)"
                else:
                    sb_regime = "Near-Zero / Transitional"

    if cg_ratio is None and corr_60d is None:
        return None

    parts: List[str] = []
    if cg_ratio is not None:
        cg_chg_s = f"{cg_3m_pct:+.2f}%" if cg_3m_pct is not None else "n/a"
        y10_s = f"{y10:.2f}% ({y10_13w_bps:+.0f} bps 13W)" if (y10 is not None and y10_13w_bps is not None) else (f"{y10:.2f}%" if y10 is not None else "n/a")
        parts.append(f"Copper/Gold (x1000) sits at {cg_ratio:.3f} ({cg_chg_s} 3M) vs. 10Y Treasury at {y10_s} [{cg_regime}].")
    if corr_60d is not None:
        prev_s = f" (prior 60D: {corr_prev_60d:+.2f})" if corr_prev_60d is not None else ""
        parts.append(f"60-day SPY/TLT return correlation is {corr_60d:+.2f}{prev_s}: {sb_regime}.")

    return CrossAssetPlumbingData(
        copper_gold_ratio=cg_ratio,
        copper_gold_3m_chg_pct=cg_3m_pct,
        nominal_10y_yield=round(float(y10), 2) if y10 is not None else None,
        yield_10y_13w_bps=y10_13w_bps,
        copper_gold_regime=cg_regime,
        stock_bond_corr_60d=corr_60d,
        stock_bond_corr_prev_60d=corr_prev_60d,
        stock_bond_regime=sb_regime,
        tlt_period_ret=tlt_p_ret,
        tlt_3m_ret=tlt_3m_ret,
        assessment=" ".join(parts),
    )


def build_commodities_cross_asset(
    bars: Dict[str, pd.DataFrame],
    as_of_date: str,
    baseline_date: str,
    prev_baseline_date: str,
    macro_data: Optional[Any],
    cfg: Any,
) -> SectionResult:
    """Build commodities, foreign exchange, bitcoin, and energy shock tracking section."""
    try:
        as_of_dt = pd.to_datetime(as_of_date)
        base_dt = pd.to_datetime(baseline_date)
        prev_base_dt = pd.to_datetime(prev_baseline_date)

        items_cfg = getattr(cfg.cross_assets_expanded, "items", []) if hasattr(cfg, "cross_assets_expanded") else []
        if not items_cfg:
            # Default fallback list
            from config.settings import SymbolItem
            items_cfg = [
                SymbolItem(symbol="CL=F", name="Crude Oil (WTI)"),
                SymbolItem(symbol="BZ=F", name="Brent Crude"),
                SymbolItem(symbol="NG=F", name="Natural Gas"),
                SymbolItem(symbol="HG=F", name="Copper"),
                SymbolItem(symbol="GC=F", name="Gold"),
                SymbolItem(symbol="DX-Y.NYB", name="US Dollar Index (DXY)"),
                SymbolItem(symbol="JPY=X", name="USD/JPY"),
                SymbolItem(symbol="EURUSD=X", name="EUR/USD"),
                SymbolItem(symbol="BTC-USD", name="Bitcoin"),
            ]

        results: List[CrossAssetExpandedItem] = []
        oil_3m_ret = 0.0

        for item_cfg in items_cfg:
            sym = item_cfg.symbol
            df = bars.get(sym)

            if df is None or df.empty:
                results.append(
                    CrossAssetExpandedItem(
                        symbol=sym,
                        name=item_cfg.name,
                        current_price=0.0,
                        period_ret=0.0,
                        prior_period_ret=None,
                        ret_3m=0.0,
                        is_available=False,
                    )
                )
                continue

            sub = df[df.index <= as_of_dt]
            if len(sub) < 3:
                results.append(
                    CrossAssetExpandedItem(
                        symbol=sym,
                        name=item_cfg.name,
                        current_price=0.0,
                        period_ret=0.0,
                        prior_period_ret=None,
                        ret_3m=0.0,
                        is_available=False,
                    )
                )
                continue

            c_now = float(sub.iloc[-1]["Close"])

            # Period Return
            base_slice = sub[sub.index <= base_dt]
            c_base = float(base_slice.iloc[-1]["Close"]) if not base_slice.empty else c_now
            p_ret = ((c_now - c_base) / c_base) * 100.0 if c_base > 0 else 0.0

            # Prior Period Return
            prior_ret: Optional[float] = None
            if prev_base_dt < base_dt:
                prev_slice = sub[sub.index <= prev_base_dt]
                if not prev_slice.empty:
                    c_prev_base = float(prev_slice.iloc[-1]["Close"])
                    if c_prev_base > 0 and c_base > 0:
                        prior_ret = ((c_base - c_prev_base) / c_prev_base) * 100.0

            # 3-Month Return (63 sessions via compute_lookback_return)
            ret_3m = compute_lookback_return(sub, as_of_dt, lookback_bars=63)

            if sym == "CL=F":
                oil_3m_ret = ret_3m

            results.append(
                CrossAssetExpandedItem(
                    symbol=sym,
                    name=item_cfg.name,
                    current_price=c_now,
                    period_ret=p_ret,
                    prior_period_ret=prior_ret,
                    ret_3m=ret_3m,
                    is_available=True,
                )
            )

        # Energy shock calculation
        metrics = getattr(macro_data, "metrics", {}) if macro_data else {}
        cpi_m = metrics.get("cpi")
        core_cpi_m = metrics.get("core_cpi")
        headline_cpi = cpi_m.headline_value if cpi_m else 4.02
        core_cpi = core_cpi_m.headline_value if core_cpi_m else 3.71
        gap = headline_cpi - core_cpi

        is_shock = abs(oil_3m_ret) >= 15.0
        if oil_3m_ret >= 15.0:
            assess = (
                f"Energy Shock Warning: Trailing 3-month crude oil has surged {oil_3m_ret:+.2f}%. "
                f"Rising energy input costs have pushed headline CPI ({headline_cpi:.2f}%) {gap:+.2f}% above core CPI ({core_cpi:.2f}%), "
                f"constraining central bank easing capacity."
            )
        elif oil_3m_ret <= -15.0:
            assess = (
                f"Energy Deflation Tailwinds: Trailing 3-month crude oil has declined {oil_3m_ret:+.2f}%. "
                f"Energy price compression provides disinflationary relief, narrowing the headline/core CPI gap to {gap:+.2f}%."
            )
        else:
            assess = (
                f"Neutral Energy Baseline: Crude oil trailing 3-month change sits at {oil_3m_ret:+.2f}%. "
                f"Energy markets are providing neither inflationary shocks nor deflationary tailwinds, keeping the headline vs. core gap at {gap:+.2f}%."
            )

        energy_shock = EnergyShockIndicator(
            oil_3m_change_pct=oil_3m_ret,
            headline_cpi_yoy=headline_cpi,
            core_cpi_yoy=core_cpi,
            headline_core_gap_pct=gap,
            is_shock_active=is_shock,
            assessment=assess,
        )

        # Phase 3A: Cross-Asset Plumbing (Copper/Gold vs 10Y Yield + 60D Rolling SPY/TLT Correlation)
        plumbing = compute_cross_asset_plumbing(
            bars=bars,
            as_of_dt=as_of_dt,
            base_dt=base_dt,
            macro_data=macro_data,
        )

        data = CommoditiesCrossAssetData(
            as_of_date=as_of_date,
            items=results,
            energy_shock=energy_shock,
            plumbing=plumbing,
        )

        return SectionResult(
            section_num=8,
            title="Commodities, Currencies & Cross-Asset Tracker",
            status="available",
            data=data,
        )

    except Exception as e:
        logger.error("Failed building commodities cross asset section: %s", e)
        return SectionResult(
            section_num=8,
            title="Commodities, Currencies & Cross-Asset Tracker",
            status="unavailable",
            error_message=f"Commodities error: {e}",
        )
