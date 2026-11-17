"""Sections 7 & 8: Conclusion and Legal Disclaimer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from indicators.engine import TechnicalSnapshot
from scoring.engine import Scorecard
from sections.base import SectionResult


@dataclass
class KeyLevel:
    label: str
    price: float
    distance_pct: float
    level_type: str  # 'Support' or 'Resistance'


@dataclass
class AssetLevels:
    symbol: str
    name: str
    current_price: float
    levels: List[KeyLevel]


@dataclass
class StrategyLens:
    strategy: str
    regime_read: str
    synthesis: str


DEFAULT_COMPLIANCE_DISCLAIMER = (
    "IMPERSONAL PUBLISHER'S EXCLUSION & HYPOTHETICAL MODEL NOTICE: "
    "This publication is strictly impersonal, bona fide financial and macroeconomic research provided solely for informational and educational purposes. "
    "Nothing herein constitutes individualized investment advice, portfolio management, or a recommendation or solicitation regarding the purchase or sale of any security, derivative, or financial instrument. "
    "The publisher is not a registered investment adviser or broker-dealer and does not tailor analysis to the investment objectives, financial situation, or risk tolerance of any specific person. "
    "All quantitative regime classifications, risk ranges, composite scores, and historical analog statistics represent hypothetical or model-derived outputs that have inherent limitations, do not reflect actual trading, and are no guarantee of future performance. "
    "Market breadth calculations use the current S&P 500 constituent roster (subject to survivorship bias in historical comparisons)."
)


@dataclass
class ConclusionData:
    verdict: str
    normalized_score: float
    top_bullish_factors: List[str]
    top_bearish_factors: List[str]
    caution_flags: List[str]
    scorecard: Scorecard
    asset_levels: List[AssetLevels]
    disclaimer: str = DEFAULT_COMPLIANCE_DISCLAIMER
    regime_intelligence: Optional[Any] = None
    cross_strategy_lenses: List[StrategyLens] = field(default_factory=list)
    seasonality: Optional[Any] = None


def compute_key_levels(
    snap: TechnicalSnapshot,
    df: Optional[pd.DataFrame] = None,
) -> AssetLevels:
    """Calculate key support and resistance levels to watch."""
    levels: List[KeyLevel] = []
    p = snap.price

    # 1. 20 DMA
    if snap.dma_20 > 0:
        l_type = "Support" if p >= snap.dma_20 else "Resistance"
        dist = ((snap.dma_20 - p) / p) * 100.0
        levels.append(KeyLevel("20-Day DMA", snap.dma_20, dist, l_type))

    # 2. 50 DMA
    if snap.dma_50 > 0:
        l_type = "Support" if p >= snap.dma_50 else "Resistance"
        dist = ((snap.dma_50 - p) / p) * 100.0
        levels.append(KeyLevel("50-Day DMA", snap.dma_50, dist, l_type))

    # 3. 200 DMA
    if snap.dma_200 > 0:
        l_type = "Support" if p >= snap.dma_200 else "Resistance"
        dist = ((snap.dma_200 - p) / p) * 100.0
        levels.append(KeyLevel("200-Day DMA", snap.dma_200, dist, l_type))

    # 4. 20-day High / Low if df available
    if df is not None and len(df) >= 20:
        recent = df.tail(20)
        h20 = float(recent["High"].max()) if "High" in recent.columns else p
        l20 = float(recent["Low"].min()) if "Low" in recent.columns else p

        if h20 > p:
            levels.append(KeyLevel("20-Day Range High", h20, ((h20 - p) / p) * 100.0, "Resistance"))
        if l20 < p:
            levels.append(KeyLevel("20-Day Range Low", l20, ((l20 - p) / p) * 100.0, "Support"))

    levels.sort(key=lambda x: x.price, reverse=True)
    return AssetLevels(
        symbol=snap.symbol,
        name=snap.name,
        current_price=p,
        levels=levels,
    )


def _build_cross_strategy_lenses(
    scorecard: Scorecard,
    spx_snap: Optional[TechnicalSnapshot] = None,
    ndx_snap: Optional[TechnicalSnapshot] = None,
    regime_intelligence: Optional[Any] = None,
    swing_data: Optional[Any] = None,
) -> List[StrategyLens]:
    """Synthesize objective, strategy-agnostic takeaways across four market participant horizons."""
    cats = {c.category_key: c for c in getattr(scorecard, "categories", []) if getattr(c, "status", "") == "active"}
    cat_breadth = cats.get("breadth")
    cat_vol = cats.get("volatility")

    # 1. Intraday & Momentum
    if spx_snap and spx_snap.dma_20 > 0:
        spx_20_dist = ((spx_snap.price - spx_snap.dma_20) / spx_snap.dma_20) * 100.0
        ndx_20_dist = ((ndx_snap.price - ndx_snap.dma_20) / ndx_snap.dma_20) * 100.0 if (ndx_snap and ndx_snap.dma_20 > 0) else None
        if spx_20_dist >= 0 and (ndx_20_dist is None or ndx_20_dist >= 0):
            momo_read = "Above 20-DMA Trend"
            momo_syn = f"S&P 500 ({spx_20_dist:+.2f}% vs 20-DMA) maintains short-term upside structure toward recent range highs."
        else:
            momo_read = "Mean-Reverting / Corrective"
            ndx_txt = f" and Nasdaq-100 {ndx_20_dist:+.2f}%" if ndx_20_dist is not None else ""
            momo_syn = f"S&P 500 ({spx_20_dist:+.2f}% vs 20-DMA{ndx_txt}) trades below short-term averages; watch 20-DMA reclaim levels as pivot triggers."
    else:
        momo_read = scorecard.verdict
        momo_syn = f"Composite score ({scorecard.normalized_score:+.2f}) guides short-term directional bias around key support and resistance bands."

    # 2. Intermediate Equity & Sector Rotation
    if regime_intelligence is not None and getattr(regime_intelligence, "rotation", None) is not None:
        rot = regime_intelligence.rotation
        p50 = getattr(swing_data, "breadth_pct_above_50dma", None) if swing_data else None
        p50_str = f"{p50:.1f}% of S&P 500 above 50-DMA; " if p50 is not None else ""
        if cat_breadth and cat_breadth.score >= 0.25:
            rot_read = "Broadening Rotation"
        elif cat_breadth and cat_breadth.score <= -0.25:
            rot_read = "Concentrated / Narrow"
        else:
            rot_read = "Selective Dispersion"
        rot_syn = f"{p50_str}RSP vs SPY spread sits at {rot.rsp_vs_spy_spread:+.2f}% with {rot.rotation_summary.rstrip('.')}."
    else:
        b_score = cat_breadth.score if cat_breadth else 0.0
        rot_read = "Supportive Breadth" if b_score >= 0.25 else ("Weak Breadth" if b_score <= -0.25 else "Mixed Breadth")
        rot_syn = f"Market breadth category sits at {b_score:+.2f}, framing sector leadership concentration vs broad participation."

    # 3. Volatility & Options
    if regime_intelligence is not None and getattr(regime_intelligence, "vol_term", None) is not None:
        vt = regime_intelligence.vol_term
        cr = getattr(regime_intelligence, "credit", None)
        vix_str = f"VIX {vt.vix_1m:.2f} (1M/3M ratio {vt.ratio:.3f}, {vt.structure})"
        cr_str = f" and HY OAS at {cr.spread_bps:.0f} bps ({cr.regime})" if cr is not None else ""
        vol_read = vt.structure
        vol_syn = f"{vix_str}{cr_str} indicate {'orderly hedging rather than systemic stress' if vt.ratio < 1.0 else 'elevated near-term hedging demand and event premium'}."
    else:
        v_score = cat_vol.score if cat_vol else 0.0
        vol_read = "Low Stress" if v_score >= 0.2 else ("Elevated Vol" if v_score <= -0.2 else "Normal / Balanced")
        vol_syn = f"Volatility & risk appetite score is {v_score:+.2f}; guides option implied move expectations and hedging costs."

    # 4. Long-Horizon & Multi-Asset
    if spx_snap and spx_snap.dma_200 > 0:
        spx_200_dist = ((spx_snap.price - spx_snap.dma_200) / spx_snap.dma_200) * 100.0
        secular_part = f"S&P 500 trades {spx_200_dist:+.2f}% vs its 200-DMA ({spx_snap.dma_200:,.2f})"
    else:
        secular_part = f"Composite regime verdict stands at {scorecard.verdict}"
    if regime_intelligence is not None and getattr(regime_intelligence, "rates", None) is not None:
        rt = regime_intelligence.rates
        core_read = regime_intelligence.quadrant
        rate_part = (
            f"10Y TIPS real yield at {rt.real_yield_10y:.2f}% keeps real discount hurdles {'elevated' if rt.real_yield_10y >= 2.0 else 'moderate'}"
        )
        core_syn = f"{secular_part}; macro quadrant '{core_read}' and {rate_part}."
    else:
        core_read = "Secular Trend Check"
        core_syn = f"{secular_part}; long-term investors weigh 200-DMA support alongside macro rate trajectories."

    return [
        StrategyLens("Intraday & Momentum", momo_read, momo_syn),
        StrategyLens("Sector & Factor Rotation", rot_read, rot_syn),
        StrategyLens("Options & Volatility", vol_read, vol_syn),
        StrategyLens("Multi-Asset & Long-Term", core_read, core_syn),
    ]


def build_conclusion(
    scorecard: Optional[Scorecard],
    spx_snap: Optional[TechnicalSnapshot],
    ndx_snap: Optional[TechnicalSnapshot],
    spx_df: Optional[pd.DataFrame] = None,
    ndx_df: Optional[pd.DataFrame] = None,
    regime_intelligence: Optional[Any] = None,
    swing_data: Optional[Any] = None,
    seasonality: Optional[Any] = None,
) -> SectionResult:
    """Build conclusion section with transparent scorecard and key price levels."""
    if not scorecard:
        return SectionResult(
            section_num=7,
            title="Conclusion & Market Outlook",
            status="unavailable",
            error_message="Scorecard unavailable.",
        )

    asset_levels: List[AssetLevels] = []
    if spx_snap:
        asset_levels.append(compute_key_levels(spx_snap, spx_df))
    if ndx_snap:
        asset_levels.append(compute_key_levels(ndx_snap, ndx_df))

    lenses = _build_cross_strategy_lenses(
        scorecard=scorecard,
        spx_snap=spx_snap,
        ndx_snap=ndx_snap,
        regime_intelligence=regime_intelligence,
        swing_data=swing_data,
    )

    data = ConclusionData(
        verdict=scorecard.verdict,
        normalized_score=scorecard.normalized_score,
        top_bullish_factors=scorecard.top_bullish_factors,
        top_bearish_factors=scorecard.top_bearish_factors,
        caution_flags=scorecard.caution_flags,
        scorecard=scorecard,
        asset_levels=asset_levels,
        regime_intelligence=regime_intelligence,
        cross_strategy_lenses=lenses,
        seasonality=seasonality,
    )

    return SectionResult(
        section_num=7,
        title="Conclusion & Market Outlook",
        status="available",
        data=data,
    )


@dataclass
class DataSourceInfo:
    source_name: str
    coverage: str
    fetch_time: str
    status: str
    notes: str


def build_footer(
    data_sources: Optional[List[DataSourceInfo]] = None,
    disclaimer: Optional[str] = None,
    overall_status: str = "PARTIAL",
    missing_inputs: Optional[List[str]] = None,
    data_warnings: Optional[List[str]] = None,
    generated_at_utc: Optional[str] = None,
) -> SectionResult:
    """Build Footer with legal disclaimer, constituent survivorship bias notice, and Data Quality provenance table."""
    return SectionResult(
        section_num=10,
        title="Footer & Disclaimer",
        status="available",
        data={
            "disclaimer": disclaimer or DEFAULT_COMPLIANCE_DISCLAIMER,
            "data_sources": data_sources or [],
            "overall_status": overall_status,
            "missing_inputs": missing_inputs or [],
            "data_warnings": data_warnings or [],
            "generated_at_utc": generated_at_utc or "",
        },
    )
