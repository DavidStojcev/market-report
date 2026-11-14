"""Tests for Stage A1: Sevens-style Editorial Brief and Contradiction-Free Sector/Tape Narrative."""

import pandas as pd
from data.calendar import PeriodInfo
from scoring.engine import Scorecard, CategoryScore
from sections.executive_summary import build_executive_summary
from sections.macro import (
    compute_sector_rotation_intelligence,
    compute_macro_regime_intelligence,
    CreditSpreadIntelligence,
    RealRateIntelligence,
    VolTermStructureIntelligence,
    MacroMetricSnapshot,
)
from render.markdown import render_markdown
from render.html import render_html


def test_narrow_tech_bounce_does_not_say_defensive_positioning():
    """When XLK/XLC lead and XLP/XLB lag with negative RSP-SPY spread, macro prose must NOT say defensive positioning."""
    dates = pd.bdate_range("2026-10-20", "2026-11-03")
    def _bars(rets):
        closes = [100.0]
        for r in rets:
            closes.append(closes[-1] * (1.0 + r))
        return pd.DataFrame({"Close": closes[1:]}, index=dates[: len(rets)])

    n = len(dates)
    bars = {s: _bars([0.0] * n) for s in ("XLE", "XLP", "XLU", "XLV", "XLB", "XLF", "XLY", "XLRE", "XLI", "XLK", "XLC", "SPY", "RSP", "^RUT", "^NDX")}
    # Make XLK +1.3%, XLC +1.2%, XLP -2.1%, XLB -2.0%, SPY +0.35%, RSP -0.35% on final day
    for sym, chg in [("XLK", 0.013), ("XLC", 0.012), ("XLP", -0.021), ("XLB", -0.020), ("SPY", 0.0035), ("RSP", -0.0035), ("^NDX", 0.007)]:
        df = bars[sym].copy()
        df.iloc[-1, df.columns.get_loc("Close")] = df.iloc[-2]["Close"] * (1.0 + chg)
        bars[sym] = df

    rot = compute_sector_rotation_intelligence(
        bars=bars,
        session_dt=pd.to_datetime("2026-11-03"),
        base_dt=pd.to_datetime("2026-11-02"),
    )
    metrics_reflation = {
        "gdp": MacroMetricSnapshot("GDPC1", "GDP", "2026-10-30", 3.37, 3.37, "%", 3.0, 3.0, 0.37),
        "core_cpi": MacroMetricSnapshot("CPILFESL", "Core CPI", "2026-10-30", 3.10, 3.10, "%", 3.0, 3.0, 0.10),
    }
    intel = compute_macro_regime_intelligence(
        metrics=metrics_reflation,
        credit=CreditSpreadIntelligence(340.0, 343.0, -3.0, "Tight / Complacent", "Sound"),
        rates=RealRateIntelligence(2.65, 2.0, 2.30, 2.0, 4.95, 4.0, "Real Rate Tightening", "Cap multiples"),
        vol_term=VolTermStructureIntelligence(20.8, 21.3, 0.979, 0.0, "Contango", "Orderly", "Normal"),
        rotation=rot,
        report_type="daily",
    )
    assert "Sector performance reflects defensive positioning" not in intel.forward_outlook
    assert "Mega-cap growth concentrated leadership" in intel.forward_outlook
    assert "Equal-weight RSP trailed cap-weighted SPY" in intel.forward_outlook

    # Also verify Stagflationary branch never mislabels tech leadership as defensive havens
    intel_stag = compute_macro_regime_intelligence(
        metrics={},
        credit=CreditSpreadIntelligence(340.0, 343.0, -3.0, "Tight / Complacent", "Sound"),
        rates=RealRateIntelligence(2.65, 2.0, 2.30, 2.0, 4.95, 4.0, "Real Rate Tightening", "Cap multiples"),
        vol_term=VolTermStructureIntelligence(20.8, 21.3, 0.979, 0.0, "Contango", "Orderly", "Normal"),
        rotation=rot,
        report_type="daily",
    )
    assert "defensive havens" not in intel_stag.forward_outlook
    assert "Mega-cap growth concentrated leadership" in intel_stag.forward_outlook


def test_executive_brief_sevens_format_renders():
    """Executive summary renders 3 Things That Mattered and What Happened -> What It Means -> The Bottom Line."""
    period = PeriodInfo(
        report_type="daily",
        as_of_date="2026-11-03",
        session_date="2026-11-03",
        current_start="2026-11-03",
        current_end="2026-11-03",
        baseline_date="2026-11-02",
        prev_period_start="2026-11-02",
        prev_period_end="2026-11-02",
        prev_baseline_date="2026-10-30",
        session_count=1,
        is_partial=False,
        notes="",
    )
    sc = Scorecard(
        normalized_score=-0.10,
        raw_score=-0.08,
        max_score=0.85,
        verdict="Neutral",
        votes=[],
        caution_flags=[],
        top_bullish_factors=["Breadth > 50-DMA"],
        top_bearish_factors=["Nasdaq-100 below 50-DMA"],
        summary_reasons=["S&P 500 below 50-DMA"],
        confidence="Agreement: 2 of 4 active categories match the verdict",
        categories=[
            CategoryScore(name="Trend / Price", category_key="trend", weight=0.30, effective_weight=0.30, score=-0.47, weighted_score=-0.141, status="active", components=[]),
            CategoryScore(name="Market Breadth", category_key="breadth", weight=0.20, effective_weight=0.20, score=0.28, weighted_score=0.056, status="active", components=[]),
        ],
    )
    sec = build_executive_summary(
        scorecard=sc,
        period_info=period,
        spx_period_ret=0.36,
        ndx_period_ret=0.68,
    )
    assert sec.data.brief is not None
    assert len(sec.data.brief.three_things) == 3

    md = render_markdown(period, [sec])
    html = render_html(period, [sec])

    for marker in ("3 Things That Mattered", "What Happened", "What It Means", "The Bottom Line"):
        assert marker in md
        assert marker in html
