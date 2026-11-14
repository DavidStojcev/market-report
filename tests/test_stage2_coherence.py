"""Acceptance & regression tests for Stage 2: Logic & Narrative Coherence (Substages 2.1 - 2.6)."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import pytest

from config.settings import MoversConfig, ScoringConfig
from data.calendar import MarketCalendar, PeriodInfo
from data.constituents import Constituent
from data.fred_provider import MacroMetricSnapshot
from data.macro_base import ReportedEarningsEvent
from indicators.breadth import BreadthMetric, MarketBreadthResult
from indicators.engine import TechnicalSnapshot
from qa_checks import (
    check_r3_cross_section_facts_registry,
    check_t1_label_vote_consistency,
)
from render.html import render_html
from render.markdown import render_markdown
from report import generate_market_report
from scoring.engine import Scorecard, TrendScoringEngine
from scoring.rules import evaluate_breadth_category
from sections.base import CanonicalFactsRegistry, SectionResult
from sections.breadth import BreadthSectionData
from sections.conclusion import ConclusionData
from sections.executive_summary import (
    ExecutiveSummaryData,
    _build_executive_brief,
)
from sections.forward_calendar import ForwardCalendarData
from sections.macro import (
    CreditSpreadIntelligence,
    FedRatesSummary,
    MacroSectionData,
    MarketRegimeIntelligence,
    RealRateIntelligence,
    SectorRotationIntelligence,
    VolTermStructureIntelligence,
    compute_macro_regime_intelligence,
)
from sections.movers import (
    MoverItem,
    PeriodMovers,
    TopMoversData,
    _calculate_mover_stats,
    _generate_movers_theme_summary,
    build_movers,
)
from sections.sectors import SectorBreadthItem
from sections.sentiment_positioning import build_sentiment_positioning
from sections.swing_regime import SwingGateData, _build_posture_triggers
from sections.valuation import ValuationData
from sections.volatility import VolatilityData
from sections.watchlist import build_watchlist_section
from sections.what_changed import DeltaMetric, WhatChangedData


def _snap_infl(series_id: str, name: str, curr: float, prior: float) -> MacroMetricSnapshot:
    delta = round(curr - prior, 4)
    return MacroMetricSnapshot(
        series_id=series_id,
        name=name,
        latest_date="2026-10-31",
        latest_value=curr,
        headline_value=curr,
        headline_unit="% YoY",
        prior_value=prior,
        prior_headline=prior,
        change=delta,
        period_change=delta,
    )


def _period(report_type: str = "daily", session_date: str = "2026-11-13") -> PeriodInfo:
    return PeriodInfo(
        report_type=report_type,
        as_of_date=session_date,
        session_date=session_date,
        current_start=session_date,
        current_end=session_date,
        baseline_date="2026-11-12",
        prev_period_start="2026-11-12",
        prev_period_end="2026-11-12",
        prev_baseline_date="2026-11-11",
        is_partial=False,
        session_count=1,
        notes="",
    )


def _tech_snap(symbol: str = "^GSPC", name: str = "S&P 500", price: float = 5900.0) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        symbol=symbol,
        name=name,
        as_of_date="2026-11-13",
        price=price,
        dma_20=price * 0.99,
        dma_50=price * 0.98,
        dma_200=price * 0.92,
        price_vs_dma20_pct=1.0,
        price_vs_dma50_pct=2.0,
        price_vs_dma200_pct=8.7,
        cross_50_200="Bullish (50 > 200)",
        rsi=58.0,
        prev_rsi=55.0,
        delta_rsi=3.0,
        rsi_flag="Neutral",
        macd=18.0,
        macd_signal=12.0,
        macd_hist=6.0,
        macd_status="Bullish (MACD > Signal)",
        prev_macd=14.0,
        delta_macd=4.0,
        adx=25.0,
        plus_di=24.0,
        minus_di=14.0,
        prev_adx=24.0,
        delta_adx=1.0,
        trend_strength="Moderate Trend",
    )


# ------------------------------------------------------------------------------
# Substage 2.1: Multi-metric inflation classifier & yield-direction prose
# ------------------------------------------------------------------------------


def test_2_1_macro_regime_multi_metric_and_yield_direction():
    """Rising-yield or hot/rising multi-metric CPI/PPI/PCE inputs never yield Goldilocks or declining-yield prose."""
    credit = CreditSpreadIntelligence(3.26, 326.0, -4.0, "Tight / Complacent", "Tight credit")
    vol_term = VolTermStructureIntelligence(18.0, 20.0, 0.90, -0.02, "Contango (Normal)", "Orderly / Calm", "Contango")
    rotation = SectorRotationIntelligence(
        cyclical_return=2.5,
        defensive_return=1.0,
        spread=1.5,
        posture="Balanced / Rotation",
        leading_cyclical="Industrials (XLI: +2.8%)",
        leading_defensive="Health Care (XLV: +1.1%)",
        assessment="Balanced",
        top_inflows=["Industrials (XLI: +2.8%)"],
        top_outflows=["Utilities (XLU: -0.5%)"],
    )

    hot_metrics = {
        "cpi": _snap_infl("CPIAUCSL", "Headline CPI", 4.02, 3.90),
        "core_cpi": _snap_infl("CPILFESL", "Core CPI", 3.45, 3.30),
        "ppi": _snap_infl("PPIFIS", "PPI Final Demand", 2.95, 2.70),
        "core_pce": _snap_infl("PCEPILFE", "Core PCE", 2.85, 2.75),
        "unemployment": _snap_infl("UNRATE", "Unemployment Rate", 4.20, 4.20),
        "payrolls": _snap_infl("PAYEMS", "Nonfarm Payrolls", 200.0, 180.0),
        "gdp": _snap_infl("GDPC1", "Real GDP", 2.60, 2.40),
    }
    rising_rates = RealRateIntelligence(2.45, 10.0, 2.50, 5.0, 4.95, 15.0, "Real Rate Tightening", "Rising yields")

    for rtype in ("daily", "weekly", "monthly"):
        intel_hot = compute_macro_regime_intelligence(
            metrics=hot_metrics,
            credit=credit,
            rates=rising_rates,
            vol_term=vol_term,
            rotation=rotation,
            report_type=rtype,
        )
        assert "Goldilocks" not in intel_hot.quadrant
        assert "Goldilocks" not in intel_hot.summary_badge
        outlook_low = intel_hot.forward_outlook.lower()
        assert "declining yield" not in outlook_low
        assert "easing yield" not in outlook_low

    # Cool/falling inflation BUT rising yields (+12 bps) -> Goldilocks still blocked!
    cool_falling_metrics = {
        "cpi": _snap_infl("CPIAUCSL", "Headline CPI", 2.10, 2.40),
        "core_cpi": _snap_infl("CPILFESL", "Core CPI", 2.20, 2.45),
        "ppi": _snap_infl("PPIFIS", "PPI Final Demand", 1.40, 1.80),
        "core_pce": _snap_infl("PCEPILFE", "Core PCE", 2.00, 2.25),
        "unemployment": _snap_infl("UNRATE", "Unemployment Rate", 3.70, 3.90),
        "payrolls": _snap_infl("PAYEMS", "Nonfarm Payrolls", 220.0, 190.0),
        "gdp": _snap_infl("GDPC1", "Real GDP", 2.80, 2.50),
        "retail_sales": _snap_infl("RSAFS", "Retail Sales", 0.50, 0.30),
    }
    intel_blocked = compute_macro_regime_intelligence(
        metrics=cool_falling_metrics,
        credit=credit,
        rates=RealRateIntelligence(2.00, 6.0, 2.15, 6.0, 4.15, 12.0, "Real Rate Tightening", "Rising"),
        vol_term=vol_term,
        rotation=rotation,
        report_type="weekly",
    )
    assert "Goldilocks" not in intel_blocked.quadrant

    # Cool/falling inflation AND falling yields (-8 bps) -> Goldilocks permitted
    intel_gold = compute_macro_regime_intelligence(
        metrics=cool_falling_metrics,
        credit=credit,
        rates=RealRateIntelligence(1.85, -5.0, 2.10, -3.0, 3.95, -8.0, "Yield Easing", "Easing"),
        vol_term=vol_term,
        rotation=rotation,
        report_type="weekly",
    )
    assert "Goldilocks" in intel_gold.quadrant


# ------------------------------------------------------------------------------
# Substage 2.2: Defensive vs cyclical rotation language & trigger prefix cleanup
# ------------------------------------------------------------------------------


def test_2_2_defensive_rotation_language_and_trigger_prefixes():
    """Staples/defensive leadership produces defensive wording and never claims cyclical rotation or risk appetite."""
    credit = CreditSpreadIntelligence(3.26, 326.0, -4.0, "Tight / Complacent", "Benign")
    vol_term = VolTermStructureIntelligence(18.5, 20.2, 0.916, -0.01, "Contango (Normal)", "Orderly / Calm", "Contango")
    rates = RealRateIntelligence(2.40, 2.0, 2.35, 1.0, 4.75, 3.0, "Balanced Rates", "Stable")
    rot_def = SectorRotationIntelligence(
        cyclical_return=1.1,
        defensive_return=5.8,
        spread=-4.7,
        posture="Defensive Flight / Capital Preservation",
        leading_cyclical="Industrials (XLI: +1.2%)",
        leading_defensive="Consumer Staples (XLP: +7.4%)",
        assessment="Defensive leadership",
        top_inflows=["Cons. Staples (XLP: +7.4%)", "Utilities (XLU: +4.9%)"],
        top_outflows=["Cons. Disc (XLY: -2.8%)", "Tech (XLK: -1.9%)"],
        rsp_vs_spy_spread=0.65,
        rut_vs_ndx_spread=0.90,
    )
    cool_metrics = {
        "cpi": _snap_infl("CPIAUCSL", "Headline CPI", 2.20, 2.40),
        "core_pce": _snap_infl("PCEPILFE", "Core PCE", 2.10, 2.30),
        "unemployment": _snap_infl("UNRATE", "Unemployment Rate", 3.80, 4.00),
        "payrolls": _snap_infl("PAYEMS", "Nonfarm Payrolls", 210.0, 180.0),
        "gdp": _snap_infl("GDPC1", "Real GDP", 2.70, 2.40),
    }

    intel = compute_macro_regime_intelligence(
        metrics=cool_metrics,
        credit=credit,
        rates=RealRateIntelligence(2.00, -4.0, 2.10, -4.0, 4.10, -8.0, "Yield Easing", "Easing"),
        vol_term=vol_term,
        rotation=rot_def,
        report_type="weekly",
    )
    assert "cyclical rotation" not in intel.forward_outlook.lower()
    assert "broad risk appetite" not in intel.forward_outlook.lower()

    sc = Scorecard(
        raw_score=1.0,
        max_score=10.0,
        normalized_score=0.10,
        verdict="Neutral (Mixed)",
        votes=[],
        caution_flags=[],
        top_bullish_factors=[],
        top_bearish_factors=[],
        summary_reasons=[],
    )
    brief = _build_executive_brief(
        scorecard=sc,
        period_info=_period("daily", "2026-11-13"),
        regime_intelligence=intel,
        spx_snap=_tech_snap("^GSPC", "S&P 500", 5900.0),
        ndx_snap=_tech_snap("^NDX", "Nasdaq-100", 20500.0),
        spx_period_ret=0.45,
        ndx_period_ret=-0.20,
    )
    combined = " ".join([brief.headline_takeaway] + brief.three_things + [brief.what_happened, brief.what_it_means]).lower()
    assert "defensive" in combined
    assert "cyclical rotation" not in combined
    assert "risk appetite" not in combined

    for posture in ("DEFENSIVE", "SELECTIVE", "REDUCED", "FULL"):
        trig = _build_posture_triggers(
            posture=posture,
            spy_price=585.0,
            spy_20ema=590.0,
            spy_50dma=580.0,
            qqq_price=495.0,
            qqq_20ema=502.0,
            qqq_50dma=498.0,
            spy_range=None,
        )
        assert not trig.upgrade_trigger.startswith("Model upgrade condition:")
        assert not trig.downgrade_trigger.startswith("Model downgrade condition:")


# ------------------------------------------------------------------------------
# Substage 2.3: Watchlist >= +5% 3M RS gate & "Why flagged" column
# ------------------------------------------------------------------------------


def test_2_3_watchlist_outlier_leader_5pct_floor_and_why_flagged():
    """Outlier Leader requires 3M RS >= +5.0% in addition to defensive sector; Why flagged column rendered."""
    dates = pd.date_range("2025-11-01", "2026-11-13", freq="B")
    n = len(dates)

    def _mk_bars(start_px: float, end_px: float, dip_mid: bool = False) -> pd.DataFrame:
        closes = np.linspace(start_px, end_px, n)
        if dip_mid:
            closes[-3] = end_px * 0.99
        return pd.DataFrame(
            {
                "Open": closes * 0.995,
                "High": closes * 1.008,
                "Low": closes * 0.992,
                "Close": closes,
                "Volume": np.full(n, 4_000_000.0),
            },
            index=dates,
        )

    # SPY gains ~10% over 3M (63 bars ~ roughly last quarter of array)
    spy_closes = np.concatenate([np.full(n - 64, 500.0), np.linspace(500.0, 550.0, 64)])
    spy_df = pd.DataFrame(
        {"Open": spy_closes, "High": spy_closes * 1.005, "Low": spy_closes * 0.995, "Close": spy_closes, "Volume": np.full(n, 50_000_000.0)},
        index=dates,
    )
    # LLY (Health Care defensive) surges +35% over last 63 bars -> 3M RS ~ +25% (>= +5%)
    lly_closes = np.concatenate([np.full(n - 64, 700.0), np.linspace(700.0, 945.0, 64)])
    lly_df = pd.DataFrame(
        {"Open": lly_closes, "High": lly_closes * 1.01, "Low": lly_closes * 0.99, "Close": lly_closes, "Volume": np.full(n, 3_000_000.0)},
        index=dates,
    )
    # PG (Consumer Staples defensive) gains +12% over last 63 bars -> 3M RS ~ +2% (< +5% -> NOT Outlier Leader!)
    pg_closes = np.concatenate([np.full(n - 64, 150.0), np.linspace(150.0, 168.0, 64)])
    pg_df = pd.DataFrame(
        {"Open": pg_closes, "High": pg_closes * 1.005, "Low": pg_closes * 0.995, "Close": pg_closes, "Volume": np.full(n, 5_000_000.0)},
        index=dates,
    )
    # JPM (Financials non-defensive) gains +5% -> negative RS vs SPY (-5% < +5% -> NOT Outlier Leader!)
    jpm_closes = np.concatenate([np.full(n - 64, 200.0), np.linspace(200.0, 210.0, 64)])
    jpm_df = pd.DataFrame(
        {"Open": jpm_closes, "High": jpm_closes * 1.005, "Low": jpm_closes * 0.995, "Close": jpm_closes, "Volume": np.full(n, 8_000_000.0)},
        index=dates,
    )

    constituents = [
        Constituent("LLY", "Eli Lilly", "Health Care", in_sp500=True),
        Constituent("PG", "Procter & Gamble", "Consumer Staples", in_sp500=True),
        Constituent("JPM", "JPMorgan Chase", "Financials", in_sp500=True),
    ]
    sec = build_watchlist_section(
        equity_bars={"LLY": lly_df, "PG": pg_df, "JPM": jpm_df},
        spy_df=spy_df,
        gate_posture="DEFENSIVE",
        sector_breadth_data=[
            SectorBreadthItem("Health Care", "XLV", 75.0, 70.0, 12),
            SectorBreadthItem("Consumer Staples", "XLP", 75.0, 70.0, 8),
            SectorBreadthItem("Financials", "XLF", 75.0, 70.0, 5),
        ],
        constituents=constituents,
        as_of_date="2026-11-13",
        custom_tickers=["LLY", "PG", "JPM"],
        fallback_to_default=False,
    )
    assert sec.is_available
    by_sym = {it.ticker: it for it in sec.data.items}

    assert by_sym["LLY"].long_permitted == "Outlier Leader (Defensive Regime)"
    assert by_sym["LLY"].readiness == "Ready"
    assert "3M RS" in by_sym["LLY"].why_flagged

    assert "Outlier Leader" not in by_sym["PG"].long_permitted
    assert "Outlier Leader" not in by_sym["PG"].readiness
    assert "3M RS" in by_sym["PG"].why_flagged

    assert "Outlier Leader" not in by_sym["JPM"].long_permitted
    assert "Outlier Leader" not in by_sym["JPM"].readiness
    assert "Lagging 3M RS" in by_sym["JPM"].why_flagged

    period = _period("daily", "2026-11-13")
    md_out = render_markdown(period, [sec])
    html_out = render_html(period, [sec])
    assert "Why flagged" in md_out
    assert "<th>Why flagged</th>" in html_out


# ------------------------------------------------------------------------------
# Substage 2.4: Movers "Technical Context" & calendar_provider scoping
# ------------------------------------------------------------------------------


class _StubCalendarProvider:
    def get_recent_reported_earnings(self, start_date: str, end_date: str, symbols=None) -> List[ReportedEarningsEvent]:
        # Report NVDA earnings only on 2026-11-04 (inside 1M window ending 2026-11-13, NOT in daily 2026-11-12..13)
        events = [
            ReportedEarningsEvent(symbol="NVDA", name="NVIDIA", date="2026-11-04", hour="amc", market_cap=3e12),
            ReportedEarningsEvent(symbol="AAPL", name="Apple", date="2026-11-13", hour="bmo", market_cap=3e12),
        ]
        return [e for e in events if start_date <= e.date <= end_date]


def test_2_4_movers_technical_context_and_earnings_window():
    """Movers uses 'Technical Context' with non-causal wording and real window earnings filtering."""
    dates = pd.date_range("2025-10-01", "2026-11-13", freq="B")
    n = len(dates)

    def _mk_stock(jump_pct: float, break_high: bool = True) -> pd.DataFrame:
        closes = np.linspace(100.0, 100.0, n)
        if not break_high:
            closes[-30] = 130.0
            closes[:-1] = np.linspace(125.0, 100.0, n - 1)
            closes[-30] = 130.0
        closes[-1] = closes[-2] * (1.0 + jump_pct / 100.0)
        vols = np.full(n, 2_000_000.0)
        vols[-1] = 4_200_000.0
        return pd.DataFrame(
            {"Open": closes, "High": closes * 1.005, "Low": closes * 0.995, "Close": closes, "Volume": vols},
            index=dates,
        )

    nvda_df = _mk_stock(8.5, break_high=True)
    aapl_df = _mk_stock(-7.2, break_high=False)
    msft_df = _mk_stock(6.5, break_high=True)

    m_non_earn = _calculate_mover_stats(
        msft_df,
        "MSFT",
        "Microsoft Corp.",
        "Information Technology",
        3e12,
        pd.Timestamp("2026-11-13"),
        pd.Timestamp("2026-11-12"),
        MoversConfig(min_price=5.0, min_avg_volume=100_000, min_market_cap=1e9, top_n=5),
        earnings_in_window=False,
    )
    assert m_non_earn is not None
    banned_causal = ("multiple compression", "relative-strength bid", "broad sector momentum / news flow", "relative weakness / sector selling")
    for phrase in banned_causal:
        assert phrase not in m_non_earn.catalyst_driver.lower()
    assert "52W high breakout" in m_non_earn.catalyst_driver

    m_earn = _calculate_mover_stats(
        aapl_df,
        "AAPL",
        "Apple Inc.",
        "Information Technology",
        3e12,
        pd.Timestamp("2026-11-13"),
        pd.Timestamp("2026-11-12"),
        MoversConfig(min_price=5.0, min_avg_volume=100_000, min_market_cap=1e9, top_n=5),
        earnings_in_window=True,
    )
    assert m_earn is not None
    assert m_earn.catalyst_driver.startswith("Post-earnings selloff / guidance cuts")

    # Full build_movers window check using StubCalendarProvider
    cal = MarketCalendar()
    constituents = [
        Constituent("NVDA", "NVIDIA Corp.", "Information Technology", in_sp500=True),
        Constituent("AAPL", "Apple Inc.", "Information Technology", in_sp500=True),
        Constituent("MSFT", "Microsoft Corp.", "Information Technology", in_sp500=True),
    ]
    infos = {
        "NVDA": {"market_cap": 3e12},
        "AAPL": {"market_cap": 3e12},
        "MSFT": {"market_cap": 3e12},
    }
    sec_mov = build_movers(
        bars={"NVDA": nvda_df, "AAPL": aapl_df, "MSFT": msft_df},
        constituents=constituents,
        ticker_infos=infos,
        calendar=cal,
        as_of_session="2026-11-13",
        movers_cfg=MoversConfig(min_price=5.0, min_avg_volume=100_000, min_market_cap=1e9, top_n=5),
        calendar_provider=_StubCalendarProvider(),
    )
    assert sec_mov.is_available
    day_syms = {m.symbol: m for m in sec_mov.data.day.gainers + sec_mov.data.day.losers}
    month_syms = {m.symbol: m for m in sec_mov.data.month.gainers + sec_mov.data.month.losers}
    assert day_syms["AAPL"].has_earnings_in_window is True
    assert day_syms["NVDA"].has_earnings_in_window is False
    assert month_syms["NVDA"].has_earnings_in_window is True

    period = _period("daily", "2026-11-13")
    md_out = render_markdown(period, [sec_mov])
    html_out = render_html(period, [sec_mov])
    assert "Technical Context" in md_out
    assert "<th>Technical Context</th>" in html_out
    assert "Move Cause / Catalyst" not in md_out
    assert "Catalyst / Driver" not in html_out

    summary = _generate_movers_theme_summary([m_non_earn], [m_earn])
    assert "multiple compression" not in summary.lower()


# ------------------------------------------------------------------------------
# Substage 2.5: Percentile-based Net New Highs & Sentiment consistency
# ------------------------------------------------------------------------------


def test_2_5_breadth_percentile_gradient_and_sentiment_consistency():
    """Net New Highs scores on 1Y percentile gradient (58th %ile -> +0.20); Sentiment notes align."""
    mbr_58 = MarketBreadthResult(
        as_of_date="2026-11-13",
        prev_date="2026-11-12",
        total_constituents=500,
        included_constituents=500,
        coverage_pct=100.0,
        pct_above_20_dma=BreadthMetric(55.0, 52.0, 3.0),
        pct_above_50_dma=BreadthMetric(58.0, 55.0, 3.0),
        pct_above_200_dma=BreadthMetric(65.0, 63.0, 2.0),
        net_new_highs=39,
        net_new_highs_pctile_1y=58.0,
    )
    _, votes_58, _ = evaluate_breadth_category(mbr_58)
    nh_vote_58 = next(v for v in votes_58 if v.rule_name == "Breadth: Net New Highs")
    assert nh_vote_58.vote == pytest.approx(0.20, abs=1e-4)
    assert "58th" in nh_vote_58.condition

    mbr_50 = MarketBreadthResult(
        as_of_date="2026-11-13",
        prev_date="2026-11-12",
        total_constituents=500,
        included_constituents=500,
        coverage_pct=100.0,
        pct_above_20_dma=BreadthMetric(50.0, 50.0, 0.0),
        pct_above_50_dma=BreadthMetric(50.0, 50.0, 0.0),
        pct_above_200_dma=BreadthMetric(50.0, 50.0, 0.0),
        net_new_highs=25,
        net_new_highs_pctile_1y=50.0,
    )
    _, votes_50, _ = evaluate_breadth_category(mbr_50)
    nh_vote_50 = next(v for v in votes_50 if v.rule_name == "Breadth: Net New Highs")
    assert nh_vote_50.vote == pytest.approx(0.0, abs=1e-6)

    # Low percentile (21.2 -> 21st, NOT '21th') produces negative vote with matching T1 label
    mbr_21 = MarketBreadthResult(
        as_of_date="2026-10-30",
        prev_date="2026-10-29",
        total_constituents=500,
        included_constituents=500,
        coverage_pct=100.0,
        pct_above_20_dma=BreadthMetric(45.0, 48.0, -3.0),
        pct_above_50_dma=BreadthMetric(45.0, 48.0, -3.0),
        pct_above_200_dma=BreadthMetric(52.0, 54.0, -2.0),
        net_new_highs=10,
        net_new_highs_pctile_1y=21.2,
    )
    _, votes_21, _ = evaluate_breadth_category(mbr_21)
    nh_vote_21 = next(v for v in votes_21 if v.rule_name == "Breadth: Net New Highs")
    assert nh_vote_21.vote == pytest.approx(-0.72, abs=1e-4)
    assert "21st" in nh_vote_21.condition
    assert "21th" not in nh_vote_21.condition
    assert "%ile 1Y %ile" not in nh_vote_21.condition

    # Scorecard check with T1 + Sentiment note check
    sc = TrendScoringEngine(
        ScoringConfig(weights={}, thresholds={"bullish": 0.3, "bearish": -0.3}, vix={})
    ).compute_scorecard(
        spx_snapshot=_tech_snap(),
        vix_current=18.5,
        vix_previous=19.0,
        breadth_data=mbr_21,
    )
    cat_sent = next(c for c in sc.categories if c.category_key == "sentiment_positioning")
    assert cat_sent.status == "unavailable"
    assert "CBOE vol surface scored in Volatility category" in cat_sent.notes

    dummy_concl = ConclusionData(
        verdict=sc.verdict,
        normalized_score=sc.normalized_score,
        top_bullish_factors=[],
        top_bearish_factors=[],
        caution_flags=[],
        scorecard=sc,
        asset_levels=[],
    )
    t1_res = check_t1_label_vote_consistency([SectionResult(9, "Conclusion", "available", dummy_concl)])
    assert t1_res.status == "PASS", t1_res.details

    dates = pd.date_range("2026-09-01", "2026-11-13", freq="B")
    spx_df = pd.DataFrame({"Close": np.linspace(5700.0, 5900.0, len(dates))}, index=dates)
    sec_sent = build_sentiment_positioning(spx_df=spx_df, vix_current=18.5, as_of_date="2026-11-13")
    assert sec_sent.is_available
    assert "Volatility & Risk Appetite category" in sec_sent.data.assessment


# ------------------------------------------------------------------------------
# Substage 2.6: CanonicalFactsRegistry & QA Check R3
# ------------------------------------------------------------------------------


def _build_consistent_r3_bundle():
    sc = Scorecard(
        raw_score=2.0,
        max_score=10.0,
        normalized_score=0.20,
        verdict="Neutral (Mixed)",
        votes=[],
        caution_flags=[],
        top_bullish_factors=[],
        top_bearish_factors=[],
        summary_reasons=[],
    )
    intel = MarketRegimeIntelligence(
        quadrant="Reflation / Late-Cycle Tension",
        growth_signal="Resilient / Expanding",
        inflation_signal="Firm / Sticky",
        summary_badge="Reflation / Late-Cycle Tension",
        forward_outlook="Test outlook",
        credit=CreditSpreadIntelligence(3.26, 326.0, -4.0, "Tight / Complacent", "Tight"),
        rates=RealRateIntelligence(2.40, 2.0, 2.35, 1.0, 4.75, 3.0, "Normal", "Normal"),
        vol_term=VolTermStructureIntelligence(19.00, 20.50, 0.927, -0.01, "Contango (Normal)", "Orderly", "Contango"),
        rotation=SectorRotationIntelligence(1.0, 2.0, -1.0, "Balanced / Rotation", "XLI", "XLP", "Balanced"),
    )
    fr = FedRatesSummary(
        fed_funds_rate=4.25,
        target_range="4.25%-4.50%",
        last_meeting_date="2026-10-28",
        next_meeting_date="2026-12-16",
        days_to_next=33,
        yield_2y=4.35,
        yield_2y_delta=0.02,
        yield_10y=4.75,
        yield_10y_delta=0.03,
        spread_2s10s=0.40,
        spread_2s10s_delta=0.01,
        is_inverted=False,
        curve_status="Normal",
    )
    val = ValuationData(nominal_10y_yield=4.75, real_10y_yield=2.40)
    mbr = MarketBreadthResult(
        as_of_date="2026-11-13",
        prev_date="2026-11-12",
        total_constituents=500,
        included_constituents=500,
        coverage_pct=100.0,
        pct_above_50_dma=BreadthMetric(62.4, 60.0, 2.4),
        net_new_highs=39,
    )
    registry = CanonicalFactsRegistry(
        as_of_date="2026-11-13",
        baseline_date="2026-11-12",
        report_type="daily",
        vix_current=19.00,
        yield_10y=4.75,
        real_yield_10y=2.40,
        pct_above_50dma=62.4,
        net_new_highs=39,
        macro_quadrant="Reflation / Late-Cycle Tension",
    )
    sections = [
        SectionResult(
            1,
            "Executive Summary",
            "available",
            ExecutiveSummaryData(
                verdict="Neutral (Mixed)",
                normalized_score=0.20,
                raw_score=2.0,
                max_score=10.0,
                reasons=[],
                report_type="Daily",
                as_of_date="2026-11-13",
                session_date="2026-11-13",
                is_partial=False,
                notes="",
                regime_summary="Reflation / Late-Cycle Tension",
                posture="REDUCED",
            ),
        ),
        SectionResult(
            2,
            "What Changed",
            "available",
            WhatChangedData(
                as_of_date="2026-11-13",
                deltas=[DeltaMetric("High-Yield Credit OAS Spread", "326 bps", "330 bps", "-4.0 bps", "positive", "Tight")],
                single_biggest_risk="Elevated yields ahead of upcoming CPI release.",
            ),
        ),
        SectionResult(
            3,
            "Swing Gate",
            "available",
            SwingGateData(
                as_of_date="2026-11-13",
                posture="REDUCED",
                posture_badge_class="badge-primary",
                triggered_rule="Test",
                spy_price=590.0,
                spy_20ema=592.0,
                spy_20ema_dist_pct=-0.34,
                spy_20ema_pass=False,
                spy_50dma=580.0,
                spy_50dma_dist_pct=1.72,
                spy_50dma_pass=True,
                qqq_price=500.0,
                qqq_20ema=502.0,
                qqq_20ema_dist_pct=-0.40,
                qqq_20ema_pass=False,
                qqq_50dma=490.0,
                qqq_50dma_dist_pct=2.04,
                qqq_50dma_pass=True,
                vix_level=19.00,
                vix_5d_trend=-0.5,
                vix_term_structure="Contango (Normal)",
                breadth_pct_above_50dma=62.4,
                breadth_ad_state="Positive",
                net_new_highs=39,
                execution_guidance="Test",
            ),
        ),
        SectionResult(
            5,
            "Volatility",
            "available",
            VolatilityData(
                symbol="^VIX",
                name="VIX",
                current_level=19.00,
                prev_level=19.20,
                change=-0.20,
                pct_change=-1.04,
                period_high=19.50,
                period_low=18.80,
                regime="Cautious",
                regime_description="Test",
                implied_daily_move_pct=1.19,
                ratio_vix_3m=0.927,
            ),
        ),
        SectionResult(
            6,
            "Macro",
            "available",
            MacroSectionData([], fr, [], [], [], {}, intelligence=intel, valuation=val),
        ),
        SectionResult(8, "Breadth", "available", BreadthSectionData(result=mbr)),
        SectionResult(
            12,
            "Forward Calendar",
            "available",
            ForwardCalendarData(as_of_date="2026-11-13", lookahead_date="2026-11-20", high_impact_count=1, implied_1d_move_pct=1.19),
        ),
        SectionResult(9, "Conclusion", "available", ConclusionData("Neutral (Mixed)", 0.20, [], [], [], sc, [])),
    ]
    return sections, registry


def test_2_6_r3_qa_check_unit_pass_and_fail():
    """R3 passes when sections agree and catches VIX, breadth, date, and event drift."""
    sections, registry = _build_consistent_r3_bundle()
    res_ok = check_r3_cross_section_facts_registry(sections, facts_registry=registry)
    assert res_ok.check_id == "R3"
    assert res_ok.status == "PASS", res_ok.details

    # 1. VIX mismatch
    s_vix, r_vix = _build_consistent_r3_bundle()
    s_vix[2].data.vix_level = 22.50
    res_vix = check_r3_cross_section_facts_registry(s_vix, facts_registry=r_vix)
    assert res_vix.status == "FAIL"
    assert any("VIX" in d for d in res_vix.details)

    # 2. Net New Highs mismatch
    s_nh, _ = _build_consistent_r3_bundle()
    s_nh[2].data.net_new_highs = 12
    res_nh = check_r3_cross_section_facts_registry(s_nh)
    assert res_nh.status == "FAIL"
    assert any("Net New Highs" in d for d in res_nh.details)

    # 3. Date mismatch
    s_dt, _ = _build_consistent_r3_bundle()
    s_dt[2].data.as_of_date = "2026-11-12"
    res_dt = check_r3_cross_section_facts_registry(s_dt)
    assert res_dt.status == "FAIL"
    assert any("Date mismatch" in d for d in res_dt.details)


def test_2_6_daily_report_populates_registry_and_passes_r3():
    """End-to-end daily generation populates CanonicalFactsRegistry and passes 15/15 QA checks including R3 and U1."""
    res = generate_market_report(report_type="daily", target_date="2026-11-13")
    assert res["status"] == "completed"
    assert res["qa_status"] == "PASSED"
    reg = res.get("facts_registry")
    assert isinstance(reg, CanonicalFactsRegistry)
    assert reg.as_of_date == "2026-11-13"
    assert reg.vix_current is not None and reg.vix_current > 0
    assert reg.net_new_highs is not None

    qa_text = Path(res["qa_report"]).read_text(encoding="utf-8")
    assert "15/15 checks passed" in qa_text
    assert "| **R3** |" in qa_text
    assert "| **U1** |" in qa_text
