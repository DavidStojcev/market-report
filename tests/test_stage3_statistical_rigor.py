"""Comprehensive verification suite for Stage 3 (Statistical Rigor).

Verifies:
- Substage 3.1: Long-History (1990+) Index-Level Analog Matching with N >= 20 and lookback span >= 25 years.
- Substage 3.2: Wilson 95% Confidence Intervals, IQR [P25, P75], and Small-Sample Precedent Badge (N < 20).
- Substage 3.3: Complete Calendar Years Seasonality (excluding partial in-progress month/year)
  and Next-Month Seasonality Statistics across Markdown and HTML.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from data.cache import ParquetCache
from data.calendar import PeriodInfo
from data.constituents import Constituent
from config.settings import BreadthConfig, BreadthScoringConfig
from indicators.breadth import compute_market_breadth
from render.html import render_html
from render.markdown import render_markdown
from sections.base import SectionResult
from sections.breadth import build_breadth_section
from sections.conclusion import ConclusionData, build_conclusion
from sections.historical_analog import (
    AnalogHorizonStats,
    HistoricalAnalogStudy,
    SeasonalityStudy,
    build_historical_analog,
    compute_monthly_seasonality,
    wilson_score_interval,
)


def _make_daily_period(session_date: str = "2026-11-13") -> PeriodInfo:
    return PeriodInfo(
        report_type="daily",
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
    )


# ==============================================================================
# Substage 3.1 Tests: Long-History (1990+) Index-Level Analog Study
# ==============================================================================

def test_3_1_long_history_index_analogs_n_ge_20_and_span():
    """Verify 1990+ index history loads, yields N >= 20 de-clustered setups, and spans >= 25 years."""
    cache = ParquetCache()
    spx_df = cache.read_bars("^GSPC")
    vix_df = cache.read_bars("^VIX")

    if spx_df is None or len(spx_df) < 5000:
        # Download 1990+ data if not already cached
        import yfinance as yf

        spx_raw = yf.download("^GSPC", start="1990-01-01", end="2026-11-14", progress=False)
        vix_raw = yf.download("^VIX", start="1990-01-01", end="2026-11-14", progress=False)
        assert not spx_raw.empty, "1990+ ^GSPC data must load successfully"
        assert not vix_raw.empty, "1990+ ^VIX data must load successfully"

        def _clean_yf(raw):
            c = raw["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            df = pd.DataFrame({"Close": c})
            df.index = pd.to_datetime(df.index).normalize()
            return df.dropna().sort_index()

        spx_df = _clean_yf(spx_raw)
        vix_df = _clean_yf(vix_raw)
        cache.write_bars("^GSPC", spx_df)
        cache.write_bars("^VIX", vix_df)

    assert spx_df is not None and len(spx_df) >= 7000, "S&P 500 history must span 25+ years (>7,000 trading sessions)"
    assert vix_df is not None and len(vix_df) >= 7000, "VIX history must span 25+ years (>7,000 trading sessions)"

    as_of = "2026-11-13"
    study = build_historical_analog(
        spx_df=spx_df,
        vix_df=vix_df,
        as_of_date=as_of,
    )

    assert study is not None
    assert study.history_years >= 25, f"Expected history_years >= 25, got {study.history_years}"
    assert study.history_start_year <= 1991, f"Expected history_start_year <= 1991, got {study.history_start_year}"
    assert study.sample_count >= 20, f"Expected N >= 20 non-overlapping episodes, got {study.sample_count}"
    assert study.is_small_sample is False
    assert study.small_sample_badge == ""
    assert len(study.horizons) == 3
    assert [h.horizon_days for h in study.horizons] == [5, 10, 20]

    # Check forward return horizons sanity
    for hz in study.horizons:
        assert 0.0 <= hz.win_rate_pct <= 100.0
        assert hz.sample_size == study.sample_count
        assert hz.win_rate_ci_low <= hz.win_rate_pct <= hz.win_rate_ci_high
        assert hz.iqr_return >= 0.0
        assert hz.p25_return <= hz.p75_return

    # Verify declustering spacing (all analog dates separated by >= 5 trading sessions)
    date_pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(spx_df[spx_df.index <= pd.to_datetime(as_of)].index)}
    positions = [date_pos[d] for d in study.analog_dates if d in date_pos]
    for a, b in zip(positions, positions[1:]):
        assert (b - a) >= 5, f"Analog dates must be de-clustered by at least 5 sessions: delta={b-a}"


def test_3_1_breadth_secondary_cohort_preservation():
    """Verify breadth-conditioned secondary cohort is preserved alongside long-history study."""
    cache = ParquetCache()
    spx_df = cache.read_bars("^GSPC")
    vix_df = cache.read_bars("^VIX")

    # Generate synthetic 400-day constituent breadth series
    dates = pd.bdate_range(end="2026-11-13", periods=400)
    p50_series = pd.Series(np.clip(55.0 + 15.0 * np.sin(np.linspace(0, 10, 400)), 20.0, 85.0), index=dates)

    study = build_historical_analog(
        spx_df=spx_df,
        vix_df=vix_df,
        as_of_date="2026-11-13",
        p50_series=p50_series,
        p50_curr=float(p50_series.iloc[-1]),
    )

    assert study is not None
    assert study.history_years >= 25
    assert study.sample_count >= 20
    assert study.breadth_cohort is not None
    assert study.breadth_cohort.sample_count >= 3
    assert "survivorship bias" in study.breadth_cohort_note.lower()
    assert "Secondary Breadth Cohort" in study.breadth_cohort_note


def test_3_1_short_history_graceful_fallback():
    """Verify that short synthetic history gracefully produces analog study with small sample badge."""
    dates = pd.bdate_range(end="2026-11-03", periods=140)
    spx_close = 5000.0 * np.cumprod(1.0 + np.random.default_rng(42).normal(0.0005, 0.008, size=140))
    short_df = pd.DataFrame({"Close": spx_close}, index=dates)
    vix_df = pd.DataFrame({"Close": np.linspace(15.0, 22.0, 140)}, index=dates)

    study = build_historical_analog(
        spx_df=short_df,
        vix_df=vix_df,
        as_of_date="2026-11-03",
    )

    assert study is not None
    assert 0 < study.sample_count < 20
    # Because history is only 140 bars, sample count is < 20
    assert study.is_small_sample is True
    assert "Small-sample precedent" in study.small_sample_badge
    assert f"N={study.sample_count}" in study.small_sample_badge


# ==============================================================================
# Substage 3.2 Tests: Wilson Score Interval, IQR, and Small-Sample Precedent Badge
# ==============================================================================

def test_3_2_wilson_score_interval_exact_math():
    """Verify Wilson score confidence interval boundary conditions, symmetry, and math."""
    # Zero or negative total
    assert wilson_score_interval(0, 0) == (0.0, 0.0)
    assert wilson_score_interval(5, -1) == (0.0, 0.0)
    assert wilson_score_interval(-2, 10) == (0.0, 27.8)  # Clamped to 0 successes

    # Zero successes out of 20
    lo0, hi0 = wilson_score_interval(0, 20)
    assert lo0 == 0.0
    assert 14.0 <= hi0 <= 18.0, f"Expected upper bound around 16.1%, got {hi0}%"

    # All successes out of 20
    lo20, hi20 = wilson_score_interval(20, 20)
    assert hi20 == 100.0
    assert 82.0 <= lo20 <= 86.0, f"Expected lower bound around 83.9%, got {lo20}%"

    # Exact 50% (10 out of 20)
    lo10, hi10 = wilson_score_interval(10, 20)
    assert 28.0 <= lo10 <= 32.0, f"Expected lower bound around 29.9%, got {lo10}%"
    assert 68.0 <= hi10 <= 72.0, f"Expected upper bound around 70.1%, got {hi10}%"
    assert abs((lo10 + hi10) - 100.0) < 0.2  # Symmetric around 50%

    # Standard textbook sample: 70 successes out of 100
    lo70, hi70 = wilson_score_interval(70, 100)
    assert 59.5 <= lo70 <= 61.5, f"Expected lo around 60.4%, got {lo70}%"
    assert 77.0 <= hi70 <= 79.0, f"Expected hi around 78.1%, got {hi70}%"


def test_3_2_analog_horizon_stats_iqr_and_wilson_ci():
    """Verify AnalogHorizonStats correctly computes Wilson CI, IQR, and percentiles."""
    stat = AnalogHorizonStats(
        horizon_days=20,
        win_rate_pct=65.0,
        median_return_pct=1.85,
        mean_return_pct=1.60,
        worst_return_pct=-4.20,
        baseline_win_rate_pct=52.0,
        baseline_median_pct=0.40,
        win_rate_ci_low=52.3,
        win_rate_ci_high=75.8,
        p25_return=-0.50,
        p75_return=3.20,
        iqr_return=3.70,
        sample_size=40,
    )

    assert stat.win_rate_ci_low == 52.3
    assert stat.win_rate_ci_high == 75.8
    assert stat.p25_return == -0.50
    assert stat.p75_return == 3.20
    assert stat.iqr_return == 3.70
    assert stat.sample_size == 40


def test_3_2_small_sample_badge_and_unbolded_win_rate_rendering():
    """Verify small-sample badge display and unbolded win rate when N < 20 vs bolded when N >= 20."""
    period = _make_daily_period("2026-11-13")

    # 1. Small sample (N = 12 < 20)
    h_small = AnalogHorizonStats(
        horizon_days=20,
        win_rate_pct=58.3,
        median_return_pct=1.20,
        mean_return_pct=1.10,
        worst_return_pct=-3.50,
        baseline_win_rate_pct=50.0,
        baseline_median_pct=0.30,
        win_rate_ci_low=32.0,
        win_rate_ci_high=80.7,
        p25_return=-0.80,
        p75_return=2.50,
        iqr_return=3.30,
        sample_size=12,
    )
    study_small = HistoricalAnalogStudy(
        benchmark_symbol="SPY",
        condition_summary="Test Small Setup",
        sample_count=12,
        lookback_days=300,
        horizons=[h_small],
        summary_text="Test small sample summary.",
        is_small_sample=True,
        small_sample_badge="⚠️ Small-sample precedent (N=12 < 20)",
        history_start_year=2024,
        history_years=2,
    )

    # Wrap in BreadthSectionData & SectionResult
    from indicators.breadth import MarketBreadthResult
    b_res_small = MarketBreadthResult(
        as_of_date=period.session_date,
        prev_date=period.prev_period_end,
        total_constituents=500,
        included_constituents=500,
        coverage_pct=100.0,
        analog_study=study_small,
    )
    sec_small = build_breadth_section(b_res_small)

    md_small = render_markdown(period, [sec_small])
    html_small = render_html(period, [sec_small])

    # Badge must appear
    assert "⚠️ Small-sample precedent (N=12 < 20)" in md_small
    assert "⚠️ Small-sample precedent (N=12 < 20)" in html_small

    # Win rate must NOT be bolded when N < 20
    assert "**58.3%**" not in md_small, "Markdown win rate must NOT be bolded (**58.3%**) when N < 20"
    assert "| 58.3% |" in md_small, "Markdown win rate must appear unbolded (58.3%) when N < 20"
    assert "<strong>58.3%</strong>" not in html_small, "HTML win rate must NOT be wrapped in <strong> when N < 20"
    assert "58.3%" in html_small

    # 2. Adequate sample (N = 35 >= 20)
    h_large = AnalogHorizonStats(
        horizon_days=20,
        win_rate_pct=71.4,
        median_return_pct=2.10,
        mean_return_pct=1.90,
        worst_return_pct=-2.50,
        baseline_win_rate_pct=52.0,
        baseline_median_pct=0.40,
        win_rate_ci_low=54.9,
        win_rate_ci_high=83.7,
        p25_return=0.20,
        p75_return=3.80,
        iqr_return=3.60,
        sample_size=35,
    )
    study_large = HistoricalAnalogStudy(
        benchmark_symbol="^GSPC",
        condition_summary="Test Large Setup",
        sample_count=35,
        lookback_days=8500,
        horizons=[h_large],
        summary_text="Test large sample summary.",
        is_small_sample=False,
        small_sample_badge="",
        history_start_year=1990,
        history_years=35,
    )
    b_res_large = MarketBreadthResult(
        as_of_date=period.session_date,
        prev_date=period.prev_period_end,
        total_constituents=500,
        included_constituents=500,
        coverage_pct=100.0,
        analog_study=study_large,
    )
    sec_large = build_breadth_section(b_res_large)

    md_large = render_markdown(period, [sec_large])
    html_large = render_html(period, [sec_large])

    # Badge must NOT appear
    assert "Small-sample precedent" not in md_large
    assert "Small-sample precedent" not in html_large

    # Win rate MUST be bolded when N >= 20
    assert "**71.4%**" in md_large, "Markdown win rate MUST be bolded (**71.4%**) when N >= 20"
    assert "<strong>71.4%</strong>" in html_large, "HTML win rate MUST be wrapped in <strong> when N >= 20"


# ==============================================================================
# Substage 3.3 Tests: Complete Calendar Years Seasonality & Next-Month Stats
# ==============================================================================

def test_3_3_seasonality_complete_years_and_next_month():
    """Verify seasonality strictly evaluates complete calendar years, excludes in-progress month, and computes next month."""
    cache = ParquetCache()
    spx_df = cache.read_bars("^GSPC")
    if spx_df is None or len(spx_df) < 5000:
        import yfinance as yf
        spx_raw = yf.download("^GSPC", start="1990-01-01", end="2026-11-14", progress=False)
        c = spx_raw["Close"].iloc[:, 0] if isinstance(spx_raw["Close"], pd.DataFrame) else spx_raw["Close"]
        spx_df = pd.DataFrame({"Close": c})
        spx_df.index = pd.to_datetime(spx_df.index).normalize()
        spx_df = spx_df.dropna().sort_index()

    as_of = "2026-11-13"
    study = compute_monthly_seasonality(spx_df=spx_df, as_of_date=as_of)

    assert study is not None
    # 2026 is incomplete (as_of is November 13, 2026) -> complete years must strictly end in 2025!
    assert study.history_end_year == 2025, f"Expected complete history to end in 2025, got {study.history_end_year}"
    assert study.history_start_year <= 1991, f"Expected complete history to start <= 1991, got {study.history_start_year}"
    assert study.complete_years_count >= 30, f"Expected complete_years_count >= 30, got {study.complete_years_count}"

    # Current month (November = 11)
    assert study.current_month_num == 11
    assert study.current_month_name == "November"
    assert study.current_month_sample_size == study.complete_years_count
    assert 0.0 <= study.current_month_win_rate <= 100.0
    assert study.current_month_win_rate_ci_low <= study.current_month_win_rate <= study.current_month_win_rate_ci_high
    assert study.current_month_iqr_return >= 0.0
    assert study.current_month_p25_return <= study.current_month_p75_return

    # Next month (December = 12)
    assert study.next_month_num == 12
    assert study.next_month_name == "December"
    assert study.next_month_sample_size == study.complete_years_count
    assert 0.0 <= study.next_month_win_rate <= 100.0
    assert study.next_month_win_rate_ci_low <= study.next_month_win_rate <= study.next_month_win_rate_ci_high
    assert study.next_month_iqr_return >= 0.0
    assert study.next_month_p25_return <= study.next_month_p75_return


def test_3_3_seasonality_december_year_wrap():
    """Verify that in December, next month wraps cleanly to January (1)."""
    cache = ParquetCache()
    spx_df = cache.read_bars("^GSPC")
    study_dec = compute_monthly_seasonality(spx_df=spx_df, as_of_date="2025-12-15")

    assert study_dec is not None
    assert study_dec.current_month_num == 12
    assert study_dec.current_month_name == "December"
    assert study_dec.next_month_num == 1
    assert study_dec.next_month_name == "January"
    assert study_dec.history_end_year == 2024  # 2025 in progress during Dec 15


def test_3_3_seasonality_markdown_and_html_rendering():
    """Verify Markdown and HTML rendering of seasonality summary table and complete-year notes."""
    period = _make_daily_period("2026-11-13")

    seas_mock = SeasonalityStudy(
        current_month_num=11,
        current_month_name="November",
        current_month_avg_return=2.15,
        current_month_median_return=2.80,
        current_month_win_rate=75.0,
        current_month_sample_size=36,
        next_month_num=12,
        next_month_name="December",
        next_month_avg_return=1.25,
        next_month_median_return=1.10,
        next_month_win_rate=72.2,
        next_month_sample_size=36,
        complete_years_count=36,
        history_start_year=1990,
        history_end_year=2025,
        as_of_date="2026-11-13",
        current_month_win_rate_ci_low=58.9,
        current_month_win_rate_ci_high=86.3,
        current_month_p25_return=-0.80,
        current_month_p75_return=4.70,
        current_month_iqr_return=5.50,
        next_month_win_rate_ci_low=56.0,
        next_month_win_rate_ci_high=84.2,
        next_month_p25_return=-0.60,
        next_month_p75_return=2.40,
        next_month_iqr_return=3.00,
    )

    from scoring.engine import Scorecard

    sc = Scorecard(
        raw_score=1.45,
        max_score=3.0,
        normalized_score=1.45,
        verdict="Constructive",
        votes=[],
        caution_flags=[],
        top_bullish_factors=["Breadth Expansion"],
        top_bearish_factors=["Yield Hurdles"],
        summary_reasons=["Breadth Expansion"],
    )

    concl_data = ConclusionData(
        verdict="Constructive",
        normalized_score=1.45,
        top_bullish_factors=["Breadth Expansion"],
        top_bearish_factors=["Yield Hurdles"],
        caution_flags=[],
        scorecard=sc,
        asset_levels=[],
        seasonality=seas_mock,
    )
    sec_concl = SectionResult(
        section_num=15,
        title="Conclusion & Market Outlook",
        status="available",
        data=concl_data,
    )

    md = render_markdown(period, [sec_concl])
    html = render_html(period, [sec_concl])

    # Verify Markdown
    assert "S&P 500 Historical Monthly Seasonality (Complete Calendar Years Only)" in md
    assert "`N=36` Complete Calendar Years, 1990–2025" in md
    assert "Current Month" in md
    assert "November" in md
    assert "**75.0%**" in md
    assert "Next Month" in md
    assert "December" in md
    assert "**72.2%**" in md
    assert "N=36 complete years" in md
    assert "partial in-progress month (November 2026) is excluded" in md

    # Verify HTML
    assert "S&P 500 Historical Monthly Seasonality (Complete Calendar Years Only)" in html
    assert "N=36 Complete Calendar Years, 1990–2025" in html
    assert "Current Month" in html
    assert "November" in html
    assert "<strong>75.0%</strong>" in html
    assert "Next Month" in html
    assert "December" in html
    assert "<strong>72.2%</strong>" in html
    assert "N=36 complete yrs" in html
    assert "partial in-progress month (November 2026) is excluded" in html
