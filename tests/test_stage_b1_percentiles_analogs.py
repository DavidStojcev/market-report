"""Tests for Stage B1: SentimenTrader 1Y Percentile Ranks & Historical Analog Base-Rate Study."""

from typing import Dict, List
import numpy as np
import pandas as pd

from config.settings import BreadthConfig, BreadthScoringConfig
from data.calendar import PeriodInfo
from data.constituents import Constituent
from indicators.breadth import (
    compute_market_breadth,
    format_percentile_ordinal,
    trailing_percentile_rank,
)
from render.html import render_html
from render.markdown import render_markdown
from sections.base import SectionResult
from sections.breadth import build_breadth_section
from sections.volatility import VolatilityData, build_volatility


def _make_daily_period(session_date: str, prev_date: str, prev_base: str) -> PeriodInfo:
    return PeriodInfo(
        report_type="daily",
        as_of_date=session_date,
        session_date=session_date,
        current_start=session_date,
        current_end=session_date,
        baseline_date=prev_date,
        prev_period_start=prev_date,
        prev_period_end=prev_date,
        prev_baseline_date=prev_base,
        is_partial=False,
        session_count=1,
    )


def test_trailing_percentile_rank_and_ordinal_formatting():
    """Verify percentile bounds, window clipping, min_obs gating, and ordinal suffixes."""
    short = pd.Series(np.linspace(10.0, 20.0, 40))
    assert trailing_percentile_rank(short, min_obs=60) is None

    ascending = pd.Series(np.arange(1, 301, dtype=float))
    hi = trailing_percentile_rank(ascending, window=252, min_obs=60)
    assert hi is not None and 99.0 <= hi <= 100.0

    descending = pd.Series(np.arange(300, 0, -1, dtype=float))
    lo = trailing_percentile_rank(descending, window=252, min_obs=60)
    assert lo is not None and 0.0 <= lo <= 1.0

    flat = pd.Series([50.0] * 100)
    assert trailing_percentile_rank(flat) == 50.0

    assert format_percentile_ordinal(None) == "—"
    assert format_percentile_ordinal(1.2) == "1st %ile"
    assert format_percentile_ordinal(2.1) == "2nd %ile"
    assert format_percentile_ordinal(3.4) == "3rd %ile"
    assert format_percentile_ordinal(11.0) == "11th %ile"
    assert format_percentile_ordinal(12.0) == "12th %ile"
    assert format_percentile_ordinal(13.0) == "13th %ile"
    assert format_percentile_ordinal(21.0) == "21st %ile"
    assert format_percentile_ordinal(84.4) == "84th %ile"


def test_breadth_percentiles_and_historical_analog_study():
    """Compute 1Y percentiles & leak-free non-overlapping Historical Analog Study and verify MD/HTML."""
    rng = np.random.default_rng(123)
    n_bars = 340
    dates = pd.bdate_range(end="2026-11-03", periods=n_bars)

    constituents: List[Constituent] = [
        Constituent(symbol=f"SYM{i}", name=f"Company {i}", sector="Technology", in_sp500=True)
        for i in range(12)
    ]

    # Oscillating market with repeated bullish and pulled-back breadth phases
    t = np.linspace(0.0, 6.0 * np.pi, n_bars)
    spy_rets = 0.0004 + 0.004 * np.sin(t) + rng.normal(0.0, 0.005, size=n_bars)
    spy_close = 500.0 * np.cumprod(1.0 + spy_rets)
    spy_df = pd.DataFrame({"Close": spy_close}, index=dates)
    rsp_df = pd.DataFrame({"Close": spy_close * (1.0 + 0.02 * np.cos(t))}, index=dates)
    vix_close = np.clip(18.5 - 3.5 * np.sin(t) + rng.normal(0.0, 0.8, size=n_bars), 12.0, 32.0)
    vix_df = pd.DataFrame({"Close": vix_close}, index=dates)

    equity_bars: Dict[str, pd.DataFrame] = {}
    for idx, c in enumerate(constituents):
        phase = idx * 0.25
        stock_rets = spy_rets + 0.003 * np.sin(t + phase) + rng.normal(0.0, 0.006, size=n_bars)
        close = 100.0 * np.cumprod(1.0 + stock_rets)
        equity_bars[c.symbol] = pd.DataFrame(
            {
                "Open": close * 0.998,
                "High": close * 1.006,
                "Low": close * 0.994,
                "Close": close,
                "Volume": 1_000_000,
            },
            index=dates,
        )

    period = _make_daily_period(
        session_date=dates[-1].strftime("%Y-%m-%d"),
        prev_date=dates[-2].strftime("%Y-%m-%d"),
        prev_base=dates[-3].strftime("%Y-%m-%d"),
    )
    cfg = BreadthConfig(
        enabled=True,
        min_history_bars=200,
        min_coverage_pct=80.0,
        mcclellan_enabled=True,
        scoring=BreadthScoringConfig(),
    )

    res = compute_market_breadth(
        equity_bars=equity_bars,
        constituents=constituents,
        period=period,
        cfg=cfg,
        spy_df=spy_df,
        rsp_df=rsp_df,
        vix_df=vix_df,
    )

    for attr in (
        "pct_above_50_dma_pctile_1y",
        "pct_above_200_dma_pctile_1y",
        "mcclellan_pctile_1y",
        "net_new_highs_pctile_1y",
    ):
        val = getattr(res, attr)
        assert val is not None, f"{attr} should be populated on 340-bar dataset"
        assert 0.0 <= val <= 100.0

    study = res.analog_study
    assert study is not None
    assert study.sample_count >= 3
    assert study.lookback_days > study.sample_count
    assert [h.horizon_days for h in study.horizons] == [5, 10, 20]
    for h in study.horizons:
        assert 0.0 <= h.win_rate_pct <= 100.0
        assert 0.0 <= h.baseline_win_rate_pct <= 100.0

    # Verify strict 5-session de-clustering and >= 20 trading-day forward isolation
    date_pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    analog_positions = [date_pos[d] for d in study.analog_dates]
    for a, b in zip(analog_positions, analog_positions[1:]):
        assert (b - a) >= 5
    assert analog_positions[-1] <= (len(dates) - 21)

    sec_breadth = build_breadth_section(breadth_result=res, section_num=6)
    md = render_markdown(period, [sec_breadth])
    html = render_html(period, [sec_breadth])

    assert "1Y %ile" in md
    assert "Historical Analog Base-Rate Study" in md
    assert "+20 Trading Days (SPY)" in md

    assert "1Y %ile" in html
    assert "Historical Analog Base-Rate Study" in html
    assert "+20 Trading Days (SPY)" in html


def test_volatility_1y_percentiles_and_short_history_fallback():
    """Verify VIX/VVIX/SKEW/VRP 1Y percentiles populate on >=60 bars and return None on short bars."""
    rng = np.random.default_rng(99)
    n_long = 140
    dates_long = pd.bdate_range(end="2026-11-03", periods=n_long)
    spx_close = 6000.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.008, size=n_long))
    vix_vals = np.linspace(14.0, 24.5, n_long)
    vvix_vals = np.linspace(82.0, 112.0, n_long)
    skew_vals = np.linspace(125.0, 148.0, n_long)

    bars_long = {
        "^GSPC": pd.DataFrame({"Close": spx_close}, index=dates_long),
        "^VIX": pd.DataFrame({"Close": vix_vals, "High": vix_vals + 0.5, "Low": vix_vals - 0.5}, index=dates_long),
        "^VVIX": pd.DataFrame({"Close": vvix_vals}, index=dates_long),
        "^SKEW": pd.DataFrame({"Close": skew_vals}, index=dates_long),
    }
    period = _make_daily_period(
        session_date=dates_long[-1].strftime("%Y-%m-%d"),
        prev_date=dates_long[-2].strftime("%Y-%m-%d"),
        prev_base=dates_long[-3].strftime("%Y-%m-%d"),
    )

    sec_vol = build_volatility(bars=bars_long, vix_symbol="^VIX", period=period, spx_df=bars_long["^GSPC"])
    assert sec_vol.status == "available"
    vd: VolatilityData = sec_vol.data

    assert vd.vix_pctile_1y is not None and vd.vix_pctile_1y > 95.0
    assert vd.vvix_pctile_1y is not None and vd.vvix_pctile_1y > 95.0
    assert vd.skew_pctile_1y is not None and vd.skew_pctile_1y > 95.0
    assert vd.vrp_pctile_1y is not None and 0.0 <= vd.vrp_pctile_1y <= 100.0
    assert "1Y percentile" in vd.surface_synthesis

    md = render_markdown(period, [sec_vol])
    html = render_html(period, [sec_vol])
    assert "1Y Rank:" in md
    assert "1Y Rank:" in html

    # Short history (<60 bars) must gracefully omit 1Y percentiles without raising
    dates_short = dates_long[-30:]
    bars_short = {k: v.loc[dates_short] for k, v in bars_long.items()}
    sec_short = build_volatility(bars=bars_short, vix_symbol="^VIX", period=period, spx_df=bars_short["^GSPC"])
    vd_short: VolatilityData = sec_short.data
    assert vd_short.vix_pctile_1y is None
    assert vd_short.vvix_pctile_1y is None
    assert vd_short.vrp_pctile_1y is None
