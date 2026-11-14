"""Stage 0 Integrity & Credibility Tests (Substages 0.1 - 0.7 & R1/R2 QA Checks)."""

import numpy as np
import pandas as pd

from config.settings import load_config
from data.cache import ParquetCache, _repair_split_discontinuities
from data.calendar import PeriodInfo
from data.calendar_provider import OfficialScheduleCalendarProvider
from indicators.breadth import format_percentile_ordinal
from qa_checks import check_r1_arithmetic_reconciliation, check_r2_calendar_chronology
from sections.base import SectionResult
from sections.earnings import EarningsLayerData, build_earnings_layer
from sections.forward_calendar import build_forward_calendar
from sections.macro import (
    FedRatesSummary,
    MacroMetricSnapshot,
    MacroSectionData,
    MarketBasedIndicatorRow,
    build_standing_macro_dashboard,
    resolve_next_release_date,
    resolve_release_metadata,
)
from sections.volatility import build_volatility
from sections.watchlist import build_watchlist_section
from sections.what_changed import build_what_changed


def test_0_1_no_hardcoded_earnings_constants():
    """0.1: Earnings layer never injects 68.0/78.4/64.2/4.3 or known_reactions fallbacks."""
    # 1. When provider is empty/None, beat rates must remain None (not 68.0/78.4/64.2/4.3)
    res_empty = build_earnings_layer(
        calendar_provider=None,
        constituents=[],
        ticker_infos={},
        as_of_date="2026-11-13",
        cfg=None,
        bars={},
    )
    assert res_empty.data.pct_reported is None
    assert res_empty.data.eps_beat_rate is None
    assert res_empty.data.rev_beat_rate is None
    assert res_empty.data.median_surprise_pct is None
    assert "n/a" in res_empty.data.status_note.lower()

    # 2. When provider has reported items but bars is empty, reaction_1d_pct stays None (no known_reactions fallback)
    cal = OfficialScheduleCalendarProvider()
    res_no_bars = build_earnings_layer(
        calendar_provider=cal,
        constituents=[],
        ticker_infos={},
        as_of_date="2026-10-30",
        cfg=None,
        bars={},
        report_type="weekly",
    )
    msft_no_bars = next(r for r in res_no_bars.data.reported_last_7_days if r.symbol == "MSFT")
    assert msft_no_bars.reaction_1d_pct is None

    # Dynamic stats reflect actual provider sample rather than hardcoded 68.0 / 78.4 / 64.2 / 4.3
    assert res_no_bars.data.eps_beat_rate is not None
    assert res_no_bars.data.eps_beat_rate != 78.4


def test_0_2_dynamic_calendar_after_nov_12_and_nvda_earnings():
    """0.2: Calendar populates post-Nov-12 events honestly, and NVDA resolves within 30d."""
    cfg = load_config()
    cal = OfficialScheduleCalendarProvider()

    events = cal.get_upcoming_events("2026-11-13", "2026-11-23", min_importance="medium")
    assert len(events) >= 3
    titles = [e.title for e in events]
    assert any("Retail Sales" in t for t in titles)
    assert any("Jobless Claims" in t for t in titles)
    # Unpolled official schedule entries after Nov 13 must not invent consensus forecasts
    for ev in events:
        if ev.forecast is None:
            assert ev.consensus_source == "—"

    # Dynamic rule generation works into 2027 without static exhaustion
    events_2027 = cal.get_upcoming_events("2027-02-28", "2027-03-31", min_importance="medium")
    assert any("ISM Manufacturing PMI" in e.title for e in events_2027)
    assert any("Employment Situation" in e.title for e in events_2027)
    assert any("Initial Jobless Claims" in e.title for e in events_2027)

    # Forward calendar data_source excludes Finnhub when no Finnhub consensus was fetched
    fc_res = build_forward_calendar(cal, as_of_date="2026-11-13", cfg=cfg, lookahead_days=7)
    assert "Finnhub" not in fc_res.data.data_source

    # NVDA earnings resolve on 2026-11-18 within 30d of 2026-11-13
    wl_res = build_watchlist_section(
        calendar_provider=cal,
        as_of_date="2026-11-13",
        custom_tickers=["NVDA", "AAPL"],
        bars={},
    )
    nvda_item = next(it for it in wl_res.data.items if it.ticker == "NVDA")
    assert nvda_item.next_earnings_date == "2026-11-18"


def test_0_3_dynamic_ism_and_0_6_release_chronology():
    """0.3 & 0.6: All Standing Macro Dashboard rows satisfy release_date <= as_of_date < next_release_date."""
    cal = OfficialScheduleCalendarProvider()
    as_of = "2026-11-13"

    # Check PPIFIS 2026-10-01 -> would have been Saturday 2026-11-14 without weekend roll
    _, ppi_rel = resolve_release_metadata("PPIFIS", "2026-10-01", as_of_date=as_of)
    assert ppi_rel == "2026-11-13"
    assert pd.to_datetime(ppi_rel).weekday() < 5

    # Check ICSA clamp when candidate would exceed as_of_date
    _, icsa_rel = resolve_release_metadata("ICSA", "2026-11-09", as_of_date=as_of)
    assert icsa_rel <= as_of
    assert pd.to_datetime(icsa_rel).weekday() < 5

    # Build dashboard with realistic snapshots and verify chronology on ALL rows (incl. ISM, JOLTS, UMCSENT)
    metrics = {
        "cpi": MacroMetricSnapshot("CPIAUCSL", "CPI", "2026-10-01", 320.0, 2.6, "% YoY", prior_value=319.0, prior_headline=2.5),
        "core_cpi": MacroMetricSnapshot("CPILFESL", "Core CPI", "2026-10-01", 330.0, 3.3, "% YoY", prior_value=329.0, prior_headline=3.3),
        "ppi": MacroMetricSnapshot("PPIFIS", "PPI", "2026-10-01", 145.0, 2.4, "% YoY", prior_value=144.5, prior_headline=2.3),
        "core_pce": MacroMetricSnapshot("PCEPILFE", "Core PCE", "2026-09-01", 125.0, 2.7, "% YoY", prior_value=124.7, prior_headline=2.7),
        "payrolls": MacroMetricSnapshot("PAYEMS", "Payrolls", "2026-10-01", 159000.0, 120.0, "k", prior_value=158880.0, prior_headline=140.0),
        "unemployment": MacroMetricSnapshot("UNRATE", "Unemployment", "2026-10-01", 4.1, 4.1, "%", prior_value=4.1, prior_headline=4.1),
        "jobless_claims": MacroMetricSnapshot("ICSA", "Claims", "2026-11-07", 215000.0, 215000.0, "k", prior_value=218000.0, prior_headline=218000.0),
        "jolts": MacroMetricSnapshot("JTSJOL", "JOLTS", "2026-09-01", 7400.0, 7400.0, "k", prior_value=7500.0, prior_headline=7500.0),
        "gdp": MacroMetricSnapshot("GDPC1", "Real GDP", "2026-07-01", 23400.0, 2.8, "% QoQ Ann.", prior_value=23240.0, prior_headline=3.0),
        "retail_sales": MacroMetricSnapshot("RSAFS", "Retail Sales", "2026-09-01", 715000.0, 0.4, "%", prior_value=712000.0, prior_headline=0.2),
        "sentiment": MacroMetricSnapshot("UMCSENT", "Sentiment", "2026-10-01", 70.5, 70.5, "index", prior_value=70.1, prior_headline=70.1),
    }
    dash_rows, _ = build_standing_macro_dashboard(
        metrics=metrics,
        dataframes={},
        as_of_date=as_of,
        calendar_provider=cal,
    )
    ism_mfg = next(r for r in dash_rows if r.indicator == "ISM Manufacturing PMI")
    assert ism_mfg.release_date == "2026-11-02"
    assert ism_mfg.reference_period == "Oct 2026"
    assert ism_mfg.next_release_date > as_of

    sec = SectionResult(
        section_num=4,
        title="Macro & Economic Events",
        status="available",
        data=MacroSectionData(
            recent_releases=[],
            fed_rates=FedRatesSummary(4.5, "4.50%-4.75%", "2026-10-28", "2026-12-09", 26, 4.86, 0.05, 5.10, 0.08, 0.24, 0.03, False, "Modestly Steep"),
            cross_assets=[],
            upcoming_events=[],
            upcoming_earnings=[],
            metrics=metrics,
            standing_dashboard=dash_rows,
        ),
    )
    r2_errors = check_r2_calendar_chronology([sec], as_of_date=as_of)
    assert r2_errors == []


def test_0_4_split_discontinuity_repair():
    """0.4: Unadjusted split jumps (like ITA 3:1) are repaired; futures (=F) are untouched."""
    dates = pd.date_range("2026-10-20", periods=6, freq="B")
    df_split = pd.DataFrame(
        {
            "Open": [205.0, 206.0, 207.0, 70.0, 71.0, 72.0],
            "High": [208.0, 208.0, 209.0, 72.0, 72.5, 73.0],
            "Low": [204.0, 205.0, 205.0, 69.5, 70.0, 71.0],
            "Close": [206.0, 207.0, 207.74, 71.43, 71.90, 72.10],
            "Volume": [700000.0, 800000.0, 677500.0, 3442800.0, 3100000.0, 3000000.0],
        },
        index=dates,
    )
    repaired = _repair_split_discontinuities(df_split, ticker="ITA")
    assert 65.0 < float(repaired["Close"].iloc[0]) < 72.0
    assert float(repaired["Volume"].iloc[2]) == 677500.0 * 3.0

    # Futures rollover (=F) must never be split-adjusted
    df_ng = pd.DataFrame(
        {"Close": [7.46, 7.40, 7.46, 3.92, 3.95, 4.00], "Volume": [1e5] * 6},
        index=dates,
    )
    ng_out = _repair_split_discontinuities(df_ng, ticker="NG=F")
    assert float(ng_out["Close"].iloc[0]) == 7.46

    # On-disk ITA parquet cache (if present) should show realistic 1M move
    cache = ParquetCache(".cache")
    ita_disk = cache.read_bars("ITA", end_date="2026-11-13")
    if ita_disk is not None and len(ita_disk) >= 22:
        ret_1m = (float(ita_disk["Close"].iloc[-1]) / float(ita_disk["Close"].iloc[-22]) - 1.0) * 100.0
        assert abs(ret_1m) < 25.0


def test_0_5_market_indicator_and_yield_curve_arithmetic():
    """0.5: Market-Based Indicators table satisfies Latest - Prior == Period Change bps and passes R1."""
    metrics = {
        "hy_credit_spread": MacroMetricSnapshot(
            series_id="BAMLH0A0HYM2",
            name="HY OAS",
            latest_date="2026-11-13",
            latest_value=3.014,
            headline_value=3.014,
            headline_unit="%",
            prior_value=3.030,       # 1-day prior (differs from period_start_value)
            prior_headline=3.030,
            period_start_value=3.144,  # 5-day baseline prior
            period_change=-0.130,
        ),
        "real_yield_10y": MacroMetricSnapshot(
            series_id="DFII10",
            name="TIPS 10Y",
            latest_date="2026-11-13",
            latest_value=2.12,
            headline_value=2.12,
            headline_unit="%",
            prior_value=2.11,
            prior_headline=2.11,
            period_start_value=2.07,
            period_change=0.05,
        ),
    }
    _, mkt_rows = build_standing_macro_dashboard(metrics=metrics, dataframes={}, as_of_date="2026-11-13")
    hy_row = next(r for r in mkt_rows if r.series_id == "BAMLH0A0HYM2")
    assert hy_row.latest_level == "3.01% (301 bps)"
    assert hy_row.prior_level == "3.14% (314 bps)"
    assert hy_row.period_change == "-13 bps"

    sec = SectionResult(
        section_num=4,
        title="Macro & Economic Events",
        status="available",
        data=MacroSectionData(
            recent_releases=[],
            fed_rates=FedRatesSummary(4.5, "4.50%-4.75%", "2026-10-28", "2026-12-09", 26, 4.86, 0.04, 5.10, 0.08, 0.24, 0.04, False, "Modestly Steep"),
            cross_assets=[],
            upcoming_events=[],
            upcoming_earnings=[],
            metrics=metrics,
            market_indicators=mkt_rows,
        ),
    )
    assert check_r1_arithmetic_reconciliation([sec]) == []


def test_0_7_ordinals_xntk_and_fomc_minutes_importance():
    """0.7: Ordinals use 1st/2nd/3rd, XNTK is Tech Breakthrough (NYSE), FOMC Minutes stays MEDIUM."""
    assert format_percentile_ordinal(1.0) == "1st %ile"
    assert format_percentile_ordinal(2.0) == "2nd %ile"
    assert format_percentile_ordinal(3.0) == "3rd %ile"
    assert format_percentile_ordinal(11.0) == "11th %ile"

    # Build volatility with extreme low VIX so percentile ranks at 1st/0th
    dates = pd.date_range("2025-10-01", "2026-11-13", freq="B")
    vix_vals = np.linspace(30.0, 12.0, len(dates))
    vix_df = pd.DataFrame({"Close": vix_vals}, index=dates)
    spx_df = pd.DataFrame({"Close": np.linspace(5000.0, 6000.0, len(dates))}, index=dates)
    period = PeriodInfo(
        report_type="daily",
        as_of_date="2026-11-13",
        session_date="2026-11-13",
        current_start="2026-11-13",
        current_end="2026-11-13",
        baseline_date="2026-11-12",
        prev_period_start="2026-11-12",
        prev_period_end="2026-11-12",
        prev_baseline_date="2026-11-11",
        is_partial=False,
        session_count=1,
    )
    vol_res = build_volatility({"^VIX": vix_df}, "^VIX", period, spx_df=spx_df)
    assert "1th" not in vol_res.data.surface_synthesis
    assert "2th" not in vol_res.data.surface_synthesis
    assert "3th" not in vol_res.data.surface_synthesis
    assert "%ile 1Y" not in vol_res.data.surface_synthesis

    # XNTK name in config.yaml
    cfg = load_config()
    xntk = next(it for it in cfg.theme_etfs.items if it.symbol == "XNTK")
    assert xntk.name == "Tech Breakthrough (NYSE)"

    # FOMC Minutes must NOT be promoted to High-Impact in what_changed when marked medium
    cal = OfficialScheduleCalendarProvider()
    events = cal.get_upcoming_events("2026-11-13", "2026-11-23", min_importance="medium")
    minutes_only = [e for e in events if "minutes" in e.title.lower()]
    assert len(minutes_only) == 1
    assert minutes_only[0].importance == "medium"

    wc_res = build_what_changed(
        scorecard=None,
        prior_scorecard=None,
        macro_data=None,
        breadth_res=None,
        sector_rotation=None,
        spx_snap=None,
        ndx_snap=None,
        as_of_date="2026-11-13",
        upcoming_events=minutes_only,
    )
    assert "FOMC Minutes" not in wc_res.data.single_biggest_risk
