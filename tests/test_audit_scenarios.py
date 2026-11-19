"""Audit regression scenarios: synthetic data run through the real calculation / verdict code.

Each test pins a failure mode found in the data-correctness & market-impression audit:
wrong data (stale, placeholder, corporate-action artifacts, period anchors) or a misleading
read (verdict conflicts, mislabelled movers, invented FOMC outcomes). No network access.
"""
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config.settings import load_config
from data.cache import ParquetCache
from data.calendar import MarketCalendar
from data.constituents import Constituent
from data.fomc import resolve_fomc_schedule
from data.fred_provider import FredProvider
from indicators.breadth import compute_market_breadth
from indicators.engine import compute_technicals
from scoring.engine import TrendScoringEngine
from sections.movers import build_movers

CFG = load_config()
CAL = MarketCalendar()
AS_OF = "2026-11-13"


def _bars(closes, end=AS_OF, vol=2e6) -> pd.DataFrame:
    idx = CAL.cal.sessions_in_range(pd.Timestamp("2022-01-01"), pd.Timestamp(end))[-len(closes):]
    idx = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tzinfo else pd.Timestamp(d) for d in idx])
    c = np.asarray(closes, float)
    return pd.DataFrame({"Open": c, "High": c * 1.003, "Low": c * 0.997, "Close": c, "Volume": np.full(len(c), vol)}, index=idx)


def _path(n, drift, vol, seed):
    rng = np.random.default_rng(seed)
    return 4000 * np.cumprod(1 + drift + vol * rng.standard_normal(n))


def _scorecard(spx, ndx, vix_now, vix_prev, rtype="weekly", breadth=None):
    p = CAL.resolve_period(rtype, date.fromisoformat(AS_OF))
    s_df, n_df = _bars(spx), _bars(ndx)
    s = compute_technicals(s_df, "^GSPC", "S&P 500", p.session_date, p.prev_period_end)
    n = compute_technicals(n_df, "^NDX", "Nasdaq-100", p.session_date, p.prev_period_end)
    eng = TrendScoringEngine(CFG.scoring, CFG.scorecard_categories)
    return eng.compute_scorecard(s, vix_now, vix_prev, n, None, CFG.macro.scoring, breadth, CFG.breadth.scoring, rtype, s_df, n_df, p, None)


# ---------------------------------------------------------------- period / calendar

def test_holiday_shortened_week_is_complete_not_partial():
    # Good Friday 2026-04-03: Thursday is the week's final session.
    p = CAL.resolve_period("weekly", date(2026, 4, 2))
    assert p.session_count == 4
    assert p.is_partial is False
    assert "Full week" in p.notes and "session counts differ" in p.notes


def test_month_to_date_flagged_partial():
    p = CAL.resolve_period("monthly", date(2026, 11, 13))
    assert p.is_partial is True and p.baseline_date == "2026-10-30"
    assert p.notes.startswith("Month-to-date")


def test_open_session_is_never_treated_as_final():
    # 20:00 UTC on 2026-11-18 = 15:00 ET: today's session has not closed.
    assert CAL.get_latest_session(None, now_utc=pd.Timestamp("2026-11-18 20:00")) == date(2026, 11, 17)
    assert CAL.get_latest_session(date(2026, 11, 18), now_utc=pd.Timestamp("2026-11-18 20:00")) == date(2026, 11, 17)
    assert CAL.get_latest_session(None, now_utc=pd.Timestamp("2026-11-18 21:31")) == date(2026, 11, 18)
    # Day after Thanksgiving closes early (13:00 ET = 18:00 UTC)
    assert CAL.get_latest_session(None, now_utc=pd.Timestamp("2026-11-27 18:31")) == date(2026, 11, 27)


# ---------------------------------------------------------------- FRED / macro placeholders

def test_fred_publication_lag_does_not_print_zero_change():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-12"]),
                       "value": [4.86, 4.88, 4.98, 5.10]})
    snap = FredProvider.compute_rate_metrics(df, "DGS10", "10Y", period_start_date="2026-11-12")
    assert snap.period_change == pytest.approx(0.12)
    assert "lagged" in snap.notes
    # Normal (non-lagged) window still measured from the baseline
    snap_w = FredProvider.compute_rate_metrics(df, "DGS10", "10Y", period_start_date="2026-11-06")
    assert snap_w.period_change == pytest.approx(0.24)


def test_macro_section_unavailable_instead_of_placeholder_yields():
    from sections.macro import build_macro_section

    class NoFred:
        api_key = ""

        def __getattr__(self, name):
            if name.startswith("compute_"):
                return getattr(FredProvider, name)
            return lambda *a, **k: None

    p = CAL.resolve_period("daily", date.fromisoformat(AS_OF))
    res = build_macro_section(fred=NoFred(), calendar_provider=SimpleNamespace(api_key=""), cross_asset_bars={},
                              period=p, cfg=CFG, market_calendar=CAL)
    assert res.status == "unavailable"
    assert "DGS10" in (res.error_message or "")


def test_unrecorded_fomc_meeting_is_not_reported_as_unanimous_hold():
    f = resolve_fomc_schedule("2026-12-11", 3.58, period_start_date="2026-12-07")
    assert f.last_meeting_date == "2026-12-09"
    assert f.outcome_action == "Unverified"
    assert "unanimous" not in f.vote_count
    assert "not in the local statement record" in f.decision_summary


# ---------------------------------------------------------------- corporate actions / cache

def test_cache_keeps_genuine_half_and_double_moves(tmp_path):
    cache = ParquetCache(str(tmp_path))
    cache.write_bars("CRSH", _bars(np.r_[np.linspace(100, 110, 299), [54.0]]))
    cache.write_bars("POP", _bars(np.r_[np.linspace(50, 52, 299), [104.0]]))
    crash = cache.read_bars("CRSH")["Close"]
    pop = cache.read_bars("POP")["Close"]
    assert crash.iloc[-1] / crash.iloc[-2] == pytest.approx(54.0 / 110.0)
    assert pop.iloc[-1] / pop.iloc[-2] == pytest.approx(2.0)


def test_cache_rescales_history_when_provider_reports_split(tmp_path):
    cache = ParquetCache(str(tmp_path))
    pre = np.linspace(100, 110, 300)
    cache.write_bars("SPLT", _bars(pre, end="2026-11-06"))
    # Provider now returns split-adjusted (2:1) bars, overlapping 5 cached sessions.
    cache.write_bars("SPLT", _bars(np.r_[pre[-5:] / 2, np.full(5, 55.5)]))
    closes = cache.read_bars("SPLT")["Close"]
    assert closes.pct_change().abs().max() < 0.05
    assert closes.iloc[0] == pytest.approx(50.0)


# ---------------------------------------------------------------- movers

def _universe(n, last_move):
    rng = np.random.default_rng(5)
    bars, consts = {}, []
    for k in range(n):
        c = 50 * np.cumprod(1 + 0.0003 + 0.01 * rng.standard_normal(400))
        c[-1] = c[-2] * (1 + last_move(k))
        bars[f"S{k}"] = _bars(c)
        consts.append(Constituent(f"S{k}", f"Stock {k}", "Industrials", True, False))
    return bars, consts


def test_losers_are_actually_down_and_label_reflects_universe():
    bars, consts = _universe(12, lambda k: 0.01 + 0.002 * k)  # every stock up on the day
    res = build_movers(bars, consts, {f"S{k}": {"marketCap": 0} for k in range(12)}, CAL, AS_OF, CFG.movers)
    assert res.is_available
    assert res.data.day.losers == []
    assert all(m.pct_change > 0 for m in res.data.day.gainers)
    assert res.title == "Top Movers (S&P 500 only)"
    assert res.data.day.gainers[0].market_cap_str == "n/a"


def test_movers_survive_a_day_with_no_gainers():
    bars, consts = _universe(12, lambda k: -0.01 - 0.002 * k)
    res = build_movers(bars, consts, {}, CAL, AS_OF, CFG.movers)
    assert res.is_available and res.data.day.gainers == [] and len(res.data.day.losers) == 5


def test_stale_ticker_excluded_from_movers():
    bars, consts = _universe(6, lambda k: 0.05 if k == 0 else 0.01)
    bars["S0"] = bars["S0"].iloc[:-3]  # feed stopped 3 sessions ago
    res = build_movers(bars, consts, {}, CAL, AS_OF, CFG.movers)
    assert "S0" not in {m.symbol for m in res.data.day.gainers}


# ---------------------------------------------------------------- breadth + verdict

def _narrow_rally_breadth(stale_from=None):
    rng = np.random.default_rng(7)
    bars, consts = {}, []
    for k in range(100):
        c = 50 * np.cumprod(1 + 0.0002 + 0.012 * rng.standard_normal(400))
        c[-5:] = c[-6] * np.cumprod([1.02] * 5) if k < 8 else c[-6] * np.cumprod([0.996] * 5)
        df = _bars(c)
        if stale_from is not None and stale_from <= k < 60:
            df = df.iloc[:-5]
        bars[f"B{k}"] = df
        consts.append(Constituent(f"B{k}", f"B{k}", "Tech", True, False))
    p = CAL.resolve_period("weekly", date.fromisoformat(AS_OF))
    return compute_market_breadth(bars, consts, p, CFG.breadth, spx_return_pct=1.5)


def test_stale_constituents_reduce_coverage_instead_of_counting_unchanged():
    br = _narrow_rally_breadth(stale_from=8)
    assert br.included_constituents == 48
    assert br.coverage_warning is not None
    assert br.advancers + br.decliners == br.included_constituents


def test_index_up_breadth_down_verdict_flags_conflict():
    br = _narrow_rally_breadth()
    assert br.ad_confirmation == "Bearish Divergence"
    spx = _path(600, 0.0005, 0.007, 3); spx[-5:] = spx[-6] * np.cumprod([1.003] * 5)
    ndx = _path(600, 0.0006, 0.009, 4); ndx[-5:] = ndx[-6] * np.cumprod([1.005] * 5)
    sc = _scorecard(spx, ndx, 15.0, 15.5, breadth=br)
    assert sc.normalized_score >= 0.15  # directional (Constructive) read from price
    assert "LOW CONVICTION" in sc.confidence
    assert any("Conflicting signals" in f for f in sc.caution_flags)


def test_high_vol_selloff_is_risk_off_and_missing_vix_not_scored():
    spx = _path(600, 0.0005, 0.007, 21)
    spx[-10:] = spx[-11] * np.cumprod(1 + np.array([-0.03, -0.025, 0.01, -0.04, -0.02, 0.015, -0.03, -0.02, 0.02, -0.035]))
    ndx = _path(600, 0.0005, 0.009, 22)
    ndx[-10:] = ndx[-11] * np.cumprod(1 + np.array([-0.04, -0.03, 0.015, -0.05, -0.025, 0.02, -0.04, -0.025, 0.025, -0.045]))
    assert _scorecard(spx, ndx, 38.0, 22.0).verdict == "Risk-off"
    sc_novix = _scorecard(spx, ndx, None, None)
    assert not any(v.rule_name.startswith("Volatility: VIX") for v in sc_novix.votes)
    assert any("VIX data unavailable" in r for r in sc_novix.summary_reasons)


def test_flat_market_is_neutral():
    spx = _path(600, 0.0, 0.004, 11); ndx = _path(600, 0.0, 0.005, 12)
    spx[-200:] = spx[-200] * (1 + 0.004 * np.sin(np.arange(200) / 3))
    ndx[-200:] = ndx[-200] * (1 + 0.005 * np.sin(np.arange(200) / 3))
    assert _scorecard(spx, ndx, 16.0, 16.1).verdict.startswith("Neutral")


def test_borderline_band_is_symmetric():
    # -0.08 is not near any band edge (-0.15 / +0.15); the old code labelled it borderline-bearish.
    from scoring import engine as eng_mod
    src = open(eng_mod.__file__, encoding="utf-8").read()
    assert "(-0.10)" not in src


# ---------------------------------------------------------------- disclosure

def test_data_warnings_flag_stale_headline_and_lags():
    from report import collect_data_warnings
    p = CAL.resolve_period("daily", date.fromisoformat(AS_OF))
    fresh = _bars(np.linspace(100, 110, 300))
    core = {"^GSPC": fresh.iloc[:-2], "^VIX": fresh, "SPY": fresh}
    macro = SimpleNamespace(is_available=True, error_message=None,
                            data=SimpleNamespace(fed_rates=SimpleNamespace(rates_as_of="2026-11-12")))
    w = collect_data_warnings(p, ["^GSPC", "^VIX", "SPY"], core, [], {},
                              [Constituent("A", "A", "X", True, False)], "^VIX", macro)
    text = " ".join(w)
    assert "^GSPC (last bar 2026-11-11)" in text
    assert "FRED" in text and "2026-11-12" in text
    assert "Nasdaq-100 constituent list unavailable" in text


def test_no_cache_flag_replaces_price_cache(tmp_path, monkeypatch):
    """--no-cache must re-download prices and replace cached bars, not just the constituent list."""
    from data.yfinance_provider import YFinanceDataProvider

    old = _bars(np.linspace(100, 110, 300))
    old.iloc[-1, old.columns.get_loc("Close")] = 55.0  # bar corrupted by older code
    ParquetCache(str(tmp_path)).write_bars("ABC", old)
    fresh = _bars(np.linspace(100, 110, 300))
    calls = []

    def fake_fetch(self, tickers, start_date, end_date, interval="1d"):
        calls.append(list(tickers))
        return pd.concat({t: fresh for t in tickers}, axis=1)

    monkeypatch.setattr(YFinanceDataProvider, "_fetch_batch_with_retry", fake_fetch)
    cached = YFinanceDataProvider(cache_dir=str(tmp_path), request_delay=0).get_history(["ABC"], "2026-01-02", AS_OF)
    assert calls == [] and cached["ABC"]["Close"].iloc[-1] == 55.0  # normal run trusts the cache

    refreshed = YFinanceDataProvider(cache_dir=str(tmp_path), request_delay=0, force_refresh=True).get_history(["ABC"], "2026-01-02", AS_OF)
    assert calls == [["ABC"]]
    assert refreshed["ABC"]["Close"].iloc[-1] == pytest.approx(110.0)
    assert ParquetCache(str(tmp_path)).read_bars("ABC")["Close"].iloc[-1] == pytest.approx(110.0)
