"""Phase 3A tests: Fed Net Liquidity, BBB/5Y5Y market indicators, and Cross-Asset Plumbing."""

import numpy as np
import pandas as pd

from data.fred_provider import MacroMetricSnapshot
from render.markdown import _esc_md_dollar
from sections.commodities_cross_asset import (
    CrossAssetPlumbingData,
    compute_cross_asset_plumbing,
)
from sections.macro import (
    FedRatesSummary,
    compute_credit_spread_intelligence,
    compute_fed_net_liquidity,
    compute_real_rate_intelligence,
)


def test_compute_fed_net_liquidity_unit_normalization_and_impulses():
    # Weekly Wednesday WALCL & WTREGEN (Millions USD), Daily RRPONTSYD (Billions USD)
    dates_w = pd.date_range("2026-07-01", periods=16, freq="W-WED")
    walcl_df = pd.DataFrame({"date": dates_w, "value": [6_800_000.0] * 16})  # $6.8T
    tga_vals = [800_000.0] * 12 + [900_000.0] * 4  # TGA drains $100B over last 4 weeks
    tga_df = pd.DataFrame({"date": dates_w, "value": tga_vals})

    dates_d = pd.date_range("2026-07-01", "2026-10-20", freq="B")
    rrp_df = pd.DataFrame({"date": dates_d, "value": [50.0] * (len(dates_d) - 1) + [10.0]})  # Latest ON RRP dropped $40B

    liq = compute_fed_net_liquidity(walcl_df, tga_df, rrp_df)
    assert liq is not None
    # Latest: 6800B - 900B - 10B = 5890B = $5.890T
    assert liq.walcl_t == 6.800
    assert liq.tga_b == 900.0
    assert liq.rrp_b == 10.0
    assert liq.net_liquidity_t == 5.890
    # Prior (4W/13W ago): 6800B - 800B - 50B = 5950B -> delta = -60.0B -> Draining
    assert liq.impulse_4w_b == -60.0
    assert liq.impulse_13w_b == -60.0
    assert liq.regime == "Draining"
    assert _esc_md_dollar(liq.assessment).count("$") == _esc_md_dollar(liq.assessment).count("\\$")


def test_compute_fed_net_liquidity_missing_data_returns_none():
    empty_df = pd.DataFrame(columns=["date", "value"])
    sample_df = pd.DataFrame({"date": ["2026-10-28"], "value": [6_700_000.0]})
    assert compute_fed_net_liquidity(empty_df, sample_df, sample_df) is None
    assert compute_fed_net_liquidity(None, sample_df, sample_df) is None


def test_credit_and_real_rate_bbb_5y5y_extensions():
    hy_snap = MacroMetricSnapshot(
        series_id="BAMLH0A0HYM2",
        name="HY OAS",
        latest_date="2026-11-03",
        latest_value=3.40,
        prior_value=3.45,
        period_start_value=3.45,
        period_change=-0.05,
        headline_value=3.40,
        headline_unit="%",
        prior_headline=3.45,
    )
    bbb_snap = MacroMetricSnapshot(
        series_id="BAMLC0A4CBBB",
        name="BBB OAS",
        latest_date="2026-11-03",
        latest_value=1.16,
        prior_value=1.20,
        period_start_value=1.20,
        period_change=-0.04,
        headline_value=1.16,
        headline_unit="%",
        prior_headline=1.20,
    )
    credit_intel = compute_credit_spread_intelligence(hy_snap, snap_bbb=bbb_snap)
    assert credit_intel.bbb_spread_bps == 116.0
    assert credit_intel.bbb_delta_bps == -4.0
    assert credit_intel.hy_bbb_compression_bps == 224.0

    f5y5y_snap = MacroMetricSnapshot(
        series_id="T5YIFR",
        name="5Y5Y Forward Inflation",
        latest_date="2026-11-03",
        latest_value=2.21,
        prior_value=2.18,
        period_start_value=2.18,
        period_change=0.03,
        headline_value=2.21,
        headline_unit="%",
        prior_headline=2.18,
    )
    real_snap = MacroMetricSnapshot(
        series_id="DFII10", name="10Y TIPS", latest_date="2026-10-30", latest_value=1.95,
        headline_value=1.95, headline_unit="%", prior_value=1.92, prior_headline=1.92, period_change=0.03,
    )
    # One leg present is enough (the other is derived via Fisher identity); placeholders are never used.
    rates_intel = compute_real_rate_intelligence(real_snap, None, 4.25, 0.05, snap_5y5y=f5y5y_snap)
    assert rates_intel.breakeven_10y == round(4.25 - 1.95, 2)
    assert rates_intel.forward_5y5y_pct == 2.21
    assert rates_intel.forward_5y5y_delta_bps == 3.0
    assert "2.21%" in rates_intel.assessment


def test_cross_asset_plumbing_copper_gold_and_spy_tlt_corr():
    dates = pd.date_range("2026-06-01", periods=100, freq="B")
    # Negative stock-bond co-movement in recent 60D
    spy_rets = np.sin(np.linspace(0, 12, 100)) * 0.01 + 0.0005
    tlt_rets = -np.sin(np.linspace(0, 12, 100)) * 0.008
    spy_close = 500.0 * np.cumprod(1.0 + spy_rets)
    tlt_close = 95.0 * np.cumprod(1.0 + tlt_rets)

    hg_close = np.linspace(4.2, 4.6, 100)
    gc_close = np.linspace(2800.0, 2700.0, 100)

    bars = {
        "SPY": pd.DataFrame({"Close": spy_close}, index=dates),
        "TLT": pd.DataFrame({"Close": tlt_close}, index=dates),
        "HG=F": pd.DataFrame({"Close": hg_close}, index=dates),
        "GC=F": pd.DataFrame({"Close": gc_close}, index=dates),
    }

    class DummyMacro:
        fed_rates = FedRatesSummary(
            fed_funds_rate=3.83,
            target_range="3.75%-4.00%",
            last_meeting_date="2026-10-28",
            next_meeting_date="2026-12-09",
            days_to_next=36,
            yield_2y=4.05,
            yield_2y_delta=0.04,
            yield_10y=4.30,
            yield_10y_delta=0.05,
            spread_2s10s=0.25,
            spread_2s10s_delta=0.01,
            is_inverted=False,
            curve_status="Modestly Steep / Normalizing",
            yield_10y_13w_delta=0.20,
        )

    pl = compute_cross_asset_plumbing(
        bars=bars,
        as_of_dt=dates[-1],
        base_dt=dates[-6],
        macro_data=DummyMacro(),
    )
    assert isinstance(pl, CrossAssetPlumbingData)
    assert pl.copper_gold_ratio == round((4.6 * 1000.0) / 2700.0, 3)
    assert pl.copper_gold_3m_chg_pct > 3.0
    assert pl.yield_10y_13w_bps == 20.0
    assert pl.copper_gold_regime == "Reflationary Growth Alignment"
    assert pl.stock_bond_corr_60d is not None and pl.stock_bond_corr_60d < -0.5
    assert "Negative" in pl.stock_bond_regime
