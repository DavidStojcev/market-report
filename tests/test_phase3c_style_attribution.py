"""Phase 3C tests: Equity Style, Size & Factor Attribution."""

import numpy as np
import pandas as pd

from config.settings import SymbolItem
from data.calendar import PeriodInfo
from sections.sectors import (
    SectorReportData,
    StyleFactorAttribution,
    build_sectors,
    compute_style_factor_attribution,
)


def test_compute_style_factor_attribution_all_pairs():
    dates = pd.date_range("2026-07-01", periods=70, freq="B")
    base_spy = np.linspace(500.0, 520.0, 70)
    # Last 5 sessions: Growth lags Value, Small-Cap & Equal-Weight rally strongly
    iwf = np.linspace(380.0, 390.0, 70)
    iwf[-5:] = np.linspace(390.0, 386.0, 5)
    iwd = np.linspace(180.0, 185.0, 70)
    iwd[-5:] = np.linspace(185.0, 190.0, 5)
    iwm = np.linspace(210.0, 220.0, 70)
    iwm[-5:] = np.linspace(220.0, 228.0, 5)
    rsp = np.linspace(170.0, 178.0, 70)
    rsp[-5:] = np.linspace(178.0, 183.0, 5)
    mtum = np.linspace(200.0, 210.0, 70)
    qual = np.linspace(180.0, 190.0, 70)

    bars = {
        "SPY": pd.DataFrame({"Close": base_spy}, index=dates),
        "IWF": pd.DataFrame({"Close": iwf}, index=dates),
        "IWD": pd.DataFrame({"Close": iwd}, index=dates),
        "IWM": pd.DataFrame({"Close": iwm}, index=dates),
        "RSP": pd.DataFrame({"Close": rsp}, index=dates),
        "MTUM": pd.DataFrame({"Close": mtum}, index=dates),
        "QUAL": pd.DataFrame({"Close": qual}, index=dates),
    }

    attr = compute_style_factor_attribution(bars=bars, curr_end_dt=dates[-1], base_dt=dates[-6])
    assert isinstance(attr, StyleFactorAttribution)
    assert len(attr.items) == 5

    gv = next(x for x in attr.items if x.long_symbol == "IWF")
    assert gv.period_spread_pct < -0.5
    assert gv.regime_label == "Value Leadership"
    assert gv.ret_1m_spread_pct is not None and gv.ret_3m_spread_pct is not None

    rsp_item = next(x for x in attr.items if x.long_symbol == "RSP")
    assert rsp_item.period_spread_pct > 1.0
    assert rsp_item.regime_label == "Broadening Participation"
    assert "Value Tilt" in attr.dominant_style
    assert "Broadening Breadth" in attr.dominant_style


def test_compute_style_factor_attribution_missing_returns_none():
    dates = pd.date_range("2026-10-01", periods=10, freq="B")
    bars = {"SPY": pd.DataFrame({"Close": np.linspace(500.0, 505.0, 10)}, index=dates)}
    assert compute_style_factor_attribution(bars=bars, curr_end_dt=dates[-1], base_dt=dates[-2]) is None


def test_build_sectors_wires_style_attribution():
    dates = pd.date_range("2026-07-01", periods=70, freq="B")
    bars = {
        "SPY": pd.DataFrame({"Close": np.linspace(500.0, 510.0, 70)}, index=dates),
        "XLK": pd.DataFrame({"Close": np.linspace(200.0, 210.0, 70)}, index=dates),
        "XLF": pd.DataFrame({"Close": np.linspace(45.0, 47.0, 70)}, index=dates),
        "IWF": pd.DataFrame({"Close": np.linspace(380.0, 395.0, 70)}, index=dates),
        "IWD": pd.DataFrame({"Close": np.linspace(180.0, 182.0, 70)}, index=dates),
    }
    period = PeriodInfo(
        report_type="weekly",
        as_of_date=dates[-1].strftime("%Y-%m-%d"),
        session_date=dates[-1].strftime("%Y-%m-%d"),
        current_start=dates[-5].strftime("%Y-%m-%d"),
        current_end=dates[-1].strftime("%Y-%m-%d"),
        baseline_date=dates[-6].strftime("%Y-%m-%d"),
        prev_period_start=dates[-10].strftime("%Y-%m-%d"),
        prev_period_end=dates[-6].strftime("%Y-%m-%d"),
        prev_baseline_date=dates[-11].strftime("%Y-%m-%d"),
        is_partial=False,
        session_count=5,
    )
    res = build_sectors(
        bars=bars,
        sectors=[SymbolItem(symbol="XLK", name="Technology"), SymbolItem(symbol="XLF", name="Financials")],
        benchmark=SymbolItem(symbol="SPY", name="SPDR S&P 500"),
        period=period,
    )
    assert res.status == "available"
    data: SectorReportData = res.data
    assert data.style_attribution is not None
    assert len(data.style_attribution.items) == 1
    assert data.style_attribution.items[0].long_symbol == "IWF"
