"""Phase 3B tests: Volatility Term Structure, VVIX/SKEW Tail Surface, and Realized Vol Cone."""

import numpy as np
import pandas as pd

from data.calendar import PeriodInfo
from sections.sentiment_positioning import build_sentiment_positioning
from sections.volatility import VolatilityData, build_volatility


def test_build_volatility_surface_and_rv_cone():
    dates = pd.date_range("2026-07-01", periods=80, freq="B")
    rng = np.random.default_rng(42)
    spx_rets = rng.normal(0.0005, 0.008, size=len(dates))
    spx_close = 6500.0 * np.cumprod(1.0 + spx_rets)

    bars = {
        "^GSPC": pd.DataFrame({"Close": spx_close}, index=dates),
        "^VIX": pd.DataFrame(
            {"Close": [18.20] * 79 + [19.50], "High": [19.0] * 79 + [20.1], "Low": [17.8] * 79 + [18.9]},
            index=dates,
        ),
        "^VIX9D": pd.DataFrame({"Close": [17.10] * 79 + [17.90]}, index=dates),
        "^VIX3M": pd.DataFrame({"Close": [20.50] * 79 + [21.20]}, index=dates),
        "^VIX6M": pd.DataFrame({"Close": [21.80] * 79 + [22.40]}, index=dates),
        "^VVIX": pd.DataFrame({"Close": [98.0] * 79 + [106.5]}, index=dates),
        "^SKEW": pd.DataFrame({"Close": [139.0] * 79 + [147.2]}, index=dates),
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

    res = build_volatility(bars=bars, vix_symbol="^VIX", period=period, spx_df=bars["^GSPC"])
    assert res.status == "available"
    vd: VolatilityData = res.data

    assert vd.vix_9d == 17.90
    assert vd.vix_3m == 21.20
    assert vd.vix_6m == 22.40
    assert vd.ratio_9d_vix == round(17.90 / 19.50, 3)
    assert vd.ratio_vix_3m == round(19.50 / 21.20, 3)
    assert vd.term_slope_pts == round(21.20 - 19.50, 2)
    assert "Contango" in vd.term_structure_label

    assert vd.vvix_level == 106.5
    assert vd.vvix_change == 8.5
    assert "Convexity" in vd.vvix_regime

    assert vd.skew_level == 147.2
    assert vd.skew_change == 8.2
    assert "Heavy Left-Tail" in vd.skew_regime

    assert vd.realized_vol_10d is not None and vd.realized_vol_10d > 0
    assert vd.realized_vol_20d is not None and vd.realized_vol_20d > 0
    assert vd.realized_vol_60d is not None and vd.realized_vol_60d > 0
    assert vd.vrp_20d == round(19.50 - vd.realized_vol_20d, 2)
    assert "VVIX" in vd.surface_synthesis and "SKEW" in vd.surface_synthesis


def test_build_volatility_gracefully_handles_missing_surface_series():
    dates = pd.date_range("2026-10-01", periods=10, freq="B")
    bars = {"^VIX": pd.DataFrame({"Close": [18.0] * 10, "High": [18.5] * 10, "Low": [17.5] * 10}, index=dates)}
    period = PeriodInfo(
        report_type="daily",
        as_of_date=dates[-1].strftime("%Y-%m-%d"),
        session_date=dates[-1].strftime("%Y-%m-%d"),
        current_start=dates[-1].strftime("%Y-%m-%d"),
        current_end=dates[-1].strftime("%Y-%m-%d"),
        baseline_date=dates[-2].strftime("%Y-%m-%d"),
        prev_period_start=dates[-2].strftime("%Y-%m-%d"),
        prev_period_end=dates[-2].strftime("%Y-%m-%d"),
        prev_baseline_date=dates[-3].strftime("%Y-%m-%d"),
        is_partial=False,
        session_count=1,
    )
    res = build_volatility(bars=bars, vix_symbol="^VIX", period=period)
    assert res.status == "available"
    assert res.data.vix_9d is None
    assert res.data.vvix_level is None
    assert res.data.skew_level is None
    assert res.data.realized_vol_20d is None


def test_sentiment_positioning_surfaces_vvix_and_skew():
    dates = pd.date_range("2026-09-01", periods=30, freq="B")
    spx_df = pd.DataFrame({"Close": np.linspace(6500.0, 6600.0, 30)}, index=dates)
    res = build_sentiment_positioning(
        spx_df=spx_df,
        vix_current=19.5,
        as_of_date=dates[-1].strftime("%Y-%m-%d"),
        vix_term_ratio=0.92,
        skew_level=147.2,
        vvix_level=106.5,
    )
    assert res.status == "available"
    names = [it.name for it in res.data.items]
    assert any("VVIX" in n for n in names)
    assert any("SKEW" in n for n in names)
