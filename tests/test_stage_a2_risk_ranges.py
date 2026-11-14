"""Tests for Stage A2: Hedgeye Risk Ranges & 3Fourteen Posture Flip Triggers."""

import numpy as np
import pandas as pd

from data.calendar import PeriodInfo
from sections.swing_regime import evaluate_swing_gate, SwingGateData
from render.markdown import render_markdown
from render.html import render_html


def _make_bars(start_px: float, n: int = 240, step_pct: float = 0.001, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-11-03", periods=n)
    closes = [start_px]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1.0 + step_pct + float(rng.normal(0.0, 0.006))))
    close_s = pd.Series(closes, index=dates)
    return pd.DataFrame({
        "Open": close_s * 0.998,
        "High": close_s * 1.007,
        "Low": close_s * 0.993,
        "Close": close_s,
        "Volume": 50_000_000,
    }, index=dates)


def test_risk_ranges_and_posture_triggers_computed_and_rendered():
    """Verify RiskRangeItem boundaries, Trade/Trend/Tail calls, and PostureTriggers in MD and HTML."""
    spy_df = _make_bars(700.0, step_pct=0.0002, seed=1)
    qqq_df = _make_bars(660.0, step_pct=-0.0005, seed=2)
    asset_bars = {
        "SPY": spy_df,
        "QQQ": qqq_df,
        "IWM": _make_bars(240.0, step_pct=0.0008, seed=3),
        "TLT": _make_bars(88.0, step_pct=-0.0003, seed=4),
        "CL=F": _make_bars(80.0, step_pct=0.0012, seed=5),
        "GC=F": _make_bars(2600.0, step_pct=0.0006, seed=6),
    }

    sec = evaluate_swing_gate(
        spy_df=spy_df,
        qqq_df=qqq_df,
        vix_current=20.8,
        vix_previous=21.1,
        vix_term_ratio=0.98,
        breadth_result=None,
        as_of_date="2026-11-03",
        asset_bars=asset_bars,
    )
    assert sec.status == "available"
    sg: SwingGateData = sec.data

    assert sg.posture_triggers is not None
    assert "Tier" in sg.posture_triggers.target_beta_band
    assert "Target Equity Beta" not in sg.posture_triggers.target_beta_band
    assert len(sg.risk_ranges) == 6
    for rr in sg.risk_ranges:
        assert rr.range_low < rr.last_price < rr.range_high
        assert 0.0 <= rr.position_pct <= 100.0
        assert rr.trade_dir in ("BULLISH", "NEUTRAL", "BEARISH")
        assert rr.trend_dir in ("BULLISH", "NEUTRAL", "BEARISH")
        assert rr.tail_dir in ("BULLISH", "NEUTRAL", "BEARISH")

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
    md = render_markdown(period, [sec])
    html = render_html(period, [sec])

    assert "Exposure Sizing & Regime Flip Triggers" in md
    assert "Quantitative Risk Ranges™" in md
    assert "Range Low (Support)" in md
    assert "Range High (Resistance)" in md
    assert "Buy / Support" not in md
    assert "TLT" in md and "CL=F" in md and "GC=F" in md
    assert "Exposure Sizing &amp; Regime Flip Triggers" in html
    assert "Quantitative Risk Ranges&trade;" in html
