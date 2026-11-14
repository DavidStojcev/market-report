"""Tests for Stage B2: JPMorgan 'If-Then' Event Scenario Matrix & Daily 48H Catalyst Playbook."""

from types import SimpleNamespace
import pandas as pd

from data.calendar import PeriodInfo
from data.macro_base import EarningsEvent, MacroEvent
from qa_checks import check_t3_claim_lint, check_t8_structure_lint
from render.html import render_html
from render.markdown import render_markdown
from sections.base import SectionResult
from sections.forward_calendar import (
    DailyCatalystPlaybook,
    EventScenarioMatrix,
    ForwardCalendarData,
    build_daily_catalyst_playbook,
    build_event_scenario_matrices,
    build_forward_calendar,
)
from sections.swing_regime import SwingGateData, RiskRangeItem


class _StubCalendarProvider:
    def __init__(self, events):
        self._events = events

    def get_upcoming_events(self, start_date: str, end_date: str, min_importance: str = "medium"):
        return list(self._events)


def test_event_scenario_matrices_vix_calibration_and_branches():
    """Verify CPI, NFP, and Claims produce calibrated 3-branch JPMorgan-style matrices."""
    events = [
        MacroEvent(
            event_id="cpi1",
            title="Consumer Price Index (CPI YoY)",
            category="inflation",
            date="2026-11-12",
            time_et="08:30 ET",
            forecast=2.9,
            previous=3.0,
            unit="%",
            importance="high",
            source="BLS",
        ),
        MacroEvent(
            event_id="nfp1",
            title="Employment Situation (Nonfarm Payrolls & Unemployment Rate)",
            category="labor",
            date="2026-11-06",
            time_et="08:30 ET",
            forecast=120.0,
            previous=165.0,
            unit="k",
            importance="high",
            source="BLS",
        ),
        MacroEvent(
            event_id="clm1",
            title="Initial Jobless Claims",
            category="labor",
            date="2026-11-05",
            time_et="08:30 ET",
            forecast=218.0,
            previous=194.0,
            unit="k",
            importance="medium",
            source="DOL",
        ),
    ]

    mats = build_event_scenario_matrices(events, vix_current=20.0, spy_price=750.0, max_matrices=2)
    assert len(mats) == 2
    for m in mats:
        assert isinstance(m, EventScenarioMatrix)
        assert len(m.outcomes) == 3
        assert m.importance == "HIGH"
        assert m.implied_1d_move_pct > 1.0
        assert any("$" in o.spy_reaction for o in m.outcomes)

    # Fallback to medium print when no high-impact events are present
    med_mats = build_event_scenario_matrices([events[2]], vix_current=18.0, spy_price=750.0, max_matrices=2)
    assert len(med_mats) == 1
    assert "Sub-Consensus Tight Labor" in [o.scenario for o in med_mats[0].outcomes]


def test_build_forward_calendar_populates_scenarios_and_renders_cleanly():
    """Verify Weekly ForwardCalendarData includes scenario_matrices and passes QA lint checks."""
    events = [
        MacroEvent(
            event_id="ref1",
            title="Treasury Quarterly Refunding Announcement",
            category="rates",
            date="2026-11-04",
            time_et="08:30 ET",
            importance="high",
            source="U.S. Department of the Treasury",
            source_url="https://home.treasury.gov",
        ),
        MacroEvent(
            event_id="nfp1",
            title="Employment Situation (Nonfarm Payrolls & Unemployment Rate)",
            category="labor",
            date="2026-11-06",
            time_et="08:30 ET",
            forecast=120.0,
            previous=165.0,
            unit="k",
            importance="high",
            source="U.S. Bureau of Labor Statistics (BLS)",
            source_url="https://www.bls.gov",
        ),
    ]
    cfg = SimpleNamespace(
        forward_calendar=SimpleNamespace(
            high_impact_rules=["employment situation", "quarterly refunding", "cpi", "fomc"]
        )
    )
    sec_cal = build_forward_calendar(
        calendar_provider=_StubCalendarProvider(events),
        as_of_date="2026-10-30",
        cfg=cfg,
        lookahead_days=10,
        vix_current=19.42,
        spy_price=750.0,
    )
    assert sec_cal.status == "available"
    fc: ForwardCalendarData = sec_cal.data
    assert len(fc.scenario_matrices) == 2

    period = PeriodInfo(
        report_type="weekly",
        as_of_date="2026-10-30",
        session_date="2026-10-30",
        current_start="2026-10-26",
        current_end="2026-10-30",
        baseline_date="2026-10-23",
        prev_period_start="2026-10-19",
        prev_period_end="2026-10-23",
        prev_baseline_date="2026-10-16",
        is_partial=False,
        session_count=5,
    )

    md = render_markdown(period, [sec_cal])
    html = render_html(period, [sec_cal])

    assert 'Event Scenario Matrix ("If-Then" Reaction Playbook)' in md
    assert "\\$" in md
    assert "Event Scenario Matrix" in html
    assert r"\$" not in html

    assert check_t3_claim_lint(md, html, [sec_cal]).status == "PASS"
    t8 = check_t8_structure_lint(md, html, [sec_cal])
    assert t8.status == "PASS", f"T8 failures: {t8.details}"


def test_daily_48h_catalyst_playbook_and_rendering():
    """Verify Daily 48H Catalyst Playbook attaches to SwingGateData and renders cleanly."""
    events = [
        MacroEvent(
            event_id="nfp1",
            title="Employment Situation (Nonfarm Payrolls & Unemployment Rate)",
            category="labor",
            date="2026-11-06",
            time_et="08:30 ET",
            forecast=120.0,
            previous=165.0,
            unit="k",
            importance="high",
            source="BLS",
        ),
        MacroEvent(
            event_id="ism1",
            title="ISM Services PMI",
            category="activity",
            date="2026-11-04",
            time_et="10:00 ET",
            forecast=54.5,
            previous=54.9,
            unit="index",
            importance="medium",
            source="ISM",
        ),
    ]
    earnings = [
        EarningsEvent(symbol="QCOM", name="QUALCOMM Inc", date="2026-11-04", hour="amc", market_cap=190e9),
        EarningsEvent(symbol="CVS", name="CVS Health Corp", date="2026-11-04", hour="bmo", market_cap=75e9),
    ]
    mats = build_event_scenario_matrices(events, vix_current=20.83, spy_price=754.35, max_matrices=1)
    spy_rr = RiskRangeItem(
        symbol="SPY",
        name="S&P 500 ETF",
        last_price=754.35,
        range_low=742.10,
        range_high=778.40,
        pivot_20ema=765.16,
        position_pct=34.0,
        atr_pct=1.12,
        trade_dir="BEARISH",
        trend_dir="BULLISH",
        tail_dir="BULLISH",
        zone_note="Near Lower Support",
    )

    pb: DailyCatalystPlaybook = build_daily_catalyst_playbook(
        as_of_date="2026-11-03",
        lookahead_date="2026-11-06",
        events=events,
        earnings_events=earnings,
        scenario_matrices=mats,
        spy_price=754.35,
        spy_20ema=765.16,
        spy_range=spy_rr,
        vix_current=20.83,
        posture="DEFENSIVE",
    )

    assert len(pb.macro_catalysts) == 2
    assert pb.earnings_tonight_tomorrow[0]["symbol"] == "QCOM"
    assert pb.earnings_tonight_tomorrow[0]["timing"] == "AMC (After Close)"
    assert pb.earnings_tonight_tomorrow[1]["timing"] == "BMO (Pre-Open)"
    assert pb.featured_scenario is not None
    assert "765.16" in pb.spy_pivot_line

    sg = SwingGateData(
        as_of_date="2026-11-03",
        posture="DEFENSIVE",
        posture_badge_class="badge-red",
        triggered_rule="Regime Tier 4 (Defensive): Dual benchmark break of 50-DMA.",
        spy_price=754.35,
        spy_20ema=765.16,
        spy_20ema_dist_pct=-1.41,
        spy_20ema_pass=False,
        spy_50dma=767.00,
        spy_50dma_dist_pct=-1.65,
        spy_50dma_pass=False,
        qqq_price=685.28,
        qqq_20ema=720.00,
        qqq_20ema_dist_pct=-4.86,
        qqq_20ema_pass=False,
        qqq_50dma=723.90,
        qqq_50dma_dist_pct=-5.34,
        qqq_50dma_pass=False,
        vix_level=20.83,
        vix_5d_trend=-0.32,
        vix_term_structure="Contango (Normal)",
        breadth_pct_above_50dma=65.4,
        breadth_ad_state="Bearish Divergence",
        net_new_highs=9,
        execution_guidance="Prioritize capital preservation.",
        risk_ranges=[spy_rr],
        catalyst_playbook=pb,
    )
    sec_swing = SectionResult(
        section_num=3,
        title="Market Regime & Risk Appetite",
        status="available",
        data=sg,
    )
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
        is_partial=False,
        session_count=1,
    )

    md = render_markdown(period, [sec_swing])
    html = render_html(period, [sec_swing])

    assert "Next 48H Catalyst Playbook" in md
    assert "QUALCOMM Inc" in md
    assert "Goldilocks / In-Line" in md
    assert r"\$" not in html
    assert "Next 48H Catalyst Playbook" in html
    assert "QUALCOMM Inc" in html
    assert check_t3_claim_lint(md, html, [sec_swing]).status == "PASS"
    t8 = check_t8_structure_lint(md, html, [sec_swing])
    assert t8.status == "PASS", f"T8 failures: {t8.details}"
