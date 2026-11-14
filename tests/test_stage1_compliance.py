"""Stage 1 acceptance and unit tests: Legal Compliance & Impersonal Language (Substages 1.1–1.3)."""

import re
import numpy as np
import pandas as pd

from data.calendar import PeriodInfo
from data.macro_base import MacroEvent
from qa_checks import check_t13_compliance_lint
from render.html import render_html
from render.markdown import render_markdown
from scoring.engine import CategoryScore, Scorecard
from sections.base import SectionResult
from sections.conclusion import DEFAULT_COMPLIANCE_DISCLAIMER, build_footer
from sections.executive_summary import build_executive_summary
from sections.forward_calendar import (
    EventScenarioMatrix,
    EventScenarioOutcome,
    ForwardCalendarData,
    build_event_scenario_matrices,
)
from sections.swing_regime import evaluate_swing_gate


def _make_bars(end_price: float, trend_step: float = 0.0, n: int = 240, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-11-13", periods=n)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1.0 + trend_step + float(rng.normal(0.0, 0.004))))
    scale = end_price / closes[-1]
    close_s = pd.Series([c * scale for c in closes], index=dates)
    return pd.DataFrame({
        "Open": close_s * 0.998,
        "High": close_s * 1.006,
        "Low": close_s * 0.994,
        "Close": close_s,
        "Volume": 45_000_000,
    }, index=dates)


def _period() -> PeriodInfo:
    return PeriodInfo(
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


def _scorecard() -> Scorecard:
    return Scorecard(
        normalized_score=-0.12,
        raw_score=-0.10,
        max_score=0.85,
        verdict="Neutral",
        votes=[],
        caution_flags=[],
        top_bullish_factors=["Credit spreads remain tight"],
        top_bearish_factors=["Index benchmarks below 50-DMA"],
        summary_reasons=["Index averages sit below 50-day moving average"],
        confidence="Agreement: 2 of 4 active categories match the verdict",
        categories=[
            CategoryScore(
                name="Trend / Price",
                category_key="trend",
                weight=0.30,
                effective_weight=0.30,
                score=-0.40,
                weighted_score=-0.12,
                status="active",
                components=[],
            ),
        ],
    )


def test_1_1_all_postures_render_model_state_without_directives():
    """1.1: All four posture tiers render impersonal model states without directive verbs or beta bands."""
    period = _period()
    sc = _scorecard()
    spy_up = _make_bars(760.0, trend_step=0.002, seed=1)
    spy_dn = _make_bars(720.0, trend_step=-0.002, seed=2)
    qqq_up = _make_bars(700.0, trend_step=0.002, seed=3)
    qqq_dn = _make_bars(660.0, trend_step=-0.002, seed=4)

    # Full, Reduced (VIX 20.5), Selective (one index down), Defensive (both down)
    configs = [
        ("FULL", spy_up, qqq_up, 16.0),
        ("REDUCED", spy_up, qqq_up, 20.5),
        ("SELECTIVE", spy_up, qqq_dn, 18.0),
        ("DEFENSIVE", spy_dn, qqq_dn, 23.0),
    ]
    banned_phrases = (
        "hold a",
        "target equity beta",
        "tighten hedge",
        "cut beta",
        "buy / support",
        "sell / resistance",
        "deploy selective",
        "stay constructive",
        "prioritize capital",
    )

    for expected_posture, spy_df, qqq_df, vix_val in configs:
        sec_swing = evaluate_swing_gate(
            spy_df=spy_df,
            qqq_df=qqq_df,
            vix_current=vix_val,
            vix_previous=vix_val,
            vix_term_ratio=0.95,
            breadth_result=None,
            as_of_date="2026-11-13",
            asset_bars={"SPY": spy_df, "QQQ": qqq_df},
        )
        assert sec_swing.data.posture == expected_posture
        sec_exec = build_executive_summary(
            scorecard=sc,
            period_info=period,
            swing_data=sec_swing.data,
            spx_period_ret=-0.25,
            ndx_period_ret=-0.40,
        )
        assert sec_exec.data.brief.bottom_line.startswith("Model posture:")
        assert "Range Low (Support)" in render_markdown(period, [sec_exec, sec_swing])

        sec_footer = build_footer()
        sections = [sec_exec, sec_swing, sec_footer]
        md = render_markdown(period, sections)
        html = render_html(period, sections)
        combined_lower = (md + "\n" + html).lower()
        for phrase in banned_phrases:
            assert not re.search(rf"\b{re.escape(phrase)}\b", combined_lower), f"Found '{phrase}' in posture {expected_posture}"
        assert "Model upgrade condition" in md and "Model downgrade condition" in md
        assert "Range Low (Support)" in md and "Range High (Resistance)" in md
        assert check_t13_compliance_lint(md, html, sections).status == "PASS"


def test_1_2_scenario_matrices_have_no_invented_probabilities_or_tilts():
    """1.2: Scenario matrices remove % probabilities, use objective conditions, and pass T13."""
    events = [
        MacroEvent("cpi", "Consumer Price Index (CPI YoY)", "inflation", "2026-11-17", "08:30 ET", previous=3.1, unit="%", importance="high", source="BLS"),
        MacroEvent("nfp", "Employment Situation (Nonfarm Payrolls)", "labor", "2026-11-18", "08:30 ET", previous=150.0, unit="k", importance="high", source="BLS"),
        MacroEvent("clm", "Initial Jobless Claims", "labor", "2026-11-19", "08:30 ET", previous=215.0, unit="k", importance="medium", source="DOL"),
        MacroEvent("fomc", "FOMC Rate Decision", "fed", "2026-11-20", "14:00 ET", importance="high", source="Federal Reserve"),
        MacroEvent("ref", "Treasury Quarterly Refunding Announcement", "rates", "2026-11-21", "08:30 ET", importance="high", source="Treasury"),
        MacroEvent("elec", "General Election Outcome", "policy", "2026-11-22", "20:00 ET", importance="high", source="Official"),
        MacroEvent("ret", "Advance Retail Sales (MoM)", "growth", "2026-11-23", "08:30 ET", previous=0.4, unit="%", importance="medium", source="Census"),
    ]
    mats = build_event_scenario_matrices(events, vix_current=19.3, spy_price=755.0, max_matrices=10)
    assert len(mats) == 5  # Only HIGH events when high events exist
    med_only = build_event_scenario_matrices([events[2], events[6]], vix_current=19.3, spy_price=755.0, max_matrices=5)
    all_mats = mats + med_only
    for m in all_mats:
        assert "Options-implied move calibrated" in m.desk_note
        for out in m.outcomes:
            assert "%" not in out.probability_hint

    sec_cal = SectionResult(
        section_num=1,
        title="Forward Calendar & Catalyst Horizon",
        status="available",
        data=ForwardCalendarData(
            as_of_date="2026-11-13",
            lookahead_date="2026-11-23",
            events=events,
            high_impact_count=5,
            scenario_matrices=all_mats,
        ),
    )
    sec_footer = build_footer()
    sections = [sec_cal, sec_footer]
    md = render_markdown(_period(), sections)
    html = render_html(_period(), sections)
    assert "Historically sensitive sectors" in md
    assert "Desk Sector Tilt" not in md
    assert "Historically sensitive sectors" in html
    assert check_t13_compliance_lint(md, html, sections).status == "PASS"


def test_1_3_t13_compliance_lint_catches_violations_and_weak_disclaimers():
    """1.3: T13 must pass clean reports and fail on advisory terms, % scenario hints, or weak disclaimers."""
    sec_footer = build_footer()
    assert "impersonal" in DEFAULT_COMPLIANCE_DISCLAIMER.lower()
    assert "registered investment adviser" in DEFAULT_COMPLIANCE_DISCLAIMER.lower()
    assert "hypothetical" in DEFAULT_COMPLIANCE_DISCLAIMER.lower()

    clean_md = render_markdown(_period(), [sec_footer])
    clean_html = render_html(_period(), [sec_footer])
    assert check_t13_compliance_lint(clean_md, clean_html, [sec_footer]).status == "PASS"

    # Negative 1: banned directive / allocation terms in MD & HTML
    for bad_phrase in (
        "Hold a Defensive posture",
        "Target Equity Beta 20-40%",
        "Tighten Hedge / Cut Beta",
        "Buy / Support",
        "Sell / Resistance",
        "Rotate toward Staples",
        "Overweight Financials",
        "Defined-risk entry with stop at 740 and target price 780",
    ):
        res_md = check_t13_compliance_lint(bad_phrase, "", [sec_footer])
        assert res_md.status == "FAIL", f"Expected T13 FAIL for Markdown phrase: {bad_phrase}"
        res_html = check_t13_compliance_lint("", f"<p>{bad_phrase}</p>", [sec_footer])
        assert res_html.status == "FAIL", f"Expected T13 FAIL for HTML phrase: {bad_phrase}"

    # Negative 2: invented scenario % probability
    bad_mat = EventScenarioMatrix(
        event_title="CPI",
        event_date="2026-11-20",
        event_time_et="08:30 ET",
        importance="HIGH",
        consensus_display="—",
        prior_display="3.0%",
        implied_1d_move_pct=1.2,
        outcomes=[EventScenarioOutcome("Base", "50% Base Case", "In line", "±1.0%", "Steady", "Broad Benchmarks")],
    )
    bad_cal_sec = SectionResult(1, "Calendar", "available", ForwardCalendarData("2026-11-13", "2026-11-20", scenario_matrices=[bad_mat]))
    res_prob = check_t13_compliance_lint(clean_md, clean_html, [bad_cal_sec, sec_footer])
    assert res_prob.status == "FAIL"
    assert any("probability percentage" in d.lower() for d in res_prob.details)

    # Negative 3: weak footer disclaimer
    weak_footer = build_footer(disclaimer="For informational purposes only. Not financial advice.")
    res_disc = check_t13_compliance_lint("Model posture: Defensive (Tier 4).", "<p>Ok</p>", [weak_footer])
    assert res_disc.status == "FAIL"
    assert any("disclaimer" in d.lower() for d in res_disc.details)
