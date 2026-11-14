"""Tests for Stage 5: Weekly Report Enhancement (Substages 5.1–5.6).

Verifies:
- Substage 5.1: Weekly layout, "Weekly Desk Takeaway", no "Morning Brief",
  visible lines <= 320, visible words <= 3000 outside <details>, single visible vol section.
- Substage 5.2: The Week Ahead (Mon-Fri catalyst grid) & Earnings on Deck.
- Substage 5.3: CFTC COT Institutional Positioning (ES, ZN, CL, GC), percentiles, regime badges.
- Substage 5.4: Model Accountability (prior week range adherence, T13 compliance, JSON round-trip).
- Substage 5.5: Sector Rotation Quadrants (10-Week RS, momentum, 4 quadrants, weekly candle structure).
- Substage 5.6: 4-Pillar Macro Cycle Dashboard visible in §17, raw tables wrapped in <details>.
"""

import json
import os
import re
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from data.calendar import PeriodInfo
from data.cot_provider import COTProvider, COTContractPositioning
from data.model_accountability import (
    WeeklyModelAdherence,
    record_weekly_snapshot,
    evaluate_weekly_range_adherence,
)
from indicators.breadth import format_percentile_ordinal
from qa_checks import check_u1_reading_time_budget
from sections.forward_calendar import (
    WeeklyAheadEvent,
    WeeklyAheadEarnings,
    classify_macro_sensitivity,
)
from sections.macro import MacroCyclePillar
from sections.sectors import compute_sector_rotation_quadrants, SectorRotationItem
from sections.swing_regime import RiskRangeItem


# ----------------------------------------------------------------------
# Substage 5.1: Weekly-Specific Layout & Compression
# ----------------------------------------------------------------------

def test_substage_5_1_no_morning_brief_in_weekly():
    """Verify 'Morning Brief' does not appear anywhere in Weekly Markdown or HTML."""
    weekly_md_path = Path("reports/weekly/2026-11-13.md")
    weekly_html_path = Path("reports/weekly/2026-11-13.html")

    assert weekly_md_path.exists(), f"Weekly markdown {weekly_md_path} missing"
    assert weekly_html_path.exists(), f"Weekly html {weekly_html_path} missing"

    md_content = weekly_md_path.read_text(encoding="utf-8")
    html_content = weekly_html_path.read_text(encoding="utf-8")

    assert "morning brief" not in md_content.lower(), "Found 'Morning Brief' in weekly markdown!"
    assert "morning brief" not in html_content.lower(), "Found 'Morning Brief' in weekly HTML!"
    assert "Weekly Desk Takeaway" in md_content, "Missing 'Weekly Desk Takeaway' in §1 markdown"
    assert "Weekly Desk Takeaway" in html_content, "Missing 'Weekly Desk Takeaway' in §1 HTML"


def test_substage_5_1_visible_line_and_word_budget():
    """Verify visible lines <= 320 and visible words <= 3000 outside <details>."""
    weekly_md_path = Path("reports/weekly/2026-11-13.md")
    assert weekly_md_path.exists()

    md_content = weekly_md_path.read_text(encoding="utf-8")

    # Strip <details> blocks
    cleaned_md = re.sub(r"<details\b[^>]*>.*?</details>", " ", md_content, flags=re.S | re.I)
    vis_lines = [l for l in cleaned_md.splitlines() if l.strip()]
    assert len(vis_lines) <= 320, f"Weekly visible lines {len(vis_lines)} exceeded max 320!"

    # QA Check U1 verification
    u1_res = check_u1_reading_time_budget([], md_content, "", "weekly")
    assert u1_res.status == "PASS", f"QA U1 check failed on weekly report: {u1_res.details}"


def test_substage_5_1_single_visible_volatility_section():
    """Verify Section 7 is primary visible vol section and §16 vol prose is wrapped in <details>."""
    weekly_md_path = Path("reports/weekly/2026-11-13.md")
    md_content = weekly_md_path.read_text(encoding="utf-8")

    # Section 7 is visible outside <details>
    cleaned_md = re.sub(r"<details\b[^>]*>.*?</details>", " ", md_content, flags=re.S | re.I)
    assert "## 7. Volatility (VIX)" in cleaned_md

    # In §16, VRP / options surface detail is inside <details>
    assert "<summary>Volatility Risk Premium (VRP) & Options Surface Detail</summary>" in md_content


# ----------------------------------------------------------------------
# Substage 5.2: "The Week Ahead" Grid & Earnings on Deck
# ----------------------------------------------------------------------

def test_substage_5_2_classify_macro_sensitivity():
    """Verify macro release sensitivity mapping produces appropriate asset classes."""
    assert "Duration & USD" in classify_macro_sensitivity("FOMC Minutes")
    assert "Yields" in classify_macro_sensitivity("CPI Report")
    assert "Cyclicals" in classify_macro_sensitivity("Retail Sales")
    assert "Fed Path" in classify_macro_sensitivity("Initial Jobless Claims")


def test_substage_5_2_the_week_ahead_rendered():
    """Verify The Week Ahead and Earnings on Deck are rendered in Section 12."""
    weekly_md_path = Path("reports/weekly/2026-11-13.md")
    weekly_html_path = Path("reports/weekly/2026-11-13.html")

    md_content = weekly_md_path.read_text(encoding="utf-8")
    html_content = weekly_html_path.read_text(encoding="utf-8")

    assert "The Week Ahead (Monday–Friday Catalyst Grid)" in md_content
    assert "The Week Ahead (Monday" in html_content and "Catalyst Grid)" in html_content
    assert "Earnings on Deck (Notable Large-Cap Reports)" in md_content
    assert "Earnings on Deck (Notable Large-Cap Reports)" in html_content

    # Check table column headers in markdown
    assert "Market Most Sensitive" in md_content
    assert "Day / Date" in md_content


# ----------------------------------------------------------------------
# Substage 5.3: Positioning: CFTC COT Institutional Futures
# ----------------------------------------------------------------------

def test_substage_5_3_cot_provider_offline_and_benchmark_contracts():
    """Verify COT provider loads benchmark contracts and formats percentiles/regimes cleanly."""
    provider = COTProvider()
    positions = provider.get_positioning("2026-11-13")

    assert len(positions) == 4, f"Expected 4 benchmark contracts, got {len(positions)}"
    symbols = {p.contract_symbol for p in positions}
    assert symbols == {"ES", "ZN", "CL", "GC"}

    valid_regimes = {
        "Crowded Long (>=80th %ile)",
        "Crowded Short (<=20th %ile)",
        "Neutral / Balanced",
    }

    for p in positions:
        assert isinstance(p.net_position, int)
        assert isinstance(p.net_change_1w, int)
        assert 0.0 <= p.percentile_3y <= 100.0
        assert p.regime in valid_regimes
        # Ensure percentile formatting ordinal does not produce '21th'
        assert not p.percentile_label.endswith("1th") or p.percentile_label.endswith("11th")


def test_substage_5_3_cot_custom_fixture(tmp_path):
    """Verify COTProvider loads cleanly from an isolated custom JSON fixture."""
    fixture_data = {
        "2026-11-13": {
            "ES": {
                "contract_symbol": "ES",
                "contract_name": "S&P 500 E-Mini (ES)",
                "net_position": 50000,
                "net_change_1w": 2500,
                "percentile_3y": 85.0,
            }
        }
    }
    fixture_file = tmp_path / "test_cot.json"
    fixture_file.write_text(json.dumps(fixture_data), encoding="utf-8")

    provider = COTProvider(cache_path=str(fixture_file))
    res = provider.get_positioning("2026-11-13")
    assert len(res) == 4
    es_item = next(p for p in res if p.contract_symbol == "ES")
    assert es_item.net_position == 50000
    assert es_item.regime == "Crowded Long (>=80th %ile)"


def test_substage_5_3_cot_rendered_in_section_16():
    """Verify CFTC COT Institutional Futures Positioning table is visible in Weekly §16."""
    weekly_md_path = Path("reports/weekly/2026-11-13.md")
    weekly_html_path = Path("reports/weekly/2026-11-13.html")

    md_content = weekly_md_path.read_text(encoding="utf-8")
    html_content = weekly_html_path.read_text(encoding="utf-8")

    assert "CFTC COT Institutional Futures Positioning" in md_content
    assert "CFTC COT Institutional Futures Positioning" in html_content
    assert "S&P 500 E-Mini (ES)" in md_content
    assert "10-Year US Treasury Note (ZN)" in md_content


# ----------------------------------------------------------------------
# Substage 5.4: Model Accountability
# ----------------------------------------------------------------------

def test_substage_5_4_model_accountability_round_trip(tmp_path):
    """Verify model accountability persists snapshot and evaluates adherence accurately."""
    storage_file = tmp_path / "weekly_accountability.json"

    ranges = [
        RiskRangeItem(
            name="S&P 500 ETF",
            symbol="SPY",
            last_price=750.0,
            range_low=730.0,
            pivot_20ema=750.0,
            range_high=770.0,
            position_pct=50.0,
            atr_pct=1.5,
            trade_dir="BEARISH",
            trend_dir="BULLISH",
            tail_dir="BULLISH",
            zone_note="Neutral",
        ),
        RiskRangeItem(
            name="Nasdaq-100 ETF",
            symbol="QQQ",
            last_price=680.0,
            range_low=650.0,
            pivot_20ema=680.0,
            range_high=710.0,
            position_pct=50.0,
            atr_pct=2.0,
            trade_dir="BEARISH",
            trend_dir="NEUTRAL",
            tail_dir="BULLISH",
            zone_note="Neutral",
        ),
    ]

    record_weekly_snapshot(
        as_of_date="2026-11-06",
        posture="DEFENSIVE",
        risk_ranges=ranges,
        file_path=str(storage_file),
    )

    assert storage_file.exists()
    with open(storage_file) as f:
        data = json.load(f)
    assert "2026-11-06" in data

    # Evaluate with current bar holding inside range
    idx_dfs = {
        "SPY": pd.DataFrame({
            "High": [760.0],
            "Low": [740.0],
            "Close": [755.0],
        }, index=pd.date_range("2026-11-13", periods=1)),
        "QQQ": pd.DataFrame({
            "High": [720.0],  # breach above
            "Low": [660.0],
            "Close": [715.0],
        }, index=pd.date_range("2026-11-13", periods=1)),
    }

    adherence = evaluate_weekly_range_adherence(
        as_of_date="2026-11-13",
        prior_week_date="2026-11-06",
        current_bars=idx_dfs,
        file_path=str(storage_file),
    )

    assert adherence.total_assets == 6
    assert adherence.ranges_held == 5
    assert "QQQ (above resistance)" in adherence.summary_text


def test_substage_5_4_model_accountability_t13_compliance():
    """Verify Weekly §3 range adherence contains strictly impersonal language (no trading verbs)."""
    weekly_md_path = Path("reports/weekly/2026-11-13.md")
    md_content = weekly_md_path.read_text(encoding="utf-8")

    m = re.search(r"Prior Week Model Range Adherence:\*{0,2}\s*(.*?)\n", md_content)
    assert m, "Missing 'Prior Week Model Range Adherence' line in §3"
    line = m.group(1).lower()

    # Verify no action verbs or hypothetical returns
    for banned in ["buy", "sell", "profit", "return", "gain", "loss", "traded", "entry", "stop"]:
        assert f" {banned} " not in f" {line} ", f"Found non-compliant verb '{banned}' in range adherence line: {line}"


# ----------------------------------------------------------------------
# Substage 5.5: Sector Rotation Quadrant & Candle Structure
# ----------------------------------------------------------------------

def test_substage_5_5_sector_rotation_quadrant_computation():
    """Verify compute_sector_rotation_quadrants classifies all 11 SPDR sectors."""
    dates = pd.date_range("2026-06-01", "2026-11-13", freq="B")
    spy_df = pd.DataFrame({
        "Open": np.linspace(500, 550, len(dates)),
        "High": np.linspace(505, 555, len(dates)),
        "Low": np.linspace(495, 545, len(dates)),
        "Close": np.linspace(500, 550, len(dates)),
    }, index=dates)

    sector_dfs = {}
    for sym in ["XLK", "XLV", "XLF", "XLY", "XLP", "XLI", "XLE", "XLB", "XLU", "XLRE", "XLC"]:
        sector_dfs[sym] = pd.DataFrame({
            "Open": np.linspace(100, 110, len(dates)),
            "High": np.linspace(102, 112, len(dates)),
            "Low": np.linspace(98, 108, len(dates)),
            "Close": np.linspace(100, 110, len(dates)),
        }, index=dates)

    curr_end_dt = dates[-1]
    base_dt = dates[-5]
    prev_end_dt = dates[-5]
    prev_base_dt = dates[-10]

    quadrants = compute_sector_rotation_quadrants(
        bars=sector_dfs,
        spy_df=spy_df,
        curr_end_dt=curr_end_dt,
        base_dt=base_dt,
        prev_end_dt=prev_end_dt,
        prev_base_dt=prev_base_dt,
    )
    assert len(quadrants) == 11

    valid_quadrants = {"Leading", "Weakening", "Lagging", "Improving"}
    valid_structures = {
        "Close Near High (Top 25%)",
        "Close Near Low (Bottom 25%)",
        "Inside Week",
        "Outside Range",
        "Mid-Range Consolidation",
    }

    for q in quadrants:
        assert q.quadrant in valid_quadrants
        assert q.candle_structure in valid_structures
        assert isinstance(q.rs_10w_pct, float)
        assert isinstance(q.rs_momentum_pct, float)


def test_substage_5_5_sector_rotation_quadrant_rendered():
    """Verify Sector Rotation Quadrants & Weekly Candle Structure table is in §8."""
    weekly_md_path = Path("reports/weekly/2026-11-13.md")
    weekly_html_path = Path("reports/weekly/2026-11-13.html")

    md_content = weekly_md_path.read_text(encoding="utf-8")
    html_content = weekly_html_path.read_text(encoding="utf-8")

    assert "Sector Rotation Quadrants & Weekly Candle Structure" in md_content
    assert "Sector Rotation Quadrants" in html_content and "Weekly Candle Structure" in html_content
    assert "10-Week RS vs. Momentum Quadrant" in md_content
    assert "Weekly Candle Structure" in md_content


# ----------------------------------------------------------------------
# Substage 5.6: 4-Pillar Macro Cycle Dashboard
# ----------------------------------------------------------------------

def test_substage_5_6_macro_cycle_dashboard_rendered():
    """Verify 4-Pillar Macro Cycle Dashboard is visible in §17 while raw dashboard is wrapped."""
    weekly_md_path = Path("reports/weekly/2026-11-13.md")
    weekly_html_path = Path("reports/weekly/2026-11-13.html")

    md_content = weekly_md_path.read_text(encoding="utf-8")
    html_content = weekly_html_path.read_text(encoding="utf-8")

    assert "4-Pillar Macro Cycle Dashboard" in md_content
    assert "4-Pillar Macro Cycle Dashboard" in html_content
    assert "Growth & Activity" in md_content
    assert "Inflation & Pricing" in md_content
    assert "Labor & Employment" in md_content
    assert "Liquidity & Credit" in md_content

    # Raw standing dashboard is inside <details>
    assert "<summary>Full Macro Indicator & Historical Appendix</summary>" in md_content
    assert "<details" in html_content


def test_weekly_full_qa_harness_pass():
    """Verify all 15 QA harness checks pass on production weekly report."""
    qa_path = Path("reports/weekly/qa_report.md")
    assert qa_path.exists(), "Weekly QA report missing"
    qa_content = qa_path.read_text(encoding="utf-8")

    assert "Status:** ✅ PASSED" in qa_content
    assert "15/15 checks passed" in qa_content
