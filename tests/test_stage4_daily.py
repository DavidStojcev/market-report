"""Stage 4 (Daily -> 10/10) Verification Suite.

Tests for Substages 4.1 to 4.5:
- 4.1: Executive Summary Restructure (Hero KPI strip, 3 core blocks, <details> Detail & Methodology)
- 4.2: Chart of the Day Deterministic Selection (max |pctile - 50|, tie-breaking)
- 4.3: Next 48 Hours Catalyst Playbook (calendar + earnings, no fabricated consensus, chronological)
- 4.4: Compact Sector Line & Cross-Asset Line + Collapsible Verbosity Compression
- 4.5: Reading-Time Budget QA Check U1 (<= 1400 visible words for daily)
"""

from pathlib import Path
import re
from typing import Any, Dict
import pytest

from data.calendar import PeriodInfo
from qa_checks import check_u1_reading_time_budget
from report import generate_market_report
from sections.executive_summary import (
    ChartOfTheDay,
    ExecutiveSummaryData,
    select_chart_of_the_day,
)


@pytest.fixture(scope="module")
def daily_report_result() -> Dict[str, Any]:
    """Generate the standard daily production report for 2026-11-13."""
    return generate_market_report(report_type="daily", target_date="2026-11-13", quiet=True)


def test_4_1_executive_summary_daily_restructure(daily_report_result):
    """Substage 4.1: Verify daily §1 structure, Hero KPI strip, 3 core blocks, and <details> wrapping."""
    md = Path(daily_report_result["markdown"]).read_text(encoding="utf-8")
    html = Path(daily_report_result["html"]).read_text(encoding="utf-8")

    # 1. Hero KPI strip in Markdown
    assert "| S&P 500 | Nasdaq-100 | 10Y Yield | CBOE VIX | Breadth >50-DMA | Model Posture |" in md
    assert "DEFENSIVE" in md

    # 2. Morning Brief Takeaway
    assert "Morning Brief Takeaway:" in md

    # 3. 3 Things That Mattered
    assert "3 Things That Mattered" in md

    # 4. What Happened -> What It Means -> Model Read
    assert "What Happened → What It Means → Model Read" in md
    assert "- **What Happened:**" in md
    assert "- **What It Means:**" in md
    assert "- **The Bottom Line (Model Read):**" in md

    # 5. Detail & Methodology collapsible wrapper
    assert "<details><summary>Detail & Methodology</summary>" in md
    assert "Regime vs. Posture Alignment:" in md
    assert "Macro Regime & Forward Implications" in md
    assert "Key Drivers & Rationale:" in md
    assert "Multi-Horizon & Cross-Strategy Synthesis" in md

    # 6. HTML structure parity
    assert "Detail &amp; Methodology" in html or "Detail & Methodology" in html
    assert "What Happened:" in html
    assert "What It Means:" in html
    assert "The Bottom Line:" in html

    # 7. Hero KPI strip 10Y Yield does not render n/a when rate data is present
    kpi_row_match = re.search(
        r"\|\s*S&P 500\s*\|\s*Nasdaq-100\s*\|\s*10Y Yield\s*\|\s*CBOE VIX\s*\|\s*Breadth >50-DMA\s*\|\s*Model Posture\s*\|\s*\n"
        r"\|\s*[-:]+\s*\|\s*[-:]+\s*\|\s*[-:]+\s*\|\s*[-:]+\s*\|\s*[-:]+\s*\|\s*[-:]+\s*\|\s*\n"
        r"\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|",
        md,
    )
    assert kpi_row_match is not None, "Hero KPI strip table row not found"
    y10_cell = kpi_row_match.group(3).strip()
    assert y10_cell != "n/a", f"10Y Yield in Daily Hero KPI Strip should not be 'n/a', got: {y10_cell}"
    assert "%" in y10_cell

    # HTML Hero KPI strip has 10Y Treasury and not n/a
    assert "10Y Treasury" in html
    assert "5.10%" in html


def test_4_2_chart_of_the_day_deterministic_selection():
    """Substage 4.2: Verify deterministic selection logic for Chart of the Day."""
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
        notes="",
    )

    class MockVol:
        vix_pctile_1y = 20.0       # |20 - 50| = 30
        current_level = 15.0
        vrp_pctile_1y = 95.0       # |95 - 50| = 45 -> winner
        vrp_20d = -3.2
        vvix_pctile_1y = 60.0      # |60 - 50| = 10
        vvix_level = 90.0
        skew_pctile_1y = 55.0      # |55 - 50| = 5
        skew_level = 125.0

    class MockBreadth:
        pct_above_50_dma_pctile_1y = 80.0   # |80 - 50| = 30
        pct_above_200_dma_pctile_1y = 70.0  # |70 - 50| = 20
        net_new_highs_pctile_1y = 10.0      # |10 - 50| = 40
        pct_above_50_dma = None
        pct_above_200_dma = None
        net_new_highs = 5

    cotd = select_chart_of_the_day(period, vol_data=MockVol(), breadth_data=MockBreadth())
    assert cotd is not None
    assert cotd.metric_name == "VRP 1Y %ile"
    assert cotd.percentile_1y == 95.0
    assert "vix_regime.png" in cotd.chart_path

    # Tie-breaker test: Candidate 1 (VIX) vs Candidate 2 (VRP) with identical extreme distance
    class MockTieVol:
        vix_pctile_1y = 90.0   # |90 - 50| = 40
        current_level = 28.0
        vrp_pctile_1y = 10.0   # |10 - 50| = 40
        vrp_20d = -4.5
        vvix_pctile_1y = 50.0
        vvix_level = 80.0
        skew_pctile_1y = 50.0
        skew_level = 120.0

    cotd_tie = select_chart_of_the_day(period, vol_data=MockTieVol(), breadth_data=None)
    assert cotd_tie is not None
    # Deterministic tie-breaker selects earlier metric order: VIX (1) before VRP (2)
    assert cotd_tie.metric_name == "VIX 1Y %ile"
    assert "%ile 1-year percentile" not in cotd_tie.annotation
    assert "(1Y)" in cotd_tie.annotation


def test_4_2_chart_of_the_day_in_report(daily_report_result):
    """Substage 4.2: Verify Chart of the Day renders in Daily report and referenced file exists."""
    md = Path(daily_report_result["markdown"]).read_text(encoding="utf-8")
    assert "Chart of the Day:" in md

    # Check caption has no duplicate %ile and redundant words
    assert "%ile 1-year percentile" not in md, "Duplicate '%ile 1-year percentile' found in Daily report"
    assert "(1Y)" in md

    # Extract chart image path
    m = re.search(r"!\[Chart of the Day:[^\]]+\]\(([^)]+)\)", md)
    assert m is not None, "Chart of the Day image markdown not found"
    img_rel_path = m.group(1)
    img_file = Path(img_rel_path)
    assert img_file.exists(), f"Chart of the day image file {img_rel_path} does not exist on disk"


def test_4_3_next_48_hours_box(daily_report_result):
    """Substage 4.3: Verify Next 48H box shows real scheduled releases and earnings without fabricated consensus."""
    md = Path(daily_report_result["markdown"]).read_text(encoding="utf-8")

    assert "Next 48H Catalyst Playbook" in md
    assert "Imminent Scheduled Macro Releases" in md
    assert "Notable S&P 500 Earnings" in md

    # Ensure unpolled consensus displays em-dash and not fabricated numbers
    macro_table_match = re.search(
        r"\*{0,2}Imminent Scheduled Macro Releases\*{0,2}\s*\n(\|.+?\n\s*\n)", md, flags=re.S
    )
    assert macro_table_match is not None
    macro_table = macro_table_match.group(1)
    # At least one consensus cell must be '—'
    assert "—" in macro_table


def test_4_4_compact_sector_line_and_cross_asset_line(daily_report_result):
    """Substage 4.4: Verify Sector Line and Cross-Asset Line in Section 4."""
    md = Path(daily_report_result["markdown"]).read_text(encoding="utf-8")

    # 1. Compact Sector Line
    assert "**11 SPDR Sectors (1D):**" in md
    for sector_etf in ("XLK", "XLF", "XLE", "XLV", "XLP", "XLU", "XLY", "XLI", "XLB", "XLC", "XLRE"):
        assert sector_etf in md

    # 2. Compact Cross-Asset Line
    assert "**Cross-Asset Line (1D):**" in md
    for asset_kw in ("10Y Yield", "DXY", "WTI", "Gold", "Bitcoin"):
        assert asset_kw in md


def test_4_5_reading_time_budget_u1_pass(daily_report_result):
    """Substage 4.5: Verify U1 passes on production daily report with visible_words <= 1400."""
    md = Path(daily_report_result["markdown"]).read_text(encoding="utf-8")
    res = check_u1_reading_time_budget(md_content=md, report_type="daily")
    assert res.status == "PASS"

    # Verify visible word count is <= 1400
    m = re.search(r"Visible words outside <details>: (\d+)", res.details[0])
    assert m is not None
    word_count = int(m.group(1))
    assert word_count <= 1400, f"Expected <= 1400 words, got {word_count}"


def test_4_5_reading_time_budget_u1_fail():
    """Substage 4.5: Verify U1 fails when visible words exceed 1400."""
    long_synthetic_md = "word " * 1500
    res = check_u1_reading_time_budget(md_content=long_synthetic_md, report_type="daily")
    assert res.status == "FAIL"
    assert any("budget exceeded" in d for d in res.details)


def test_4_5_reading_time_budget_weekly_and_monthly():
    """Substage 4.5: Verify U1 passes for weekly and monthly report types under their bounds."""
    text_2000 = "market analysis " * 1000  # 2000 words
    res_w = check_u1_reading_time_budget(md_content=text_2000, report_type="weekly")
    assert res_w.status == "PASS"

    res_m = check_u1_reading_time_budget(md_content=text_2000, report_type="monthly")
    assert res_m.status == "PASS"


def test_4_all_15_qa_checks_pass(daily_report_result):
    """Verify that all 15 QA checks (T1-T13, R3, U1) pass on the daily report."""
    qa_md = Path(daily_report_result["qa_report"]).read_text(encoding="utf-8")
    assert "15/15 checks passed" in qa_md
    assert "| **R3** |" in qa_md
    assert "| **U1** |" in qa_md
    assert "❌ FAILED" not in qa_md
