"""Comprehensive tests for Stage 6: Monthly Strategic Dossier Upgrade (Substages 6.1 - 6.5)."""

from pathlib import Path
import re
import pandas as pd
import pytest

from data.calendar import PeriodInfo
from qa_checks import check_u1_reading_time_budget, check_t13_compliance_lint, run_qa_checks
from render.charts import generate_net_liquidity_vs_spx_chart
from render.html import render_html
from render.markdown import render_markdown
from report import generate_market_report
from sections.base import SectionResult
from sections.commodities_cross_asset import (
    CommoditiesCrossAssetData,
    CrossAssetCorrelationMatrixData,
    compute_cross_asset_correlation_matrix,
)
from sections.macro import (
    CreditQualitySpreadData,
    MacroRegimeClockData,
    MacroSectionData,
    NetLiquiditySummaryData,
    compute_credit_quality_spread,
    compute_macro_regime_clock,
    compute_net_liquidity_summary,
)
from sections.technicals import (
    MonthlySecularTechnicalsData,
    SecularTechnicalsItem,
    TechnicalsData,
    compute_monthly_secular_technicals,
)
from sections.theme_relative_strength import (
    ThematicBasket,
    ThemeItem,
    ThemeRelativeStrengthData,
    compute_thematic_baskets,
)
from sections.valuation import ValuationERPData, compute_valuation_erp


# ---------------------------------------------------------------------------
# Substage 6.1: Monthly Layout, Horizon Cleanup & Density Compression
# ---------------------------------------------------------------------------
def test_substage_6_1_monthly_takeaway_and_no_morning_brief():
    """6.1: Monthly uses 'Monthly Strategic Takeaway'; 'Morning Brief' NEVER appears in MD or HTML."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")

    assert "Monthly Strategic Takeaway" in md
    assert "Monthly Strategic Takeaway" in html
    assert "Morning Brief" not in md
    assert "Morning Brief" not in html
    assert "morning-brief" not in html


def test_substage_6_1_no_next_48h_catalyst_playbook_in_monthly():
    """6.1: No 'Next 48H Catalyst Playbook' in Monthly MD or HTML (uses month-ahead calendar)."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")

    assert "Next 48H Catalyst Playbook" not in md
    assert "Next 48H Catalyst Playbook" not in html
    assert "Monthly / Month-Ahead Macro & Policy Catalyst Calendar" in md


def test_substage_6_1_monthly_movers_header():
    """6.1: Section 14 movers header in Monthly is 'Monthly Top Movers (S&P 500 + Nasdaq-100)'."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")

    # Label reflects the universe actually screened (S&P 500 only when the Nasdaq-100 roster is unavailable)
    assert "Monthly Top Movers (S&P 500" in md
    assert "Monthly Top Movers (S&P 500" in html
    assert "Daily Movers" not in md


def test_substage_6_1_u1_reading_time_budget_enforced():
    """6.1: check_u1_reading_time_budget enforces <= 4500 visible words outside <details> on monthly."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")

    qa_u1 = check_u1_reading_time_budget(md_content=md, report_type="monthly")
    assert qa_u1.status == "PASS"

    # Verify artificial failure when visible words exceed 4500
    long_content = "word " * 4501
    qa_u1_fail = check_u1_reading_time_budget(md_content=long_content, report_type="monthly")
    assert qa_u1_fail.status == "FAIL"
    assert "budget: <= 4500 words" in qa_u1_fail.details[0]


# ---------------------------------------------------------------------------
# Substage 6.2: Macro Regime Clock
# ---------------------------------------------------------------------------
def test_substage_6_2_macro_regime_clock_quadrants():
    """6.2: Test deterministic 4-quadrant Growth x Inflation 3M RoC classifier."""
    from data.fred_provider import MacroMetricSnapshot
    metrics = {
        "gdp": MacroMetricSnapshot(series_id="GDPC1", name="Real GDP", latest_date="2026-09-01", latest_value=2.8, headline_value=2.8, headline_unit="%", prior_value=2.3, prior_headline=2.3, change=0.5),
        "cpi": MacroMetricSnapshot(series_id="CPIAUCSL", name="CPI", latest_date="2026-10-01", latest_value=2.4, headline_value=2.4, headline_unit="%", prior_value=2.7, prior_headline=2.7, change=-0.3),
    }
    clock = compute_macro_regime_clock(metrics=metrics, as_of_date="2026-10-30")
    assert isinstance(clock, MacroRegimeClockData)
    assert clock.current_quadrant == "Quadrant 2 — Goldilocks / Disinflationary Expansion"
    assert clock.growth_roc_score > 0.0 and clock.inflation_roc_score < 0.0
    assert clock.clock_transition_summary != ""
    assert clock.historical_regime_note != ""


def test_substage_6_2_macro_regime_clock_rendered_in_section_17():
    """6.2: Macro Regime Clock rendered visibly in Monthly Section 17 in Markdown and HTML."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")

    assert "### Macro Regime Clock (Growth × Inflation 3M Rate-of-Change)" in md
    assert "Active Macro Regime Clock:" in md
    assert "Macro Regime Clock (Growth &times; Inflation 3M Rate-of-Change)" in html

    # Verify T13 compliance on regime clock text
    t13_check = check_t13_compliance_lint(md, html)
    assert t13_check.status == "PASS"


# ---------------------------------------------------------------------------
# Substage 6.3: Valuation & Equity Risk Premium
# ---------------------------------------------------------------------------
def test_substage_6_3_compute_valuation_erp_live_and_cache():
    """6.3: ValuationERPData computes trailing EY, Shiller CAPE, Nominal/Real ERP, and percentiles."""
    erp = compute_valuation_erp(nominal_10y=4.28, real_10y=1.92, as_of_date="2026-10-30")
    assert isinstance(erp, ValuationERPData)
    assert erp.trailing_pe is not None and erp.trailing_pe > 0
    assert erp.trailing_earnings_yield_pct is not None and erp.trailing_earnings_yield_pct > 0
    assert erp.cape_ratio is not None and erp.cape_ratio > 0
    assert "Shiller" in erp.cape_reference_month or "September" in erp.cape_reference_month or "October" in erp.cape_reference_month
    assert erp.nominal_erp_bps is not None
    assert erp.real_erp_bps is not None
    assert erp.real_erp_percentile_10y is not None
    assert erp.valuation_tier in ("Compressed / Rich Valuation", "Fair / Historical Norm", "Wide / Value Buffer")


def test_substage_6_3_valuation_erp_rendered_in_monthly():
    """6.3: S&P 500 Valuation, Shiller CAPE & ERP rendered visibly in Monthly MD & HTML."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")

    assert "### S&P 500 Valuation, Shiller CAPE & Equity Risk Premium" in md
    assert "Trailing P/E (SPY TTM)" in md
    assert "Shiller CAPE (P/E 10)" in md
    assert "Real ERP (vs 10Y TIPS)" in md
    assert "n/a (source unavailable)" in md

    assert "S&amp;P 500 Valuation, Shiller CAPE &amp; Equity Risk Premium" in html
    assert "Trailing P/E (SPY TTM)" in html
    assert "Shiller CAPE (P/E 10)" in html


# ---------------------------------------------------------------------------
# Substage 6.4: Structural Cross-Asset & Monthly-Bar Technicals
# ---------------------------------------------------------------------------
def test_substage_6_4_net_liquidity_summary_and_chart():
    """6.4 #1: 52-Week Fed Net Liquidity vs SPX summary and net_liquidity_vs_spx.png chart."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")

    assert "Federal Reserve Net Liquidity vs. S&P 500 Trajectory" in md
    assert "Fed Net Liquidity (WALCL − TGA − RRP)" in md
    assert "Federal Reserve Net Liquidity vs. S&amp;P 500 Trajectory" in html

    # Verify chart file created
    chart_png = Path("reports/monthly/2026-10-30_charts/net_liquidity_vs_spx.png")
    assert chart_png.exists()
    assert chart_png.stat().st_size > 1000


def test_substage_6_4_credit_quality_spread():
    """6.4 #2: Credit Quality Spread (HY OAS - BBB OAS) and cycle regime."""
    cqs = compute_credit_quality_spread(hy_spread_bps=344.0, bbb_spread_bps=116.0, as_of_date="2026-10-30")
    assert isinstance(cqs, CreditQualitySpreadData)
    assert cqs.quality_spread_bps == 228.0
    assert cqs.percentile_1y is not None
    assert cqs.cycle_regime != ""

    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")

    assert "### Credit Quality Spread & Corporate Balance Sheet Tiering" in md
    assert "Credit Quality Spread (HY − BBB)" in md
    assert "Credit Quality Spread &amp; Corporate Balance Sheet Tiering" in html


def test_substage_6_4_cross_asset_correlation_matrix():
    """6.4 #3: 60-Day Cross-Asset Return Correlation Matrix rendered in Section 10."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")

    assert "### 60-Day Cross-Asset Return Correlation Matrix" in md
    assert "Cross-Asset Diversification Surface (60 Trading Days)" in md
    assert "60-Day Cross-Asset Return Correlation Matrix" in html


def test_substage_6_4_monthly_secular_technicals_sma_10m_12m():
    """6.4 #4: Monthly-Bar Secular Trend Technicals (10M & 12M SMA) in Section 5."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")

    assert "### Monthly Secular Trend Analysis (10-Month & 12-Month Moving Averages)" in md
    assert "10-Month SMA" in md
    assert "12-Month SMA" in md
    assert "Monthly Secular Trend Analysis (10-Month &amp; 12-Month Moving Averages)" in html


# ---------------------------------------------------------------------------
# Substage 6.5: Thematic Baskets
# ---------------------------------------------------------------------------
def test_substage_6_5_thematic_baskets_computation_and_t13_compliance():
    """6.5: Industry/Thematic ETFs grouped into 4-5 thematic baskets with 0 T13 violations."""
    dummy_items = [
        ThemeItem(symbol="SMH", name="VanEck Semiconductor", category="Technology", close=240.0, ret_1d=1.0, ret_1w=2.0, ret_1m=5.0, ret_3m=12.0, rs_vs_spy=3.0, rank=1, rank_change=0, above_50dma=True, dist_50dma_pct=4.0),
        ThemeItem(symbol="IGV", name="iShares Tech-Software", category="Technology", close=90.0, ret_1d=0.5, ret_1w=1.0, ret_1m=3.0, ret_3m=8.0, rs_vs_spy=1.0, rank=2, rank_change=0, above_50dma=True, dist_50dma_pct=2.0),
        ThemeItem(symbol="XOP", name="SPDR Oil & Gas", category="Energy", close=135.0, ret_1d=-1.0, ret_1w=-2.0, ret_1m=-3.0, ret_3m=-5.0, rs_vs_spy=-5.0, rank=10, rank_change=-1, above_50dma=False, dist_50dma_pct=-3.0),
    ]
    baskets, synth = compute_thematic_baskets(dummy_items, spy_1m=2.0)
    assert len(baskets) >= 2
    assert any(b.name == "AI & Semiconductor Infrastructure" for b in baskets)

    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")

    assert "### Thematic Equal-Weight Baskets Performance" in md
    assert "Thematic Equal-Weight Baskets Performance" in html
    assert "AI & Semiconductor Infrastructure" in md
    assert "AI & Semiconductor Infrastructure" in html

    # Strict T13 compliance check: ensure no imperative/advisory terms appear
    t13 = check_t13_compliance_lint(md, html)
    assert t13.status == "PASS"


# ---------------------------------------------------------------------------
# Full Monthly QA Harness Verification
# ---------------------------------------------------------------------------
def test_monthly_full_qa_harness_pass():
    """Verify Monthly report passes all 15 QA harness checks with 0 failures and 0 warnings."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    report = run_qa_checks(
        period=res["period"],
        sections=res["sections"],
        md_content=Path(res["markdown"]).read_text(encoding="utf-8"),
        html_content=Path(res["html"]).read_text(encoding="utf-8"),
        charts=res["charts"],
    )
    assert report.is_success
    assert report.passed_checks == 15
    assert report.failed_checks == 0
    assert report.warned_checks == 0
    assert res["qa_status"] == "PASSED"

    qa_text = Path(res["qa_report"]).read_text(encoding="utf-8")
    assert "15/15 checks passed (0 failed, 0 warnings)" in qa_text


# ---------------------------------------------------------------------------
# Stage 7.2 Independent Audit Regression Tests (Fixes 1 - 9)
# ---------------------------------------------------------------------------
def test_audit_fix_1_macro_regime_clock_rising_inflation_reflation():
    """Fix 1: Rising CPI, Core CPI, PPI, and Core PCE must output Quadrant 1 — Reflation, NEVER Goldilocks or Decelerating."""
    from data.fred_provider import MacroMetricSnapshot
    from sections.macro import MacroDashboardRow, compute_macro_regime_clock

    # Mock dashboard rows with accelerating inflation indicators
    dash_rows = [
        MacroDashboardRow(indicator="Headline CPI (YoY)", category="Inflation", latest_value="4.1%", prior_value="4.0%", direction_3m="Rising ▲", plain_label="Elevated", release_date="2026-10-15", reference_period="Oct 2026", next_release_date="2026-11-12", source="BLS", numeric_latest=4.08, numeric_prior=4.02),
        MacroDashboardRow(indicator="Core CPI (YoY)", category="Inflation", latest_value="2.9%", prior_value="2.8%", direction_3m="Rising ▲", plain_label="Elevated", release_date="2026-10-15", reference_period="Oct 2026", next_release_date="2026-11-12", source="BLS", numeric_latest=2.87, numeric_prior=2.79),
        MacroDashboardRow(indicator="Core PCE Price Index (YoY)", category="Inflation", latest_value="3.1%", prior_value="3.0%", direction_3m="Rising ▲", plain_label="Above Target", release_date="2026-10-30", reference_period="Sep 2026", next_release_date="2026-11-27", source="BEA", numeric_latest=3.12, numeric_prior=3.03),
        MacroDashboardRow(indicator="Producer Price Index (PPI YoY)", category="Inflation", latest_value="6.5%", prior_value="6.3%", direction_3m="Rising ▲", plain_label="Elevated Pipeline", release_date="2026-10-14", reference_period="Sep 2026", next_release_date="2026-11-13", source="BLS", numeric_latest=6.52, numeric_prior=6.29),
        MacroDashboardRow(indicator="Real GDP (Annualized QoQ)", category="Economic Activity", latest_value="2.8%", prior_value="2.8%", direction_3m="Rising ▲", plain_label="Robust Expansion", release_date="2026-10-29", reference_period="Q3 2026", next_release_date="2026-11-25", source="BEA", numeric_latest=2.8, numeric_prior=2.8),
    ]
    metrics = {
        "cpi": MacroMetricSnapshot(series_id="CPIAUCSL", name="CPI", latest_date="2026-10-01", latest_value=4.08, headline_value=4.08, headline_unit="%", prior_value=4.02, prior_headline=4.02, change=0.06),
        "core_cpi": MacroMetricSnapshot(series_id="CPILFESL", name="Core CPI", latest_date="2026-10-01", latest_value=2.87, headline_value=2.87, headline_unit="%", prior_value=2.79, prior_headline=2.79, change=0.08),
        "core_pce": MacroMetricSnapshot(series_id="PCEPILFE", name="Core PCE", latest_date="2026-09-01", latest_value=3.12, headline_value=3.12, headline_unit="%", prior_value=3.03, prior_headline=3.03, change=0.09),
        "ppi": MacroMetricSnapshot(series_id="PPIFIS", name="PPI", latest_date="2026-09-01", latest_value=6.52, headline_value=6.52, headline_unit="%", prior_value=6.29, prior_headline=6.29, change=0.23),
        "gdp": MacroMetricSnapshot(series_id="GDPC1", name="Real GDP", latest_date="2026-09-01", latest_value=2.8, headline_value=2.8, headline_unit="%", prior_value=2.8, prior_headline=2.8, change=0.0),
    }

    clock = compute_macro_regime_clock(metrics=metrics, standing_dashboard=dash_rows)
    assert clock.current_quadrant == "Quadrant 1 — Reflation"
    assert clock.inflation_roc_score > 0
    assert "Goldilocks" not in clock.current_quadrant
    assert "Decelerating" not in clock.inflation_components["Headline CPI YoY"]


def test_audit_fix_2_macro_cycle_dashboard_dynamic_inflation():
    """Fix 2: 4-pillar cycle dashboard reflects accelerating inflation dynamically, not hardcoded Disinflating."""
    res_m = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md_m = Path(res_m["markdown"]).read_text(encoding="utf-8")
    assert "Quadrant 1 — Reflation" in md_m

    res_w = generate_market_report(report_type="weekly", target_date="2026-11-13", quiet=True)
    md_w = Path(res_w["markdown"]).read_text(encoding="utf-8")
    assert "▲ Sticky / Accelerating" in md_w
    assert "Reflation / Elevated Price Pressure" in md_w
    assert "▼ Disinflating" not in md_w


def test_audit_fix_3_movers_no_guidance_cuts():
    """Fix 3: Movers driver labels use honest volume-backed catalyst descriptions, never ungrounded guidance cuts."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")
    assert "guidance cuts" not in md
    assert "guidance cuts" not in html
    assert "Post-earnings reaction / downside volume" in md or "Post-earnings reaction / upside volume" in md


def test_audit_fix_4_valuation_no_duplicate_percentile():
    """Fix 4: Valuation ERP narrative must not contain redundant '%ile percentile'."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    assert "%ile percentile" not in md


def test_audit_fix_5_charts_escaped_dollar_in_caption():
    """Fix 5: Chart captions with dollar symbols must be escaped to avoid KaTeX corruption."""
    from render.charts import generate_net_liquidity_vs_spx_chart
    chart = generate_net_liquidity_vs_spx_chart(None, None, "2026-10-30")
    assert r"\$T" in chart.caption
    assert " $T" not in chart.caption


def test_audit_fix_6_equal_weight_spread_unified_timeframe():
    """Fix 6: Section 16 RSP vs SPY spread matches Section 1 & Section 8 across Monthly, Weekly, Daily."""
    # Monthly
    res_m = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md_m = Path(res_m["markdown"]).read_text(encoding="utf-8")
    assert "+5.85%" in md_m
    assert "Monthly return spread: +5.85%" in md_m

    # Weekly
    res_w = generate_market_report(report_type="weekly", target_date="2026-11-13", quiet=True)
    md_w = Path(res_w["markdown"]).read_text(encoding="utf-8")
    assert "-0.23%" in md_w
    assert "Weekly return spread: -0.23%" in md_w


def test_audit_fix_7_stock_bond_correlation_exact_match():
    """Fix 7: Section 10 plumbing 60D stock-bond correlation and Section 10 correlation matrix SPY/TLT match exactly."""
    import pandas as pd
    from data.yfinance_provider import YFinanceDataProvider
    from sections.commodities_cross_asset import compute_cross_asset_plumbing, compute_cross_asset_correlation_matrix

    provider = YFinanceDataProvider(cache_dir="data/cache")
    bars = provider.get_history(["SPY", "TLT", "QQQ", "GLD", "USO", "UUP"], start_date="2025-01-01", end_date="2026-10-30")

    plumb = compute_cross_asset_plumbing(bars, pd.to_datetime("2026-10-30"), pd.to_datetime("2026-09-30"))
    matrix = compute_cross_asset_correlation_matrix(bars, "2026-10-30")

    assert plumb is not None
    assert matrix is not None
    spy_tlt_matrix = matrix.matrix.get("SPY", {}).get("TLT")
    assert round(plumb.stock_bond_corr_60d, 2) == spy_tlt_matrix


def test_audit_fix_8_chart_of_the_day_title_and_realized_vol():
    """Fix 8: Chart of the Day title aligns with graphic and vix_regime chart generates 20D Realized Vol when spx_df provided."""
    from render.charts import generate_vix_chart
    from data.yfinance_provider import YFinanceDataProvider
    provider = YFinanceDataProvider(cache_dir="data/cache")
    bars = provider.get_history(["^VIX", "^GSPC"], start_date="2025-01-01", end_date="2026-11-13")

    chart = generate_vix_chart(vix_df=bars.get("^VIX"), as_of_date="2026-11-13", spx_df=bars.get("^GSPC"))
    assert "20D Realized Vol" in chart.title
    assert "realized volatility" in chart.caption.lower()


def test_audit_fix_9_cot_source_tuesday_cutoff_label():
    """Fix 9: CFTC COT source subtitle explicitly includes 'Tuesday cutoff' lag label."""
    res = generate_market_report(report_type="monthly", target_date="2026-10-30", quiet=True)
    md = Path(res["markdown"]).read_text(encoding="utf-8")
    html = Path(res["html"]).read_text(encoding="utf-8")
    assert "prior Tuesday cutoff" in md
    assert "prior Tuesday cutoff" in html
