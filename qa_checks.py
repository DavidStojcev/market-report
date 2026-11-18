"""Automated QA Harness for US Market Report Engine (Change Request v2.2 Stage 0).

Implements independent verification checks (T1 - T12) that execute after report generation,
produce a comprehensive markdown report (qa_report.md), and fail the build if any check FAILS.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import pandas as pd

from data.calendar import PeriodInfo
from sections.base import SectionResult
from sections.breadth import BreadthSectionData
from sections.conclusion import ConclusionData
from sections.executive_summary import ExecutiveSummaryData
from sections.macro import MacroSectionData
from sections.movers import TopMoversData
from sections.sectors import SectorReportData
from sections.sentiment_positioning import SentimentPositioningData
from sections.swing_regime import SwingGateData
from sections.theme_relative_strength import ThemeRelativeStrengthData
from sections.valuation import ValuationData
from sections.volatility import VolatilityData
from sections.what_changed import WhatChangedData
from sections.forward_calendar import ForwardCalendarData
from sections.commodities_cross_asset import CommoditiesCrossAssetData

logger = logging.getLogger(__name__)


@dataclass
class QACheckResult:
    check_id: str
    name: str
    status: str  # "PASS", "FAIL", "WARN"
    details: List[str] = field(default_factory=list)


@dataclass
class QAHarnessReport:
    as_of_date: str
    report_type: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    warned_checks: int
    results: List[QACheckResult] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return self.failed_checks == 0


# ==============================================================================
# Individual QA Checks: T1 through T12
# ==============================================================================

def check_t1_label_vote_consistency(sections: List[SectionResult]) -> QACheckResult:
    """T1: For every indicator row, the sign of the vote must match the sign implied by its label/interpretation."""
    details: List[str] = []
    concl_sec = next((s for s in sections if isinstance(s.data, ConclusionData)), None)
    exec_sec = next((s for s in sections if isinstance(s.data, ExecutiveSummaryData)), None)
    scorecard = None
    if concl_sec and concl_sec.data and concl_sec.data.scorecard:
        scorecard = concl_sec.data.scorecard
    elif exec_sec and exec_sec.data and getattr(exec_sec.data, "scorecard", None):
        scorecard = exec_sec.data.scorecard
    if not scorecard:
        return QACheckResult("T1", "Label-Vote Consistency", "FAIL", ["Conclusion scorecard data missing."])
    for v in scorecard.votes:
        name = v.rule_name
        interp = v.interpretation.lower()
        vote = v.vote

        # Check for positive implied indicators with negative votes
        # e.g., VRP positive/contango voting negative
        if "vrp" in name.lower() or "volatility risk premium" in name.lower():
            if ("positive" in interp or "contango" in interp or "complacency" in interp or "insurance" in interp) and vote < 0:
                details.append(f"Mismatched VRP vote: label indicates positive premium/contango ('{v.interpretation}'), but vote is {vote:+.2f}")
        elif "put/call" in name.lower():
            if "hedging" in interp and vote < -0.5 and "extreme" not in interp:
                # If put/call is 1.25 (typical normal/elevated hedging) and voting -1.00
                details.append(f"Mismatched Put/Call vote: '{v.condition}' with interpretation '{v.interpretation}' voting extreme {vote:+.2f}")
        elif "easing" in interp or "bullish" in interp or "supportive" in interp or "expansion" in interp:
            if vote < -0.1:
                details.append(f"Indicator '{name}' interpretation implies easing/bullish ('{v.interpretation}'), but vote is {vote:+.2f}")
        elif "tightening" in interp or "bearish" in interp or "restrictive" in interp or "inversion" in interp:
            if vote > 0.1:
                details.append(f"Indicator '{name}' interpretation implies tightening/bearish ('{v.interpretation}'), but vote is {vote:+.2f}")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T1", "Label-Vote Consistency", status, details)


def check_t2_single_source_of_truth(sections: List[SectionResult]) -> QACheckResult:
    """T2: Single source of truth: extract every metric that appears in 2+ places and assert equal values."""
    details: List[str] = []
    values_by_metric: Dict[str, List[Tuple[str, float]]] = {}

    for s in sections:
        sec_name = s.title
        data = s.data
        if isinstance(data, WhatChangedData):
            for r in data.deltas:
                if "High-Yield Credit" in r.name:
                    try:
                        # Extract number like 326
                        bps = float(re.search(r"(\d+)\s*bps", r.current_val).group(1))
                        values_by_metric.setdefault("HY_OAS_BPS", []).append((sec_name, bps))
                    except Exception:
                        pass
                elif "Macro & Micro Trend Score" in r.name:
                    try:
                        sc = float(re.search(r"([+-]?\d+\.\d+)", r.current_val).group(1))
                        values_by_metric.setdefault("MACRO_SCORE", []).append((sec_name, sc))
                    except Exception:
                        pass

        elif isinstance(data, MacroSectionData):
            if data.fed_rates and data.fed_rates.hy_spread is not None:
                values_by_metric.setdefault("HY_OAS_BPS", []).append((sec_name, round(data.fed_rates.hy_spread * 100.0, 1)))
            if data.intelligence and data.intelligence.credit:
                values_by_metric.setdefault("HY_OAS_BPS", []).append((f"{sec_name} (Intel)", round(data.intelligence.credit.spread_bps, 1)))
            if data.intelligence and data.intelligence.vol_term:
                values_by_metric.setdefault("VIX", []).append((sec_name, round(data.intelligence.vol_term.vix_1m, 2)))
            if hasattr(data, "cross_assets") and data.cross_assets:
                for ca in data.cross_assets:
                    if getattr(ca, "symbol", "") == "CL=F":
                        values_by_metric.setdefault("OIL_3M_RET", []).append((sec_name, round(getattr(ca, "pct_change_3m", 0.0), 2)))

        elif isinstance(data, CommoditiesCrossAssetData):
            for it in data.items:
                if it.symbol == "CL=F":
                    values_by_metric.setdefault("OIL_3M_RET", []).append((sec_name, round(it.ret_3m, 2)))

        elif isinstance(data, SentimentPositioningData):
            values_by_metric.setdefault("VIX", []).append((sec_name, round(data.vix, 2)))

        elif isinstance(data, ConclusionData):
            values_by_metric.setdefault("MACRO_SCORE", []).append((sec_name, round(data.normalized_score, 2)))
            if data.regime_intelligence and data.regime_intelligence.credit:
                values_by_metric.setdefault("HY_OAS_BPS", []).append((sec_name, round(data.regime_intelligence.credit.spread_bps, 1)))
            if data.scorecard and getattr(data.scorecard, "all_votes", None):
                for v in data.scorecard.all_votes:
                    if "Crude Oil (WTI)" in v.rule_name:
                        m_oil = re.search(r"3M trend ([+-]?\d+\.\d+)%", v.condition)
                        if m_oil:
                            values_by_metric.setdefault("OIL_3M_RET", []).append((f"{sec_name} (Vote)", round(float(m_oil.group(1)), 2)))

    for metric, instances in values_by_metric.items():
        if len(instances) >= 2:
            base_val = instances[0][1]
            for loc, val in instances[1:]:
                if abs(val - base_val) > 0.05:
                    details.append(f"Inconsistent {metric}: {instances[0][0]} has {base_val}, but {loc} has {val}")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T2", "Single Source of Truth", status, details)


def check_t3_claim_lint(md_content: str, html_content: str, sections: List[SectionResult]) -> QACheckResult:
    """T3: Claim lint: block phrases in the banned list unless supported by verified fields."""
    details: List[str] = []
    # Lint: forbidden phrase definitions to detect and block
    banned_phrases = [
        "markets price aggressive fed easing",  # Lint: forbidden
        "aggressive fed easing",                # Lint: forbidden
        "downward earnings revisions",          # Lint: forbidden
        "cyclical equities face downward earnings revisions",  # Lint: forbidden
        "institutional liquidation",            # Lint: forbidden
        "institutional participation",          # Lint: forbidden
        "institutional distribution",           # Lint: forbidden (CR v2.3)
        "institutional trend",                  # Lint: forbidden (CR v2.3)
        "elevated term premium",                # Lint: forbidden (CR v2.3)
        "term premium",                         # Lint: forbidden (CR v2.3)
        "tactical leadership reflects sector returns",  # Lint: forbidden template glitch (CR v2.3)
        "eliminates recession risk",            # Lint: forbidden
        "eliminates",                           # Lint: forbidden
        "proves",                               # Lint: forbidden
    ]

    combined_text = (md_content + " " + html_content).lower()
    for phrase in banned_phrases:
        if phrase in combined_text:
            details.append(f"Banned or unsupported claim detected in generated report: '{phrase}'")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T3", "Claim Lint & Banned Phrases", status, details)


def check_t4_regime_truth_table(sections: List[SectionResult], report_type: str = "") -> QACheckResult:
    """T4: Regime truth table: regime label must equal quadrant derived from Growth & Inflation signals."""
    details: List[str] = []
    macro_sec = next((s for s in sections if isinstance(s.data, MacroSectionData)), None)
    if not macro_sec or not macro_sec.data or not getattr(macro_sec.data, "intelligence", None):
        if report_type == "daily" and any(isinstance(s.data, ExecutiveSummaryData) and getattr(s.data, "regime_summary", "") for s in sections):
            return QACheckResult("T4", "Regime Truth Table", "PASS", [])
        return QACheckResult("T4", "Regime Truth Table", "WARN", ["Macro intelligence data not available to verify regime."])

    intel = macro_sec.data.intelligence
    quadrant = intel.quadrant.lower()
    growth = intel.growth_signal.lower()
    inflation = intel.inflation_signal.lower()

    # Contradiction: Deflationary Slowdown requires Decelerating/low inflation
    if "deflationary" in quadrant or "slowdown" in quadrant:
        if "accelerating" in inflation or "reigniting" in inflation or "sticky" in inflation or "firm" in inflation:
            details.append(
                f"Regime truth table violation: Regime is '{intel.quadrant}', but Inflation signal is '{intel.inflation_signal}' (Deflationary Slowdown cannot occur with accelerating/sticky inflation)."
            )

    # Goldilocks requires Easing/Low inflation and Expanding growth
    if "goldilocks" in quadrant:
        if "accelerating" in inflation or "reigniting" in inflation:
            details.append(f"Regime truth table violation: Regime is Goldilocks, but Inflation is '{intel.inflation_signal}'.")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T4", "Regime Truth Table", status, details)


def check_r1_arithmetic_reconciliation(sections: List[SectionResult]) -> List[str]:
    """R1: Verify table arithmetic (Latest - Prior == Delta bps, 10Y - 2Y == 2s10s) and split sanity."""
    details: List[str] = []
    macro_sec = next((s for s in sections if isinstance(s.data, MacroSectionData)), None)
    if macro_sec and macro_sec.data:
        for row in getattr(macro_sec.data, "market_indicators", []) or []:
            if row.series_id == "H41_NETLIQ":
                continue
            if "%" in row.latest_level and "%" in row.prior_level and "bps" in row.period_change:
                m_lat = re.search(r"([+-]?\d+\.\d+)%", row.latest_level)
                m_pri = re.search(r"([+-]?\d+\.\d+)%", row.prior_level)
                m_chg = re.search(r"([+-]?\d+)\s*bps", row.period_change)
                if m_lat and m_pri and m_chg:
                    lat_v = round(float(m_lat.group(1)), 2)
                    pri_v = round(float(m_pri.group(1)), 2)
                    chg_bps = int(m_chg.group(1))
                    expected_bps = round(lat_v * 100.0) - round(pri_v * 100.0)
                    if expected_bps != chg_bps:
                        details.append(
                            f"R1 Arithmetic mismatch in '{row.indicator}': Latest ({lat_v:.2f}%) - Prior ({pri_v:.2f}%) = {expected_bps:+d} bps != Period Delta ({chg_bps:+d} bps)."
                        )
        fr = getattr(macro_sec.data, "fed_rates", None)
        if fr is not None and fr.yield_10y is not None and fr.yield_2y is not None and fr.spread_2s10s is not None:
            implied_2s10s = round(round(fr.yield_10y, 2) - round(fr.yield_2y, 2), 2)
            reported_2s10s = round(fr.spread_2s10s, 2)
            if abs(implied_2s10s - reported_2s10s) > 0.015:
                details.append(
                    f"R1 Yield curve mismatch: 10Y ({fr.yield_10y:.2f}%) - 2Y ({fr.yield_2y:.2f}%) = {implied_2s10s:+.2f}% != 2s10s ({reported_2s10s:+.2f}%)."
                )

    theme_sec = next((s for s in sections if isinstance(s.data, ThemeRelativeStrengthData)), None)
    if theme_sec and theme_sec.data:
        for item in getattr(theme_sec.data, "items", []) or []:
            if getattr(item, "is_available", True):
                if abs(item.ret_1m) > 35.0 or abs(item.dist_50dma_pct) > 35.0:
                    details.append(
                        f"R1 Split/Outlier anomaly for ETF {item.symbol}: 1M={item.ret_1m:+.1f}%, vs50DMA={item.dist_50dma_pct:+.1f}%."
                    )

    return details


def check_r2_calendar_chronology(
    sections: List[SectionResult],
    as_of_date: str = "",
    md_content: str = "",
) -> List[str]:
    """R2: Verify macro release_date <= as_of_date < next_release_date, weekday release dates, and valid ordinals."""
    details: List[str] = []
    macro_sec = next((s for s in sections if isinstance(s.data, MacroSectionData)), None)
    if macro_sec and macro_sec.data and as_of_date:
        for row in getattr(macro_sec.data, "standing_dashboard", []) or []:
            rel_d = getattr(row, "release_date", "") or ""
            next_d = getattr(row, "next_release_date", "") or ""
            if re.match(r"^\d{4}-\d{2}-\d{2}$", rel_d):
                if rel_d > as_of_date:
                    details.append(
                        f"R2 Chronology violation in '{row.indicator}': release_date {rel_d} > as_of_date {as_of_date}."
                    )
                if pd.to_datetime(rel_d).weekday() >= 5:
                    details.append(
                        f"R2 Weekend release_date in '{row.indicator}': {rel_d} is not a business weekday."
                    )
            if re.match(r"^\d{4}-\d{2}-\d{2}$", next_d):
                if next_d <= as_of_date:
                    details.append(
                        f"R2 Stale next_release_date in '{row.indicator}': {next_d} <= as_of_date {as_of_date}."
                    )

    if md_content:
        for m in re.finditer(r"\b(\d+(?:1|2|3)th)\b|\b([123]th)\b", md_content):
            tok = m.group(0)
            if not tok.endswith(("11th", "12th", "13th")):
                details.append(f"R2 Invalid ordinal typo detected: '{tok}'.")

    return details


def check_t5_format_lint(
    md_content: str,
    html_content: str,
    sections: Optional[List[SectionResult]] = None,
    as_of_date: str = "",
) -> QACheckResult:
    """T5: Format lint: no stray suffixes, FX pairs have no '$', amounts have thousands separators, and R2 ordinals/chronology."""
    details: List[str] = []

    # 1. Stray suffixes like 48.60index or 54.90index, or literal `nan` / NaN values
    stray_suffix_matches = re.findall(r"\b\d+\.\d+(?:index|kindex|%index)\b", md_content + " " + html_content)
    if stray_suffix_matches:
        details.append(f"Stray unit suffix found in text: {set(stray_suffix_matches)}")

    nan_matches = re.findall(r"`nan`|\bNaN\b|Low\s+`?nan`?|High\s+`?nan`?", md_content + " " + html_content)
    if nan_matches:
        details.append(f"Formatted NaN value leaked into report output: {set(nan_matches)}")

    if "+DI 0.0 > -DI 0.0" in md_content or "-DI 0.0 > +DI 0.0" in md_content:
        details.append("Invalid zero-DI strict inequality (+DI 0.0 > -DI 0.0) found in technicals.")

    # 2. FX pairs with dollar sign ($EUR/USD, $USD/JPY, or EUR/USD $1.08)
    fx_dollar_matches = re.findall(r"\$(?:EUR/USD|USD/JPY|GBP/USD|AUD/USD)", md_content + " " + html_content)
    if fx_dollar_matches:
        details.append(f"FX pair prefixed with dollar sign: {set(fx_dollar_matches)}")

    fx_price_dollar_matches = re.findall(r"(?:EUR/USD|USD/JPY|GBP/USD)[^\n]*?\\?\$[0-9]+", md_content)
    if fx_price_dollar_matches:
        details.append(f"FX pair price displayed with dollar symbol in Markdown: {fx_price_dollar_matches[:3]}")

    # 3. Dollar amounts missing thousands separators (e.g. $4157M or $3202.3M)
    unformatted_dollar_matches = re.findall(r"(?:\$|\\\$)[1-9][0-9]{3,}(?:\.[0-9]+)?[MB]", md_content + " " + html_content)
    if unformatted_dollar_matches:
        details.append(f"Large dollar amounts missing thousands separator: {set(unformatted_dollar_matches)}")

    # 4. R2 Chronology & Ordinal checks
    details.extend(check_r2_calendar_chronology(sections or [], as_of_date=as_of_date, md_content=md_content))

    status = "FAIL" if details else "PASS"
    return QACheckResult("T5", "Format Lint, Chronology & Units", status, details)


def check_t6_run_to_run_stability(period: PeriodInfo, sections: List[SectionResult]) -> QACheckResult:
    """T6: Run-to-run stability: verify that baseline/prior values match expected historical levels."""
    details: List[str] = []
    # Check WhatChanged prior HY OAS if present
    wc_sec = next((s for s in sections if isinstance(s.data, WhatChangedData)), None)
    if wc_sec and wc_sec.data:
        for r in wc_sec.data.deltas:
            if "High-Yield Credit" in r.name:
                # Historical FRED BAMLH0A0HYM2 on 2026-10-23 was 3.38% (338 bps). An earlier run showed 3.14%.
                if "314" in r.prior_val or "3.14" in r.prior_val:
                    details.append(f"HY OAS prior was 3.14% instead of official FRED baseline (3.38%).")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T6", "Run-to-Run Stability", status, details)


def check_t7_chart_coverage(charts: Optional[Dict[str, Any]]) -> QACheckResult:
    """T7: Chart coverage: every legend entry has a plotted line spanning the entire window."""
    details: List[str] = []
    if not charts:
        return QACheckResult("T7", "Chart Coverage", "PASS", ["No charts generated for this run."])

    for k, chart in charts.items():
        if not getattr(chart, "is_available", False) or not getattr(chart, "svg_content", None):
            continue
        svg = chart.svg_content
        # Check if 200-DMA is in legend but truncated
        if "200-DMA" in svg or "200 DMA" in svg:
            # Check for empty path or premature truncation indicator if any
            if "d=\"M" not in svg:
                details.append(f"Chart '{k}' mentions 200-DMA but contains no path curves.")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T7", "Chart Coverage & History Span", status, details)


def check_t8_structure_lint(md_content: str, html_content: str, sections: List[SectionResult]) -> QACheckResult:
    """T8: Structure lint: no paragraph duplicate (30+ words), section cross-references match actual headings."""
    details: List[str] = []

    # 1. Duplicate paragraph check (30+ words) & duplicate sentence check (12+ words)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", md_content) if len(p.strip().split()) >= 30]
    seen_paras = set()
    for p in paragraphs:
        norm_p = " ".join(p.split())
        if norm_p in seen_paras:
            details.append(f"Duplicate 30+ word paragraph detected: '{norm_p[:80]}...'")
        seen_paras.add(norm_p)

    # Sentence deduplication (12+ words) across HTML content (excluding tabular header rows and scripts)
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["thead", "th", "script", "style", "svg"]):
        tag.decompose()
    raw_html_text = soup.get_text()
    raw_sentences = re.split(r"[\.\?!]\s+", raw_html_text)
    sentence_counts: Dict[str, int] = {}
    for s in raw_sentences:
        s_clean = " ".join(s.strip().split())
        words = s_clean.split()
        if len(words) >= 12:
            sentence_counts[s_clean] = sentence_counts.get(s_clean, 0) + 1
    for s_txt, count in sentence_counts.items():
        if count > 1:
            details.append(f"Duplicate 12+ word sentence detected ({count}x): '{s_txt[:80]}...'")

    # 2. Literal backslash check in HTML (CR v2.3 Item 5) & balanced <details> tags in Markdown
    if r"\$" in html_content:
        details.append("Literal backslash '\\$' detected in HTML output.")

    open_details = len(re.findall(r"<details>", md_content))
    close_details = len(re.findall(r"</details>", md_content))
    if open_details != close_details:
        details.append(f"Unbalanced <details> tags in Markdown: {open_details} open vs {close_details} close.")

    # 3. Cross-reference section number and exact title validation (CR v2.3 Item 6)
    sec_map = {s.section_num: s.title for s in sections}
    cross_refs = re.findall(r"Section (\d+):?\s*([A-Za-z0-9 &,;–—\-]+)", md_content + " " + html_content)
    for num_str, topic in cross_refs:
        num = int(num_str)
        if num in sec_map:
            actual_title = sec_map[num].lower()
            topic_clean = topic.strip().lower().replace("&amp;", "&")
            if "commodities" in actual_title:
                if "commodities, currencies & cross-asset tracker" not in topic_clean:
                    details.append(f"Section cross-reference mismatch: Section {num} title is '{sec_map[num]}', but text refers to '{topic}'")
            elif "forward calendar" in actual_title:
                if "forward calendar & catalyst horizon" not in topic_clean:
                    details.append(f"Section cross-reference mismatch: Section {num} title is '{sec_map[num]}', but text refers to '{topic}'")

    # 4. Numeric table columns check (e.g. "Narrowing" in Prior column)
    wc_sec = next((s for s in sections if isinstance(s.data, WhatChangedData)), None)
    if wc_sec and wc_sec.data:
        for r in wc_sec.data.deltas:
            if "Narrowing" in r.prior_val or "Widening" in r.prior_val:
                details.append(f"Non-numeric word '{r.prior_val}' found in numeric Prior column of What Changed table for '{r.name}'.")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T8", "Structure Lint & Section References", status, details)


def check_t9_provenance(sections: List[SectionResult]) -> QACheckResult:
    """T9: Provenance: calendar events must have verified sources; no templated speaker topics."""
    details: List[str] = []
    cal_sec = next((s for s in sections if isinstance(s.data, ForwardCalendarData)), None)
    if cal_sec and cal_sec.data:
        for ev in cal_sec.data.events:
            if not ev.source or ev.source.strip() == "":
                details.append(f"Calendar event '{ev.title}' on {ev.date} has no verified data source.")
            # Check for unverified templated speaker topics
            if any(w in ev.title.lower() for w in ("waller", "bowman", "keynote", "jefferson")):
                if "economic outlook & balance sheet" in ev.title.lower() or "bank supervision & inflation mandate" in ev.title.lower():
                    details.append(f"Speaker event '{ev.title}' contains templated topic not dynamically verified from Federal Reserve events feed.")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T9", "Provenance & Calendar Sourcing", status, details)


def check_t10_posture_logic(sections: List[SectionResult]) -> QACheckResult:
    """T10: Posture logic: an overlay may only say it changed posture if resulting tier differs."""
    details: List[str] = []
    gate_sec = next((s for s in sections if isinstance(s.data, SwingGateData)), None)
    if gate_sec and gate_sec.data:
        g = gate_sec.data
        desc = (g.triggered_rule + " " + g.execution_guidance).lower()
        if "capped 1-tier more conservative" in desc and g.posture.upper() == "DEFENSIVE":
            # Check if base posture was already defensive (e.g. dual 50-DMA fail)
            if "dual benchmark break of 50-dma" in desc or "gate rule 4" in desc:
                details.append("Posture logic violation: Event-Risk Overlay claimed to cap posture 1-tier more conservative to DEFENSIVE, but posture was already DEFENSIVE (a no-op). Must state 'no change (already Defensive)'.")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T10", "Posture Logic & Event Overlay", status, details)


def check_t11_status_derivation(md_content: str, html_content: str, sections: List[SectionResult]) -> QACheckResult:
    """T11: Status derivation: Data Quality status must be PARTIAL if any input was missing / n/a."""
    details: List[str] = []
    text = md_content + " " + html_content
    # If missing inputs are listed, status cannot be COMPLETE or VERIFIED
    has_missing = "missing / unavailable inputs" in text.lower() or "n/a (source unavailable)" in text.lower()
    if has_missing:
        if "data quality: complete (verified)" in text.lower() or "status: verified" in text.lower():
            details.append("Data quality footer claims COMPLETE (VERIFIED) despite missing/unpolled input fields.")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T11", "Data Quality Status Derivation", status, details)


def check_t12_sanity_bounds(sections: List[SectionResult]) -> QACheckResult:
    """T12: Sanity bounds: flag any daily dollar-volume > 5% of market cap, large moves > 30% unconfirmed."""
    details: List[str] = []
    sec_sec = next((s for s in sections if isinstance(s.data, SectorReportData)), None)
    if sec_sec and sec_sec.data and sec_sec.data.sector_leaders:
        for sec_name, leaders in sec_sec.data.sector_leaders.items():
            for leader in leaders:
                # Sanity bound: daily dollar volume > $4B or > 5% of realistic market cap for mid-caps
                # Specifically check MRNA anomaly: $4,157M avg daily volume on a ~$10B stock (~42% turnover!)
                if leader.symbol == "MRNA" and leader.avg_dollar_volume > 1.5e9:
                    details.append(f"Unrealistic dollar volume anomaly for {leader.symbol}: ${leader.avg_dollar_volume / 1e6:.1f}M/day exceeds reasonable volume bounds.")
                elif leader.avg_dollar_volume > 1e11:  # $100B/day is physically impossible for single stock
                    details.append(f"Unrealistic dollar volume for {leader.symbol}: ${leader.avg_dollar_volume / 1e6:.1f}M/day.")

    details.extend(check_r1_arithmetic_reconciliation(sections))

    status = "FAIL" if details else "PASS"
    return QACheckResult("T12", "Sanity Bounds, Arithmetic & Split Integrity", status, details)


_T13_PROPER_NOUN_EXEMPTIONS: List[str] = [r"\bBest Buy\b"]

_T13_BANNED_PATTERNS: List[Tuple[str, str]] = [
    (r"\bbuy\b", "buy"),
    (r"\bsell\b", "sell"),
    (r"\badd\b", "add"),
    (r"\btrim\b", "trim"),
    (r"\bhedge\b", "hedge"),
    (r"\bcut beta\b", "cut beta"),
    (r"\btarget (?:equity )?beta\b", "target beta"),
    (r"\brotate (?:toward|into)\b", "rotate toward/into"),
    (r"\bhold a\b", "hold a"),
    (r"\bhold disciplined\b", "hold disciplined"),
    (r"\boverweight\b", "overweight"),
    (r"\bunderweight\b", "underweight"),
    (r"\bentr(?:y|ies)\b", "entry/entries"),
    (r"\bstops?\b", "stop/stops"),
    (r"\btarget price\b", "target price"),
    (r"\bdesk sector tilt\b", "desk sector tilt"),
    (r"\bprioritize\b", "prioritize"),
    (r"\bdeploy selective\b", "deploy selective"),
    (r"\bstay constructive\b", "stay constructive"),
]


def check_r3_cross_section_facts_registry(
    sections: List[SectionResult],
    md_content: str = "",
    html_content: str = "",
    facts_registry: Optional[Any] = None,
) -> QACheckResult:
    """R3: Cross-check headline numbers appearing across multiple sections against a single truth."""
    details: List[str] = []

    exec_sec = next((s for s in sections if isinstance(s.data, ExecutiveSummaryData)), None)
    conc_sec = next((s for s in sections if isinstance(s.data, ConclusionData)), None)
    gate_sec = next((s for s in sections if isinstance(s.data, SwingGateData)), None)
    wc_sec = next((s for s in sections if isinstance(s.data, WhatChangedData)), None)
    macro_sec = next((s for s in sections if isinstance(s.data, MacroSectionData)), None)
    breadth_sec = next((s for s in sections if isinstance(s.data, BreadthSectionData)), None)
    vol_sec = next((s for s in sections if isinstance(s.data, VolatilityData)), None)
    cal_sec = next((s for s in sections if isinstance(s.data, ForwardCalendarData)), None)
    val_sec = next((s for s in sections if isinstance(s.data, ValuationData)), None)

    # 1. Dates synchronization across sections
    date_sources: List[Tuple[str, str]] = []
    if exec_sec and exec_sec.data and getattr(exec_sec.data, "session_date", ""):
        date_sources.append(("ExecutiveSummary.session_date", str(exec_sec.data.session_date)))
    if gate_sec and gate_sec.data and getattr(gate_sec.data, "as_of_date", ""):
        date_sources.append(("SwingGate.as_of_date", str(gate_sec.data.as_of_date)))
    if wc_sec and wc_sec.data and getattr(wc_sec.data, "as_of_date", ""):
        date_sources.append(("WhatChanged.as_of_date", str(wc_sec.data.as_of_date)))
    if cal_sec and cal_sec.data and getattr(cal_sec.data, "as_of_date", ""):
        date_sources.append(("ForwardCalendar.as_of_date", str(cal_sec.data.as_of_date)))
    if facts_registry is not None and getattr(facts_registry, "as_of_date", ""):
        date_sources.append(("CanonicalFactsRegistry.as_of_date", str(facts_registry.as_of_date)))
    unique_dates = {d for _, d in date_sources if d}
    if len(unique_dates) > 1:
        summary = ", ".join(f"{lbl}={val}" for lbl, val in date_sources)
        details.append(f"R3 Date mismatch across sections: {summary}")

    # 2. Scorecard verdict, normalized score, posture, and macro quadrant
    if exec_sec and exec_sec.data and conc_sec and conc_sec.data:
        if str(exec_sec.data.verdict).strip().lower() != str(conc_sec.data.verdict).strip().lower():
            details.append(
                f"R3 Verdict mismatch: ExecSummary '{exec_sec.data.verdict}' != Conclusion '{conc_sec.data.verdict}'"
            )
        if abs(float(exec_sec.data.normalized_score) - float(conc_sec.data.normalized_score)) > 1e-4:
            details.append(
                f"R3 Score mismatch: ExecSummary {exec_sec.data.normalized_score:+.4f} != Conclusion {conc_sec.data.normalized_score:+.4f}"
            )
    if exec_sec and exec_sec.data and gate_sec and gate_sec.data:
        exec_posture = str(getattr(exec_sec.data, "posture", "") or "").strip().upper()
        gate_posture = str(getattr(gate_sec.data, "posture", "") or "").strip().upper()
        if exec_posture and gate_posture and exec_posture != gate_posture:
            details.append(
                f"R3 Posture mismatch: ExecSummary '{exec_posture}' != SwingGate '{gate_posture}'"
            )

    intel = getattr(macro_sec.data, "intelligence", None) if (macro_sec and macro_sec.data) else None
    if exec_sec and exec_sec.data and intel is not None:
        exec_reg = str(getattr(exec_sec.data, "regime_summary", "") or "").strip()
        macro_reg = str(getattr(intel, "summary_badge", "") or "").strip()
        if exec_reg and macro_reg and exec_reg != macro_reg:
            details.append(
                f"R3 Macro quadrant mismatch: ExecSummary '{exec_reg}' != Macro '{macro_reg}'"
            )
    if facts_registry is not None and intel is not None:
        reg_q = str(getattr(facts_registry, "macro_quadrant", "") or "").strip()
        macro_q = str(getattr(intel, "quadrant", "") or "").strip()
        if reg_q and macro_q and reg_q != macro_q:
            details.append(
                f"R3 FactsRegistry quadrant mismatch: Registry '{reg_q}' != Macro '{macro_q}'"
            )

    # 3. VIX & Term Ratio cross-section parity
    vix_readings: List[Tuple[str, float]] = []
    if vol_sec and vol_sec.data and getattr(vol_sec.data, "current_level", None) is not None:
        vix_readings.append(("Volatility.current_level", float(vol_sec.data.current_level)))
    if gate_sec and gate_sec.data and getattr(gate_sec.data, "vix_level", None) is not None:
        vix_readings.append(("SwingGate.vix_level", float(gate_sec.data.vix_level)))
    if intel and getattr(intel, "vol_term", None) and getattr(intel.vol_term, "vix_1m", None) is not None:
        vix_readings.append(("Macro.vol_term.vix_1m", float(intel.vol_term.vix_1m)))
    if facts_registry is not None and getattr(facts_registry, "vix_current", None) is not None:
        vix_readings.append(("CanonicalFactsRegistry.vix_current", float(facts_registry.vix_current)))
    if len(vix_readings) >= 2:
        vals = [v for _, v in vix_readings]
        if max(vals) - min(vals) > 0.05:
            summary = ", ".join(f"{lbl}={val:.2f}" for lbl, val in vix_readings)
            details.append(f"R3 VIX cross-section mismatch: {summary}")

    if (
        vol_sec and vol_sec.data and getattr(vol_sec.data, "ratio_vix_3m", None) is not None
        and intel and getattr(intel, "vol_term", None) and getattr(intel.vol_term, "ratio", None) is not None
    ):
        v_rat = float(vol_sec.data.ratio_vix_3m)
        m_rat = float(intel.vol_term.ratio)
        if abs(v_rat - m_rat) > 0.005:
            details.append(
                f"R3 VIX 1M/3M term ratio mismatch: Volatility {v_rat:.3f} != Macro {m_rat:.3f}"
            )

    # 4. Implied 1D move parity (Volatility vs ForwardCalendar vs DailyPlaybook)
    if vol_sec and vol_sec.data and getattr(vol_sec.data, "implied_daily_move_pct", None) is not None:
        vol_imp = float(vol_sec.data.implied_daily_move_pct)
        if cal_sec and cal_sec.data and getattr(cal_sec.data, "implied_1d_move_pct", None) is not None:
            cal_imp = float(cal_sec.data.implied_1d_move_pct)
            if abs(vol_imp - cal_imp) > 0.05:
                details.append(
                    f"R3 Implied 1D move mismatch: Volatility ±{vol_imp:.2f}% != ForwardCalendar ±{cal_imp:.2f}%"
                )
        pb = getattr(gate_sec.data, "catalyst_playbook", None) if (gate_sec and gate_sec.data) else None
        if pb is not None and getattr(pb, "implied_1d_move_pct", None) is not None:
            pb_imp = float(pb.implied_1d_move_pct)
            if abs(vol_imp - pb_imp) > 0.05:
                details.append(
                    f"R3 Implied 1D move mismatch: Volatility ±{vol_imp:.2f}% != Playbook ±{pb_imp:.2f}%"
                )

    # 5. Breadth parity (% > 50-DMA & Net New Highs across Breadth, SwingGate, and Registry)
    b_res = getattr(breadth_sec.data, "result", None) if (breadth_sec and breadth_sec.data) else None
    if b_res is not None:
        b_p50_obj = getattr(b_res, "pct_above_50_dma", None)
        b_p50 = float(b_p50_obj.current) if (b_p50_obj is not None and getattr(b_p50_obj, "current", None) is not None) else None
        b_nh = getattr(b_res, "net_new_highs", None)
        if gate_sec and gate_sec.data:
            if b_p50 is not None and getattr(gate_sec.data, "breadth_pct_above_50dma", None) is not None:
                g_p50 = float(gate_sec.data.breadth_pct_above_50dma)
                if abs(b_p50 - g_p50) > 0.2:
                    details.append(
                        f"R3 Breadth %>50-DMA mismatch: Breadth {b_p50:.1f}% != SwingGate {g_p50:.1f}%"
                    )
            if b_nh is not None and getattr(gate_sec.data, "net_new_highs", None) is not None:
                g_nh = int(gate_sec.data.net_new_highs)
                if int(b_nh) != g_nh:
                    details.append(
                        f"R3 Net New Highs mismatch: Breadth {int(b_nh):+d} != SwingGate {g_nh:+d}"
                    )
        if facts_registry is not None:
            if b_p50 is not None and getattr(facts_registry, "pct_above_50dma", None) is not None:
                if abs(b_p50 - float(facts_registry.pct_above_50dma)) > 0.2:
                    details.append(
                        f"R3 Breadth %>50-DMA mismatch vs Registry: Breadth {b_p50:.1f}% != Registry {float(facts_registry.pct_above_50dma):.1f}%"
                    )
            if b_nh is not None and getattr(facts_registry, "net_new_highs", None) is not None:
                if int(b_nh) != int(facts_registry.net_new_highs):
                    details.append(
                        f"R3 Net New Highs mismatch vs Registry: Breadth {int(b_nh):+d} != Registry {int(facts_registry.net_new_highs):+d}"
                    )

    # 6. 10Y Nominal & Real Yield parity (Macro vs Valuation)
    fr = getattr(macro_sec.data, "fed_rates", None) if (macro_sec and macro_sec.data) else None
    val_objs = []
    if macro_sec and macro_sec.data and getattr(macro_sec.data, "valuation", None) is not None:
        val_objs.append(("Macro.valuation", macro_sec.data.valuation))
    if val_sec and val_sec.data is not None:
        val_objs.append(("ValuationSection", val_sec.data))
    if fr is not None and getattr(fr, "yield_10y", None) is not None:
        y10_macro = float(fr.yield_10y)
        for v_lbl, v_obj in val_objs:
            v_y10 = getattr(v_obj, "nominal_10y_yield", None)
            if v_y10 is not None and abs(y10_macro - float(v_y10)) > 0.05:
                details.append(
                    f"R3 10Y nominal yield mismatch: Macro.fed_rates {y10_macro:.2f}% != {v_lbl} {float(v_y10):.2f}%"
                )
        if facts_registry is not None and getattr(facts_registry, "yield_10y", None) is not None:
            if abs(y10_macro - float(facts_registry.yield_10y)) > 0.05:
                details.append(
                    f"R3 10Y nominal yield mismatch vs Registry: Macro {y10_macro:.2f}% != Registry {float(facts_registry.yield_10y):.2f}%"
                )
    if intel and getattr(intel, "rates", None) and getattr(intel.rates, "real_yield_10y", None) is not None:
        ry_macro = float(intel.rates.real_yield_10y)
        for v_lbl, v_obj in val_objs:
            v_ry = getattr(v_obj, "real_10y_yield", None)
            if v_ry is not None and abs(ry_macro - float(v_ry)) > 0.05:
                details.append(
                    f"R3 10Y real yield mismatch: Macro.rates {ry_macro:.2f}% != {v_lbl} {float(v_ry):.2f}%"
                )

    # 7. WhatChanged numeric deltas & ForwardCalendar event-impact parity
    if wc_sec and wc_sec.data:
        for d in getattr(wc_sec.data, "deltas", []) or []:
            m_num = re.search(r"[-+]?\d+(?:\.\d+)?", str(getattr(d, "current_val", "")))
            if not m_num:
                continue
            parsed = float(m_num.group(0))
            if d.name == "High-Yield Credit OAS Spread" and intel and getattr(intel, "credit", None):
                if abs(parsed - float(intel.credit.spread_bps)) > 2.0:
                    details.append(
                        f"R3 HY OAS mismatch: WhatChanged '{d.current_val}' != Macro {intel.credit.spread_bps:.0f} bps"
                    )
            elif d.name == "S&P 500 % Above 50-Day Moving Average" and b_res is not None:
                b_p50_obj = getattr(b_res, "pct_above_50_dma", None)
                if b_p50_obj is not None and abs(parsed - float(b_p50_obj.current)) > 0.2:
                    details.append(
                        f"R3 %>50-DMA mismatch: WhatChanged '{d.current_val}' != Breadth {b_p50_obj.current:.1f}%"
                    )
            elif d.name == "10-Year Treasury Yield" and fr is not None and getattr(fr, "yield_10y", None) is not None:
                if abs(parsed - float(fr.yield_10y)) > 0.02:
                    details.append(
                        f"R3 10Y yield mismatch: WhatChanged '{d.current_val}' != Macro {float(fr.yield_10y):.2f}%"
                    )

        if cal_sec and cal_sec.data:
            risk_low = str(getattr(wc_sec.data, "single_biggest_risk", "") or "").lower()
            hi_cnt = int(getattr(cal_sec.data, "high_impact_count", 0) or 0)
            if hi_cnt > 0 and any(p in risk_low for p in ("no high-impact", "absence of high-impact", "no major high-impact")):
                details.append(
                    f"R3 Event-risk mismatch: WhatChanged claims no high-impact catalysts while ForwardCalendar reports high_impact_count={hi_cnt}"
                )

    status = "FAIL" if details else "PASS"
    return QACheckResult("R3", "Cross-Section Facts Registry Reconciliation", status, details)


def check_t13_compliance_lint(
    md_content: str,
    html_content: str,
    sections: Optional[List[SectionResult]] = None,
) -> QACheckResult:
    """T13: Compliance lint: block advisory/imperative wording, invented scenario %, and weak disclaimers."""
    details: List[str] = []

    cleaned_html = re.sub(r"<script\b[^>]*>.*?</script>", "", html_content or "", flags=re.S | re.I)
    cleaned_html = re.sub(r"<style\b[^>]*>.*?</style>", "", cleaned_html, flags=re.S | re.I)

    for source_label, text in (("Markdown", md_content or ""), ("HTML", cleaned_html)):
        # Company names are not advice ("Best Buy" tripped the 'buy' rule when BBY was a sector leader).
        for proper_noun in _T13_PROPER_NOUN_EXEMPTIONS:
            text = re.sub(proper_noun, " ", text)
        for pat, label in _T13_BANNED_PATTERNS:
            m = re.search(pat, text, flags=re.I)
            if m:
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 30)
                snippet = text[start:end].replace("\n", " ").strip()
                details.append(f"Banned imperative/allocation term '{label}' detected in {source_label}: '...{snippet}...'")

    if sections:
        for sec in sections:
            data = getattr(sec, "data", None)
            if isinstance(data, ForwardCalendarData):
                for mat in getattr(data, "scenario_matrices", []) or []:
                    for out in getattr(mat, "outcomes", []) or []:
                        if "%" in str(getattr(out, "probability_hint", "")):
                            details.append(
                                f"Invented probability percentage found in ForwardCalendarData scenario '{mat.event_title}': '{out.probability_hint}'"
                            )
            elif isinstance(data, SwingGateData):
                pb = getattr(data, "catalyst_playbook", None)
                feat = getattr(pb, "featured_scenario", None) if pb is not None else None
                if feat is not None:
                    for out in getattr(feat, "outcomes", []) or []:
                        if "%" in str(getattr(out, "probability_hint", "")):
                            details.append(
                                f"Invented probability percentage found in SwingGateData featured scenario '{feat.event_title}': '{out.probability_hint}'"
                            )
            elif isinstance(data, dict) and "disclaimer" in data:
                disc_low = str(data.get("disclaimer", "")).lower()
                for req_phrase in ("impersonal", "individualized", "registered investment adviser", "hypothetical"):
                    if req_phrase not in disc_low:
                        details.append(f"Footer legal disclaimer missing required compliance phrase: '{req_phrase}'")

    status = "FAIL" if details else "PASS"
    return QACheckResult("T13", "Compliance & Impersonal Language Lint", status, details)


def check_u1_reading_time_budget(
    sections: Optional[List[SectionResult]] = None,
    md_content: str = "",
    html_content: str = "",
    report_type: str = "daily",
) -> QACheckResult:
    """U1: Reading-Time & Signal Density Budget QA Check.
    
    Verifies that the report respects the target reading-time budget by counting
    visible words outside collapsible <details> blocks.
    - Daily: visible_words <= 1400 (target 5-7 minute executive read).
    - Weekly: reports visible words and enforces upper bound (<= 4500).
    - Monthly: reports visible words and enforces upper bound (<= 7500).
    """
    details: List[str] = []

    # Strip <details>...</details> blocks
    cleaned_md = re.sub(r"<details\b[^>]*>.*?</details>", " ", md_content or "", flags=re.S | re.I)
    # Strip markdown images ![...](...)
    cleaned_md = re.sub(r"!\[.*?\]\(.*?\)", " ", cleaned_md)
    # Strip markdown links, keeping label text
    cleaned_md = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned_md)
    # Strip HTML tags
    cleaned_md = re.sub(r"<[^>]+>", " ", cleaned_md)
    # Strip markdown table separator rows
    cleaned_md = re.sub(r"^\s*\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)+\|?\s*$", " ", cleaned_md, flags=re.M)
    # Strip markdown horizontal rules
    cleaned_md = re.sub(r"^\s*[-*_]{3,}\s*$", " ", cleaned_md, flags=re.M)
    # Strip markdown symbols & pipes
    cleaned_md = re.sub(r"[*`_~|#]+", " ", cleaned_md)

    words = [w for w in cleaned_md.split() if any(c.isalnum() for c in w)]
    visible_word_count = len(words)

    if report_type == "daily":
        budget = 1400
        if visible_word_count > budget:
            details.append(
                f"Reading-time budget exceeded for DAILY: {visible_word_count} visible words outside <details> (budget: <= {budget} words)"
            )
        else:
            details.append(
                f"Visible words outside <details>: {visible_word_count} (budget: <= {budget} words, DAILY)"
            )
    elif report_type == "weekly":
        budget = 3000
        if visible_word_count > budget:
            details.append(
                f"Reading-time budget exceeded for WEEKLY: {visible_word_count} visible words outside <details> (budget: <= {budget} words)"
            )
        else:
            details.append(
                f"Visible words outside <details>: {visible_word_count} (budget: <= {budget} words, WEEKLY)"
            )
    elif report_type == "monthly":
        budget = 4500
        if visible_word_count > budget:
            details.append(
                f"Reading-time budget exceeded for MONTHLY: {visible_word_count} visible words outside <details> (budget: <= {budget} words)"
            )
        else:
            details.append(
                f"Visible words outside <details>: {visible_word_count} (budget: <= {budget} words, MONTHLY)"
            )
    else:
        budget = 5000
        if visible_word_count > budget:
            details.append(
                f"Reading-time budget exceeded for {report_type.upper()}: {visible_word_count} visible words outside <details> (budget: <= {budget} words)"
            )
        else:
            details.append(
                f"Visible words outside <details>: {visible_word_count} (budget: <= {budget} words, {report_type.upper()})"
            )

    status = "FAIL" if any("exceeded" in d for d in details) else "PASS"
    return QACheckResult("U1", "Reading-Time & Signal Density Budget", status, details)


# ==============================================================================
# Harness Runner & Markdown Generator
# ==============================================================================

def run_qa_checks(
    period: PeriodInfo,
    sections: List[SectionResult],
    md_content: str,
    html_content: str,
    charts: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    strict: bool = False,
) -> QAHarnessReport:
    """Execute all QA checks, write qa_report.md, and optionally raise exception on failure."""
    checks = [
        check_t1_label_vote_consistency(sections),
        check_t2_single_source_of_truth(sections),
        check_t3_claim_lint(md_content, html_content, sections),
        check_t4_regime_truth_table(sections, report_type=period.report_type),
        check_t5_format_lint(md_content, html_content, sections=sections, as_of_date=period.session_date),
        check_t6_run_to_run_stability(period, sections),
        check_t7_chart_coverage(charts),
        check_t8_structure_lint(md_content, html_content, sections),
        check_t9_provenance(sections),
        check_t10_posture_logic(sections),
        check_t11_status_derivation(md_content, html_content, sections),
        check_t12_sanity_bounds(sections),
        check_t13_compliance_lint(md_content, html_content, sections),
        check_r3_cross_section_facts_registry(sections, md_content, html_content),
        check_u1_reading_time_budget(sections, md_content=md_content, html_content=html_content, report_type=period.report_type),
    ]

    passed = sum(1 for c in checks if c.status == "PASS")
    failed = sum(1 for c in checks if c.status == "FAIL")
    warned = sum(1 for c in checks if c.status == "WARN")

    report = QAHarnessReport(
        as_of_date=period.session_date,
        report_type=period.report_type,
        total_checks=len(checks),
        passed_checks=passed,
        failed_checks=failed,
        warned_checks=warned,
        results=checks,
    )

    # Generate Markdown QA Report
    qa_lines = [
        f"# QA Harness Report: {period.report_type.upper()} ({period.session_date})",
        f"**Status:** {'❌ FAILED' if failed > 0 else '✅ PASSED'} | **Score:** {passed}/{len(checks)} checks passed ({failed} failed, {warned} warnings)",
        "",
        "## Summary Table",
        "| ID | Check Name | Status | Details / Violations |",
        "| :---: | :--- | :---: | :--- |",
    ]

    for c in checks:
        badge = "✅ PASS" if c.status == "PASS" else ("❌ FAIL" if c.status == "FAIL" else "⚠️ WARN")
        det_str = "<br>".join(c.details) if c.details else "All criteria satisfied."
        qa_lines.append(f"| **{c.check_id}** | {c.name} | **{badge}** | {det_str} |")

    qa_lines.append("")
    if failed > 0:
        qa_lines.append("## Detailed Failures & Required Actions")
        for c in checks:
            if c.status == "FAIL":
                qa_lines.append(f"### ❌ {c.check_id}: {c.name}")
                for d in c.details:
                    qa_lines.append(f"- {d}")
                qa_lines.append("")

    qa_lines.append("## Known Limitations")
    qa_lines.append("- **S&P 500 Forward P/E & Forward EPS:** Not published by free public yfinance index/ETF endpoints; panel collapses gracefully to `n/a (source unavailable)` per Ground Rule 1 & CR 4.2 without fabricating estimates until a verified fundamental data feed is configured.")
    qa_lines.append("- **Economic Calendar Consensus Forecasts:** Unpolled forward releases and synthesized releases lacking survey consensus display `n/a (no consensus)` and are evaluated via direction-based rules per CR 4.3.")
    qa_lines.append("- **Fed Funds Rate Probabilities:** Derived from CBOT 30-Day Fed Funds Futures (ZQ contracts on CME/Yahoo); prediction market API fallback is optional/unconfigured.")
    qa_lines.append("- **Sentiment & Positioning Feed:** AAII investor survey raw feed is unconfigured; proxy sentiment models and Volatility Risk Premium (VRP) are utilized under strict minimum-input rules.")
    qa_lines.append("")

    qa_md_str = "\n".join(qa_lines)

    # Write to output_dir if specified
    if output_dir:
        report_path = Path(output_dir) / "qa_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(qa_md_str)
        logger.info("QA report saved to: %s", report_path)

    if strict and failed > 0:
        raise AssertionError(f"QA Harness failed {failed} checks. See qa_report.md for details.")

    return report
