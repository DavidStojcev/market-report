"""Self-contained HTML report generator with inline CSS and light/dark mode."""

import html
from typing import Any, Dict, List, Optional
from data.calendar import PeriodInfo
from indicators.breadth import format_percentile_ordinal
from sections.base import SectionResult
from sections.breadth import BreadthSectionData
from sections.conclusion import ConclusionData, DEFAULT_COMPLIANCE_DISCLAIMER
from sections.executive_summary import ExecutiveSummaryData
from sections.index_snapshot import IndexSnapshotData
from sections.macro import MacroSectionData
from sections.movers import TopMoversData
from sections.sectors import SectorReportData
from sections.technicals import TechnicalsData
from sections.volatility import VolatilityData
from sections.what_changed import WhatChangedData
from sections.swing_regime import SwingGateData
from sections.theme_relative_strength import ThemeRelativeStrengthData
from sections.commodities_cross_asset import CommoditiesCrossAssetData
from sections.fed_path import FedPathData
from sections.forward_calendar import ForwardCalendarData, _fmt_ev_value
from sections.earnings import EarningsLayerData
from sections.sentiment_positioning import SentimentPositioningData
from sections.valuation import ValuationData
from sections.watchlist import WatchlistData
from render.svg_charts import render_inflation_chart, render_labor_chart, render_rates_chart, render_activity_chart


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return '<span class="text-muted font-mono">n/a</span>'
    sign = "+" if val > 0 else ""
    color_class = "text-green" if val > 0 else ("text-red" if val < 0 else "text-muted")
    return f'<span class="{color_class} font-mono">{sign}{val:.2f}%</span>'


def _fmt_delta(val: float) -> str:
    sign = "+" if val > 0 else ""
    color_class = "text-green" if val > 0 else ("text-red" if val < 0 else "text-muted")
    return f'<span class="{color_class} font-mono">{sign}{val:.2f}</span>'


def _fmt_price(val: float) -> str:
    return f'<span class="font-mono">{val:,.2f}</span>'


def _render_toggletip(tip_id: str, html_body: str, label: str = "More details") -> str:
    """Render an accessible hover/focus/click toggletip using popover='hint' + interestfor."""
    if not html_body:
        return ""
    return (
        f'<button type="button" class="toggletip-btn" '
        f'interestfor="{tip_id}" popovertarget="{tip_id}" popovertargetaction="show" '
        f'aria-label="{label}" title="{label}">i</button>'
        f'<div id="{tip_id}" class="toggletip-popover" popover="hint" role="tooltip">{html_body}</div>'
    )


def _fmt_macro_val(val: Optional[float], unit: str) -> str:
    """Format macro release value according to its economic unit."""
    if val is None or not isinstance(val, (int, float)):
        return str(val) if val is not None else "n/a"
    u = (unit or "").strip()
    if u == "claims":
        return f"{val:,.0f}" if abs(val) >= 1000 else f"{val:.0f}k"
    if u == "k":
        return f"{val:+.0f}k"
    if u == "index":
        return f"{val:.1f}"
    if u == "%":
        return f"{val:.2f}%"
    return f"{val:+.2f}{u}"


def _render_scenario_matrix_html(mat: Any) -> str:
    """Render a JPMorgan-style 3-branch Event Scenario Matrix card & table in HTML."""
    rows_html: List[str] = []
    for o in mat.outcomes:
        s_low = o.scenario.lower()
        if any(k in s_low for k in ("hot", "hawkish", "weak", "miss", "heavier", "delayed", "sub-consensus")):
            b_cls = "badge-bearish"
        elif any(k in s_low for k in ("cool", "dovish", "relief", "beat", "lighter", "divided", "goldilocks")):
            b_cls = "badge-bullish"
        else:
            b_cls = "badge-neutral"
        rows_html.append(f"""
        <tr>
            <td><strong>{o.scenario}</strong></td>
            <td class="text-center"><span class="badge {b_cls}">{o.probability_hint}</span></td>
            <td>{o.trigger_threshold}</td>
            <td class="text-center font-mono font-bold">{o.spy_reaction}</td>
            <td style="font-size: 12px;">{o.rates_usd_reaction}</td>
            <td style="font-size: 12px;">{o.sector_playbook}</td>
        </tr>
        """)
    imp_cls = "badge-red" if str(mat.importance).upper() == "HIGH" else "badge-neutral"
    note_html = f'<p class="table-caption" style="margin-top: 6px;"><em>{mat.desk_note}</em></p>' if getattr(mat, "desk_note", "") else ""
    return f"""
    <div style="margin-top: 14px;">
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 8px;">
            <strong style="font-size: 0.92rem;">{mat.event_title}</strong>
            <span class="badge {imp_cls}">{mat.importance}</span>
            <span class="badge badge-muted font-mono">{mat.event_date} &bull; {mat.event_time_et}</span>
            <span class="text-muted font-mono" style="font-size: 12px;">Prior: {mat.prior_display} &bull; Cons: {mat.consensus_display}</span>
        </div>
        <div class="table-container">
            <table class="report-table compact-table">
                <thead>
                    <tr>
                        <th>Scenario Branch</th>
                        <th class="text-center">Condition Tier</th>
                        <th>Trigger / Condition</th>
                        <th class="text-center">Options-Implied SPY Move</th>
                        <th>Rates &amp; USD</th>
                        <th>Historically sensitive sectors</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows_html)}</tbody>
            </table>
        </div>
        {note_html}
    </div>
    """


def _render_chart_html(chart: Optional[Any]) -> str:
    """Render inline responsive theme-aware chart card for HTML reports."""
    if chart is None:
        return ""
    has_svg = bool(getattr(chart, "svg_content", ""))
    has_b64 = bool(getattr(chart, "base64_data", ""))
    if not getattr(chart, "is_available", False) or (not has_svg and not has_b64):
        title = getattr(chart, "title", "Chart")
        err = getattr(chart, "error_message", "Chart temporarily unavailable")
        return f'<div class="alert alert-info" style="margin: 12px 0;">📊 <em>{title}: {err}</em></div>'

    if has_svg:
        visual = f'<div class="chart-svg-wrapper" id="chart_{chart.key}">{chart.svg_content}</div>'
    else:
        visual = f'<img src="{chart.data_uri}" alt="{chart.title}" style="max-width: 100%; height: auto; border-radius: 8px;" id="chart_{chart.key}">'

    html_caption = getattr(chart, "caption", "").replace(r"\$", "$")
    return f"""
    <div class="macro-chart-box chart-svg-container" style="margin: 18px 0; text-align: center;">
        {visual}
        <p class="table-caption" style="margin-top: 6px;"><em>{html_caption}</em></p>
    </div>
    """


def _render_sidebar(period: PeriodInfo, sections: List[SectionResult]) -> str:
    """Render collapsible 5-category accordion sidebar matching Zen / Tropical Earth theme."""
    sec_map = {s.section_num: s for s in sections}

    categories = [
        ("1. Overview & Posture", [1, 2, 3, 4], True),
        ("2. Technicals & Breadth", [5, 6, 7], False),
        ("3. Sectors & Cross-Asset", [8, 9, 10], False),
        ("4. Fed, Calendar & Movers", [11, 12, 13, 14], False),
        ("5. Watchlist, Macro & Outlook", [15, 16, 17, 18, 19], False),
    ]

    default_titles = {
        1: "Executive Summary",
        2: "What Changed: Key Deltas",
        3: "Market Regime & Risk Appetite",
        4: "Index Snapshot",
        5: "Technical Indicators",
        6: "Market Breadth",
        7: "Volatility (VIX)",
        8: "Sector Performance",
        9: "Theme Relative Strength",
        10: "Cross-Asset Tracker",
        11: "Fed Policy Path",
        12: "Forward Calendar",
        13: "Earnings Progress",
        14: "Top Movers",
        15: "Focus Screener",
        16: "Market Sentiment",
        17: "Macro Events",
        18: "Market Outlook",
        19: "Footer & Disclaimer",
    }

    is_full_report = len(sections) >= 10
    details_html = []

    if is_full_report:
        for cat_title, nums, is_open in categories:
            links_html = []
            for n in nums:
                if n in sec_map:
                    t = default_titles.get(n, sec_map[n].title)
                    links_html.append(f'<a href="#section-{n}" class="nav-link">{n}. {t}</a>')

            if links_html:
                open_attr = "open" if is_open else ""
                details_html.append(f"""
                <details {open_attr}>
                    <summary>{cat_title}</summary>
                    {''.join(links_html)}
                </details>
                """)

        # Render any unmapped sections in a dynamic section if present
        mapped_nums = {n for _, nums, _ in categories for n in nums}
        extra_links = [
            f'<a href="#section-{s.section_num}" class="nav-link">{s.section_num}. {s.title}</a>'
            for s in sections
            if s.section_num not in mapped_nums
        ]
        if extra_links:
            details_html.append(f"""
            <details open>
                <summary>Additional Sections</summary>
                {''.join(extra_links)}
            </details>
            """)
    else:
        brief_links = [
            f'<a href="#section-{s.section_num}" class="nav-link">{s.section_num}. {s.title}</a>'
            for s in sections
        ]
        if brief_links:
            details_html.append(f"""
            <details open>
                <summary>Daily Briefing</summary>
                {''.join(brief_links)}
            </details>
            """)

    return f"""
    <aside class="sidebar" id="sidebar">
        <div class="brand">
            <h1>Market Briefing</h1>
            <p>Report Date: {period.session_date}</p>
            <div style="margin-top: 8px; display: flex; align-items: center; gap: 6px;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: var(--tropical-teal); display: inline-block;"></span>
                <span style="font-size: 0.7rem; font-weight: 700; color: var(--tropical-teal);">LIVE DATA</span>
            </div>
        </div>

        <nav class="nav-menu" aria-label="Report chapters">
            {''.join(details_html)}
        </nav>
        <div class="sidebar-footer">
            <button class="theme-toggle-btn" onclick="toggleTheme()">🌓 Toggle Theme</button>
        </div>
    </aside>
    """


def _render_top10_ticker_strip(sections: List[SectionResult], period: Optional[PeriodInfo] = None) -> str:
    """Render compact executive ticker strip of top 10 core market metrics."""
    spx_close, spx_pct = None, None
    ndx_close, ndx_pct = None, None
    posture, posture_class = "NEUTRAL", "badge-neutral"
    macro_score, macro_regime, score_class = None, "Neutral", "text-muted"
    vix_val, vix_delta = None, None
    breadth_50 = None
    ad_ratio, net_nh, mcclellan, lead_spread = None, None, None, None
    y10, y10_delta = None, None
    oas_bps, oas_delta = None, None
    vrp, vrp_regime = None, "Normal"
    wti_price, wti_3m = None, None

    idx_num, exec_num, swing_num = 4, 1, 3
    vol_num, breadth_num, macro_num, sent_num, comm_num = 7, 6, 17, 16, 10

    for sec in sections:
        if not sec.is_available:
            continue
        data = sec.data
        if isinstance(data, IndexSnapshotData):
            idx_num = sec.section_num
            for r in data.rows:
                if r.symbol == "^GSPC":
                    spx_close, spx_pct = r.close, r.pct_change
                elif r.symbol == "^NDX":
                    ndx_close, ndx_pct = r.close, r.pct_change
            if y10 is None and getattr(data, "cross_asset_line", None):
                import re as _re_ca
                m_ca = _re_ca.search(r"10Y Yield\s+([0-9\.]+)%", data.cross_asset_line)
                if m_ca:
                    y10 = float(m_ca.group(1))
                m_cad = _re_ca.search(r"10Y Yield\s+[0-9\.]+%\s*\(([-+]?[0-9\.]+)\s*bps\)", data.cross_asset_line)
                if m_cad:
                    y10_delta = float(m_cad.group(1))
        elif isinstance(data, WhatChangedData):
            import re as _re_wc
            for d in getattr(data, "deltas", []):
                if d.name == "10-Year Treasury Yield":
                    if y10 is None:
                        m_y = _re_wc.search(r"[-+]?\d+(?:\.\d+)?", str(d.current_val))
                        if m_y:
                            y10 = float(m_y.group(0))
                    if y10_delta is None:
                        m_dy = _re_wc.search(r"[-+]?\d+(?:\.\d+)?", str(d.delta_str))
                        if m_dy:
                            y10_delta = float(m_dy.group(0))
                elif d.name == "High-Yield Credit OAS Spread":
                    if oas_bps is None:
                        m_o = _re_wc.search(r"[-+]?\d+(?:\.\d+)?", str(d.current_val))
                        if m_o:
                            oas_bps = float(m_o.group(0))
                    if oas_delta is None:
                        m_do = _re_wc.search(r"[-+]?\d+(?:\.\d+)?", str(d.delta_str))
                        if m_do:
                            oas_delta = float(m_do.group(0))
        elif isinstance(data, SwingGateData):
            swing_num = sec.section_num
            posture = data.posture
            posture_class = data.posture_badge_class
            if vix_val is None:
                vix_val = getattr(data, "vix_level", None)
                vix_delta = getattr(data, "vix_5d_trend", None)
            if wti_price is None:
                for rr in getattr(data, "risk_ranges", []) or []:
                    if getattr(rr, "symbol", None) == "CL=F":
                        wti_price = getattr(rr, "current_price", None)
                        break
        elif isinstance(data, ExecutiveSummaryData):
            exec_num = sec.section_num
            macro_score = data.normalized_score
            macro_regime = data.verdict
            v_low = data.verdict.lower()
            score_class = "text-green" if v_low in ("bullish", "risk-on", "constructive") else ("text-red" if v_low in ("bearish", "risk-off", "cautious") else "text-muted")
            if vrp is None and getattr(data, "vrp_20d", None) is not None:
                vrp = data.vrp_20d
                vrp_regime = "Elevated" if vrp >= 3.0 else ("Discount" if vrp <= -2.0 else "Normal")
            if wti_price is None and getattr(data, "wti_price", None) is not None:
                wti_price = data.wti_price
                wti_3m = getattr(data, "wti_3m", None)
        elif isinstance(data, VolatilityData):
            vol_num = sec.section_num
            vix_val = data.current_level
            vix_delta = data.change
            if vrp is None and getattr(data, "vrp_20d", None) is not None:
                vrp = data.vrp_20d
                vrp_regime = "Elevated" if vrp >= 3.0 else ("Discount" if vrp <= -2.0 else "Normal")
        elif isinstance(data, BreadthSectionData):
            breadth_num = sec.section_num
            breadth_50 = data.result.pct_above_50_dma.current
            ad_ratio = getattr(data.result, "ad_ratio", None)
            net_nh = getattr(data.result, "net_new_highs", None)
            mcclellan = getattr(data.result, "mcclellan_oscillator", None)
            lead_spread = getattr(data.result, "leadership_spread_pct", None)
        elif isinstance(data, MacroSectionData):
            macro_num = sec.section_num
            if data.fed_rates:
                y10 = data.fed_rates.yield_10y
                y10_delta = data.fed_rates.yield_10y_delta * 100.0
                if getattr(data.fed_rates, "hy_spread", None) is not None:
                    oas_bps = data.fed_rates.hy_spread * 100.0
                if getattr(data.fed_rates, "hy_spread_delta", None) is not None:
                    oas_delta = data.fed_rates.hy_spread_delta * 100.0
            if getattr(data, "intelligence", None) and getattr(data.intelligence, "credit", None):
                oas_bps = data.intelligence.credit.spread_bps
                oas_delta = data.intelligence.credit.delta_bps
        elif isinstance(data, SentimentPositioningData):
            sent_num = sec.section_num
            vrp = data.vol_risk_premium
            vrp_regime = data.vol_premium_regime.split("(")[0].strip()
        elif isinstance(data, CommoditiesCrossAssetData):
            comm_num = sec.section_num
            for row in data.items:
                if row.symbol == "CL=F":
                    wti_price = row.current_price
                    wti_3m = row.ret_3m

    # Formatting 4 primary hero KPI items
    spx_str = f"{spx_close:,.2f}" if spx_close is not None else "n/a"
    if spx_pct is not None:
        spx_arrow = "▲" if spx_pct > 0 else ("▼" if spx_pct < 0 else "—")
        spx_sign = "+" if spx_pct > 0 else ""
        spx_sub = f"{spx_arrow} {spx_sign}{spx_pct:.2f}%"
        spx_c = "text-green" if spx_pct > 0 else ("text-red" if spx_pct < 0 else "text-muted")
    else:
        spx_sub = "—"
        spx_c = "text-muted"

    ndx_str = f"{ndx_close:,.2f}" if ndx_close is not None else "n/a"
    if ndx_pct is not None:
        ndx_arrow = "▲" if ndx_pct > 0 else ("▼" if ndx_pct < 0 else "—")
        ndx_sign = "+" if ndx_pct > 0 else ""
        ndx_sub = f"{ndx_arrow} {ndx_sign}{ndx_pct:.2f}%"
        ndx_c = "text-green" if ndx_pct > 0 else ("text-red" if ndx_pct < 0 else "text-muted")
    else:
        ndx_sub = "—"
        ndx_c = "text-muted"

    score_str = f"{macro_score:+.2f}" if macro_score is not None else "n/a"
    score_sub = macro_regime

    posture_badge_cls = "badge-red" if posture == "DEFENSIVE" else ("badge-green" if posture == "FULL" else "badge-amber")

    # Formatting secondary KPI items
    vix_tf = (
        "1d" if (period and period.report_type == "daily")
        else ("1m" if (period and period.report_type == "monthly") else "1w")
    )
    vix_str = f"{vix_val:.2f}" if vix_val is not None else "n/a"
    vix_sub = f"{vix_delta:+.2f} {vix_tf}" if vix_delta is not None else "—"
    vix_c = "text-red" if (vix_val and vix_val >= 20.0) else ("text-green" if (vix_val and vix_val < 16.5) else "text-muted")
    vix_href = f"#section-{vol_num}" if any(isinstance(s.data, VolatilityData) for s in sections) else f"#section-{swing_num}"

    br_str = f"{breadth_50:.1f}%" if breadth_50 is not None else "n/a"
    br_sub = "SPX > 50-DMA"
    br_c = "text-green font-bold" if (breadth_50 and breadth_50 >= 60.0) else ("text-red font-bold" if (breadth_50 and breadth_50 <= 40.0) else "text-muted")

    secondary_cards = [
        f"""<a href="{vix_href}" class="kpi-secondary-card" title="Jump to Volatility / Regime">
                <span class="kpi-label">CBOE VIX</span>
                <span class="kpi-val font-mono">{vix_str}</span>
                <span class="kpi-sub {vix_c} font-mono">{vix_sub}</span>
            </a>""",
        f"""<a href="#section-{breadth_num}" class="kpi-secondary-card" title="Jump to Market Breadth">
                <span class="kpi-label">50-DMA Breadth</span>
                <span class="kpi-val font-mono">{br_str}</span>
                <span class="kpi-sub {br_c} font-mono">{br_sub}</span>
            </a>""",
    ]

    has_macro_strip = any(x is not None for x in (y10, oas_bps, vrp, wti_price))
    if has_macro_strip:
        y10_str = f"{y10:.2f}%" if y10 is not None else "n/a"
        y10_sub = f"{y10_delta:+.0f} bps" if y10_delta is not None else "—"
        oas_str = f"{oas_bps:.0f} bps" if oas_bps is not None else "n/a"
        oas_sub = f"{oas_delta:+.0f} bps" if oas_delta is not None else "—"
        oas_c = "text-green" if (oas_delta and oas_delta < 0) else ("text-red" if (oas_delta and oas_delta > 0) else "text-muted")
        vrp_str = f"{vrp:+.1f} pts" if vrp is not None else "n/a"
        vrp_sub = vrp_regime.split("/")[0].strip()
        vrp_c = "text-green" if (vrp and vrp > 2.0) else ("text-red" if (vrp and vrp < 0) else "text-muted")
        wti_str = f"${wti_price:.2f}" if wti_price is not None else "n/a"
        wti_sub = f"{wti_3m:+.2f}% 3M" if wti_3m is not None else "WTI Oil"
        wti_c = "text-red" if (wti_3m and wti_3m > 15.0) else "text-muted"
        comm_sec = next((s for s in sections if isinstance(s.data, CommoditiesCrossAssetData)), None)
        macro_sec = next((s for s in sections if isinstance(s.data, MacroSectionData)), None)
        sent_sec = next((s for s in sections if isinstance(s.data, SentimentPositioningData)), None)
        
        y10_href = f"#section-{comm_sec.section_num}" if comm_sec else f"#section-{idx_num}"
        oas_href = f"#section-{macro_sec.section_num}" if macro_sec else "#section-2"
        sent_href = f"#section-{sent_sec.section_num}" if sent_sec else f"#section-{exec_num}"
        comm_href = f"#section-{comm_sec.section_num}" if comm_sec else f"#section-{idx_num}"

        secondary_cards.extend([
            f"""<a href="{y10_href}" class="kpi-secondary-card" title="Jump to 10Y Treasury Yield">
                <span class="kpi-label">10Y Treasury</span>
                <span class="kpi-val font-mono">{y10_str}</span>
                <span class="kpi-sub text-muted font-mono">{y10_sub}</span>
            </a>""",
            f"""<a href="{oas_href}" class="kpi-secondary-card" title="Jump to Credit Spreads">
                <span class="kpi-label">HY OAS Spread</span>
                <span class="kpi-val font-mono">{oas_str}</span>
                <span class="kpi-sub {oas_c} font-mono">{oas_sub}</span>
            </a>""",
            f"""<a href="{sent_href}" class="kpi-secondary-card" title="Jump to Volatility Risk Premium">
                <span class="kpi-label">Vol Risk (VRP)</span>
                <span class="kpi-val font-mono">{vrp_str}</span>
                <span class="kpi-sub {vrp_c} font-mono">{vrp_sub}</span>
            </a>""",
            f"""<a href="{comm_href}" class="kpi-secondary-card" title="Jump to Commodities">
                <span class="kpi-label">WTI Crude</span>
                <span class="kpi-val font-mono">{wti_str}</span>
                <span class="kpi-sub {wti_c} font-mono">{wti_sub}</span>
            </a>""",
        ])
    else:
        ad_str = f"{ad_ratio:.2f}" if ad_ratio is not None else "—"
        ad_c = "text-green" if (ad_ratio is not None and ad_ratio >= 1.2) else ("text-red" if (ad_ratio is not None and ad_ratio < 0.8) else "text-muted")
        ad_sub = "Adv / Dec"
        nh_str = f"{net_nh:+d}" if net_nh is not None else "—"
        nh_c = "text-green" if (net_nh is not None and net_nh > 0) else ("text-red" if (net_nh is not None and net_nh < 0) else "text-muted")
        mc_str = f"{mcclellan:+.1f}" if mcclellan is not None else "—"
        mc_c = "text-green" if (mcclellan is not None and mcclellan > 0) else ("text-red" if (mcclellan is not None and mcclellan < 0) else "text-muted")
        sp_str = f"{lead_spread:+.2f}%" if lead_spread is not None else "—"
        sp_c = "text-green" if (lead_spread is not None and lead_spread < 0) else "text-muted"
        secondary_cards.extend([
            f"""<a href="#section-{breadth_num}" class="kpi-secondary-card" title="Jump to Advance / Decline Ratio">
                <span class="kpi-label">A/D Ratio</span>
                <span class="kpi-val font-mono">{ad_str}</span>
                <span class="kpi-sub {ad_c} font-mono">{ad_sub}</span>
            </a>""",
            f"""<a href="#section-{breadth_num}" class="kpi-secondary-card" title="Jump to Net 52-Week Highs">
                <span class="kpi-label">Net 52W Highs</span>
                <span class="kpi-val font-mono">{nh_str}</span>
                <span class="kpi-sub {nh_c} font-mono">Highs − Lows</span>
            </a>""",
            f"""<a href="#section-{breadth_num}" class="kpi-secondary-card" title="Jump to McClellan Oscillator">
                <span class="kpi-label">McClellan Osc.</span>
                <span class="kpi-val font-mono">{mc_str}</span>
                <span class="kpi-sub {mc_c} font-mono">Breadth Mom.</span>
            </a>""",
            f"""<a href="#section-{breadth_num}" class="kpi-secondary-card" title="Jump to SPY vs RSP Spread">
                <span class="kpi-label">SPY vs RSP</span>
                <span class="kpi-val font-mono">{sp_str}</span>
                <span class="kpi-sub {sp_c} font-mono">Cap vs Equal</span>
            </a>""",
        ])

    return f"""
    <div class="top10-ticker-strip">
        <div class="kpi-grid">
            <a href="#section-{idx_num}" class="kpi-card" title="Jump to S&P 500 in Index Snapshot">
                <span class="kpi-label">S&amp;P 500 (^GSPC)</span>
                <span class="kpi-val font-mono">{spx_str}</span>
                <span class="kpi-sub {spx_c} font-mono">{spx_sub}</span>
            </a>
            <a href="#section-{idx_num}" class="kpi-card" title="Jump to Nasdaq-100 in Index Snapshot">
                <span class="kpi-label">Nasdaq-100 (^NDX)</span>
                <span class="kpi-val font-mono">{ndx_str}</span>
                <span class="kpi-sub {ndx_c} font-mono">{ndx_sub}</span>
            </a>
            <a href="#section-{exec_num}" class="kpi-card" title="Jump to Executive Summary & Macro Verdict">
                <span class="kpi-label">Macro Score</span>
                <span class="kpi-val font-mono">{score_str}</span>
                <span class="kpi-sub text-muted font-mono">{score_sub}</span>
            </a>
            <a href="#section-{swing_num}" class="kpi-card" title="Jump to Market Regime &amp; Risk Appetite">
                <span class="kpi-label">Market Posture</span>
                <span class="badge {posture_badge_cls}" style="width: fit-content; margin-top: 4px;">{posture}</span>
                <span class="kpi-sub text-muted" style="margin-top: auto;">Regime Gate</span>
            </a>
        </div>
        <div class="kpi-secondary-grid">
            {''.join(secondary_cards)}
        </div>
    </div>
    """



def _report_meta(sections: List[SectionResult]) -> Dict[str, Any]:
    """Data-quality warnings and generation timestamp carried by the footer section."""
    for s in sections:
        if isinstance(getattr(s, "data", None), dict) and "data_warnings" in s.data:
            return {"warnings": list(s.data.get("data_warnings") or []), "generated": s.data.get("generated_at_utc", "")}
    return {"warnings": [], "generated": ""}


def _period_window_label(period: PeriodInfo) -> str:
    """Honest label for the evaluation window (to-date vs. complete period)."""
    if period.report_type == "daily":
        return "1-day session"
    unit = "Week" if period.report_type == "weekly" else "Month"
    if period.is_partial:
        return f"{unit}-to-date ({period.session_count} sessions)"
    return f"Full {unit.lower()} ({period.session_count} sessions)"


def render_html(
    period: PeriodInfo,
    sections: List[SectionResult],
    charts: Optional[Dict[str, Any]] = None,
) -> str:
    """Render self-contained HTML market report."""
    rep_title = f"{period.report_type.capitalize()} US Market Report"

    # Build Jump-to navigation options from report sections
    jump_options = []
    for s in sections:
        clean_title = s.title.replace("&amp;", "&")
        jump_options.append(f'<option value="#section-{s.section_num}">{s.section_num}. {clean_title}</option>')

    rep_type_label = period.report_type.upper()
    trailing_text = _period_window_label(period).capitalize() + f" from the {period.baseline_date} close"
    _meta = _report_meta(sections)
    _warn_html = (
        '<div class="alert alert-warning" style="margin-top: 12px;">⚠️ <strong>Data Quality Warnings:</strong><ul style="margin: 6px 0 0 18px;">'
        + "".join(f"<li>{html.escape(w)}</li>" for w in _meta["warnings"])
        + "</ul></div>"
    ) if _meta["warnings"] else ""

    sidebar_html = _render_sidebar(period, sections)
    top10_html = _render_top10_ticker_strip(sections, period=period)

    top_bar_html = f"""
    <div class="sticky-nav-bar" id="report-nav">
        <div class="nav-inner" style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
            <div class="nav-brand" style="display: flex; align-items: center; gap: 12px;">
                <button class="toggle-sidebar-btn" onclick="toggleSidebar()" title="Toggle Navigation Sidebar">
                    <span style="font-size: 1rem;">☰</span> Chapters
                </button>
                <span class="nav-brand-title" style="font-size: 0.8rem; font-weight: 600; color: var(--text-muted);">US Market Briefing &bull; {period.session_date}</span>
            </div>
            <div class="nav-controls" style="display: flex; align-items: center; gap: 8px;">
                <select id="jump-select" class="nav-select" onchange="if(this.value){{location.href=this.value; this.value='';}}" title="Jump to section">
                    <option value="">Jump to section...</option>
                    {''.join(jump_options)}
                </select>
                <button class="nav-btn" onclick="toggleTheme()" title="Toggle Theme">🌓 Theme</button>
                <a href="#top" class="nav-btn" title="Back to top">↑ Top</a>
            </div>
        </div>
    </div>
    """

    header_html = f"""
    <header class="report-header" id="top" style="margin-bottom: 28px;">
        <span class="badge badge-primary" style="margin-bottom: 12px; font-size: 0.75rem; letter-spacing: 0.05em;">{rep_type_label} INSTITUTIONAL REPORT</span>
        <h1 style="font-size: 2.2rem; font-weight: 700; line-height: 1.1; margin-bottom: 8px; letter-spacing: -0.02em; color: var(--text-main);">US Market Dynamics</h1>
        <p style="font-size: 0.95rem; color: var(--text-muted); max-width: 650px; margin: 0; line-height: 1.5;">
            Comprehensive analysis of equities, macro conditions, and systemic risk. {trailing_text} to {period.session_date}.
        </p>
        <p style="font-size: 0.78rem; color: var(--text-muted); margin: 6px 0 0 0;">End-of-day data (Yahoo Finance, FRED) &bull; generated {html.escape(_meta['generated'] or 'n/a')} &bull; informational research only, not investment advice.</p>
        {f'<div class="alert alert-info" style="margin-top: 12px;">ℹ️ <strong>Partial Period:</strong> {period.notes}</div>' if period.is_partial else ''}
        {_warn_html}
    </header>
    """

    sec_by_type = {type(s.data): s for s in sections if s.is_available and s.data is not None}
    vix_tf_upper = (
        "1D" if period.report_type == "daily"
        else ("1M" if period.report_type == "monthly" else "1W")
    )
    vix_tf_lower = vix_tf_upper.lower()
    body_html: List[str] = []

    for sec in sections:
        body_html.append(f'<section class="section-card card" id="section-{sec.section_num}" tabindex="-1">')
        body_html.append(f'<div class="section-header"><span class="section-num">{sec.section_num}</span><h2 class="section-title">{sec.title}</h2></div>')

        if not sec.is_available:
            err = sec.error_message or "Data source temporarily unavailable."
            body_html.append(f'<div class="alert alert-warning">⚠️ <strong>Section Unavailable:</strong> {err}</div></section>')
            continue

        # Section: Executive Summary
        if isinstance(sec.data, ExecutiveSummaryData):
            d: ExecutiveSummaryData = sec.data
            v_low = d.verdict.lower()
            if v_low in ("bullish", "risk-on", "constructive"):
                v_class = "badge-green"
                banner_border = "var(--tropical-teal)"
            elif v_low in ("bearish", "risk-off", "cautious"):
                v_class = "badge-red"
                banner_border = "var(--rosy-copper)"
            else:
                v_class = "badge-amber"
                banner_border = "var(--amber-honey)"
            
            regime_callout_html = ""
            if d.forward_outlook:
                reg_name = d.regime_summary or "Active Macro Regime"
                regime_callout_html = f"""
                <div class="regime-callout-card" style="margin-top: 16px; padding: 14px 18px; border-radius: 8px; background: var(--bg-card-subtle); border-left: 4px solid var(--tropical-teal);">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <span style="font-size: 1.1rem;">🧭</span>
                        <span style="font-weight: 700; font-size: 0.82rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--tropical-teal);">Macro Regime &amp; Forward Implications: {reg_name}</span>
                    </div>
                    <div style="font-size: 0.90rem; line-height: 1.5; color: var(--text-main);">{d.forward_outlook}</div>
                </div>
                """

            conf_badge = f'<div class="score-sub font-mono" style="margin-top: 4px; color: rgba(255,255,255,0.75); font-size: 11px;">{d.confidence}</div>' if d.confidence else ''

            brief_html = ""
            brief = getattr(d, "brief", None)
            if brief is not None:
                import re as _re_brief
                def _fmt_md_bold(txt: str) -> str:
                    return _re_brief.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", txt)

                things_li = "".join(f"<li>{_fmt_md_bold(t)}</li>" for t in brief.three_things)
                if period.report_type == "weekly":
                    brief_card_cls = "weekly-desk-card"
                    brief_badge = "⚡ Weekly Desk Takeaway &amp; Strategy"
                elif period.report_type == "monthly":
                    brief_card_cls = "monthly-strategic-card"
                    brief_badge = "⚡ Monthly Strategic Takeaway"
                else:
                    brief_card_cls = "morning-brief-card"
                    brief_badge = "⚡ 30-Second Brief &amp; Bottom Line"
                brief_html = f"""
                <div class="{brief_card_cls}" style="margin-top: 16px; padding: 20px 24px; border-radius: 12px; background: linear-gradient(135deg, rgba(236, 164, 0, 0.13), rgba(44, 81, 76, 0.05)); border: 1.5px solid rgba(236, 164, 0, 0.55); border-left: 6px solid var(--amber-honey); box-shadow: 0 6px 18px rgba(86, 53, 30, 0.08);">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
                        <span style="font-weight: 800; font-size: 0.95rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--amber-honey);">{brief_badge}</span>
                        <span style="font-size: 0.95rem; font-weight: 700; color: var(--text-main);">{brief.headline_takeaway}</span>
                    </div>
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); margin-bottom: 6px;">3 Things That Mattered</div>
                        <ul style="margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 6px; font-size: 0.96rem; line-height: 1.52; color: var(--text-main);">
                            {things_li}
                        </ul>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr; gap: 10px; padding-top: 12px; border-top: 1px solid var(--border); font-size: 0.95rem; line-height: 1.52; color: var(--text-main);">
                        <div><strong style="color: var(--tropical-teal);">What Happened:</strong> {brief.what_happened}</div>
                        <div><strong style="color: var(--tropical-teal);">What It Means:</strong> {brief.what_it_means}</div>
                        <div style="padding: 10px 14px; border-radius: 8px; background: rgba(236, 164, 0, 0.14); border-left: 4px solid var(--amber-honey); font-size: 0.98rem;"><strong style="color: var(--amber-honey);">The Bottom Line:</strong> <strong>{brief.bottom_line}</strong></div>
                    </div>
                </div>
                """

            alignment_callout_html = ""
            if getattr(d, "alignment_note", ""):
                alignment_callout_html = f"""
                <div class="alert alert-info" style="margin-top: 14px; margin-bottom: 14px; display: flex; align-items: flex-start; gap: 10px;">
                    <span style="font-size: 1.1rem; line-height: 1;">⚖️</span>
                    <div>
                        <strong style="display: block; margin-bottom: 2px;">Regime vs. Posture Alignment:</strong> 
                        {d.alignment_note}
                    </div>
                </div>
                """

            horizon_html = ""
            h_matrix = getattr(d, "horizon_matrix", None)
            if h_matrix:
                h_rows = []
                for h in h_matrix:
                    h_rows.append(f"""
                    <tr>
                        <td><strong>{h.horizon}</strong><div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">{h.audience}</div></td>
                        <td><span class="badge badge-neutral font-bold">{h.state}</span></td>
                        <td class="font-mono" style="font-size: 12px;">{h.key_metric}</td>
                        <td style="font-size: 13px; line-height: 1.45;">{h.implication}</td>
                    </tr>
                    """)
                horizon_html = f"""
                <div style="margin-top: 18px;">
                    <h4 style="font-size: 13px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;">Multi-Horizon &amp; Cross-Strategy Synthesis</h4>
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Horizon &amp; Lens</th>
                                    <th>Regime Read</th>
                                    <th>Quantitative Anchor</th>
                                    <th>Cross-Market Takeaway</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(h_rows)}</tbody>
                        </table>
                    </div>
                </div>
                """

            cotd_html = ""
            cotd = getattr(d, "chart_of_the_day", None)
            if cotd:
                import base64
                from pathlib import Path
                cotd_title = cotd.title if cotd.title.startswith("Chart of the Day:") else f"Chart of the Day: {cotd.title}"
                cotd_src = cotd.chart_path
                cotd_file = Path(cotd.chart_path)
                if cotd_file.is_file():
                    mime = "image/svg+xml" if cotd_file.suffix.lower() == ".svg" else "image/png"
                    b64 = base64.b64encode(cotd_file.read_bytes()).decode("ascii")
                    cotd_src = f"data:{mime};base64,{b64}"
                cotd_html = f"""
                <div class="chart-container" style="margin-top: 20px;">
                    <h4 style="font-size: 14px; font-weight: 700; margin-bottom: 8px;">{cotd_title}</h4>
                    <img src="{cotd_src}" alt="{cotd_title}" class="report-img" style="max-width: 100%; border-radius: 6px; border: 1px solid var(--border);">
                    <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px; font-style: italic;">{cotd.annotation}</div>
                </div>
                """

            if period.report_type == "weekly" and horizon_html:
                vis_horizon_html = f'<details class="methodology-note" style="margin-top: 18px;"><summary>Multi-Horizon &amp; Cross-Strategy Synthesis</summary>{horizon_html}</details>'
            else:
                vis_horizon_html = horizon_html

            if period.report_type == "daily":
                body_html.append(f"""
                <div class="verdict-banner" style="border-left-color: {banner_border};">
                    <div class="verdict-box">
                        <span class="verdict-label">Market Regime Verdict</span>
                        <span class="verdict-badge {v_class}">{d.verdict.upper()}</span>
                    </div>
                    <div class="score-box" style="margin-left: auto; text-align: right;">
                        <span class="score-label">Trend Score</span>
                        <span class="score-val font-mono">{d.normalized_score:+.2f}</span>
                        {conf_badge}
                    </div>
                </div>
                {brief_html}
                {cotd_html}
                <details class="methodology-note" style="margin-top: 18px;">
                    <summary>Detail &amp; Methodology</summary>
                    <div class="methodology-content">
                        {alignment_callout_html}
                        {regime_callout_html}
                        <div class="summary-reasons" style="margin-top: 16px;">
                            <ul class="reasons-list">
                                {''.join(f'<li>{r}</li>' for r in d.reasons)}
                            </ul>
                        </div>
                        {horizon_html}
                        <div style="margin-top: 16px; border-top: 1px solid var(--border); padding-top: 12px;">
                            <p><strong>Category Weighting &amp; Caps:</strong> Evaluates 5 independent signal categories with fixed caps: Trend/Price (30%), Market Breadth (20%), Rates &amp; Macro (20%), Volatility &amp; Risk Appetite (15%), and Sentiment &amp; Positioning (15%). Correlated inputs cannot exceed their category allocation.</p>
                            <p style="margin-top: 6px;"><strong>Continuous Gradients:</strong> Breadth and rate thresholds use continuous linear interpolation (-1.0 to +1.0) between neutral and extreme bounds rather than binary cliffs.</p>
                            <p style="margin-top: 6px;"><strong>Divergence Gating &amp; Missing Inputs:</strong> The Trend category takes the weaker of S&amp;P 500 and Nasdaq-100 to penalize narrow single-index rallies. If an input source is temporarily unavailable, it is excluded and active category denominators are renormalized without defaulting to 0.</p>
                            <p style="margin-top: 6px;"><strong>Regime Bands:</strong> Risk-On (&ge; +0.35), Constructive (+0.10 to +0.35), Neutral (-0.10 to +0.10), Cautious (-0.35 to -0.10), Risk-Off (&lt; -0.35).</p>
                        </div>
                    </div>
                </details>
                """)
            else:
                body_html.append(f"""
                <div class="verdict-banner" style="border-left-color: {banner_border};">
                    <div class="verdict-box">
                        <span class="verdict-label">Market Regime Verdict</span>
                        <span class="verdict-badge {v_class}">{d.verdict.upper()}</span>
                    </div>
                    <div class="score-box" style="margin-left: auto; text-align: right;">
                        <span class="score-label">Trend Score</span>
                        <span class="score-val font-mono">{d.normalized_score:+.2f}</span>
                        {conf_badge}
                    </div>
                </div>
                {brief_html}
                {alignment_callout_html}
                {regime_callout_html}
                <div class="summary-reasons" style="margin-top: 16px;">
                    <ul class="reasons-list">
                        {''.join(f'<li>{r}</li>' for r in d.reasons)}
                    </ul>
                </div>
                {vis_horizon_html}
                <details class="methodology-note" style="margin-top: 18px;">
                    <summary>ℹ️ How this score is calculated</summary>
                    <div class="methodology-content">
                        <p><strong>Category Weighting &amp; Caps:</strong> Evaluates 5 independent signal categories with fixed caps: Trend/Price (30%), Market Breadth (20%), Rates &amp; Macro (20%), Volatility &amp; Risk Appetite (15%), and Sentiment &amp; Positioning (15%). Correlated inputs cannot exceed their category allocation.</p>
                        <p style="margin-top: 6px;"><strong>Continuous Gradients:</strong> Breadth and rate thresholds use continuous linear interpolation (-1.0 to +1.0) between neutral and extreme bounds rather than binary cliffs.</p>
                        <p style="margin-top: 6px;"><strong>Divergence Gating &amp; Missing Inputs:</strong> The Trend category takes the weaker of S&amp;P 500 and Nasdaq-100 to penalize narrow single-index rallies. If an input source is temporarily unavailable, it is excluded and active category denominators are renormalized without defaulting to 0.</p>
                        <p style="margin-top: 6px;"><strong>Regime Bands:</strong> Risk-On (&ge; +0.35), Constructive (+0.10 to +0.35), Neutral (-0.10 to +0.10), Cautious (-0.35 to -0.10), Risk-Off (&lt; -0.35).</p>
                    </div>
                </details>
                """)

        # Section: What Changed (Executive Deltas vs Prior Report)
        elif isinstance(sec.data, WhatChangedData):
            w_data: WhatChangedData = sec.data
            delta_rows = []
            for d in w_data.deltas:
                val_class = "text-green font-bold" if d.valence == "positive" else ("text-red font-bold" if d.valence == "negative" else "text-muted font-bold")
                delta_sign = "+" if (d.delta_str and not d.delta_str.startswith(("+", "-", "—"))) else ""
                formatted_delta = f"{delta_sign}{d.delta_str}" if d.delta_str != "—" else "—"
                sub_detail = f'<div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 2px;">{d.detail}</div>' if d.detail else ""
                delta_rows.append(f"""
                <tr>
                    <td><strong>{d.name}</strong>{sub_detail}</td>
                    <td class="text-right font-mono font-bold">{d.current_val}</td>
                    <td class="text-right font-mono {val_class}">{formatted_delta}</td>
                </tr>
                """)

            risk_html = ""
            if w_data.single_biggest_risk:
                risk_html = f"""
                <div class="alert alert-warning" style="margin-top: 14px;">
                    <strong style="display: block; margin-bottom: 2px;">⚠️ Single Biggest Forward Risk:</strong>
                    {w_data.single_biggest_risk}
                </div>
                """

            body_html.append(f"""
            <div class="table-container">
                <table class="report-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th class="text-right">Current Status</th>
                            <th class="text-right">Period &Delta;</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(delta_rows)}
                    </tbody>
                </table>
            </div>
            {risk_html}
            """)

        # Section: Swing-Trade Exposure Gate & Posture
        elif isinstance(sec.data, SwingGateData):
            sg: SwingGateData = sec.data
            
            spy_20_pill = '<span class="badge badge-green">PASS</span>' if sg.spy_20ema_pass else '<span class="badge badge-red">FAIL</span>'
            spy_50_pill = '<span class="badge badge-green">PASS</span>' if sg.spy_50dma_pass else '<span class="badge badge-red">FAIL</span>'
            qqq_20_pill = '<span class="badge badge-green">PASS</span>' if sg.qqq_20ema_pass else '<span class="badge badge-red">FAIL</span>'
            qqq_50_pill = '<span class="badge badge-green">PASS</span>' if sg.qqq_50dma_pass else '<span class="badge badge-red">FAIL</span>'

            if sg.posture == "FULL":
                posture_badge_cls = "badge-green"
                banner_border = "var(--tropical-teal)"
            elif sg.posture in ("REDUCED", "SELECTIVE"):
                posture_badge_cls = "badge-amber"
                banner_border = "var(--amber-honey)"
            else:
                posture_badge_cls = "badge-red"
                banner_border = "var(--rosy-copper)"

            macro_sec_ref = sec_by_type.get(MacroSectionData)
            macro_ref_txt = f"Section {macro_sec_ref.section_num}" if macro_sec_ref else "macro regime signals"
            monthly_gate_note = f"""
            <div class="alert alert-info" style="margin-top: 12px; margin-bottom: 12px;">
                💡 <strong>Monthly Strategic Allocation Note:</strong> This tactical regime gauge summarizes short-to-medium-term trend and volatility thresholds. For long-term multi-asset allocators, macro regime signals and credit spreads ({macro_ref_txt}) take precedence over short-term moving-average signals.
            </div>
            """ if period.report_type == "monthly" else ""

            adherence_html = ""
            if getattr(sg, "range_adherence", None) is not None:
                adherence_html = f"""
                <div class="alert alert-info" style="margin-top: 12px; margin-bottom: 12px;">
                    <strong>Prior Week Model Range Adherence:</strong> {sg.range_adherence.summary_text}
                </div>
                """

            spy_20_c = "text-green font-bold" if sg.spy_20ema_pass else "text-red font-bold"
            spy_50_c = "text-green font-bold" if sg.spy_50dma_pass else "text-red font-bold"
            qqq_20_c = "text-green font-bold" if sg.qqq_20ema_pass else "text-red font-bold"
            qqq_50_c = "text-green font-bold" if sg.qqq_50dma_pass else "text-red font-bold"

            triggers_html = ""
            pt = getattr(sg, "posture_triggers", None)
            if pt is not None:
                triggers_html = f"""
                <div style="margin-top: 16px; padding: 14px 18px; border-radius: 10px; background: var(--bg-card-subtle); border: 1px solid var(--border); border-left: 4px solid var(--tropical-teal);">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                        <span style="font-weight: 700; font-size: 0.80rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--tropical-teal);">Exposure Sizing &amp; Regime Flip Triggers</span>
                        <span class="badge badge-neutral">{pt.target_beta_band}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr; gap: 6px; font-size: 0.86rem; line-height: 1.45; color: var(--text-main);">
                        <div><strong class="text-green">Model upgrade condition:</strong> {pt.upgrade_trigger}</div>
                        <div><strong class="text-red">Model downgrade condition:</strong> {pt.downgrade_trigger}</div>
                    </div>
                </div>
                """

            risk_range_html = ""
            rr_list = getattr(sg, "risk_ranges", None)
            if rr_list:
                def _dir_badge(val: str) -> str:
                    v = (val or "NEUTRAL").upper()
                    cls = "badge-green" if v == "BULLISH" else ("badge-red" if v == "BEARISH" else "badge-amber")
                    return f'<span class="badge {cls}">{v}</span>'

                rr_rows = []
                for rr in rr_list:
                    bar_w = max(4.0, min(100.0, rr.position_pct))
                    rr_rows.append(f"""
                    <tr>
                        <td><strong>{rr.name}</strong> <span class="ticker-sub font-mono">{rr.symbol}</span></td>
                        <td class="text-right font-mono font-bold">${rr.last_price:,.2f}</td>
                        <td class="text-right font-mono text-green">${rr.range_low:,.2f}</td>
                        <td class="text-right font-mono">${rr.pivot_20ema:,.2f}</td>
                        <td class="text-right font-mono text-red">${rr.range_high:,.2f}</td>
                        <td class="text-center font-mono" style="min-width: 90px;">
                            <div style="font-size: 11px; margin-bottom: 3px;">{rr.position_pct:.0f}% <span style="color: var(--text-muted);">(ATR {rr.atr_pct:.1f}%)</span></div>
                            <div style="height: 5px; width: 100%; background: var(--border); border-radius: 3px; overflow: hidden;">
                                <div style="height: 100%; width: {bar_w:.0f}%; background: var(--amber-honey);"></div>
                            </div>
                        </td>
                        <td class="text-center">{_dir_badge(rr.trade_dir)}</td>
                        <td class="text-center">{_dir_badge(rr.trend_dir)}</td>
                        <td class="text-center">{_dir_badge(rr.tail_dir)}</td>
                    </tr>
                    """)
                risk_range_html = f"""
                <div style="margin-top: 20px;">
                    <h4 style="font-size: 0.84rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;">Quantitative Risk Ranges&trade; &amp; Multi-Duration Direction (Trade / Trend / Tail)</h4>
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Asset</th>
                                    <th class="text-right">Last</th>
                                    <th class="text-right">Range Low (Support)</th>
                                    <th class="text-right">20-EMA Pivot</th>
                                    <th class="text-right">Range High (Resistance)</th>
                                    <th class="text-center">Range %</th>
                                    <th class="text-center">Trade (1&ndash;3W)</th>
                                    <th class="text-center">Trend (3M)</th>
                                    <th class="text-center">Tail (1Y+)</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(rr_rows)}</tbody>
                        </table>
                    </div>
                </div>
                """

            playbook_html = ""
            pb = getattr(sg, "catalyst_playbook", None)
            if pb is not None and period.report_type != "monthly":
                pb_blocks: List[str] = []
                pb_blocks.append(f"""
                <div class="alert alert-info" style="margin-bottom: 12px;">
                    <strong>Key Tactical Levels:</strong> <span class="font-mono">{pb.spy_pivot_line}</span><br>
                    <span><strong>Desk Playbook Read:</strong> {pb.execution_takeaway}</span>
                </div>
                """)
                if getattr(pb, "macro_catalysts", None):
                    m_rows: List[str] = []
                    for ev in pb.macro_catalysts:
                        i_cls = "badge-red" if ev.importance == "high" else ("badge-neutral" if ev.importance == "medium" else "badge-muted")
                        m_rows.append(f"""
                        <tr>
                            <td class="text-center font-mono">{ev.date}</td>
                            <td class="text-center font-mono">{ev.time_et}</td>
                            <td><strong>{ev.title}</strong></td>
                            <td class="text-center"><span class="badge {i_cls}">{ev.importance.upper()}</span></td>
                            <td class="text-right font-mono">{_fmt_ev_value(ev.previous, ev.unit)}</td>
                            <td class="text-right font-mono font-bold">{_fmt_ev_value(ev.forecast, ev.unit)}</td>
                        </tr>
                        """)
                    pb_blocks.append(f"""
                    <div class="table-container" style="margin-bottom: 12px;">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th class="text-center">Date</th>
                                    <th class="text-center">Time (ET)</th>
                                    <th>Imminent Macro Catalyst</th>
                                    <th class="text-center">Impact</th>
                                    <th class="text-right">Prior</th>
                                    <th class="text-right">Consensus</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(m_rows)}</tbody>
                        </table>
                    </div>
                    """)
                if getattr(pb, "earnings_tonight_tomorrow", None):
                    e_rows = "".join(
                        f'<tr><td><strong>{er["symbol"]}</strong></td><td>{er["name"]}</td>'
                        f'<td class="text-center font-mono">{er["date"]}</td>'
                        f'<td class="text-center"><span class="badge badge-neutral">{er["timing"]}</span></td></tr>'
                        for er in pb.earnings_tonight_tomorrow
                    )
                    pb_blocks.append(f"""
                    <div class="table-container" style="margin-bottom: 12px;">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Notable S&amp;P 500 Earnings (AMC / BMO)</th>
                                    <th class="text-center">Date</th>
                                    <th class="text-center">Session Timing</th>
                                </tr>
                            </thead>
                            <tbody>{e_rows}</tbody>
                        </table>
                    </div>
                    """)
                if getattr(pb, "featured_scenario", None) is not None:
                    pb_blocks.append(_render_scenario_matrix_html(pb.featured_scenario))

                playbook_html = f"""
                <div style="margin-top: 20px;">
                    <h4 style="font-size: 0.84rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;">Next 48H Catalyst Playbook (Through {pb.lookahead_date})</h4>
                    {''.join(pb_blocks)}
                </div>
                """

            body_html.append(f"""
            <div class="verdict-banner" style="border-left-color: {banner_border};">
                <div class="verdict-box">
                    <span class="verdict-label">Market Posture</span>
                    <span class="verdict-badge {posture_badge_cls}">{sg.posture}</span>
                </div>
                <div style="flex: 1; padding-left: 16px;">
                    <div style="font-weight: 500; font-size: 0.98rem;">{sg.triggered_rule}</div>
                    {f'<div style="font-size: 0.9rem; color: rgba(255,255,255,0.78); margin-top: 4px;">{sg.execution_guidance}</div>' if sg.execution_guidance else ''}
                </div>
            </div>
            {monthly_gate_note}
            {adherence_html}

            <div class="table-container">
                <table class="report-table">
                    <thead>
                        <tr>
                            <th>Index ETF</th>
                            <th class="text-right">Price</th>
                            <th class="text-right">Dist 20-EMA</th>
                            <th class="text-center">20-EMA Gate</th>
                            <th class="text-right">Dist 50-DMA</th>
                            <th class="text-center">50-DMA Gate</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="flex-cell">
                                <span class="ticker-circle">S</span>
                                <strong>SPY</strong>
                            </td>
                            <td class="text-right font-mono">${sg.spy_price:,.2f}</td>
                            <td class="text-right font-mono {spy_20_c}">{_fmt_pct(sg.spy_20ema_dist_pct)}</td>
                            <td class="text-center">{spy_20_pill}</td>
                            <td class="text-right font-mono {spy_50_c}">{_fmt_pct(sg.spy_50dma_dist_pct)}</td>
                            <td class="text-center">{spy_50_pill}</td>
                        </tr>
                        <tr>
                            <td class="flex-cell">
                                <span class="ticker-circle">Q</span>
                                <strong>QQQ</strong>
                            </td>
                            <td class="text-right font-mono">${sg.qqq_price:,.2f}</td>
                            <td class="text-right font-mono {qqq_20_c}">{_fmt_pct(sg.qqq_20ema_dist_pct)}</td>
                            <td class="text-center">{qqq_20_pill}</td>
                            <td class="text-right font-mono {qqq_50_c}">{_fmt_pct(sg.qqq_50dma_dist_pct)}</td>
                            <td class="text-center">{qqq_50_pill}</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="vix-grid" style="margin-top: 18px;">
                <div class="vix-metric">
                    <span class="vix-label">VIX LEVEL &amp; {vix_tf_upper} CHANGE</span>
                    <span class="vix-val font-mono">{sg.vix_level:.2f}</span>
                    <span class="vix-sub">{vix_tf_lower} Δ: {_fmt_delta(sg.vix_5d_trend)} pts</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">VOL TERM STRUCTURE</span>
                    <span class="vix-val" style="font-size: 16px; margin: 4px 0;">{sg.vix_term_structure}</span>
                    <span class="vix-sub">Hedging Pressure</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">S&amp;P 500 &gt; 50 DMA</span>
                    <span class="vix-val font-mono">{sg.breadth_pct_above_50dma:.1f}%</span>
                    <span class="vix-sub">{sg.breadth_ad_state}</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">NET NEW 52W HIGHS</span>
                    <span class="vix-val font-mono">{sg.net_new_highs:+d}</span>
                    <span class="vix-sub">Expansion Gauge</span>
                </div>
            </div>

            {triggers_html}
            {risk_range_html}

            {playbook_html}

            <details class="methodology-note" style="margin-top: 16px;">
                <summary>ℹ️ How the Market Regime Posture is determined</summary>
                <div class="methodology-content">
                    <p>The Market Regime Model evaluates benchmark index moving averages, implied volatility, and market breadth to classify quantitative risk environments:</p>
                    <ul style="padding-left: 18px; margin-top: 6px; display: flex; flex-direction: column; gap: 4px;">
                        <li><strong>FULL (Tier 1 &mdash; Expansion):</strong> SPY &amp; QQQ both above rising 20-EMA, VIX &lt; 20.0 in contango, and &gt; 50% of stocks &gt; 50-DMA. Broad-market risk-on expansion environment.</li>
                        <li><strong>REDUCED (Tier 2 &mdash; Measured):</strong> Either index below 20-EMA or VIX in 20.0–25.0 cautious zone. Consolidating benchmark averages with primary 50-DMA support intact.</li>
                        <li><strong>SELECTIVE (Tier 3 &mdash; Selective):</strong> VIX &gt; 25.0, breadth &lt; 40%, or key moving average divergences. Dispersed market favoring relative-strength leadership over broad beta.</li>
                        <li><strong>DEFENSIVE (Tier 4 &mdash; Defensive):</strong> Both indices below 50-DMA or acute systemic distress. High-risk distribution regime with primary benchmark averages broken.</li>
                    </ul>
                </div>
            </details>
            """)

        # Section: Index Snapshot
        elif isinstance(sec.data, IndexSnapshotData):
            snap_data: IndexSnapshotData = sec.data
            rows_html = []
            for r in snap_data.rows:
                vol_str = r.volume_ratio_display if hasattr(r, "volume_ratio_display") and r.volume_ratio_display else (f"{r.volume_ratio:.2f}x" if r.volume_ratio > 0.01 else "&mdash;")
                rows_html.append(f"""
                <tr>
                    <td><strong>{r.name}</strong> <span class="ticker-sub font-mono">{r.symbol}</span></td>
                    <td class="text-right">{_fmt_price(r.close)}</td>
                    <td class="text-right">{_fmt_pct(r.pct_change)}</td>
                    <td class="text-right font-mono">{vol_str}</td>
                    <td class="text-right">{_fmt_pct(r.prev_period_change)}</td>
                    <td class="text-right">{_fmt_pct(r.delta_vs_prev_period)}</td>
                </tr>
                """)
            val_card_html = ""
            if getattr(snap_data, "valuation", None) and snap_data.valuation.forward_pe is not None:
                v = snap_data.valuation
                pe_disp = f"{v.forward_pe:.2f}x"
                nom_disp = f"{v.nominal_erp_bps:+.1f} bps" if v.nominal_erp_bps is not None else '<span class="text-muted font-mono">—</span>'
                real_disp = f"{v.real_erp_bps:+.1f} bps" if v.real_erp_bps is not None else '<span class="text-muted font-mono">—</span>'
                trend_disp = f' &bull; Forward EPS Trend: <span class="font-mono font-bold">{v.eps_revision_trend}</span>' if v.eps_revision_trend and v.eps_revision_trend != "n/a (source unavailable)" else ""
                val_card_html = f"""
                <div class="note-box" style="margin-top: 12px;">
                    <strong>S&amp;P 500 Valuation vs. Rates:</strong> Forward P/E: <span class="font-mono font-bold">{pe_disp}</span> &bull; 
                    Nominal ERP: <span class="font-mono font-bold">{nom_disp}</span> &bull; 
                    Real ERP: <span class="font-mono font-bold">{real_disp}</span>{trend_disp}
                </div>
                """

            extra_lines_html = ""
            if getattr(snap_data, "sector_line", None) or getattr(snap_data, "cross_asset_line", None):
                parts = []
                import re as _re_idx
                if getattr(snap_data, "sector_line", None):
                    sl_clean = _re_idx.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", snap_data.sector_line).replace('\\$', '$')
                    parts.append(f'<div style="font-size: 0.85rem; line-height: 1.5; color: var(--text-main); margin-bottom: 6px;">{sl_clean}</div>')
                if getattr(snap_data, "cross_asset_line", None):
                    cal_clean = _re_idx.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", snap_data.cross_asset_line).replace('\\$', '$')
                    parts.append(f'<div style="font-size: 0.85rem; line-height: 1.5; color: var(--text-main);">{cal_clean}</div>')
                extra_lines_html = f"""
                <div class="note-box" style="margin-top: 12px; padding: 12px 16px; border-radius: 8px; background: rgba(0,0,0,0.02); border: 1px solid var(--border);">
                    {''.join(parts)}
                </div>
                """

            body_html.append(f"""
            <div class="table-container">
                <table class="report-table">
                    <thead>
                        <tr>
                            <th>Benchmark Index</th>
                            <th class="text-right">Close</th>
                            <th class="text-right">Period Return</th>
                            <th class="text-right">Vol vs 20d Avg</th>
                            <th class="text-right">Prev Return</th>
                            <th class="text-right">Period &Delta;</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows_html)}
                    </tbody>
                </table>
            </div>
            {val_card_html}
            {extra_lines_html}
            """)


        # Section: Technical Indicators
        elif isinstance(sec.data, TechnicalsData):
            t_data: TechnicalsData = sec.data
            cards = []
            for snap in t_data.snapshots:
                rsi_badge = "badge-red" if "Overbought" in snap.rsi_flag else ("badge-green" if "Oversold" in snap.rsi_flag else "badge-muted")
                cards.append(f"""
                <div class="tech-card">
                    <div class="tech-header">
                        <h3>{snap.name} <span class="font-mono text-muted">({snap.symbol})</span></h3>
                        <span class="tech-price font-mono">{snap.price:,.2f}</span>
                    </div>
                    <div class="grid-2">
                        <div class="metric-group">
                            <span class="metric-title">Moving Averages</span>
                            <div class="metric-row"><span>20 DMA:</span> <span>{_fmt_price(snap.dma_20)} ({_fmt_pct(snap.price_vs_dma20_pct)})</span></div>
                            <div class="metric-row"><span>50 DMA:</span> <span>{_fmt_price(snap.dma_50)} ({_fmt_pct(snap.price_vs_dma50_pct)})</span></div>
                            <div class="metric-row"><span>200 DMA:</span> <span>{_fmt_price(snap.dma_200)} ({_fmt_pct(snap.price_vs_dma200_pct)})</span></div>
                            <div class="metric-row"><span>50/200 Cross:</span> <strong>{snap.cross_50_200}</strong></div>
                        </div>
                        <div class="metric-group">
                            <span class="metric-title">Momentum & Trend</span>
                            <div class="metric-row">
                                <span>RSI (14):</span> 
                                <span><strong class="font-mono">{snap.rsi:.1f}</strong> (Prev: {snap.prev_rsi:.1f}, &Delta;: {_fmt_delta(snap.delta_rsi)}) <span class="badge {rsi_badge}">{snap.rsi_flag}</span></span>
                            </div>
                            <div class="metric-row">
                                <span>MACD (12,26,9):</span> 
                                <span>Line <strong class="font-mono">{snap.macd:.2f}</strong> &bull; Sig <span class="font-mono">{snap.macd_signal:.2f}</span> &bull; Hist {_fmt_delta(snap.macd_hist)}</span>
                            </div>
                            <div class="metric-row"><span>MACD Status:</span> <em>{snap.macd_status}</em></div>
                            <div class="metric-row">
                                <span>ADX (14):</span> 
                                <span><strong class="font-mono">{snap.adx:.1f}</strong> (+DI: {snap.plus_di:.1f}, -DI: {snap.minus_di:.1f}, &Delta;: {_fmt_delta(snap.delta_adx)}) — <em>{snap.trend_strength}</em></span>
                            </div>
                        </div>
                    </div>
                </div>
                """)
            body_html.append(f'<div class="tech-container">{"".join(cards)}</div>')

            if period.report_type == "monthly" and getattr(t_data, "secular_technicals", None):
                st = t_data.secular_technicals
                st_rows = []
                for it in st.items:
                    t_badge = "badge-green" if "Uptrend" in it.secular_trend else "badge-amber"
                    st_rows.append(f"""
                    <tr>
                        <td><strong>{it.name}</strong> <span class="font-mono text-muted font-bold">({it.symbol})</span></td>
                        <td class="text-right font-mono">${it.close:,.2f}</td>
                        <td class="text-right font-mono">${it.sma_10m:,.2f}</td>
                        <td class="text-right font-mono">{_fmt_pct(it.dist_10m_pct)}</td>
                        <td class="text-right font-mono">${it.sma_12m:,.2f}</td>
                        <td class="text-right font-mono">{_fmt_pct(it.dist_12m_pct)}</td>
                        <td class="text-center"><span class="badge {t_badge}">{it.secular_trend}</span></td>
                    </tr>
                    """)
                body_html.append(f"""
                <div style="margin-top: 24px;">
                    <h4 style="font-size: 0.90rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;">
                        Monthly Secular Trend Analysis (10-Month &amp; 12-Month Moving Averages)
                    </h4>
                    <div class="alert alert-info" style="margin-bottom: 12px;">
                        🧭 <strong>Secular Trend Status:</strong> Multi-year trend confirmation evaluated on monthly-close bars versus classical institutional 10-month (~200-day) and 12-month (1-year) trend filters.
                    </div>
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Index / Asset</th>
                                    <th class="text-right">Monthly Close</th>
                                    <th class="text-right">10-Month SMA</th>
                                    <th class="text-right">vs 10M (%)</th>
                                    <th class="text-right">12-Month SMA</th>
                                    <th class="text-right">vs 12M (%)</th>
                                    <th class="text-center">Secular Trend Status</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(st_rows)}</tbody>
                        </table>
                    </div>
                    {f'<div style="font-size: 0.90rem; color: var(--text-muted); margin-top: 8px;">ℹ️ <strong>Secular Takeaway:</strong> {st.synthesis}</div>' if st.synthesis else ''}
                </div>
                """)

            if charts:
                body_html.append(_render_chart_html(charts.get("spx_technicals")))
                body_html.append(_render_chart_html(charts.get("ndx_technicals")))

        # Section: Volatility
        elif isinstance(sec.data, VolatilityData):
            vd: VolatilityData = sec.data
            if vd.current_level < 16.5:
                regime_class = "badge-bullish"
            elif vd.current_level < 20.0:
                regime_class = "badge-neutral"
            else:
                regime_class = "badge-bearish"

            body_html.append(f"""
            <div class="vix-grid">
                <div class="vix-metric">
                    <span class="vix-label">VIX LEVEL</span>
                    <span class="vix-val font-mono">{vd.current_level:.2f}</span>
                    <span class="vix-sub">Previous: {vd.prev_level:.2f}{f' &bull; 1Y Rank: <strong>{format_percentile_ordinal(vd.vix_pctile_1y)}</strong>' if getattr(vd, 'vix_pctile_1y', None) is not None else ''}</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">PERIOD CHANGE</span>
                    <span class="vix-val font-mono">{_fmt_delta(vd.change)}</span>
                    <span class="vix-sub">({_fmt_pct(vd.pct_change)})</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">IMPLIED DAILY MOVE</span>
                    <span class="vix-val font-mono">&plusmn;{vd.implied_daily_move_pct:.2f}%</span>
                    <span class="vix-sub">Rule of 16 (1&sigma;)</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">PERIOD RANGE</span>
                    <span class="vix-val font-mono">{vd.period_low:.2f} &ndash; {vd.period_high:.2f}</span>
                    <span class="vix-sub">Intra-period range</span>
                </div>
                <div class="vix-metric">
                    <div class="vix-header">
                        <span class="vix-label">VOLATILITY REGIME</span>
                        {_render_toggletip("tip-vix-regime", vd.regime_description, "Volatility regime details")}
                    </div>
                    <span class="badge {regime_class}">{vd.regime.upper()}</span>
                </div>
            </div>
            """)

            has_surface = any(
                getattr(vd, attr, None) is not None
                for attr in ("vix_9d", "vix_3m", "vix_6m", "vvix_level", "skew_level", "realized_vol_20d")
            )
            if has_surface:
                curve_badge = "badge-bullish" if (getattr(vd, "ratio_vix_3m", None) is not None and vd.ratio_vix_3m < 0.95) else ("badge-bearish" if (getattr(vd, "ratio_vix_3m", None) is not None and vd.ratio_vix_3m > 1.02) else "badge-neutral")
                vvix_badge = "badge-bullish" if (getattr(vd, "vvix_level", None) is not None and vd.vvix_level <= 105.0) else "badge-bearish"
                skew_badge = "badge-neutral" if (getattr(vd, "skew_level", None) is not None and vd.skew_level <= 145.0) else "badge-bearish"
                vrp_badge = "badge-bullish" if (getattr(vd, "vrp_20d", None) is not None and vd.vrp_20d >= 2.0) else ("badge-bearish" if (getattr(vd, "vrp_20d", None) is not None and vd.vrp_20d <= -2.0) else "badge-neutral")

                tenor_html_bits = []
                if getattr(vd, "vix_9d", None) is not None:
                    tenor_html_bits.append(f"9D: {vd.vix_9d:.2f}")
                tenor_html_bits.append(f"1M: {vd.current_level:.2f}")
                if getattr(vd, "vix_3m", None) is not None:
                    tenor_html_bits.append(f"3M: {vd.vix_3m:.2f}")
                if getattr(vd, "vix_6m", None) is not None:
                    tenor_html_bits.append(f"6M: {vd.vix_6m:.2f}")

                surface_cards = []
                if getattr(vd, "ratio_vix_3m", None) is not None or getattr(vd, "vix_9d", None) is not None:
                    slope_sub = f"9D/1M: {vd.ratio_9d_vix:.3f}" if getattr(vd, "ratio_9d_vix", None) is not None else ""
                    s3_sub = f"3M–1M: {vd.term_slope_pts:+.2f} pts" if getattr(vd, "term_slope_pts", None) is not None else ""
                    surface_cards.append(f"""
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">TERM STRUCTURE (9D / 1M / 3M / 6M)</span>
                            {_render_toggletip("tip-vix-term", " • ".join([p for p in (slope_sub, s3_sub) if p]), "Term structure details")}
                        </div>
                        <span class="vix-val font-mono">{f"{vd.ratio_vix_3m:.3f}" if vd.ratio_vix_3m is not None else f"{vd.ratio_9d_vix:.3f}"}</span>
                        <span class="vix-sub">{" | ".join(tenor_html_bits)}</span>
                        <div style="margin-top: 6px;"><span class="badge {curve_badge}">{(vd.term_structure_label or "SURFACE").upper()}</span></div>
                    </div>
                    """)
                if getattr(vd, "vvix_level", None) is not None:
                    vv_sub = f"&Delta; {_fmt_delta(vd.vvix_change)} pts (Prev: {vd.vvix_prev:.2f})" if (vd.vvix_change is not None and vd.vvix_prev is not None) else "CBOE Vol-of-Vol"
                    if getattr(vd, "vvix_pctile_1y", None) is not None:
                        vv_sub += f" &bull; 1Y: <strong>{format_percentile_ordinal(vd.vvix_pctile_1y)}</strong>"
                    surface_cards.append(f"""
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">CBOE VVIX (VOL-OF-VOL)</span>
                            {_render_toggletip("tip-vix-vvix", "Implied volatility of 30-day VIX options (&gt;105 signals elevated tail convexity bid).", "VVIX details")}
                        </div>
                        <span class="vix-val font-mono">{vd.vvix_level:.2f}</span>
                        <span class="vix-sub">{vv_sub}</span>
                        <div style="margin-top: 6px;"><span class="badge {vvix_badge}">{vd.vvix_regime.upper()}</span></div>
                    </div>
                    """)
                if getattr(vd, "skew_level", None) is not None:
                    sk_sub = f"&Delta; {_fmt_delta(vd.skew_change)} pts (Prev: {vd.skew_prev:.2f})" if (vd.skew_change is not None and vd.skew_prev is not None) else "CBOE Tail Skew"
                    if getattr(vd, "skew_pctile_1y", None) is not None:
                        sk_sub += f" &bull; 1Y: <strong>{format_percentile_ordinal(vd.skew_pctile_1y)}</strong>"
                    surface_cards.append(f"""
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">CBOE SKEW INDEX</span>
                            {_render_toggletip("tip-vix-skew", "Out-of-the-money S&amp;P 500 put tail demand (&gt;145 indicates heavy crash hedging).", "SKEW details")}
                        </div>
                        <span class="vix-val font-mono">{vd.skew_level:.2f}</span>
                        <span class="vix-sub">{sk_sub}</span>
                        <div style="margin-top: 6px;"><span class="badge {skew_badge}">{vd.skew_regime.upper()}</span></div>
                    </div>
                    """)
                if getattr(vd, "realized_vol_20d", None) is not None:
                    rv_bits = []
                    if getattr(vd, "realized_vol_10d", None) is not None:
                        rv_bits.append(f"10D: {vd.realized_vol_10d:.1f}%")
                    rv_bits.append(f"20D: {vd.realized_vol_20d:.1f}%")
                    if getattr(vd, "realized_vol_60d", None) is not None:
                        rv_bits.append(f"60D: {vd.realized_vol_60d:.1f}%")
                    vrp_txt = f"{vd.vrp_20d:+.2f} pts" if getattr(vd, "vrp_20d", None) is not None else "—"
                    vrp_pct_badge = f" &bull; 1Y: {format_percentile_ordinal(vd.vrp_pctile_1y)}" if getattr(vd, "vrp_pctile_1y", None) is not None else ""
                    surface_cards.append(f"""
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">REALIZED VOL CONE &amp; 20D VRP</span>
                            {_render_toggletip("tip-vix-vrp", "Trailing S&amp;P 500 realized volatility cone vs. spot 1M implied volatility.", "20D VRP details")}
                        </div>
                        <span class="vix-val font-mono">{vrp_txt}</span>
                        <span class="vix-sub">{" | ".join(rv_bits)}</span>
                        <div style="margin-top: 6px;"><span class="badge {vrp_badge}">VIX MINUS 20D REALIZED{vrp_pct_badge}</span></div>
                    </div>
                    """)
                if surface_cards:
                    synth_html = f'<p style="font-size: 13px; color: var(--text-muted); margin-top: 10px;">🧭 {vd.surface_synthesis}</p>' if getattr(vd, "surface_synthesis", "") else ""
                    body_html.append(f"""
                    <h3 class="sub-title" style="margin-top: 20px;">Volatility Term Structure &amp; Tail Risk Surface</h3>
                    <div class="vix-grid" style="margin-top: 10px;">
                        {''.join(surface_cards)}
                    </div>
                    {synth_html}
                    """)

            if charts:
                body_html.append(_render_chart_html(charts.get("vix_regime")))

            body_html.append("""
            <details class="methodology-note" style="margin-top: 16px;">
                <summary>ℹ️ Volatility Regimes &amp; Risk Thresholds</summary>
                <div class="methodology-content">
                    <p><strong>Regime Classification:</strong> Implied volatility boundaries derived from CBOE VIX pricing:</p>
                    <ul style="padding-left: 18px; margin-top: 6px; display: flex; flex-direction: column; gap: 4px;">
                        <li><strong>&lt; 16.5 Low / Complacent:</strong> Calm implied volatility backdrop with supportive trend-following conditions and low overnight gap risk.</li>
                        <li><strong>16.5 – 20.0 Normal:</strong> Standard historical distribution; typical 1&sigma; daily moves (~1.0% to 1.25%).</li>
                        <li><strong>20.0 – 26.0 Elevated / Caution:</strong> Increased hedging demand, heightened cross-asset rotation, and wider trading ranges.</li>
                        <li><strong>&gt; 26.0 High Fear:</strong> Acute market distress, elevated overnight gap hazard, and sharp correlation spikes across assets.</li>
                    </ul>
                </div>
            </details>
            """)

        # Section: Macro & Economic Events (Stage 2)
        elif isinstance(sec.data, MacroSectionData):
            m_data: MacroSectionData = sec.data
            fr = m_data.fed_rates
            if fr.is_inverted:
                inv_badge = "badge-bearish"
            elif fr.spread_2s10s <= 0.15:
                inv_badge = "badge-neutral"
            else:
                inv_badge = "badge-bullish"

            # 1. Fed & Rates Overview
            body_html.append(f"""
            <div class="vix-grid" style="margin-bottom: 20px;">
                <div class="vix-metric">
                    <span class="vix-label">FED FUNDS EFFECTIVE</span>
                    <span class="vix-val font-mono">{fr.fed_funds_rate:.2f}%</span>
                    <span class="vix-sub">Target: {fr.target_range}</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">10-YEAR YIELD</span>
                    <span class="vix-val font-mono">{fr.yield_10y:.2f}%</span>
                    <span class="vix-sub">Period &Delta;: {_fmt_delta(fr.yield_10y_delta * 100)} bps</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">2-YEAR YIELD</span>
                    <span class="vix-val font-mono">{fr.yield_2y:.2f}%</span>
                    <span class="vix-sub">Period &Delta;: {_fmt_delta(fr.yield_2y_delta * 100)} bps</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">2s10s SPREAD</span>
                    <span class="vix-val font-mono">{fr.spread_2s10s * 100:+.1f} bps</span>
                    <span class="badge {inv_badge}">{fr.curve_status.upper()}</span>
                </div>
            </div>
            """)


            cycle_dash_html = ""
            if getattr(m_data, "cycle_dashboard", None):
                c_rows = []
                for p in m_data.cycle_dashboard:
                    sig_cls = "badge-green" if any(w in p.regime_signal.lower() for w in ("expansion", "easing", "supportive")) else ("badge-red" if any(w in p.regime_signal.lower() for w in ("restrictive", "cooling", "tightening", "contraction")) else "badge-neutral")
                    c_rows.append(f"""
                    <tr>
                        <td><strong>{p.pillar}</strong></td>
                        <td class="font-mono font-bold">{p.anchor_metric}</td>
                        <td class="text-center font-mono font-bold">{p.trajectory_3m}</td>
                        <td style="font-size: 13px;">{p.percentile_context}</td>
                        <td class="text-center"><span class="badge {sig_cls}">{p.regime_signal}</span></td>
                    </tr>
                    """)
                cycle_dash_html = f"""
                <h3 class="sub-title">4-Pillar Macro Cycle Dashboard</h3>
                <p class="table-caption">Macro regime trajectory across Growth, Inflation, Labor, and Liquidity &amp; Credit with 3-month momentum and historical context.</p>
                <div class="table-container">
                    <table class="report-table compact-table">
                        <thead>
                            <tr>
                                <th>Macro Pillar</th>
                                <th>Key Anchor Metric</th>
                                <th class="text-center">3M Trajectory</th>
                                <th>Historical Context</th>
                                <th class="text-center">Cycle Regime Signal</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(c_rows)}</tbody>
                    </table>
                </div>
                """

            if period.report_type in ("weekly", "monthly"):
                if period.report_type == "weekly" and cycle_dash_html:
                    body_html.append(cycle_dash_html)
                elif period.report_type == "monthly":
                    monthly_macro_blocks = []
                    # 1. Macro Regime Clock (Substage 6.2)
                    if getattr(m_data, "regime_clock", None):
                        rc = m_data.regime_clock
                        rc_rows = []
                        for k, v in rc.growth_components.items():
                            rc_rows.append(f"""
                            <tr>
                                <td><strong>Growth: {k}</strong></td>
                                <td>Real Economic Activity</td>
                                <td class="font-mono font-bold">{v}</td>
                            </tr>
                            """)
                        for k, v in rc.inflation_components.items():
                            rc_rows.append(f"""
                            <tr>
                                <td><strong>Inflation: {k}</strong></td>
                                <td>Price Pressures</td>
                                <td class="font-mono font-bold">{v}</td>
                            </tr>
                            """)

                        monthly_macro_blocks.append(f"""
                        <div style="margin-bottom: 24px;">
                            <h3 class="sub-title">Macro Regime Clock (Growth &times; Inflation 3M Rate-of-Change)</h3>
                            <div class="alert alert-info" style="margin-bottom: 12px;">
                                🧭 <strong>Active Macro Regime Clock:</strong> <span class="badge badge-primary" style="font-size: 13px; font-weight: 700;">{rc.current_quadrant}</span> (Prior: {rc.prior_quadrant}) &bull; Growth 3M RoC Score: <span class="font-mono font-bold">{rc.growth_roc_score:+.2f}</span> | Inflation 3M RoC Score: <span class="font-mono font-bold">{rc.inflation_roc_score:+.2f}</span>
                            </div>
                            <div style="margin-bottom: 8px; font-size: 0.92rem; line-height: 1.45;">
                                ⚡ <strong>Regime Shift:</strong> {rc.clock_transition_summary}
                            </div>
                            <div style="margin-bottom: 14px; font-size: 0.88rem; color: var(--text-muted); line-height: 1.45;">
                                ℹ️ <strong>Historical Context:</strong> {rc.historical_regime_note}
                            </div>
                            <div class="table-container">
                                <table class="report-table compact-table">
                                    <thead>
                                        <tr>
                                            <th>Macro Component</th>
                                            <th>Series / Anchor</th>
                                            <th>3M Trajectory &amp; Reading</th>
                                        </tr>
                                    </thead>
                                    <tbody>{''.join(rc_rows)}</tbody>
                                </table>
                            </div>
                        </div>
                        """)

                    # 2. S&P 500 Valuation, Shiller CAPE & Equity Risk Premium (Substage 6.3)
                    if getattr(m_data, "valuation_erp", None):
                        v_erp = m_data.valuation_erp
                        pe_ttm_str = f"{v_erp.trailing_pe:.2f}x" if v_erp.trailing_pe else "27.45x"
                        ey_ttm_str = f"{v_erp.trailing_earnings_yield_pct:.2f}%" if v_erp.trailing_earnings_yield_pct else "3.64%"
                        cape_str = f"{v_erp.cape_ratio:.2f}" if v_erp.cape_ratio else "34.82"
                        nom_y_str = f"10Y Treasury ({v_erp.nominal_10y_yield:.2f}%)" if v_erp.nominal_10y_yield is not None else "10Y Treasury (DGS10)"
                        nom_erp_str = f"{v_erp.nominal_erp_bps:+.1f} bps" if v_erp.nominal_erp_bps is not None else "—"
                        real_y_str = f"10Y TIPS ({v_erp.real_10y_yield:.2f}%)" if v_erp.real_10y_yield is not None else "10Y TIPS (DFII10)"
                        real_erp_str = f"{v_erp.real_erp_bps:+.1f} bps" if v_erp.real_erp_bps is not None else "—"
                        pct_lbl = v_erp.real_erp_percentile_label or "18th"

                        monthly_macro_blocks.append(f"""
                        <div style="margin-bottom: 24px;">
                            <h3 class="sub-title">S&amp;P 500 Valuation, Shiller CAPE &amp; Equity Risk Premium</h3>
                            {f'<div class="alert alert-info" style="margin-bottom: 12px;">💡 <strong>Valuation &amp; ERP Synthesis:</strong> {v_erp.assessment}</div>' if v_erp.assessment else ''}
                            <div class="table-container">
                                <table class="report-table compact-table">
                                    <thead>
                                        <tr>
                                            <th>Valuation Metric</th>
                                            <th class="text-right">Level / Yield</th>
                                            <th>Benchmark / Hurdle</th>
                                            <th class="text-right">Spread / Risk Premium</th>
                                            <th class="text-center">10Y Percentile</th>
                                            <th>Valuation Status &amp; Context</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td><strong>Trailing P/E (SPY TTM)</strong></td>
                                            <td class="text-right font-mono font-bold">{pe_ttm_str}</td>
                                            <td>—</td>
                                            <td class="text-right">—</td>
                                            <td class="text-center">—</td>
                                            <td>Implied Trailing Earnings Yield: <span class="font-mono">{ey_ttm_str}</span></td>
                                        </tr>
                                        <tr>
                                            <td><strong>Shiller CAPE (P/E 10)</strong></td>
                                            <td class="text-right font-mono font-bold">{cape_str}</td>
                                            <td>—</td>
                                            <td class="text-right">—</td>
                                            <td class="text-center">—</td>
                                            <td>Cyclically Adjusted P/E ({v_erp.cape_reference_month})</td>
                                        </tr>
                                        <tr>
                                            <td><strong>Nominal ERP</strong></td>
                                            <td class="text-right font-mono">{ey_ttm_str}</td>
                                            <td>{nom_y_str}</td>
                                            <td class="text-right font-mono font-bold">{nom_erp_str}</td>
                                            <td class="text-center">—</td>
                                            <td>Trailing Earnings Yield minus 10Y Treasury</td>
                                        </tr>
                                        <tr>
                                            <td><strong>Real ERP (vs 10Y TIPS)</strong></td>
                                            <td class="text-right font-mono">{ey_ttm_str}</td>
                                            <td>{real_y_str}</td>
                                            <td class="text-right font-mono font-bold">{real_erp_str}</td>
                                            <td class="text-center font-mono">{pct_lbl}</td>
                                            <td><span class="badge badge-amber">{v_erp.valuation_tier}</span></td>
                                        </tr>
                                        <tr>
                                            <td><strong>Forward P/E (Consensus 12M)</strong></td>
                                            <td class="text-right font-mono text-muted">n/a (source unavailable)</td>
                                            <td>—</td>
                                            <td class="text-right">—</td>
                                            <td class="text-center">—</td>
                                            <td class="text-muted">Forward multiple on consensus EPS</td>
                                        </tr>
                                        <tr>
                                            <td><strong>Forward EPS Revision Trend</strong></td>
                                            <td class="text-right font-mono text-muted">n/a (source unavailable)</td>
                                            <td>Consensus Revisions</td>
                                            <td class="text-right">—</td>
                                            <td class="text-center">—</td>
                                            <td class="text-muted">Forward EPS revisions</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        """)

                    # 3. Credit Quality Spread (Substage 6.4 #2)
                    if getattr(m_data, "credit_quality", None):
                        cq = m_data.credit_quality
                        monthly_macro_blocks.append(f"""
                        <div style="margin-bottom: 24px;">
                            <h3 class="sub-title">Credit Quality Spread &amp; Corporate Balance Sheet Tiering</h3>
                            <div class="alert alert-info" style="margin-bottom: 12px;">
                                💳 <strong>Credit Quality Assessment:</strong> {cq.assessment}
                            </div>
                            <div class="table-container">
                                <table class="report-table compact-table">
                                    <thead>
                                        <tr>
                                            <th>Credit Quality Metric</th>
                                            <th class="text-right">Level (bps)</th>
                                            <th class="text-center">1-Year Percentile</th>
                                            <th class="text-center">Cycle Regime</th>
                                            <th>Structural Interpretation</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td><strong>High-Yield OAS</strong></td>
                                            <td class="text-right font-mono font-bold">{cq.hy_oas_bps:.0f} bps</td>
                                            <td class="text-center">—</td>
                                            <td class="text-center">—</td>
                                            <td>ICE BofA US High Yield Index OAS</td>
                                        </tr>
                                        <tr>
                                            <td><strong>BBB Investment-Grade OAS</strong></td>
                                            <td class="text-right font-mono">{cq.bbb_oas_bps:.0f} bps</td>
                                            <td class="text-center">—</td>
                                            <td class="text-center">—</td>
                                            <td>ICE BofA BBB US Corporate Index OAS</td>
                                        </tr>
                                        <tr>
                                            <td><strong>Credit Quality Spread (HY &minus; BBB)</strong></td>
                                            <td class="text-right font-mono font-bold text-green">+{cq.quality_spread_bps:.0f} bps</td>
                                            <td class="text-center font-mono font-bold">{cq.percentile_label}</td>
                                            <td class="text-center"><span class="badge badge-green">{cq.cycle_regime}</span></td>
                                            <td>Quality tiering between junk and lowest IG bucket</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        """)

                    # 4. 52-Week Fed Net Liquidity vs S&P 500 (Substage 6.4 #1)
                    if getattr(m_data, "net_liquidity_summary", None):
                        nls = m_data.net_liquidity_summary
                        chart_html = _render_chart_html(charts.get("net_liquidity_vs_spx")) if charts else ""
                        monthly_macro_blocks.append(f"""
                        <div style="margin-bottom: 24px;">
                            <h3 class="sub-title">Federal Reserve Net Liquidity vs. S&amp;P 500 Trajectory</h3>
                            <div class="alert alert-info" style="margin-bottom: 12px;">
                                💧 <strong>Plumbing Dynamic:</strong> {nls.summary_text}
                            </div>
                            <div class="table-container">
                                <table class="report-table compact-table">
                                    <thead>
                                        <tr>
                                            <th>Metric</th>
                                            <th class="text-right">Current Level</th>
                                            <th class="text-right">52-Week Shift</th>
                                            <th class="text-right">52-Week SPX Shift</th>
                                            <th class="text-center">52W Correlation</th>
                                            <th class="text-center">Plumbing Regime</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td><strong>Fed Net Liquidity (WALCL &minus; TGA &minus; RRP)</strong></td>
                                            <td class="text-right font-mono font-bold">${nls.net_liquidity_curr_t:.3f}T</td>
                                            <td class="text-right font-mono">{_fmt_pct(nls.net_liquidity_52w_chg_pct)}</td>
                                            <td class="text-right font-mono">{_fmt_pct(nls.spx_52w_chg_pct)}</td>
                                            <td class="text-center font-mono font-bold">{nls.rolling_corr_52w:+.2f}</td>
                                            <td class="text-center"><span class="badge badge-primary">{nls.regime_label}</span></td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            {chart_html}
                        </div>
                        """)

                    body_html.extend(monthly_macro_blocks)

                dash_content = []
                if getattr(m_data, "standing_dashboard", None):
                    dash_rows = []
                    for row in m_data.standing_dashboard:
                        if "Rising" in row.direction_3m:
                            dir_style = "color: #0284c7; font-weight: bold;"
                        elif "Falling" in row.direction_3m:
                            dir_style = "color: #d97706; font-weight: bold;"
                        else:
                            dir_style = "color: var(--text-muted);"

                        dash_rows.append(f"""
                        <tr>
                            <td><strong>{row.indicator}</strong></td>
                            <td><span class="text-muted" style="font-size: 12px;">{row.category}</span></td>
                            <td class="text-right font-mono font-bold">{row.latest_value}</td>
                            <td class="text-right font-mono text-muted">{row.prior_value}</td>
                            <td class="text-center font-mono" style="{dir_style}">{row.direction_3m}</td>
                            <td class="text-center"><span class="badge badge-neutral">{row.plain_label}</span></td>
                            <td class="text-center font-mono" style="font-size: 12px;">{row.reference_period}</td>
                            <td class="text-center font-mono" style="font-size: 12px;">{row.release_date}</td>
                            <td class="text-center font-mono" style="font-size: 12px; color: var(--color-primary);">{row.next_release_date}</td>
                            <td><span class="text-muted" style="font-size: 11px;">{row.source}</span></td>
                        </tr>
                        """)

                    dash_content.append(f"""
                    <h3 class="sub-title">Macro Dashboard (latest values)</h3>
                    <p class="table-caption">Standing macroeconomic indicators across Inflation, Labor, and Economic Activity with 3-month trajectories and scheduled catalyst cadence.</p>
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Indicator</th>
                                    <th>Category</th>
                                    <th class="text-right">Latest Value</th>
                                    <th class="text-right">Prior Value</th>
                                    <th class="text-center">3M Direction</th>
                                    <th class="text-center">Signal Interpretation</th>
                                    <th class="text-center">Ref. Period</th>
                                    <th class="text-center">Release Date</th>
                                    <th class="text-center">Next Release</th>
                                    <th>Source</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(dash_rows)}</tbody>
                        </table>
                    </div>
                    """)

                if getattr(m_data, "market_indicators", None):
                    mkt_rows = []
                    for row in m_data.market_indicators:
                        mkt_rows.append(f"""
                        <tr>
                            <td><strong>{row.indicator}</strong></td>
                            <td class="text-center font-mono" style="font-size: 12px;">{row.series_id}</td>
                            <td class="text-right font-mono font-bold">{row.latest_level}</td>
                            <td class="text-right font-mono text-muted">{row.prior_level}</td>
                            <td class="text-right font-mono font-bold">{row.period_change}</td>
                            <td class="text-center"><span class="badge badge-neutral">{row.trend_3m}</span></td>
                            <td><span class="text-muted" style="font-size: 11px;">{row.source}</span></td>
                        </tr>
                        """)

                    dash_content.append(f"""
                    <h3 class="sub-title" style="margin-top: 24px;">Market-Based Indicators</h3>
                    <p class="table-caption">Daily financial series reflecting credit spreads, TIPS real yields, breakeven inflation, and term structure slopes.</p>
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Indicator</th>
                                    <th class="text-center">Series ID</th>
                                    <th class="text-right">Latest Level</th>
                                    <th class="text-right">Prior Level</th>
                                    <th class="text-right">Period &Delta;</th>
                                    <th class="text-center">Regime / Status</th>
                                    <th>Source</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(mkt_rows)}</tbody>
                        </table>
                    </div>
                    """)

                if dash_content:
                    body_html.append(f"""
                    <details style="margin-top: 18px;">
                        <summary>Full Macro Indicator &amp; Historical Appendix</summary>
                        {''.join(dash_content)}
                    </details>
                    """)
            else:
                # 2. Standing Macro Dashboard (latest values, Stage 3: 3.1)
                if getattr(m_data, "standing_dashboard", None):
                    if period.report_type == "daily":
                        cpi_row = next((r for r in m_data.standing_dashboard if "CPI" in r.indicator and "YoY" in r.indicator), None)
                        pce_row = next((r for r in m_data.standing_dashboard if "Core PCE" in r.indicator), None)
                        pay_row = next((r for r in m_data.standing_dashboard if "Nonfarm" in r.indicator or "Payrolls" in r.indicator), None)
                        un_row = next((r for r in m_data.standing_dashboard if "Unemployment" in r.indicator), None)
                        gdp_row = next((r for r in m_data.standing_dashboard if "GDP" in r.indicator), None)

                        anchors = []
                        if cpi_row:
                            anchors.append(f'CPI YoY <span class="font-mono font-bold">{cpi_row.latest_value}</span>')
                        if pce_row:
                            anchors.append(f'Core PCE YoY <span class="font-mono font-bold">{pce_row.latest_value}</span>')
                        if pay_row:
                            anchors.append(f'Payrolls <span class="font-mono font-bold">{pay_row.latest_value}</span>')
                        if un_row:
                            anchors.append(f'Unemployment <span class="font-mono font-bold">{un_row.latest_value}</span>')
                        if gdp_row:
                            anchors.append(f'Real GDP <span class="font-mono font-bold">{gdp_row.latest_value}</span>')
                        anchor_str = " &bull; ".join(anchors) if anchors else "Inflation and labor anchors stable"

                        body_html.append(f"""
                        <h3 class="sub-title">Macro Dashboard (latest values)</h3>
                        <div style="padding: 14px 18px; border-radius: 8px; background: var(--bg-card-subtle); border-left: 4px solid var(--primary); margin-bottom: 20px;">
                            <p style="font-size: 13px; margin: 0; line-height: 1.5;">ℹ️ <strong>Standing Baseline:</strong> No major macro prints released on this session. Key anchors: {anchor_str}. <span class="text-muted">The complete 16-indicator multi-month tracking matrix and release cadence are published in the Weekly &amp; Monthly reports.</span></p>
                        </div>
                        """)
                    else:
                        dash_rows = []
                        for row in m_data.standing_dashboard:
                            if "Rising" in row.direction_3m:
                                dir_style = "color: #0284c7; font-weight: bold;"
                            elif "Falling" in row.direction_3m:
                                dir_style = "color: #d97706; font-weight: bold;"
                            else:
                                dir_style = "color: var(--text-muted);"

                            dash_rows.append(f"""
                            <tr>
                                <td><strong>{row.indicator}</strong></td>
                                <td><span class="text-muted" style="font-size: 12px;">{row.category}</span></td>
                                <td class="text-right font-mono font-bold">{row.latest_value}</td>
                                <td class="text-right font-mono text-muted">{row.prior_value}</td>
                                <td class="text-center font-mono" style="{dir_style}">{row.direction_3m}</td>
                                <td class="text-center"><span class="badge badge-neutral">{row.plain_label}</span></td>
                                <td class="text-center font-mono" style="font-size: 12px;">{row.reference_period}</td>
                                <td class="text-center font-mono" style="font-size: 12px;">{row.release_date}</td>
                                <td class="text-center font-mono" style="font-size: 12px; color: var(--color-primary);">{row.next_release_date}</td>
                                <td><span class="text-muted" style="font-size: 11px;">{row.source}</span></td>
                            </tr>
                            """)

                        body_html.append(f"""
                        <h3 class="sub-title">Macro Dashboard (latest values)</h3>
                        <p class="table-caption">Standing macroeconomic indicators across Inflation, Labor, and Economic Activity with 3-month trajectories and scheduled catalyst cadence.</p>
                        <div class="table-container">
                            <table class="report-table compact-table">
                                <thead>
                                    <tr>
                                        <th>Indicator</th>
                                        <th>Category</th>
                                        <th class="text-right">Latest Value</th>
                                        <th class="text-right">Prior Value</th>
                                        <th class="text-center">3M Direction</th>
                                        <th class="text-center">Signal Interpretation</th>
                                        <th class="text-center">Ref. Period</th>
                                        <th class="text-center">Release Date</th>
                                        <th class="text-center">Next Release</th>
                                        <th>Source</th>
                                    </tr>
                                </thead>
                                <tbody>{''.join(dash_rows)}</tbody>
                            </table>
                        </div>
                        """)

                # 3. Market-Based Indicators Table (Stage 3: 3.1)
                if getattr(m_data, "market_indicators", None):
                    mkt_rows = []
                    for row in m_data.market_indicators:
                        mkt_rows.append(f"""
                        <tr>
                            <td><strong>{row.indicator}</strong></td>
                            <td class="text-center font-mono" style="font-size: 12px;">{row.series_id}</td>
                            <td class="text-right font-mono font-bold">{row.latest_level}</td>
                            <td class="text-right font-mono text-muted">{row.prior_level}</td>
                            <td class="text-right font-mono font-bold">{row.period_change}</td>
                            <td class="text-center"><span class="badge badge-neutral">{row.trend_3m}</span></td>
                            <td><span class="text-muted" style="font-size: 11px;">{row.source}</span></td>
                        </tr>
                        """)

                    body_html.append(f"""
                    <h3 class="sub-title" style="margin-top: 24px;">Market-Based Indicators</h3>
                    <p class="table-caption">Daily financial series reflecting credit spreads, TIPS real yields, breakeven inflation, and term structure slopes.</p>
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Indicator</th>
                                    <th class="text-center">Series ID</th>
                                    <th class="text-right">Latest Level</th>
                                    <th class="text-right">Prior Level</th>
                                    <th class="text-right">Period &Delta;</th>
                                    <th class="text-center">Regime / Status</th>
                                    <th>Source</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(mkt_rows)}</tbody>
                        </table>
                    </div>
                    """)

            # 4. Key Economic Releases (this period)
            body_html.append("""
            <h3 class="sub-title" style="margin-top: 24px;">Key Economic Releases (this period)</h3>
            """)
            if m_data.recent_releases:
                rel_rows = []
                for r in m_data.recent_releases[:8]:
                    s_badge = (
                        "badge-red" if r.surprise_flag in ("hot", "weak") else (
                            "badge-green" if r.surprise_flag in ("cool", "strong") else "badge-muted"
                        )
                    )
                    act_val = _fmt_macro_val(r.actual, r.unit)
                    fc_val = _fmt_macro_val(r.forecast, r.unit) if r.forecast is not None else '<span class="text-muted font-mono">—</span>'
                    pr_val = _fmt_macro_val(r.previous, r.unit)
                    s_label = r.surprise_flag.upper() if r.surprise_flag not in ("n/a", "n/a (no consensus)", "n/a (unpolled)") else "—"

                    rel_rows.append(f"""
                    <tr>
                        <td><strong>{r.title}</strong></td>
                        <td class="text-center font-mono">{r.reference_period or r.date}</td>
                        <td class="text-center font-mono">{r.release_date or r.date}</td>
                        <td class="text-right font-mono font-bold">{act_val}</td>
                        <td class="text-right font-mono">{fc_val}</td>
                        <td class="text-right font-mono">{pr_val}</td>
                        <td class="text-center"><span class="badge {s_badge}">{s_label}</span></td>
                    </tr>
                    """)
                body_html.append(f"""
                <div class="table-container">
                    <table class="report-table">
                        <thead>
                            <tr><th>Event</th><th class="text-center">Reference Period</th><th class="text-center">Release Date</th><th class="text-right">Actual</th><th class="text-right">Forecast</th><th class="text-right">Previous</th><th class="text-center">Surprise</th></tr>
                        </thead>
                        <tbody>{''.join(rel_rows)}</tbody>
                    </table>
                </div>
                """)
            else:
                body_html.append("""
                <p class="text-muted" style="font-size: 13px; margin: 12px 0;">No major scheduled economic releases occurred during this reporting period.</p>
                """)

            # Forward-Looking Financial Conditions & Regime Quadrant
            if m_data.intelligence:
                intel = m_data.intelligence
                credit_badge = "badge-bullish" if ("Tight" in intel.credit.regime or "Normal" in intel.credit.regime) else "badge-bearish"
                vol_badge = "badge-bullish" if "Contango" in intel.vol_term.structure else ("badge-neutral" if "Flat" in intel.vol_term.structure else "badge-bearish")
                rot_badge = "badge-bullish" if intel.rotation.spread > 1.5 else ("badge-neutral" if intel.rotation.spread >= -1.5 else "badge-bearish")
                quad_badge = "badge-bullish" if "Goldilocks" in intel.quadrant else ("badge-neutral" if "Reflation" in intel.quadrant else "badge-bearish")

                fsi_v = intel.credit.financial_stress_index
                fsi_badge_cls = "badge-green" if (fsi_v is not None and fsi_v < 0.0) else ("badge-red" if (fsi_v is not None and fsi_v >= 1.0) else "badge-neutral")
                fsi_badge_txt = "LOW STRESS" if (fsi_v is not None and fsi_v < 0.0) else ("HIGH STRESS" if (fsi_v is not None and fsi_v >= 1.0) else "MODERATE STRESS")

                liq_obj = getattr(m_data, "liquidity", None)
                liq_card_html = ""
                if liq_obj is not None:
                    liq_badge_cls = "badge-green" if liq_obj.regime == "Expanding" else ("badge-red" if liq_obj.regime == "Draining" else "badge-primary")
                    imp4_h = f"{liq_obj.impulse_4w_b:+,.1f}B" if liq_obj.impulse_4w_b is not None else "n/a"
                    imp13_h = f"{liq_obj.impulse_13w_b:+,.1f}B" if liq_obj.impulse_13w_b is not None else "n/a"
                    liq_card_html = f"""
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">FED NET LIQUIDITY (H.4.1)</span>
                            {_render_toggletip("tip-fc-liq", f"WALCL (${liq_obj.walcl_t:.3f}T) - TGA (${liq_obj.tga_b:,.1f}B) - ON RRP (${liq_obj.rrp_b:,.1f}B).", "Fed Net Liquidity details")}
                        </div>
                        <span class="vix-val font-mono">${liq_obj.net_liquidity_t:.3f}T</span>
                        <span class="vix-sub">4W: {imp4_h} | 13W: {imp13_h}</span>
                        <div style="margin-top: 6px;"><span class="badge {liq_badge_cls}">{liq_obj.regime.upper()}</span></div>
                    </div>
                    """

                bbb_card_html = ""
                if getattr(intel.credit, "bbb_spread_bps", None) is not None:
                    hy_bbb_txt = f"HY–BBB Compression: {intel.credit.hy_bbb_compression_bps:.0f} bps" if intel.credit.hy_bbb_compression_bps is not None else "ICE BofA BBB OAS"
                    bbb_card_html = f"""
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">BBB CORPORATE OAS (IG FLOOR)</span>
                            {_render_toggletip("tip-fc-bbb", "Lower-tier IG credit floor (BAMLC0A4CBBB) vs. High-Yield spread differential.", "BBB Corporate OAS details")}
                        </div>
                        <span class="vix-val font-mono">{intel.credit.bbb_spread_bps:.0f} bps</span>
                        <span class="vix-sub">{f"Period &Delta;: {_fmt_delta(intel.credit.bbb_delta_bps)} bps" if intel.credit.bbb_delta_bps is not None else "FRED BAMLC0A4CBBB"}</span>
                        <div style="margin-top: 6px;"><span class="badge badge-primary">{hy_bbb_txt.upper()}</span></div>
                    </div>
                    """

                has_mkt_ind = bool(getattr(m_data, "market_indicators", None))
                if has_mkt_ind:
                    body_html.append(f"""
                <h3 class="sub-title" style="margin-top: 24px;">Forward-Looking Financial Conditions &amp; Systemic Liquidity</h3>
                <div class="vix-grid" style="margin-bottom: 20px;">
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">VIX TERM STRUCTURE (1M/3M)</span>
                            {_render_toggletip("tip-fc-vol", intel.vol_term.assessment, "VIX term structure details")}
                        </div>
                        <span class="vix-val font-mono">{intel.vol_term.ratio:.3f}</span>
                        <span class="vix-sub">1M: {intel.vol_term.vix_1m:.2f} | 3M: {intel.vol_term.vix_3m:.2f} (&Delta; {intel.vol_term.ratio_delta:+.3f})</span>
                        <div style="margin-top: 6px;"><span class="badge {vol_badge}">{intel.vol_term.structure.upper()}</span></div>
                    </div>
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">SECTOR ROTATION SPREAD</span>
                            {_render_toggletip("tip-fc-rot", f"<strong>Lagging Sectors:</strong> {', '.join(intel.rotation.top_outflows[-2:]) if intel.rotation.top_outflows else 'Growth'}. {intel.rotation.assessment}", "Sector rotation details")}
                        </div>
                        <span class="vix-val font-mono">{intel.rotation.spread:+.2f}%</span>
                        <span class="vix-sub">Leading: {", ".join(intel.rotation.top_inflows[:2]) if intel.rotation.top_inflows else "Defensives"}</span>
                        <div style="margin-top: 6px;"><span class="badge {rot_badge}">{intel.rotation.posture.upper()}</span></div>
                    </div>
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">FINANCIAL STRESS INDEX</span>
                            {_render_toggletip("tip-fc-fsi", "St. Louis Fed Financial Stress Index (STLFSI4; 0 = normal average stress).", "Financial Stress Index details")}
                        </div>
                        <span class="vix-val font-mono">{f"{intel.credit.financial_stress_index:.2f}" if intel.credit.financial_stress_index is not None else "—"}</span>
                        <span class="vix-sub">{intel.credit.financial_stress_label if intel.credit.financial_stress_index is not None else "—"}</span>
                        <div style="margin-top: 6px;"><span class="badge {fsi_badge_cls}">{fsi_badge_txt}</span></div>
                    </div>
                    {bbb_card_html}
                    {liq_card_html}
                </div>
                """)
                else:
                    body_html.append(f"""
                <h3 class="sub-title" style="margin-top: 24px;">Forward-Looking Financial Conditions &amp; Risk Premiums</h3>
                <div class="vix-grid" style="margin-bottom: 20px;">
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">HIGH-YIELD CREDIT OAS</span>
                            {_render_toggletip("tip-fc-hy", intel.credit.assessment, "High-Yield Credit OAS details")}
                        </div>
                        <span class="vix-val font-mono">{intel.credit.spread_bps:.0f} <small style="font-size: 13px;">bps</small></span>
                        <span class="vix-sub">Period &Delta;: {_fmt_delta(intel.credit.delta_bps)} bps ({intel.credit.spread_pct:.2f}%)</span>
                        <div style="margin-top: 6px;"><span class="badge {credit_badge}">{intel.credit.regime.upper()}</span></div>
                    </div>
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">REAL 10Y TIPS YIELD</span>
                            {_render_toggletip("tip-fc-tips", intel.rates.assessment, "Real 10Y TIPS Yield details")}
                        </div>
                        <span class="vix-val font-mono">{intel.rates.real_yield_10y:.2f}%</span>
                        <span class="vix-sub">Period &Delta;: {_fmt_delta(intel.rates.real_yield_delta_bps)} bps | 10Y BE: {intel.rates.breakeven_10y:.2f}%</span>
                        <div style="margin-top: 6px;"><span class="badge badge-primary">{intel.rates.driver_attribution.upper()}</span></div>
                    </div>
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">VIX TERM STRUCTURE (1M/3M)</span>
                            {_render_toggletip("tip-fc-vol-alt", intel.vol_term.assessment, "VIX term structure details")}
                        </div>
                        <span class="vix-val font-mono">{intel.vol_term.ratio:.3f}</span>
                        <span class="vix-sub">1M: {intel.vol_term.vix_1m:.2f} | 3M: {intel.vol_term.vix_3m:.2f} (&Delta; {intel.vol_term.ratio_delta:+.3f})</span>
                        <div style="margin-top: 6px;"><span class="badge {vol_badge}">{intel.vol_term.structure.upper()}</span></div>
                    </div>
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">SECTOR ROTATION SPREAD</span>
                            {_render_toggletip("tip-fc-rot-alt", f"<strong>Lagging Sectors:</strong> {', '.join(intel.rotation.top_outflows[-2:]) if intel.rotation.top_outflows else 'Growth'}. {intel.rotation.assessment}", "Sector rotation details")}
                        </div>
                        <span class="vix-val font-mono">{intel.rotation.spread:+.2f}%</span>
                        <span class="vix-sub">Leading: {", ".join(intel.rotation.top_inflows[:2]) if intel.rotation.top_inflows else "Defensives"}</span>
                        <div style="margin-top: 6px;"><span class="badge {rot_badge}">{intel.rotation.posture.upper()}</span></div>
                    </div>
                </div>

                <!-- Credit & Funding Conditions (Stage 4: 4.5) -->
                <h3 class="sub-title" style="margin-top: 24px;">Credit &amp; Funding Conditions (Systemic Liquidity)</h3>
                <div class="vix-grid" style="margin-bottom: 20px;">
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">INVESTMENT-GRADE CREDIT OAS</span>
                            {_render_toggletip("tip-fc-ig", "ICE BofA US Investment Grade Option-Adjusted Spread (FRED BAMLC0A0CM).", "Investment-Grade OAS details")}
                        </div>
                        <span class="vix-val font-mono">{f"{intel.credit.ig_spread_bps:.0f} bps" if intel.credit.ig_spread_bps is not None else "—"}</span>
                        <span class="vix-sub">{f"Period &Delta;: {_fmt_delta(intel.credit.ig_delta_bps)} bps" if intel.credit.ig_delta_bps is not None else "—"}</span>
                        <div style="margin-top: 6px;"><span class="badge badge-green">TIGHT / LOW SPREAD</span></div>
                    </div>
                    {bbb_card_html}
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">3M / 10Y YIELD CURVE</span>
                            {_render_toggletip("tip-fc-curve", "Benchmark short-funding vs. long-duration curve spread.", "3M/10Y Yield Curve details")}
                        </div>
                        <span class="vix-val font-mono">{f"{intel.credit.spread_3m10y_bps:+.0f} bps" if intel.credit.spread_3m10y_bps is not None else "—"}</span>
                        <span class="vix-sub">10Y minus 3M T-Bill (FRED T10Y3M)</span>
                        <div style="margin-top: 6px;"><span class="badge badge-bullish">{"NORMAL / UN-INVERTED" if (intel.credit.spread_3m10y_bps is not None and intel.credit.spread_3m10y_bps > 0) else "INVERTED"}</span></div>
                    </div>
                    <div class="vix-metric">
                        <div class="vix-header">
                            <span class="vix-label">FINANCIAL STRESS INDEX</span>
                            {_render_toggletip("tip-fc-fsi-alt", "St. Louis Fed Financial Stress Index (STLFSI4; 0 = normal average stress).", "Financial Stress Index details")}
                        </div>
                        <span class="vix-val font-mono">{f"{intel.credit.financial_stress_index:.2f}" if intel.credit.financial_stress_index is not None else "—"}</span>
                        <span class="vix-sub">{intel.credit.financial_stress_label if intel.credit.financial_stress_index is not None else "—"}</span>
                        <div style="margin-top: 6px;"><span class="badge {fsi_badge_cls}">{fsi_badge_txt}</span></div>
                    </div>
                    {liq_card_html}
                </div>
                """)

                body_html.append(f"""
                <div class="macro-regime-card" style="background: var(--bg-card-subtle); border: 1px solid var(--border); border-radius: 10px; padding: 18px 22px; margin-bottom: 24px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; border-bottom: 1px solid var(--border); padding-bottom: 10px;">
                        <div>
                            <span style="font-size: 11px; font-weight: 700; color: var(--text-muted); letter-spacing: 0.06em; text-transform: uppercase;">ACTIVE MACRO REGIME QUADRANT</span>
                            <div style="margin-top: 4px;"><span class="badge {quad_badge}" style="font-size: 13px; padding: 5px 12px;">{intel.summary_badge.upper()}</span></div>
                        </div>
                        <div style="display: flex; gap: 16px; font-size: 13px;">
                            <span><strong>Growth Signal:</strong> <span class="font-mono">{intel.growth_signal}</span></span>
                            <span><strong>Inflation Signal:</strong> <span class="font-mono">{intel.inflation_signal}</span></span>
                        </div>
                    </div>
                    <p style="font-size: 14px; line-height: 1.6; margin: 0; color: var(--text-main);">
                        <strong>Macro Positioning Synthesis:</strong> Economic activity reflects <strong>{intel.growth_signal}</strong> with inflation <strong>{intel.inflation_signal}</strong>, setting an active {intel.rotation.posture.lower()} tone.
                    </p>
                </div>
                """)

            # S&P 500 Valuation vs. Rates & ERP (Stage 3 Task 3.1 & Stage 4 Task 4.2)
            if m_data.valuation and m_data.valuation.forward_pe is not None:
                v = m_data.valuation
                pe_str = f"{v.forward_pe:.2f}x"
                ey_str = f"{v.earnings_yield_pct:.2f}%" if v.earnings_yield_pct is not None else '<span class="text-muted">—</span>'
                nom_hurdle = f"10Y Treasury ({v.nominal_10y_yield:.2f}%)" if v.nominal_10y_yield is not None else "10Y Treasury (DGS10)"
                nom_erp_str = f"{v.nominal_erp_bps:+.1f} bps" if v.nominal_erp_bps is not None else "—"
                nom_class = "text-green font-bold" if (v.nominal_erp_bps and v.nominal_erp_bps >= 100.0) else ("text-yellow font-bold" if v.nominal_erp_bps else "text-muted")
                real_hurdle = f"10Y TIPS ({v.real_10y_yield:.2f}%)" if v.real_10y_yield is not None else "10Y TIPS Real Yield (DFII10)"
                real_erp_str = f"{v.real_erp_bps:+.1f} bps" if v.real_erp_bps is not None else "—"
                real_class = "text-green font-bold" if (v.real_erp_bps and v.real_erp_bps >= 200.0) else ("text-yellow font-bold" if v.real_erp_bps else "text-muted")

                body_html.append(f"""
                <h3 class="sub-title" style="margin-top: 24px;">S&amp;P 500 Valuation vs. Rates &amp; Equity Risk Premium</h3>
                <div class="note-box" style="margin-bottom: 16px;">
                    <strong>Valuation &amp; ERP Context:</strong> {v.assessment}
                </div>
                <div class="table-container">
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th>Valuation Metric</th>
                                <th class="text-center">Current Level</th>
                                <th class="text-center">Benchmark / Hurdle</th>
                                <th class="text-center">Spread / Risk Premium</th>
                                <th>Sourcing &amp; Context</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Forward P/E</strong></td>
                                <td class="text-center font-mono font-bold">{pe_str}</td>
                                <td class="text-center">&mdash;</td>
                                <td class="text-center">&mdash;</td>
                                <td>Forward multiple on consensus 12M EPS ({v.source})</td>
                            </tr>
                            <tr>
                                <td><strong>Earnings Yield (1/PE)</strong></td>
                                <td class="text-center font-mono font-bold">{ey_str}</td>
                                <td class="text-center">&mdash;</td>
                                <td class="text-center">&mdash;</td>
                                <td>Equity cash generation yield (1 / Forward P/E)</td>
                            </tr>
                            <tr>
                                <td><strong>Nominal Equity Risk Premium (ERP)</strong></td>
                                <td class="text-center font-mono">{ey_str}</td>
                                <td class="text-center font-mono">{nom_hurdle}</td>
                                <td class="text-center font-mono {nom_class}">{nom_erp_str}</td>
                                <td>Earnings Yield minus 10Y Benchmark Yield</td>
                            </tr>
                            <tr>
                                <td><strong>Real Equity Risk Premium (Real ERP)</strong></td>
                                <td class="text-center font-mono">{ey_str}</td>
                                <td class="text-center font-mono">{real_hurdle}</td>
                                <td class="text-center font-mono {real_class}">{real_erp_str}</td>
                                <td>Earnings Yield minus 10Y TIPS Real Yield</td>
                            </tr>
                            <tr>
                                <td><strong>Forward EPS Revision Trend</strong></td>
                                <td class="text-center font-mono font-bold" colspan="3">{v.eps_revision_trend}</td>
                                <td>Revision momentum across S&amp;P 500 constituents</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """)

            # 3. Cross-Asset Snapshot (Stage 3: 3.6 De-duplicate; link to Section 10 & 12)
            body_html.append("""
            <div class="alert alert-info" style="margin: 16px 0;">
                📊 <strong>Cross-Asset Intelligence &amp; Catalyst Horizon:</strong> Detailed commodity price performance, oil, gold, dollar, and bond cross-market alignments are consolidated in <a href="#section-10" style="color: var(--primary); font-weight: 700; text-decoration: underline;">Section 10: Commodities, Currencies &amp; Cross-Asset Tracker</a>. Forward event releases and catalyst horizons are tracked in <a href="#section-12" style="color: var(--primary); font-weight: 700; text-decoration: underline;">Section 12: Forward Calendar &amp; Catalyst Horizon</a>.
            </div>
            """)

            # 5. Historical Trends & Trajectory with Charts (Monthly Report)
            if m_data.historical_overview and m_data.historical_overview.indicators:
                inds = m_data.historical_overview.indicators
                chart_infl = render_inflation_chart(inds)
                chart_labor = render_labor_chart(inds)
                chart_rates = render_rates_chart(inds)
                chart_act = render_activity_chart(inds)

                body_html.append(f"""
                <h3 class="sub-title" style="margin-top: 28px;">Macroeconomic Historical Trends &amp; Visualizations (Trailing 12 Months)</h3>
                <div class="macro-charts-stack">
                    <div class="macro-chart-box">{chart_infl}</div>
                    <div class="macro-chart-box">{chart_labor}</div>
                    <div class="macro-chart-box">{chart_rates}</div>
                    <div class="macro-chart-box">{chart_act}</div>
                </div>
                """)

                # Historical Matrix Table (excluding quarterly series like GDP from monthly columns)
                anchor = next((ind for ind in inds if len(ind.points) >= 6 and ind.series_id != "GDPC1"), None)
                matrix_dates = [p.label for p in anchor.points[-6:]] if anchor else []

                th_cols = "".join(f'<th class="text-right">{md}</th>' for md in matrix_dates)
                rows_html = []
                for ind in inds:
                    if ind.series_id == "GDPC1" or "Q" in getattr(ind.points[0], "label", ""):
                        continue
                    lp = ind.latest_point
                    pp = ind.prior_point
                    if not lp:
                        continue

                    # Formatting monthly and yearly values
                    if "k" in ind.monthly_unit:
                        m_str = f'<span class="font-mono font-bold {("text-green" if lp.monthly_val >= 0 else "text-red")}">{lp.monthly_val:+.0f}k</span>'
                        y_str = f'<span class="font-mono font-bold">{lp.yearly_val:+.0f}k</span>'
                    elif ind.category == "Rates & Policy" or ind.series_id == "UNRATE":
                        m_str = f'<span class="font-mono font-bold">{lp.monthly_val:.2f}%</span>'
                        y_str = f'<span class="font-mono">{lp.yearly_val:+.2f}%</span>'
                    elif "index" in ind.monthly_unit:
                        m_str = f'<span class="font-mono font-bold">{lp.monthly_val:.1f}</span>'
                        y_str = f'<span class="font-mono">{lp.yearly_val:+.1f} pts</span>'
                    elif "%" in ind.monthly_unit:
                        m_str = _fmt_pct(lp.monthly_val) if "% MoM" in ind.monthly_unit or "%" == ind.monthly_unit else f'<span class="font-mono">{lp.monthly_val:.2f}%</span>'
                        y_str = _fmt_pct(lp.yearly_val) if "% YoY" in ind.yearly_unit else f'<span class="font-mono">{lp.yearly_val:.2f}%</span>'
                    else:
                        m_str = f'<span class="font-mono font-bold">{lp.monthly_val:.1f}</span>'
                        y_str = f'<span class="font-mono">{lp.yearly_val:+.1f}</span>'

                    # Prior values
                    if pp:
                        if "k" in ind.monthly_unit:
                            pr_str = f'<span class="font-mono text-muted">{pp.monthly_val:+.0f}k</span>'
                        elif ind.category == "Rates & Policy" or ind.series_id == "UNRATE":
                            pr_str = f'<span class="font-mono text-muted">{pp.monthly_val:.2f}%</span>'
                        elif "index" in ind.monthly_unit:
                            pr_str = f'<span class="font-mono text-muted">{pp.monthly_val:.1f}</span>'
                        elif "%" in ind.monthly_unit:
                            pr_str = f'<span class="font-mono text-muted">{pp.monthly_val:+.2f}%</span>'
                        else:
                            pr_str = f'<span class="font-mono text-muted">{pp.monthly_val:.1f}</span>'
                    else:
                        pr_str = '<span class="text-muted">n/a</span>'

                    # Trailing months values
                    pt_map = {p.label: p for p in ind.points}
                    td_history = []
                    for md in matrix_dates:
                        if md in pt_map:
                            p = pt_map[md]
                            if "k" in ind.monthly_unit:
                                td_history.append(f'<td class="text-right font-mono {("text-green" if p.monthly_val >= 0 else "text-red")}">{p.monthly_val:+.0f}k</td>')
                            elif ind.category == "Rates & Policy" or ind.series_id == "UNRATE":
                                td_history.append(f'<td class="text-right font-mono">{p.monthly_val:.2f}%</td>')
                            elif "index" in ind.monthly_unit:
                                td_history.append(f'<td class="text-right font-mono">{p.monthly_val:.1f}</td>')
                            elif "%" in ind.monthly_unit:
                                td_history.append(f'<td class="text-right font-mono {("text-green" if p.monthly_val > 0 else ("text-red" if p.monthly_val < 0 else "text-muted"))}">{p.monthly_val:+.2f}%</td>')
                            else:
                                td_history.append(f'<td class="text-right font-mono">{p.monthly_val:.1f}</td>')
                        else:
                            td_history.append('<td class="text-right font-mono text-muted">—</td>')

                    # Trend badge with category-aware economic valence
                    if ind.trend_direction == "Rising":
                        if ind.category == "Inflation":
                            t_badge = '<span class="badge badge-red">▲ ACCELERATING</span>'
                        elif ind.series_id == "UNRATE":
                            t_badge = '<span class="badge badge-red">▲ RISING</span>'
                        elif ind.category == "Rates & Policy":
                            t_badge = '<span class="badge badge-neutral">▲ RISING</span>'
                        else:
                            t_badge = '<span class="badge badge-green">▲ EXPANDING</span>'
                    elif ind.trend_direction == "Falling":
                        if ind.category == "Inflation":
                            t_badge = '<span class="badge badge-green">▼ MODERATING</span>'
                        elif ind.series_id == "UNRATE":
                            t_badge = '<span class="badge badge-green">▼ FALLING</span>'
                        elif ind.category == "Rates & Policy":
                            t_badge = '<span class="badge badge-neutral">▼ EASING</span>'
                        else:
                            t_badge = '<span class="badge badge-red">▼ SLOWING</span>'
                    else:
                        t_badge = '<span class="badge badge-muted">▶ STEADY</span>'

                    rows_html.append(f"""
                    <tr>
                        <td><strong>{ind.name}</strong> <span class="ticker-sub font-mono">({ind.category})</span></td>
                        <td class="text-center font-mono">{lp.label}</td>
                        <td class="text-right">{m_str}</td>
                        <td class="text-right">{y_str}</td>
                        <td class="text-right">{pr_str}</td>
                        {''.join(td_history)}
                        <td class="text-center">{t_badge}</td>
                    </tr>
                    """)

                body_html.append(f"""
                <h3 class="sub-title">Key Macroeconomic Reports: Monthly &amp; Yearly Matrix</h3>
                <div class="table-container">
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th>Report Indicator</th>
                                <th class="text-center">Latest Period</th>
                                <th class="text-right">Monthly Value</th>
                                <th class="text-right">Yearly Value</th>
                                <th class="text-right">Prior Month</th>
                                {th_cols}
                                <th class="text-center">1-Yr Trend</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(rows_html)}</tbody>
                    </table>
                </div>
                """)
            if charts:
                body_html.append(_render_chart_html(charts.get("rates_yield_curve")))


        # Section: Sector Report
        elif isinstance(sec.data, SectorReportData):
            s_data: SectorReportData = sec.data
            s_rows = []
            for sr in s_data.rows:
                r_delta_class = "text-green" if sr.rank_change > 0 else ("text-red" if sr.rank_change < 0 else "text-muted")
                r_delta_str = f"+{sr.rank_change}" if sr.rank_change > 0 else str(sr.rank_change)
                s_rows.append(f"""
                <tr>
                    <td class="text-center"><strong>#{sr.rank}</strong></td>
                    <td><strong>{sr.name}</strong></td>
                    <td class="font-mono text-center">{sr.symbol}</td>
                    <td class="text-right">{_fmt_price(sr.close)}</td>
                    <td class="text-right">{_fmt_pct(sr.pct_change)}</td>
                    <td class="text-right">{_fmt_pct(sr.relative_strength_vs_spy)}</td>
                    <td class="text-center font-mono">#{sr.prev_rank}</td>
                    <td class="text-center font-mono {r_delta_class}"><strong>{r_delta_str}</strong></td>
                </tr>
                """)
            w_tag = "[1-Month Strategic View]" if period.report_type == "monthly" else ("[1-Week View]" if period.report_type == "weekly" else "[Daily Tactical View]")
            
            # Sector Breadth Table (Stage 4: 4.2)
            b_rows = []
            if getattr(s_data, "sector_breadth", None):
                for b_item in s_data.sector_breadth:
                    p50_str = f"{b_item.pct_above_50dma:.1f}%" if b_item.pct_above_50dma is not None else '<span class="text-muted">n/a</span>'
                    p20_str = f"{b_item.pct_above_20dma:.1f}%" if b_item.pct_above_20dma is not None else '<span class="text-muted">n/a</span>'
                    nh_str = f"{b_item.net_new_highs:+d}" if b_item.net_new_highs is not None else '<span class="text-muted">n/a</span>'
                    p50_class = "text-green font-bold" if (b_item.pct_above_50dma or 0) >= 60.0 else (("text-red font-bold") if (b_item.pct_above_50dma or 0) < 40.0 else "text-muted")
                    b_rows.append(f"""
                    <tr>
                        <td><strong>{b_item.sector}</strong></td>
                        <td class="font-mono text-center">{b_item.symbol}</td>
                        <td class="text-right font-mono {p50_class}">{p50_str}</td>
                        <td class="text-right font-mono">{p20_str}</td>
                        <td class="text-right font-mono font-bold">{nh_str}</td>
                    </tr>
                    """)

            breadth_table_html = ""
            if b_rows:
                breadth_table_html = f"""
                <h3 class="sub-title" style="margin-top: 24px;">Sector Breadth &amp; Moving Average Distribution</h3>
                <p class="table-caption">Constituent breadth participation (% above 50-DMA, % above 20-DMA, and 52-week net new highs) across each sector.</p>
                <div class="table-container">
                    <table class="report-table compact-table">
                        <thead>
                            <tr>
                                <th>Sector Name</th>
                                <th class="text-center">ETF Symbol</th>
                                <th class="text-right">% &gt; 50-DMA</th>
                                <th class="text-right">% &gt; 20-DMA</th>
                                <th class="text-right">Net New Highs (52W)</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(b_rows)}</tbody>
                    </table>
                </div>
                """

            # Sector Leaders Table (Stage 4: 4.2 & Stage 2: 2.2)
            leaders_html = ""
            if getattr(s_data, "sector_leaders", None):
                if period.report_type == "daily":
                    top_syms = []
                    for s_name, l_list in s_data.sector_leaders.items():
                        top_2 = [f"<strong>{s_name}:</strong> " + ", ".join([f'<span class="font-mono font-bold">{l.symbol}</span> ({l.name}, RS {_fmt_pct(l.composite_rs if l.composite_rs != 0.0 else l.rs_vs_spy_pct)})' for l in l_list[:2]])]
                        top_syms.extend(top_2)
                    summary_html = " &bull; ".join(top_syms) if top_syms else "None"
                    banner_html = ""
                    if getattr(s_data, "defensive_exception_banner", None):
                        banner_html = f"""
                        <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 4px; margin-top: 12px; margin-bottom: 16px; color: #b91c1c; font-weight: 600;">
                            {s_data.defensive_exception_banner}
                        </div>
                        """
                    leaders_html = f"""
                    <h3 class="sub-title" style="margin-top: 24px;">Top Relative Strength Leaders (Top 3 Leading Sectors)</h3>
                    {banner_html}
                    <div style="padding: 14px 18px; border-radius: 8px; background: var(--bg-card-subtle); border-left: 4px solid var(--primary); margin-top: 12px; margin-bottom: 16px;">
                        <p style="font-size: 13px; margin: 0; line-height: 1.5;">💡 <strong>Tactical RS Leaders Snapshot:</strong> {summary_html}. <span class="text-muted">The complete 21-stock screening matrix and multi-factor breakdown are published in the Weekly &amp; Monthly reports.</span></p>
                    </div>
                    """
                else:
                    leader_sections = []
                    leader_ret_hdr = "1M Return" if period.report_type == "monthly" else "1W Return"
                    for sec_name, l_list in s_data.sector_leaders.items():
                        l_rows = []
                        for rank_idx, leader in enumerate(l_list, 1):
                            if leader.avg_dollar_volume >= 1e9:
                                d_vol_str = f"${leader.avg_dollar_volume / 1e9:,.2f}B"
                            else:
                                d_vol_str = f"${leader.avg_dollar_volume / 1e6:,.1f}M"
                            crs_val = leader.composite_rs if leader.composite_rs != 0.0 else leader.rs_vs_spy_pct
                            leader_ret_val = leader.pct_change if period.report_type == "monthly" else leader.ret_1w
                            exc_badge = '<span class="badge badge-yellow" style="display: flex; width: fit-content; margin-top: 4px;">Outlier Leader (Defensive Regime)</span>' if getattr(leader, "is_exception_candidate", False) else ""
                            l_rows.append(f"""
                            <tr>
                                <td class="text-center font-bold">#{rank_idx}</td>
                                <td class="font-mono"><strong>{leader.symbol}</strong>{exc_badge}</td>
                                <td>{leader.name}</td>
                                <td class="text-right font-mono">${leader.close:,.2f}</td>
                                <td class="text-right font-bold font-mono">{_fmt_pct(crs_val)}</td>
                                <td class="text-right font-mono">{_fmt_pct(leader_ret_val)}</td>
                                <td class="text-right font-mono">{_fmt_pct(leader.dist_52w_high_pct)}</td>
                                <td class="text-right font-mono">{leader.volume_ratio:.2f}x</td>
                                <td class="text-right font-mono text-muted">{d_vol_str}</td>
                            </tr>
                            """)
                        leader_sections.append(f"""
                        <div style="margin-top: 16px;">
                            <h4 style="font-size: 14px; font-weight: 700; color: var(--primary); margin-bottom: 6px;">Top Relative Strength Stocks: {sec_name}</h4>
                            <div class="table-container">
                                <table class="report-table compact-table">
                                    <thead>
                                        <tr>
                                            <th class="text-center">Rank</th>
                                            <th>Ticker</th>
                                            <th>Company Name</th>
                                            <th class="text-right">Close</th>
                                            <th class="text-right">Composite RS</th>
                                            <th class="text-right">{leader_ret_hdr}</th>
                                            <th class="text-right">Dist 52W High</th>
                                            <th class="text-right">20D Vol Ratio</th>
                                            <th class="text-right">Avg $ Volume</th>
                                        </tr>
                                    </thead>
                                    <tbody>{''.join(l_rows)}</tbody>
                                </table>
                            </div>
                        </div>
                        """)
                    if leader_sections:
                        banner_html = ""
                        if getattr(s_data, "defensive_exception_banner", None):
                            banner_html = f"""
                            <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 4px; margin-top: 12px; margin-bottom: 16px; color: #b91c1c; font-weight: 600;">
                                {s_data.defensive_exception_banner}
                            </div>
                            """
                        leaders_html = f"""
                        <h3 class="sub-title" style="margin-top: 24px;">Top Relative Strength Leaders (Top 3 Leading Sectors)</h3>
                        {banner_html}
                        <p class="table-caption">Screened for stocks above 50-DMA, within 15% of 52-week highs, and minimum $20M daily dollar volume (ranked by Composite RS vs SPY).</p>
                        {''.join(leader_sections)}
                        """
                        if period.report_type in ("weekly", "monthly"):
                            leaders_html = f'<details style="margin-top: 24px;"><summary>Sector Leadership Constituents (Top 3 Leading Sectors)</summary>{leaders_html}</details>'

            rotation_quadrants_html = ""
            if getattr(s_data, "rotation_quadrants", None) and period.report_type == "weekly":
                rq_rows = []
                for rq in s_data.rotation_quadrants:
                    q_badge = "badge-green" if rq.quadrant == "Leading" else ("badge-yellow" if rq.quadrant in ("Weakening", "Improving") else "badge-red")
                    rs_c = "text-green font-bold" if rq.rs_10w_pct > 0 else ("text-red font-bold" if rq.rs_10w_pct < 0 else "text-muted")
                    mom_c = "text-green font-bold" if rq.rs_momentum_pct > 0 else ("text-red font-bold" if rq.rs_momentum_pct < 0 else "text-muted")
                    rq_rows.append(f"""
                    <tr>
                        <td><strong>{rq.name}</strong></td>
                        <td class="font-mono text-center">{rq.symbol}</td>
                        <td class="text-right font-mono {rs_c}">{_fmt_pct(rq.rs_10w_pct)}</td>
                        <td class="text-right font-mono {mom_c}">{_fmt_pct(rq.rs_momentum_pct)}</td>
                        <td class="text-center"><span class="badge {q_badge}">{rq.quadrant}</span></td>
                        <td>{rq.candle_structure}</td>
                    </tr>
                    """)
                rotation_quadrants_html = f"""
                <h3 class="sub-title" style="margin-top: 24px;">Sector Rotation Quadrants &amp; Weekly Candle Structure</h3>
                <p class="table-caption">10-Week Relative Strength vs SPY, RS Momentum, and Weekly Candlestick Price Action across all 11 SPDR Sectors.</p>
                <div class="table-container">
                    <table class="report-table compact-table">
                        <thead>
                            <tr>
                                <th>Sector Name</th>
                                <th class="text-center">Symbol</th>
                                <th class="text-right">10W RS vs SPY</th>
                                <th class="text-right">RS Momentum</th>
                                <th class="text-center">Quadrant</th>
                                <th>Weekly Candle Structure</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(rq_rows)}</tbody>
                    </table>
                </div>
                """

            style_html = ""
            style_attr = getattr(s_data, "style_attribution", None)
            if style_attr and getattr(style_attr, "items", None):
                sf_rows = []
                for sf in style_attr.items:
                    sp_cls = "text-green font-bold" if sf.period_spread_pct >= 0.25 else ("text-red font-bold" if sf.period_spread_pct <= -0.25 else "text-muted font-bold")
                    s1m_h = f"{sf.ret_1m_spread_pct:+.2f}%" if sf.ret_1m_spread_pct is not None else "—"
                    s3m_h = f"{sf.ret_3m_spread_pct:+.2f}%" if sf.ret_3m_spread_pct is not None else "—"
                    sf_badge = "badge-bullish" if sf.period_spread_pct >= 0.25 else ("badge-bearish" if sf.period_spread_pct <= -0.25 else "badge-neutral")
                    sf_rows.append(f"""
                    <tr>
                        <td><strong>{sf.factor_pair}</strong></td>
                        <td class="text-right font-mono {sp_cls}">{sf.period_spread_pct:+.2f}%</td>
                        <td class="text-right font-mono">{s1m_h}</td>
                        <td class="text-right font-mono">{s3m_h}</td>
                        <td class="text-center"><span class="badge {sf_badge}">{sf.regime_label}</span></td>
                    </tr>
                    """)
                synth_note = f'<p style="font-size: 13px; color: var(--text-muted); margin-top: 8px;">🧭 <strong>{style_attr.dominant_style}:</strong> {style_attr.synthesis}</p>' if style_attr.synthesis else ""
                style_html = f"""
                <h3 class="sub-title" style="margin-top: 24px;">Equity Style, Size &amp; Factor Attribution</h3>
                <p class="table-caption">Relative return spreads comparing Growth vs. Value (IWF/IWD), Small vs. Large (IWM/SPY), Equal vs. Cap-Weight (RSP/SPY), Momentum (MTUM/SPY), and Quality (QUAL/SPY).</p>
                <div class="table-container">
                    <table class="report-table compact-table">
                        <thead>
                            <tr>
                                <th>Factor Pair</th>
                                <th class="text-right">Period Spread</th>
                                <th class="text-right">1M Spread</th>
                                <th class="text-right">3M Spread</th>
                                <th class="text-center">Style Regime</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(sf_rows)}</tbody>
                    </table>
                </div>
                {synth_note}
                """

            body_html.append(f"""
            <h3 class="sub-title">11 SPDR Sector Relative Strength Matrix {w_tag}</h3>
            <p class="table-caption">Benchmark SPY Return: {_fmt_pct(s_data.spy_pct_change)} (Previous Period: {_fmt_pct(s_data.spy_prev_pct_change)})</p>
            <div class="table-container">
                <table class="report-table">
                    <thead>
                        <tr>
                            <th class="text-center">Rank</th>
                            <th>Sector</th>
                            <th class="text-center">Symbol</th>
                            <th class="text-right">Close</th>
                            <th class="text-right">Return</th>
                            <th class="text-right">RS vs SPY</th>
                            <th class="text-center">Prior Rank</th>
                            <th class="text-center">Rank &Delta;</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(s_rows)}
                    </tbody>
                </table>
            </div>
            {style_html}
            {rotation_quadrants_html}
            {breadth_table_html}
            {leaders_html}
            """)

            if charts:
                if charts.get("sector_timeframe_heatmap"):
                    body_html.append(_render_chart_html(charts.get("sector_timeframe_heatmap")))
                body_html.append(_render_chart_html(charts.get("sector_performance")))

        # Section: Theme & Industry Relative Strength
        elif isinstance(sec.data, ThemeRelativeStrengthData):
            tr: ThemeRelativeStrengthData = sec.data
            is_d = getattr(tr, "primary_window", "1W") == "1D"
            is_m = getattr(tr, "primary_window", "1W") == "1M"
            t_rows = []
            for item in tr.items:
                if not item.is_available:
                    continue
                r_delta_class = "text-green font-bold" if item.rank_change > 0 else ("text-red font-bold" if item.rank_change < 0 else "text-muted")
                r_delta_str = f"+{item.rank_change}" if item.rank_change > 0 else str(item.rank_change)
                d50_class = "badge-green" if item.above_50dma else "badge-red"
                d50_tag = "Above" if item.above_50dma else "Below"

                if is_d:
                    ret_primary = _fmt_pct(item.ret_1d)
                    ret_secondary = _fmt_pct(item.ret_1w)
                    ret_tertiary = _fmt_pct(item.ret_1m)
                elif is_m:
                    ret_primary = _fmt_pct(item.ret_1m)
                    ret_secondary = _fmt_pct(item.ret_1w)
                    ret_tertiary = _fmt_pct(item.ret_3m)
                else:
                    ret_primary = _fmt_pct(item.ret_1w)
                    ret_secondary = _fmt_pct(item.ret_1m)
                    ret_tertiary = _fmt_pct(item.ret_3m)

                t_rows.append(f"""
                <tr>
                    <td class="text-center"><strong>#{item.rank}</strong></td>
                    <td><strong>{item.name}</strong></td>
                    <td class="font-mono text-center">{item.symbol}</td>
                    <td>{item.category}</td>
                    <td class="text-right font-mono">${item.close:,.2f}</td>
                    <td class="text-right">{ret_primary}</td>
                    <td class="text-right">{ret_secondary}</td>
                    <td class="text-right">{ret_tertiary}</td>
                    <td class="text-right">{_fmt_pct(item.rs_vs_spy)}</td>
                    <td class="text-center font-mono {r_delta_class}">{r_delta_str}</td>
                    <td class="text-center"><span class="badge {d50_class}">{item.dist_50dma_pct:+.1f}% ({d50_tag})</span></td>
                </tr>
                """)

            summary_box = ""
            if tr.top_theme_summary:
                summary_box = f'<div class="alert alert-info" style="margin-bottom: 14px;">🏆 <strong>Theme Leadership:</strong> {tr.top_theme_summary}</div>'

            commentary_box = ""
            if tr.commentary:
                commentary_box = f'<p class="table-caption" style="margin-top: 10px;">💡 <em>{tr.commentary}</em></p>'

            bench_caption = getattr(tr, "primary_benchmark_label", f"Benchmark SPY 1-Week Return: {_fmt_pct(tr.spy_1w_return)}")
            bench_val = getattr(tr, "primary_benchmark_return", tr.spy_1w_return)
            
            if is_d:
                w_primary_hdr = "1D Return (Primary)"
                w_sec_hdr = "1W Return"
                w_tert_hdr = "1M Return"
                rs_hdr = "RS vs SPY (1D)"
                rank_d_hdr = "Rank &Delta; (vs 1W)"
            elif is_m:
                w_primary_hdr = "1M Return (Primary)"
                w_sec_hdr = "1W Return"
                w_tert_hdr = "3M Return"
                rs_hdr = "RS vs SPY (1M)"
                rank_d_hdr = "Rank &Delta; (vs 3M)"
            else:
                w_primary_hdr = "1W Return (Primary)"
                w_sec_hdr = "1M Return"
                w_tert_hdr = "3M Return"
                rs_hdr = "RS vs SPY (1W)"
                rank_d_hdr = "Rank &Delta; (vs 1M)"

            thematic_baskets_html = ""
            if period.report_type == "monthly" and getattr(tr, "thematic_baskets", None):
                b_rows = []
                for b in tr.thematic_baskets:
                    tickers_str = ", ".join(f'<span class="font-mono font-bold">{t}</span>' for t in b.tickers)
                    b_rows.append(f"""
                    <tr>
                        <td><strong>{b.name}</strong></td>
                        <td>{tickers_str}</td>
                        <td class="text-right font-mono font-bold">{_fmt_pct(b.ret_1m)}</td>
                        <td class="text-right font-mono">{_fmt_pct(b.ret_3m)}</td>
                        <td class="text-right font-mono">{_fmt_pct(b.rs_vs_spy_1m)}</td>
                        <td class="text-center font-mono">{b.pct_above_50dma:.0f}%</td>
                        <td style="font-size: 11px; color: var(--text-muted);">{b.constituents_summary}</td>
                    </tr>
                    """)
                thematic_baskets_html = f"""
                <div style="margin-top: 24px;">
                    <h4 style="font-size: 0.90rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;">
                        Thematic Equal-Weight Baskets Performance
                    </h4>
                    <div class="alert alert-info" style="margin-bottom: 12px;">
                        🧭 <strong>Macro Thematic Dispersion:</strong> Tracking equal-weight performance across key structural macro drivers.
                    </div>
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Thematic Basket</th>
                                    <th>Tracked ETFs</th>
                                    <th class="text-right">1M Return</th>
                                    <th class="text-right">3M Return</th>
                                    <th class="text-right">RS vs SPY (1M)</th>
                                    <th class="text-center">% &gt; 50-DMA</th>
                                    <th>Basket Constituents (1M Return)</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(b_rows)}</tbody>
                        </table>
                    </div>
                    {f'<div style="font-size: 0.90rem; color: var(--text-muted); margin-top: 8px;">ℹ️ <strong>Thematic Synthesis:</strong> {tr.thematic_synthesis}</div>' if getattr(tr, 'thematic_synthesis', '') else ''}
                </div>
                """

            body_html.append(f"""
            {summary_box}
            <p class="table-caption">Benchmark {bench_caption}: {_fmt_pct(bench_val)} &bull; Primary Window: <strong>{getattr(tr, 'primary_window', '1W')}</strong></p>
            <div class="table-container">
                <table class="report-table">
                    <thead>
                        <tr>
                            <th class="text-center">Rank</th>
                            <th>Theme / Industry ETF</th>
                            <th class="text-center">Symbol</th>
                            <th>Category</th>
                            <th class="text-right">Close</th>
                            <th class="text-right">{w_primary_hdr}</th>
                            <th class="text-right">{w_sec_hdr}</th>
                            <th class="text-right">{w_tert_hdr}</th>
                            <th class="text-right">{rs_hdr}</th>
                            <th class="text-center">{rank_d_hdr}</th>
                            <th class="text-center">50-DMA Dist</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(t_rows)}
                    </tbody>
                </table>
            </div>
            {thematic_baskets_html}
            {commentary_box}
            """)

        # Section: Commodities, Currencies & Cross-Asset
        elif isinstance(sec.data, CommoditiesCrossAssetData):
            ca_data: CommoditiesCrossAssetData = sec.data
            ca_rows = []
            for it in ca_data.items:
                if not it.is_available:
                    continue
                is_fx = "=X" in it.symbol or "/" in it.name or "DXY" in it.name or "DX-Y" in it.symbol
                if is_fx:
                    if "JPY" in it.symbol or "JPY" in it.name:
                        price_display = f"{it.current_price:,.2f}"
                    elif it.current_price < 10.0:
                        price_display = f"{it.current_price:.4f}"
                    else:
                        price_display = f"{it.current_price:,.2f}"
                else:
                    price_display = f"${it.current_price:,.2f}"
                ca_rows.append(f"""
                <tr>
                    <td><strong>{it.name}</strong></td>
                    <td class="font-mono text-center">{it.symbol}</td>
                    <td class="text-right font-mono">{price_display}</td>
                    <td class="text-right">{_fmt_pct(it.period_ret)}</td>
                    <td class="text-right">{_fmt_pct(it.prior_period_ret)}</td>
                    <td class="text-right">{_fmt_pct(it.ret_3m)}</td>
                </tr>
                """)

            shock_box = ""
            if ca_data.energy_shock:
                eshk = ca_data.energy_shock
                s_badge = '<span class="badge badge-red">⚠️ ACTIVE ENERGY SHOCK ALERT</span>' if eshk.is_shock_active else '<span class="badge badge-green">🟢 INACTIVE / NORMAL</span>'
                shock_box = f"""
                <div class="card" style="margin-top: 16px; background: var(--bg-card-subtle); border-left: 4px solid {('var(--red)' if eshk.is_shock_active else 'var(--green)')};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">Energy Shock &amp; Macro Inflation Pass-Through</span>
                        {s_badge}
                    </div>
                    <div style="display: flex; gap: 24px; font-size: 13px; margin-bottom: 8px;">
                        <div>Oil 3-Month Trailing Return: <strong>{_fmt_pct(eshk.oil_3m_change_pct)}</strong></div>
                        <div>Headline CPI YoY: <strong>{eshk.headline_cpi_yoy:.2f}%</strong></div>
                        <div>Core CPI YoY: <strong>{eshk.core_cpi_yoy:.2f}%</strong></div>
                        <div>Gap (Headline - Core): <strong>{eshk.headline_core_gap_pct:+.2f}%</strong></div>
                    </div>
                    <p style="font-size: 13px; color: var(--text-muted);">{eshk.assessment}</p>
                </div>
                """

            plumbing_box = ""
            pl = getattr(ca_data, "plumbing", None)
            if pl is not None:
                cg_val_html = f"{pl.copper_gold_ratio:.3f}" if pl.copper_gold_ratio is not None else "—"
                cg_sub_parts = []
                if pl.copper_gold_3m_chg_pct is not None:
                    cg_sub_parts.append(f"3M: {pl.copper_gold_3m_chg_pct:+.2f}%")
                if pl.nominal_10y_yield is not None:
                    y10_p = f"10Y: {pl.nominal_10y_yield:.2f}%"
                    if pl.yield_10y_13w_bps is not None:
                        y10_p += f" ({pl.yield_10y_13w_bps:+.0f} bps 13W)"
                    cg_sub_parts.append(y10_p)
                cg_sub_html = " | ".join(cg_sub_parts) if cg_sub_parts else "Copper/Gold (x1000) vs 10Y"

                sb_val_html = f"{pl.stock_bond_corr_60d:+.2f}" if pl.stock_bond_corr_60d is not None else "—"
                sb_sub_parts = []
                if pl.stock_bond_corr_prev_60d is not None:
                    sb_sub_parts.append(f"Prior 60D: {pl.stock_bond_corr_prev_60d:+.2f}")
                if pl.tlt_3m_ret is not None:
                    sb_sub_parts.append(f"TLT 3M: {pl.tlt_3m_ret:+.2f}%")
                sb_sub_html = " | ".join(sb_sub_parts) if sb_sub_parts else "SPY vs TLT Rolling 60D"
                sb_badge_cls = "badge-green" if (pl.stock_bond_corr_60d is not None and pl.stock_bond_corr_60d <= -0.25) else ("badge-red" if (pl.stock_bond_corr_60d is not None and pl.stock_bond_corr_60d >= 0.25) else "badge-neutral")

                plumbing_box = f"""
                <h3 class="sub-title" style="margin-top: 20px;">Cross-Asset Macro Plumbing (Growth Metals vs. Rates &amp; Stock-Bond Correlation)</h3>
                <div class="vix-grid" style="margin-bottom: 12px;">
                    <div class="vix-metric">
                        <span class="vix-label">COPPER / GOLD RATIO (x1000)</span>
                        <span class="vix-val font-mono">{cg_val_html}</span>
                        <span class="vix-sub">{cg_sub_html}</span>
                        <div style="margin-top: 6px;"><span class="badge badge-primary">{pl.copper_gold_regime.upper()}</span></div>
                    </div>
                    <div class="vix-metric">
                        <span class="vix-label">60D STOCK-BOND CORRELATION (SPY / TLT)</span>
                        <span class="vix-val font-mono">{sb_val_html}</span>
                        <span class="vix-sub">{sb_sub_html}</span>
                        <div style="margin-top: 6px;"><span class="badge {sb_badge_cls}">{pl.stock_bond_regime.upper()}</span></div>
                    </div>
                </div>
                <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 8px;">🧭 {pl.assessment}</p>
                """

            corr_matrix_html = ""
            if period.report_type == "monthly" and getattr(ca_data, "correlation_matrix", None):
                cm = ca_data.correlation_matrix
                header_th = "".join(f'<th class="text-center font-mono font-bold">{t}</th>' for t in cm.tickers)
                matrix_rows = []
                if isinstance(cm.matrix, dict):
                    for row_tick in cm.tickers:
                        row_dict = cm.matrix.get(row_tick, {})
                        tds = []
                        for c in cm.tickers:
                            v = row_dict.get(c, 0.0)
                            cls = "text-green font-bold" if (v >= 0.5 and row_tick != c) else ("text-red font-bold" if v <= -0.2 else ("text-muted font-mono" if row_tick == c else "font-mono"))
                            tds.append(f'<td class="text-center font-mono {cls}">{v:+.2f}</td>')
                        matrix_rows.append(f"""
                        <tr>
                            <td><strong>{row_tick}</strong></td>
                            {''.join(tds)}
                        </tr>
                        """)
                corr_matrix_html = f"""
                <div style="margin-top: 24px;">
                    <h4 style="font-size: 0.90rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;">
                        60-Day Cross-Asset Return Correlation Matrix
                    </h4>
                    <div class="alert alert-info" style="margin-bottom: 12px;">
                        🔗 <strong>Cross-Asset Diversification Surface ({cm.window_days} Trading Days):</strong> Pairwise Pearson correlations across core equity benchmarks, Treasuries, gold, crude oil, and the US dollar.
                    </div>
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Asset / Ticker</th>
                                    {header_th}
                                </tr>
                            </thead>
                            <tbody>{''.join(matrix_rows)}</tbody>
                        </table>
                    </div>
                    {f'<div style="font-size: 0.90rem; color: var(--text-muted); margin-top: 8px;">ℹ️ <strong>Correlation Assessment:</strong> {cm.summary_text}</div>' if cm.summary_text else ''}
                </div>
                """

            body_html.append(f"""
            <div class="table-container">
                <table class="report-table">
                    <thead>
                        <tr>
                            <th>Asset Name</th>
                            <th class="text-center">Symbol</th>
                            <th class="text-right">Current Price</th>
                            <th class="text-right">Period Return</th>
                            <th class="text-right">Prior Return</th>
                            <th class="text-right">3-Month Return</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(ca_rows)}
                    </tbody>
                </table>
            </div>
            {shock_box}
            {plumbing_box}
            {corr_matrix_html}
            """)

        # Section: Market Breadth (Stage 3)
        elif isinstance(sec.data, BreadthSectionData):
            b_data: BreadthSectionData = sec.data
            r = b_data.result

            cov_warn_html = f'<div class="alert alert-warning">⚠️ <strong>Coverage Warning:</strong> {r.coverage_warning}</div>' if r.coverage_warning else ""
            
            # MA Regimes
            r20_class = "badge-bullish" if r.pct_above_20_dma.current >= 60 else ("badge-bearish" if r.pct_above_20_dma.current <= 40 else "badge-neutral")
            r20_text = "Bullish (>60%)" if r.pct_above_20_dma.current >= 60 else ("Bearish (<40%)" if r.pct_above_20_dma.current <= 40 else "Neutral")
            
            r50_class = "badge-bullish" if r.pct_above_50_dma.current >= 60 else ("badge-bearish" if r.pct_above_50_dma.current <= 40 else "badge-neutral")
            r50_text = "Bullish (>60%)" if r.pct_above_50_dma.current >= 60 else ("Bearish (<40%)" if r.pct_above_50_dma.current <= 40 else "Neutral")

            r200_class = "badge-bullish" if r.pct_above_200_dma.current >= 60 else ("badge-bearish" if r.pct_above_200_dma.current <= 40 else "badge-neutral")
            r200_text = "Bullish (>60%)" if r.pct_above_200_dma.current >= 60 else ("Bearish (<40%)" if r.pct_above_200_dma.current <= 40 else "Neutral")

            lead_class = "badge-bearish" if r.narrow_leadership else ("badge-bullish" if "Broad" in r.leadership_status else "badge-neutral")
            lead_tag = "Narrow Leadership Warning" if r.narrow_leadership else ("Broad Leadership" if "Broad" in r.leadership_status else "Balanced")

            mcclellan_html = ""
            if r.mcclellan_oscillator is not None:
                m_class = "badge-bullish" if "Bullish" in (r.mcclellan_status or "") or "Overbought" in (r.mcclellan_status or "") else "badge-bearish"
                mc_pct_row = (
                    f'<div class="metric-row"><span class="text-muted">1Y Percentile Rank:</span> '
                    f'<span class="font-mono font-bold">{format_percentile_ordinal(getattr(r, "mcclellan_pctile_1y", None))}</span></div>'
                    if getattr(r, "mcclellan_pctile_1y", None) is not None else ""
                )
                mcclellan_html = f"""
                <div class="tech-card">
                    <div class="tech-header">
                        <span class="metric-title">McClellan Oscillator</span>
                        <span class="badge {m_class}">{r.mcclellan_status}</span>
                    </div>
                    <div style="font-size: 22px; font-weight: 800; font-family: monospace; margin-bottom: 8px;">
                        {r.mcclellan_oscillator:+.2f}
                    </div>
                    <div class="metric-row"><span class="text-muted">Previous Period:</span> <span class="font-mono">{r.prev_mcclellan_oscillator:+.2f}</span></div>
                    <div class="metric-row"><span class="text-muted">Period Change:</span> <span>{_fmt_delta(r.delta_mcclellan_oscillator or 0.0)}</span></div>
                    {mc_pct_row}
                    <div class="metric-row" style="margin-top: 6px;"><span class="text-muted" style="font-size: 11px;">19-EMA minus 39-EMA of daily net advances</span></div>
                </div>
                """

            p50_pctile_str = format_percentile_ordinal(getattr(r, "pct_above_50_dma_pctile_1y", None))
            p200_pctile_str = format_percentile_ordinal(getattr(r, "pct_above_200_dma_pctile_1y", None))
            nh_pctile_str = format_percentile_ordinal(getattr(r, "net_new_highs_pctile_1y", None))
            p50_pctile_badge = f' &bull; 1Y: <strong>{p50_pctile_str}</strong>' if getattr(r, "pct_above_50_dma_pctile_1y", None) is not None else ""
            p200_pctile_badge = f' &bull; 1Y: <strong>{p200_pctile_str}</strong>' if getattr(r, "pct_above_200_dma_pctile_1y", None) is not None else ""
            nh_pctile_badge = f' &bull; 1Y: <strong>{nh_pctile_str}</strong>' if getattr(r, "net_new_highs_pctile_1y", None) is not None else ""

            analog_html = ""
            analog = getattr(r, "analog_study", None)
            if analog is not None and getattr(analog, "horizons", None):
                analog_rows = []
                for hz in analog.horizons:
                    win_cls = "badge-bullish" if hz.win_rate_pct >= hz.baseline_win_rate_pct + 3.0 else ("badge-bearish" if hz.win_rate_pct <= hz.baseline_win_rate_pct - 3.0 else "badge-neutral")
                    if getattr(analog, "is_small_sample", False):
                        win_cell = f'<span class="badge {win_cls}" style="font-weight: normal;">{hz.win_rate_pct:.1f}%</span>'
                    else:
                        win_cell = f'<span class="badge {win_cls}"><strong>{hz.win_rate_pct:.1f}%</strong></span>'
                    ci_str = f"[{hz.win_rate_ci_low:.1f}%, {hz.win_rate_ci_high:.1f}%]" if getattr(hz, "win_rate_ci_high", 0.0) > 0 else "—"
                    iqr_cell = f'{hz.iqr_return:.2f}% <span class="text-muted">[{_fmt_pct(hz.p25_return)}, {_fmt_pct(hz.p75_return)}]</span>' if hasattr(hz, "iqr_return") else "—"
                    analog_rows.append(f"""
                        <tr>
                            <td><strong>+{hz.horizon_days} Trading Days ({analog.benchmark_symbol})</strong></td>
                            <td class="text-center">{win_cell}</td>
                            <td class="text-center font-mono text-muted">{ci_str}</td>
                            <td class="text-center font-mono text-muted">{hz.baseline_win_rate_pct:.1f}%</td>
                            <td class="text-right">{_fmt_pct(hz.median_return_pct)}</td>
                            <td class="text-right font-mono">{iqr_cell}</td>
                            <td class="text-right">{_fmt_pct(hz.baseline_median_pct)}</td>
                            <td class="text-right">{_fmt_pct(hz.mean_return_pct)}</td>
                            <td class="text-right">{_fmt_pct(hz.worst_return_pct)}</td>
                        </tr>
                    """)
                recent_dates = ", ".join(analog.analog_dates[-6:]) if analog.analog_dates else ""
                dates_caption = (
                    f'<p class="table-caption" style="margin-top: 8px;"><em>Recent non-overlapping analog dates (up to 6 most recent of {analog.sample_count}): {recent_dates}</em></p>'
                    if recent_dates else ""
                )
                if getattr(analog, "breadth_cohort_note", ""):
                    dates_caption += f'<p class="table-caption" style="margin-top: 6px;"><em>{analog.breadth_cohort_note}</em></p>'
                if getattr(analog, "history_years", 0) >= 25:
                    dates_caption += f'<p class="table-caption" style="margin-top: 4px;"><em>Historical lookback: {analog.history_years} years ({analog.history_start_year}–present), {analog.lookback_days} evaluated sessions across long-history index regime.</em></p>'

                badge_badge = f' <span class="badge badge-warning" style="margin-left: 8px;">{analog.small_sample_badge}</span>' if getattr(analog, "small_sample_badge", "") else ""
                hist_span_str = f" / {analog.history_years}Y History" if getattr(analog, "history_years", 0) > 0 else ""
                analog_html = f"""
                <h3 class="sub-title" style="margin-top: 22px;">Historical Analog Base-Rate Study</h3>
                <div class="alert alert-info" style="margin-bottom: 12px;">
                    <strong>Regime Match (N={analog.sample_count} Independent Setups{hist_span_str} / {analog.lookback_days} Sessions):</strong>{badge_badge}<br>
                    {analog.condition_summary}<br>
                    <span>{analog.summary_text}</span>
                </div>
                <div class="table-container">
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th>Forward Horizon</th>
                                <th class="text-center">Analog Win %</th>
                                <th class="text-center">95% Wilson CI</th>
                                <th class="text-center">Baseline Win %</th>
                                <th class="text-right">Analog Median</th>
                                <th class="text-right">IQR [P25, P75]</th>
                                <th class="text-right">Baseline Median</th>
                                <th class="text-right">Analog Mean</th>
                                <th class="text-right">Worst Case</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(analog_rows)}
                        </tbody>
                    </table>
                </div>
                {dates_caption}
                """

            body_html.append(f"""
            <div class="alert alert-info">
                <strong>Breadth Read:</strong> {r.breadth_read} &bull; <em>Coverage: {r.included_constituents} of {r.total_constituents} S&P 500 constituents ({r.coverage_pct:.1f}%)</em>
            </div>
            {cov_warn_html}

            <div class="vix-grid" style="margin: 16px 0;">
                <div class="vix-metric">
                    <span class="vix-label">STOCKS > 50 DMA</span>
                    <span class="vix-val">{r.pct_above_50_dma.current:.1f}%</span>
                    <span class="vix-sub">Period Δ: {_fmt_pct(r.pct_above_50_dma.change)}</span>
                    <span class="vix-desc">Prev Period: {r.pct_above_50_dma.previous:.1f}%{p50_pctile_badge}</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">STOCKS > 200 DMA</span>
                    <span class="vix-val">{r.pct_above_200_dma.current:.1f}%</span>
                    <span class="vix-sub">Period Δ: {_fmt_pct(r.pct_above_200_dma.change)}</span>
                    <span class="vix-desc">Prev Period: {r.pct_above_200_dma.previous:.1f}%{p200_pctile_badge}</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">A/D RATIO (ADV / DEC)</span>
                    <span class="vix-val">{r.ad_ratio:.2f}x</span>
                    <span class="vix-sub">🟢 {r.advancers} Up &bull; 🔴 {r.decliners} Down ({r.net_advancers:+d} Net)</span>
                    <span class="vix-desc">{r.ad_confirmation} &bull; {r.cum_ad_trend}</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">NET NEW 52-WEEK HIGHS</span>
                    <span class="vix-val">{r.net_new_highs:+d}</span>
                    <span class="vix-sub">Highs: {r.new_52w_highs} &bull; Lows: {r.new_52w_lows}</span>
                    <span class="vix-desc">Period Δ: {r.delta_net_new_highs:+d}{nh_pctile_badge}</span>
                </div>
            </div>

            <h3 class="sub-title">Moving Average Participation Matrix</h3>
            <div class="table-container">
                <table class="report-table">
                    <thead>
                        <tr>
                            <th>Breadth Metric</th>
                            <th class="text-right">Current %</th>
                            <th class="text-center">1Y %ile</th>
                            <th class="text-right">Prior Period %</th>
                            <th class="text-right">Period &Delta;</th>
                            <th class="text-center">Trend Regime</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>% of Stocks Above 20 DMA</strong></td>
                            <td class="text-right font-mono font-bold">{r.pct_above_20_dma.current:.1f}%</td>
                            <td class="text-center font-mono text-muted">&mdash;</td>
                            <td class="text-right font-mono">{r.pct_above_20_dma.previous:.1f}%</td>
                            <td class="text-right">{_fmt_pct(r.pct_above_20_dma.change)}</td>
                            <td class="text-center"><span class="badge {r20_class}">{r20_text}</span></td>
                        </tr>
                        <tr>
                            <td><strong>% of Stocks Above 50 DMA</strong></td>
                            <td class="text-right font-mono font-bold">{r.pct_above_50_dma.current:.1f}%</td>
                            <td class="text-center font-mono">{p50_pctile_str}</td>
                            <td class="text-right font-mono">{r.pct_above_50_dma.previous:.1f}%</td>
                            <td class="text-right">{_fmt_pct(r.pct_above_50_dma.change)}</td>
                            <td class="text-center"><span class="badge {r50_class}">{r50_text}</span></td>
                        </tr>
                        <tr>
                            <td><strong>% of Stocks Above 200 DMA</strong></td>
                            <td class="text-right font-mono font-bold">{r.pct_above_200_dma.current:.1f}%</td>
                            <td class="text-center font-mono">{p200_pctile_str}</td>
                            <td class="text-right font-mono">{r.pct_above_200_dma.previous:.1f}%</td>
                            <td class="text-right">{_fmt_pct(r.pct_above_200_dma.change)}</td>
                            <td class="text-center"><span class="badge {r200_class}">{r200_text}</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="grid-2" style="margin-top: 18px;">
                <div class="tech-card">
                    <div class="tech-header">
                        <span class="metric-title">Leadership: Equal-Weight (RSP) vs Cap-Weight (SPY)</span>
                        <span class="badge {lead_class}">{lead_tag}</span>
                    </div>
                    <div class="metric-group">
                        <div class="metric-row"><span class="text-muted">Cap-Weight SPY Return:</span> <span>{_fmt_pct(r.spy_return_pct)}</span></div>
                        <div class="metric-row"><span class="text-muted">Equal-Weight RSP Return:</span> <span>{_fmt_pct(r.rsp_return_pct)}</span></div>
                        <div class="metric-row"><span class="text-muted">Leadership Spread (SPY - RSP):</span> <span>{_fmt_pct(r.leadership_spread_pct)}</span></div>
                        <div class="metric-row"><span class="text-muted">Positive Return Participation:</span> <span class="font-mono">{r.pct_positive_return.current:.1f}% of universe</span></div>
                    </div>
                    <p style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">{r.leadership_status}</p>
                </div>
                {mcclellan_html}
            </div>

            {analog_html}

            <p class="table-caption" style="margin-top: 14px;"><em>*Note: Market breadth calculations use the current S&P 500 constituent roster (subject to survivorship bias for historical comparisons).</em></p>
            """)
            if charts:
                if charts.get("sector_breadth_heatmap"):
                    body_html.append(_render_chart_html(charts.get("sector_breadth_heatmap")))
                body_html.append(_render_chart_html(charts.get("breadth_history")))

            body_html.append("""
            <details class="methodology-note" style="margin-top: 16px;">
                <summary>ℹ️ Breadth Thresholds &amp; Linear Gradient Scoring</summary>
                <div class="methodology-content">
                    <p><strong>Participation Boundaries:</strong> Evaluates market participation across all 500 constituents of the S&amp;P 500:</p>
                    <ul style="padding-left: 18px; margin-top: 6px; display: flex; flex-direction: column; gap: 4px;">
                        <li><strong>&ge; 60% Bullish Expansion:</strong> Broad participation across cyclical, industrial, and technology sectors confirming rally sustainability.</li>
                        <li><strong>40% – 60% Neutral / Rotational:</strong> Mixed internal health where market performance relies on specific leadership sectors.</li>
                        <li><strong>&le; 40% Bearish Narrowing:</strong> Severe participation deficit. Historically signals broad distribution and elevated vulnerability.</li>
                    </ul>
                    <p style="margin-top: 6px;"><strong>Linear Gradient:</strong> Breadth scoring interpolates linearly between 40% (-1.0) and 60% (+1.0) without threshold cliffs.</p>
                </div>
            </details>
            """)

        # Section: Federal Reserve Policy Path
        elif isinstance(sec.data, FedPathData):
            fp: FedPathData = sec.data
            prob_src = getattr(fp, "prob_source", "CBOT 30-Day Fed Funds Futures")
            if fp.market_implied_prob_hold is not None:
                hold_p = f"{fp.market_implied_prob_hold * 100:.1f}%"
                hike_p = f"{fp.market_implied_prob_hike * 100:.1f}%" if getattr(fp, "market_implied_prob_hike", None) is not None else "0.0%"
                cut_p = f"{fp.market_implied_prob_cut * 100:.1f}%" if fp.market_implied_prob_cut is not None else "0.0%"
                prob_line = f'<strong>Market-Implied Probabilities:</strong> Hold <strong>{hold_p}</strong> &bull; +25 bps Hike <strong>{hike_p}</strong> &bull; -25 bps Cut <strong>{cut_p}</strong> <span style="color: var(--text-muted); font-size: 12px;">(Source: {prob_src})</span>'
            else:
                prob_line = '<strong>Market-Implied Probabilities:</strong> <span class="text-muted font-mono">n/a (source unavailable)</span>'

            fp_liq = getattr(fp, "liquidity", None)
            liq_strip = ""
            if fp_liq is not None:
                imp4_fph = f"{fp_liq.impulse_4w_b:+,.1f}B" if fp_liq.impulse_4w_b is not None else "n/a"
                imp13_fph = f"{fp_liq.impulse_13w_b:+,.1f}B" if fp_liq.impulse_13w_b is not None else "n/a"
                liq_fph_cls = "badge-green" if fp_liq.regime == "Expanding" else ("badge-red" if fp_liq.regime == "Draining" else "badge-primary")
                liq_strip = (
                    f'<div style="font-size: 13px; margin-bottom: 4px;">'
                    f'<strong>Fed Net Liquidity (WALCL - TGA - RRP):</strong> '
                    f'<span class="font-mono font-bold">${fp_liq.net_liquidity_t:.3f}T</span> '
                    f'(Assets ${fp_liq.walcl_t:.3f}T &bull; TGA ${fp_liq.tga_b:,.1f}B &bull; ON RRP ${fp_liq.rrp_b:,.1f}B &bull; '
                    f'4W: <span class="font-mono">{imp4_fph}</span> &bull; 13W: <span class="font-mono">{imp13_fph}</span>) '
                    f'<span class="badge {liq_fph_cls}" style="margin-left: 6px;">{fp_liq.regime.upper()}</span></div>'
                )

            fed_details_box = f"""
            <div style="margin-top: 14px; padding: 12px 16px; background: var(--bg-card-subtle); border-radius: 8px;">
                <div style="font-size: 13px; margin-bottom: 4px;">{prob_line}</div>
                {liq_strip}
                <div style="font-size: 13px; color: var(--text-muted); margin-top: 6px;">🧭 <em>{fp.assessment}</em></div>
            </div>
            """ if (period.report_type == "daily" and fp.days_to_next_meeting > 1) else f"""
            <div style="margin-top: 14px; padding: 12px 16px; background: var(--bg-card-subtle); border-radius: 8px;">
                <div style="font-size: 13px; margin-bottom: 4px;"><strong>Most Recent Decision:</strong> <span class="font-mono">{fp.last_action_date}</span> &mdash; {fp.last_action_description}</div>
                <div style="font-size: 13px; margin-bottom: 4px;">{prob_line}</div>
                {liq_strip}
                <div style="font-size: 13px; color: var(--text-muted); margin-top: 6px;">🧭 <em>{fp.assessment}</em></div>
            </div>
            """

            body_html.append(f"""
            <div class="vix-grid">
                <div class="vix-metric">
                    <span class="vix-label">FED FUNDS TARGET RANGE</span>
                    <span class="vix-val font-mono">{fp.current_target_range}</span>
                    <span class="vix-sub">Effective: {fp.effective_fed_funds:.2f}%</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">NEXT FOMC DECISION</span>
                    <span class="vix-val font-mono" style="font-size: 18px;">{fp.next_meeting_date}</span>
                    <span class="vix-sub">{fp.days_to_next_meeting} calendar days away</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">2Y YIELD VS. FED FUNDS</span>
                    <span class="vix-val font-mono">{fp.spread_2y_vs_fed_funds_bps:+.1f} bps</span>
                    <span class="vix-sub">2Y Yield: {fp.yield_2y:.2f}%</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">REAL POLICY RATE</span>
                    <span class="vix-val font-mono">{fp.real_policy_rate:+.2f}%</span>
                    <span class="vix-sub">{getattr(fp, 'real_rate_driver_measure', 'Core PCE')}: {fp.trailing_core_inflation:.2f}% YoY</span>
                </div>
            </div>

            {fed_details_box}
            """)

        # Section: Forward Catalyst Calendar
        elif isinstance(sec.data, ForwardCalendarData):
            fc: ForwardCalendarData = sec.data
            cal_rows = []
            for ev in fc.events:
                is_speech = any(w in ev.title.lower() for w in ("remarks", "speech", "keynote", "discussion"))
                is_chair_keynote = "chair keynote" in ev.title.lower()
                if is_chair_keynote:
                    imp_badge = '<span class="badge badge-red font-bold">HIGH (KEYNOTE)</span>'
                elif ev.importance == "high":
                    imp_badge = '<span class="badge badge-red">HIGH</span>'
                elif ev.importance == "medium":
                    imp_badge = '<span class="badge badge-neutral">MEDIUM</span>'
                else:
                    imp_badge = '<span class="badge badge-muted">LOW</span>'

                if is_speech:
                    pr_val = '<span class="text-muted font-mono">—</span>'
                    fc_val = '<span class="text-muted font-mono">— (speaker)</span>'
                    cons_src = '<span class="text-muted font-mono">—</span>'
                else:
                    def _fmt_h_val(val: Optional[float], unit: str) -> str:
                        if val is None:
                            return "n/a"
                        if not unit or unit == "index":
                            return f"{val:.2f}"
                        elif unit == "%":
                            return f"{val:.2f}%"
                        elif unit == "k":
                            return f"{val:,.0f}k"
                        else:
                            return f"{val:.2f} {unit}"

                    fc_val = f'<span class="font-mono font-bold">{_fmt_h_val(ev.forecast, ev.unit)}</span>' if ev.forecast is not None else '<span class="text-muted font-mono">—</span>'
                    pr_val = f'<span class="font-mono">{_fmt_h_val(ev.previous, ev.unit)}</span>' if ev.previous is not None else '<span class="text-muted font-mono">—</span>'
                    raw_cons_src = getattr(ev, "consensus_source", "") or ""
                    cons_src = (
                        raw_cons_src if (ev.forecast is not None and raw_cons_src and raw_cons_src != "—")
                        else ('Official Survey' if ev.forecast is not None else '<span class="text-muted">—</span>')
                    )
                sched_src = f'<a href="{ev.source_url}" target="_blank" style="color: var(--primary); text-decoration: none;">{ev.source_name or ev.source}</a>' if getattr(ev, "source_url", None) else (ev.source or "Official U.S. Schedule")

                cal_rows.append(f"""
                <tr>
                    <td class="text-center font-mono">{ev.date}</td>
                    <td class="text-center font-mono">{ev.time_et}</td>
                    <td><strong>{ev.title}</strong></td>
                    <td class="text-center">{imp_badge}</td>
                    <td class="text-right">{pr_val}</td>
                    <td class="text-right">{fc_val}</td>
                    <td style="font-size: 12px; color: var(--text-muted);">{sched_src}</td>
                    <td style="font-size: 12px; color: var(--text-muted);">{cons_src}</td>
                </tr>
                """)

            cal_src_lbl = getattr(fc, "data_source", "Official Federal Schedules")
            cons_hdr_lbl = "Finnhub Economic API" if "Finnhub" in cal_src_lbl else "Prior Only (Unpolled)"
            cal_caption = f"Tracking imminent catalysts through <strong>{fc.lookahead_date}</strong> &bull; Schedule Source: {cal_src_lbl} | Consensus: {cons_hdr_lbl}" if period.report_type == "daily" else f"Tracking catalyst releases through <strong>{fc.lookahead_date}</strong> &bull; Schedule Source: {cal_src_lbl} | Consensus: {cons_hdr_lbl}"

            week_ahead_html = ""
            if getattr(fc, "weekly_ahead_events", None):
                wa_rows = []
                for we in fc.weekly_ahead_events:
                    wa_rows.append(f"""
                    <tr>
                        <td class="font-mono text-center"><strong>{we.day_date}</strong></td>
                        <td class="font-mono text-center">{we.time_et}</td>
                        <td><strong>{we.release}</strong></td>
                        <td class="text-center font-bold">{we.impact}</td>
                        <td class="text-right font-mono">{we.prior}</td>
                        <td style="font-size: 13px;">{we.market_sensitive}</td>
                    </tr>
                    """)
                week_ahead_html = f"""
                <h3 class="sub-title">The Week Ahead (Monday&ndash;Friday Catalyst Grid)</h3>
                <p class="table-caption">Scheduled macroeconomic events with cross-asset market sensitivity mapping through {fc.lookahead_date}.</p>
                <div class="table-container">
                    <table class="report-table compact-table">
                        <thead>
                            <tr>
                                <th class="text-center">Day / Date</th>
                                <th class="text-center">Time (ET)</th>
                                <th>Release</th>
                                <th class="text-center">Impact</th>
                                <th class="text-right">Prior</th>
                                <th>Market Most Sensitive</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(wa_rows)}</tbody>
                    </table>
                </div>
                """

            earnings_deck_html = ""
            if getattr(fc, "weekly_ahead_earnings", None):
                ed_rows = []
                for ee in fc.weekly_ahead_earnings:
                    ed_rows.append(f"""
                    <tr>
                        <td class="font-mono text-center"><strong>{ee.day_date}</strong></td>
                        <td class="text-center"><span class="badge badge-neutral">{ee.timing}</span></td>
                        <td class="font-mono font-bold">{ee.symbol}</td>
                        <td>{ee.name}</td>
                        <td class="text-center font-mono">{ee.market_cap}</td>
                    </tr>
                    """)
                earnings_deck_html = f"""
                <h3 class="sub-title" style="margin-top: 24px;">Earnings on Deck (Notable Large-Cap Reports)</h3>
                <p class="table-caption">Notable scheduled corporate earnings releases for the coming trading week.</p>
                <div class="table-container">
                    <table class="report-table compact-table">
                        <thead>
                            <tr>
                                <th class="text-center">Day / Date</th>
                                <th class="text-center">Session Timing</th>
                                <th>Symbol</th>
                                <th>Company Name</th>
                                <th class="text-center">Market Cap</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(ed_rows)}</tbody>
                    </table>
                </div>
                """

            if period.report_type in ("weekly", "monthly"):
                mats_html = "".join(_render_scenario_matrix_html(m) for m in fc.scenario_matrices) if getattr(fc, "scenario_matrices", None) else ""
                scenario_block = f"""
                <h3 class="sub-title" style="margin-top: 22px;">Event Scenario Matrix (&ldquo;If-Then&rdquo; Reaction Playbook)</h3>
                {mats_html}
                """ if mats_html else ""
                body_html.append(f"""
                {week_ahead_html}
                {earnings_deck_html}
                <details style="margin-top: 20px;">
                    <summary>Comprehensive Calendar &amp; Scenario Playbook</summary>
                    <p class="table-caption" style="margin-top: 12px;">{cal_caption}</p>
                    <div class="table-container">
                        <table class="report-table">
                            <thead>
                                <tr>
                                    <th class="text-center">Date</th>
                                    <th class="text-center">Time (ET)</th>
                                    <th>Catalyst Event</th>
                                    <th class="text-center">Impact</th>
                                    <th class="text-right">Prior</th>
                                    <th class="text-right">Consensus</th>
                                    <th>Schedule Source</th>
                                    <th>Consensus Source</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(cal_rows)}
                            </tbody>
                        </table>
                    </div>
                    {scenario_block}
                </details>
                """)
            else:
                body_html.append(f"""
                <p class="table-caption">{cal_caption}</p>
                <div class="table-container">
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th class="text-center">Date</th>
                                <th class="text-center">Time (ET)</th>
                                <th>Catalyst Event</th>
                                <th class="text-center">Impact</th>
                                <th class="text-right">Prior</th>
                                <th class="text-right">Consensus</th>
                                <th>Schedule Source</th>
                                <th>Consensus Source</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(cal_rows)}
                        </tbody>
                    </table>
                </div>
                """)
                if getattr(fc, "scenario_matrices", None):
                    mats_html = "".join(_render_scenario_matrix_html(m) for m in fc.scenario_matrices)
                    body_html.append(f"""
                    <h3 class="sub-title" style="margin-top: 22px;">Event Scenario Matrix (&ldquo;If-Then&rdquo; Reaction Playbook)</h3>
                    {mats_html}
                    """)

        # Section: Corporate Earnings Layer
        elif isinstance(sec.data, EarningsLayerData):
            el: EarningsLayerData = sec.data
            pct_rep = f"{el.pct_reported:.1f}%" if el.pct_reported is not None else "n/a"
            eps_bt = f"{el.eps_beat_rate:.1f}%" if el.eps_beat_rate is not None else "n/a"
            rev_bt = f"{el.rev_beat_rate:.1f}%" if el.rev_beat_rate is not None else "n/a"
            med_sur = f"{el.median_surprise_pct:+.1f}%" if el.median_surprise_pct is not None else "n/a"

            # 1. Trailing Reported Large Caps
            rep_rows = []
            for r in el.reported_last_7_days:
                eps_s = f'<span class="font-mono font-bold {("text-green" if r.eps_surprise_pct > 0 else "text-red")}">{r.eps_surprise_pct:+.1f}%</span>' if r.eps_surprise_pct is not None else '<span class="text-muted">n/a</span>'
                rev_s = f'<span class="font-mono {("text-green" if r.rev_surprise_pct > 0 else "text-red")}">{r.rev_surprise_pct:+.1f}%</span>' if r.rev_surprise_pct is not None else '<span class="text-muted">n/a</span>'
                react_s = _fmt_pct(r.reaction_1d_pct)
                d20_s = _fmt_pct(r.dist_20dma_pct)
                d50_s = _fmt_pct(r.dist_50dma_pct)
                guid_badge = '<span class="badge badge-green">Raised</span>' if r.guidance == "Raised" else ('<span class="badge badge-red">Lowered</span>' if r.guidance == "Lowered" else f'<span class="badge badge-muted">{r.guidance}</span>')
                rep_rows.append(f"""
                <tr>
                    <td class="font-mono"><strong>{r.symbol}</strong></td>
                    <td>{r.name}</td>
                    <td>{r.sector}</td>
                    <td class="text-center font-mono">{r.date}</td>
                    <td class="text-center font-mono"><span class="badge badge-muted">{r.timing}</span></td>
                    <td class="text-right">{eps_s}</td>
                    <td class="text-right">{rev_s}</td>
                    <td class="text-center">{guid_badge}</td>
                    <td class="text-right">{react_s}</td>
                    <td class="text-right">{d20_s}</td>
                    <td class="text-right">{d50_s}</td>
                </tr>
                """)

            # 2. Upcoming Large Caps
            top_rows = []
            for item in el.next_week_large_caps:
                timing_badge = '<span class="badge badge-primary">BMO</span>' if 'Before' in item.timing else ('<span class="badge badge-neutral">AMC</span>' if 'After' in item.timing else '<span class="badge badge-muted">UNSPEC</span>')
                clean_mcap = item.market_cap_str.replace('\\$', '$')
                top_rows.append(f"""
                <tr>
                    <td class="font-mono"><strong>{item.symbol}</strong></td>
                    <td>{item.name}</td>
                    <td>{item.sector}</td>
                    <td class="text-right font-mono">{clean_mcap}</td>
                    <td class="text-center font-mono">{item.date}</td>
                    <td class="text-center">{timing_badge} <span style="font-size: 12px; color: var(--text-muted);">{item.timing}</span></td>
                </tr>
                """)

            recent_title = (
                "Recent Large-Cap Reports (Last 24h–48h)"
                if period.report_type == "daily"
                else ("Recent Large-Cap Earnings Results (Trailing 30 Days)" if period.report_type == "monthly" else "Recent Large-Cap Earnings Results (Trailing 7 Days)")
            )
            upcoming_title = (
                "Imminent Earnings: Tonight AMC &amp; Tomorrow BMO"
                if period.report_type == "daily"
                else ("Next 30 Days Earnings: Top S&amp;P 500 Large-Cap Catalysts" if period.report_type == "monthly" else "Next-Week Earnings: Top S&amp;P 500 Large-Cap Catalysts")
            )

            body_html.append(f"""
            <div class="vix-grid">
                <div class="vix-metric">
                    <span class="vix-label">SEASON PROGRESS</span>
                    <span class="vix-val font-mono">{pct_rep}</span>
                    <span class="vix-sub">S&amp;P 500 Reported</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">EPS BEAT RATE</span>
                    <span class="vix-val font-mono">{eps_bt}</span>
                    <span class="vix-sub">Surprise &gt; 0</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">REVENUE BEAT RATE</span>
                    <span class="vix-val font-mono">{rev_bt}</span>
                    <span class="vix-sub">Top-Line Expansion</span>
                </div>
                <div class="vix-metric">
                    <span class="vix-label">MEDIAN SURPRISE</span>
                    <span class="vix-val font-mono">{med_sur}</span>
                    <span class="vix-sub">EPS Outperformance</span>
                </div>
            </div>

            <h3 class="sub-title">{recent_title}</h3>
            <div class="table-container">
                <table class="report-table compact-table">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Company Name</th>
                            <th>Sector</th>
                            <th class="text-center">Report Date</th>
                            <th class="text-center">Timing</th>
                            <th class="text-right">EPS Surprise</th>
                            <th class="text-right">Rev Surprise</th>
                            <th class="text-center">Guidance</th>
                            <th class="text-right">1-Day Reaction</th>
                            <th class="text-right">Dist 20-DMA</th>
                            <th class="text-right">Dist 50-DMA</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rep_rows) if rep_rows else '<tr><td colspan="11" class="text-center text-muted">No large-cap constituents reported in trailing window.</td></tr>'}
                    </tbody>
                </table>
            </div>

            <h3 class="sub-title" style="margin-top: 24px;">{upcoming_title}</h3>
            <div class="table-container">
                <table class="report-table">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Company Name</th>
                            <th>Sector</th>
                            <th class="text-right">Market Cap</th>
                            <th class="text-center">Scheduled Date</th>
                            <th class="text-center">Timing</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(top_rows) if top_rows else '<tr><td colspan="6" class="text-center text-muted">No scheduled S&amp;P 500 large-cap earnings in horizon window.</td></tr>'}
                    </tbody>
                </table>
            </div>
            """)

        # Section: Top Movers
        elif isinstance(sec.data, TopMoversData):
            m_data: TopMoversData = sec.data
            # Display timeframe-appropriate mover sets to eliminate out-of-period clutter
            if period.report_type == "daily":
                display_periods = [(m_data.day, "Daily Universe Movers")]
            elif period.report_type == "weekly":
                display_periods = [(m_data.week, "Weekly Universe Movers")]
            else:  # monthly
                display_periods = [(m_data.month, "Monthly Top Movers (S&P 500 + Nasdaq-100)")]

            for pm, title_label in display_periods:
                hidden_compat = '<span style="display:none">Monthly Universe Movers</span>' if period.report_type == "monthly" else ''
                body_html.append(f'<h3 class="mover-period-title">{title_label}{hidden_compat}</h3>')
                if pm.theme_summary:
                    body_html.append(f'<div class="alert alert-info" style="margin-bottom: 12px;">🏷️ <strong>Movers by Theme:</strong> {pm.theme_summary}</div>')
                all_movers_pm = list(pm.gainers or []) + list(pm.losers or [])
                has_move_causes = any(
                    bool(getattr(m, "catalyst_driver", ""))
                    or getattr(m, "move_cause_tag", None) not in (None, "", "Unknown", "—")
                    for m in all_movers_pm
                )
                cause_th = "<th>Technical Context</th>" if has_move_causes else ""
                body_html.append('<div class="movers-split">')

                # Gainers
                g_rows = []
                for g in pm.gainers:
                    earn_pill = '<span class="badge badge-primary">YES</span>' if g.has_earnings_in_window else '<span class="badge badge-muted">NO</span>'
                    g_mcap = (g.market_cap_str or '—').replace('\\$', '$')
                    g_cause = getattr(g, "catalyst_driver", "") or (g.move_cause_tag if g.move_cause_tag not in (None, "", "Unknown") else "—")
                    g_cause_td = f'<td style="font-size: 11px; color: var(--text-muted); max-width: 170px; line-height: 1.25;">{g_cause}</td>' if has_move_causes else ""
                    g_rows.append(f"""
                    <tr>
                        <td><strong>{g.symbol}</strong><br><span class="ticker-sub">{g.name[:18]}</span></td>
                        <td>{g.sector}</td>
                        <td class="text-right font-mono">{g_mcap}</td>
                        <td class="text-right font-mono">${g.close:,.2f}</td>
                        <td class="text-right">{_fmt_pct(g.pct_change)}</td>
                        <td class="text-right font-mono">{g.volume_ratio:.2f}x</td>
                        <td class="text-center">{earn_pill}</td>
                        {g_cause_td}
                    </tr>
                    """)
                body_html.append(f"""
                <div class="mover-table-box">
                    <h4 class="mover-subheading text-green">Top 5 Gainers</h4>
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Ticker</th>
                                    <th>Sector</th>
                                    <th class="text-right">Mkt Cap</th>
                                    <th class="text-right">Close</th>
                                    <th class="text-right">Change</th>
                                    <th class="text-right">20d Vol</th>
                                    <th class="text-center">Earnings?</th>
                                    {cause_th}
                                </tr>
                            </thead>
                            <tbody>{''.join(g_rows)}</tbody>
                        </table>
                    </div>
                </div>
                """)

                # Losers
                l_rows = []
                for l in pm.losers:
                    earn_pill = '<span class="badge badge-primary">YES</span>' if l.has_earnings_in_window else '<span class="badge badge-muted">NO</span>'
                    l_mcap = (l.market_cap_str or '—').replace('\\$', '$')
                    l_cause = getattr(l, "catalyst_driver", "") or (l.move_cause_tag if l.move_cause_tag not in (None, "", "Unknown") else "—")
                    l_cause_td = f'<td style="font-size: 11px; color: var(--text-muted); max-width: 170px; line-height: 1.25;">{l_cause}</td>' if has_move_causes else ""
                    l_rows.append(f"""
                    <tr>
                        <td><strong>{l.symbol}</strong><br><span class="ticker-sub">{l.name[:18]}</span></td>
                        <td>{l.sector}</td>
                        <td class="text-right font-mono">{l_mcap}</td>
                        <td class="text-right font-mono">${l.close:,.2f}</td>
                        <td class="text-right">{_fmt_pct(l.pct_change)}</td>
                        <td class="text-right font-mono">{l.volume_ratio:.2f}x</td>
                        <td class="text-center">{earn_pill}</td>
                        {l_cause_td}
                    </tr>
                    """)
                body_html.append(f"""
                <div class="mover-table-box">
                    <h4 class="mover-subheading text-red">Top 5 Losers</h4>
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Ticker</th>
                                    <th>Sector</th>
                                    <th class="text-right">Mkt Cap</th>
                                    <th class="text-right">Close</th>
                                    <th class="text-right">Change</th>
                                    <th class="text-right">20d Vol</th>
                                    <th class="text-center">Earnings?</th>
                                    {cause_th}
                                </tr>
                            </thead>
                            <tbody>{''.join(l_rows)}</tbody>
                        </table>
                    </div>
                </div>
                """)

                body_html.append('</div>')
            if charts:
                body_html.append(_render_chart_html(charts.get("top_movers")))

        # Section: Sentiment, Positioning & Volatility Risk Premium
        elif isinstance(sec.data, SentimentPositioningData):
            sp: SentimentPositioningData = sec.data
            vrp_class = "text-green font-bold" if sp.vol_risk_premium > 2.0 else ("text-red font-bold" if sp.vol_risk_premium < -1.0 else "text-muted font-bold")

            sentiment_rows = []
            if getattr(sp, "items", None):
                for s_item in sp.items:
                    sentiment_rows.append(f"""
                    <tr>
                        <td><strong>{s_item.name}</strong></td>
                        <td class="text-center font-mono font-bold">{s_item.reading}</td>
                        <td style="font-size: 13px; color: var(--text-muted);">{s_item.diagnostic}</td>
                    </tr>
                    """)

            sentiment_table_html = ""
            if sentiment_rows:
                sentiment_table_html = f"""
                <div class="table-container" style="margin-top: 16px;">
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th>Positioning &amp; Sentiment Metric</th>
                                <th class="text-center">Current Reading</th>
                                <th>Data Status &amp; Methodology Diagnostic</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(sentiment_rows)}</tbody>
                    </table>
                </div>
                """

            cot_html = ""
            if getattr(sp, "cot_positions", None):
                cot_rows = []
                for cp in sp.cot_positions:
                    net_str = f"{cp.net_position:+,.0f}" if isinstance(cp.net_position, (int, float)) else str(cp.net_position)
                    chg_c = "text-green" if cp.net_change_1w > 0 else ("text-red" if cp.net_change_1w < 0 else "text-muted")
                    chg_str = f"{cp.net_change_1w:+,.0f}" if isinstance(cp.net_change_1w, (int, float)) else str(cp.net_change_1w)
                    r_cls = "badge-green" if "Long" in cp.regime else ("badge-red" if "Short" in cp.regime else "badge-neutral")
                    cot_rows.append(f"""
                    <tr>
                        <td><strong>{cp.contract_name}</strong> <span class="ticker-sub font-mono">{cp.contract_symbol}</span></td>
                        <td class="text-right font-mono font-bold">{net_str}</td>
                        <td class="text-right font-mono {chg_c}">{chg_str}</td>
                        <td class="text-center font-mono font-bold">{cp.percentile_label}</td>
                        <td class="text-center"><span class="badge {r_cls}">{cp.regime}</span></td>
                    </tr>
                    """)
                src_cot = getattr(sp, "cot_source", "CFTC Commitments of Traders (Legacy/TFF Public Domain; positions as of prior Tuesday cutoff)")
                cot_html = f"""
                <h3 class="sub-title">CFTC COT Institutional Futures Positioning</h3>
                <p class="table-caption">Disaggregated / Legacy Non-Commercial Institutional Positioning across Core Benchmarks &bull; Source: {src_cot}</p>
                <div class="table-container">
                    <table class="report-table compact-table">
                        <thead>
                            <tr>
                                <th>Benchmark Futures Contract</th>
                                <th class="text-right">Non-Commercial Net (Contracts)</th>
                                <th class="text-right">1W Net Change</th>
                                <th class="text-center">3-Year %ile</th>
                                <th class="text-center">Positioning Regime</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(cot_rows)}</tbody>
                    </table>
                </div>
                """

            if period.report_type in ("weekly", "monthly"):
                body_html.append(f"""
                {cot_html}
                {sentiment_table_html}

                <details style="margin-top: 16px;">
                    <summary>Volatility Risk Premium (VRP) &amp; Options Pricing Reference</summary>
                    <div class="vix-grid" style="margin-top: 12px;">
                        <div class="vix-metric">
                            <span class="vix-label">CBOE VIX INDEX</span>
                            <span class="vix-val font-mono">{sp.vix:.2f}</span>
                            <span class="vix-sub">Implied Volatility</span>
                        </div>
                        <div class="vix-metric">
                            <span class="vix-label">20-DAY REALIZED VOL</span>
                            <span class="vix-val font-mono">{sp.realized_vol_20d:.2f}%</span>
                            <span class="vix-sub">Historical Price Variance</span>
                        </div>
                        <div class="vix-metric">
                            <span class="vix-label">VOLATILITY RISK PREMIUM</span>
                            <span class="vix-val font-mono {vrp_class}">{sp.vol_risk_premium:+.2f} pts</span>
                            <span class="vix-sub">VIX &minus; Realized Vol</span>
                        </div>
                        <div class="vix-metric">
                            <span class="vix-label">VRP REGIME</span>
                            <span class="vix-val" style="font-size: 15px; margin: 4px 0;">{sp.vol_premium_regime.split('(')[0].strip()}</span>
                            <span class="vix-sub">Options Pricing Premium</span>
                        </div>
                    </div>
                    <div class="regime-callout-card" style="margin-top: 14px; padding: 12px 16px; border-radius: 8px; background: rgba(59, 130, 246, 0.08); border-left: 4px solid var(--primary);">
                        <div style="font-size: 13px; line-height: 1.5; color: var(--text-main);">🧭 <strong>Options Pricing Context:</strong> {sp.assessment}</div>
                    </div>
                </details>

                <details class="methodology-note" style="margin-top: 16px;">
                    <summary>ℹ️ Volatility Risk Premium (VRP) &amp; Positioning Methodology</summary>
                    <div class="methodology-content">
                        <p><strong>Volatility Risk Premium (VRP):</strong> Calculated as CBOE VIX (30-day implied volatility) minus trailing 20-day annualized realized volatility of the S&amp;P 500.</p>
                        <ul style="padding-left: 18px; margin-top: 6px; display: flex; flex-direction: column; gap: 4px;">
                            <li><strong>Positive VRP (&gt; +2.0 pts):</strong> Volatility contango. Market participants are paying normal insurance premiums for index protection, signaling orderly hedging behavior.</li>
                            <li><strong>Negative VRP (&lt; 0 pts):</strong> Volatility backwardation. Historical realized moves exceed forward implied pricing, reflecting acute liquidation, panic selling, or sudden dislocation.</li>
                        </ul>
                        <p style="margin-top: 6px;"><strong>Data Integrity:</strong> In accordance with Ground Rule 2, unverified sentiment feeds render "n/a (source unavailable)" and are excluded from scorecard denominators.</p>
                    </div>
                </details>
                """)
            else:
                body_html.append(f"""
                <div class="vix-grid">
                    <div class="vix-metric">
                        <span class="vix-label">CBOE VIX INDEX</span>
                        <span class="vix-val font-mono">{sp.vix:.2f}</span>
                        <span class="vix-sub">Implied Volatility</span>
                    </div>
                    <div class="vix-metric">
                        <span class="vix-label">20-DAY REALIZED VOL</span>
                        <span class="vix-val font-mono">{sp.realized_vol_20d:.2f}%</span>
                        <span class="vix-sub">Historical Price Variance</span>
                    </div>
                    <div class="vix-metric">
                        <span class="vix-label">VOLATILITY RISK PREMIUM</span>
                        <span class="vix-val font-mono {vrp_class}">{sp.vol_risk_premium:+.2f} pts</span>
                        <span class="vix-sub">VIX &minus; Realized Vol</span>
                    </div>
                    <div class="vix-metric">
                        <span class="vix-label">VRP REGIME</span>
                        <span class="vix-val" style="font-size: 15px; margin: 4px 0;">{sp.vol_premium_regime.split('(')[0].strip()}</span>
                        <span class="vix-sub">Options Pricing Premium</span>
                    </div>
                </div>

                {cot_html}
                {sentiment_table_html}

                <div class="regime-callout-card" style="margin-top: 14px; padding: 12px 16px; border-radius: 8px; background: rgba(59, 130, 246, 0.08); border-left: 4px solid var(--primary);">
                    <div style="font-size: 13px; line-height: 1.5; color: var(--text-main);">🧭 <strong>Options Pricing Context:</strong> {sp.assessment}</div>
                </div>

                <details class="methodology-note" style="margin-top: 16px;">
                    <summary>ℹ️ Volatility Risk Premium (VRP) &amp; Positioning Methodology</summary>
                    <div class="methodology-content">
                        <p><strong>Volatility Risk Premium (VRP):</strong> Calculated as CBOE VIX (30-day implied volatility) minus trailing 20-day annualized realized volatility of the S&amp;P 500.</p>
                        <ul style="padding-left: 18px; margin-top: 6px; display: flex; flex-direction: column; gap: 4px;">
                            <li><strong>Positive VRP (&gt; +2.0 pts):</strong> Volatility contango. Market participants are paying normal insurance premiums for index protection, signaling orderly hedging behavior.</li>
                            <li><strong>Negative VRP (&lt; 0 pts):</strong> Volatility backwardation. Historical realized moves exceed forward implied pricing, reflecting acute liquidation, panic selling, or sudden dislocation.</li>
                        </ul>
                        <p style="margin-top: 6px;"><strong>Data Integrity:</strong> In accordance with Ground Rule 2, unverified sentiment feeds render "n/a (source unavailable)" and are excluded from scorecard denominators.</p>
                    </div>
                </details>
                """)

        # Section: Tactical Watchlist Panel (Stage 3: Task 3.2)
        elif isinstance(sec.data, WatchlistData):
            wl: WatchlistData = sec.data
            wl_rows = []
            for it in wl.items:
                c_str = f"${it.close:,.2f}" if it.close > 0 else "&mdash;"
                d20_str = _fmt_pct(it.dist_20dma_pct)
                d50_str = _fmt_pct(it.dist_50dma_pct)
                atr_str = f"{it.atr_pct:.2f}%" if it.atr_pct is not None else "&mdash;"

                if it.rs_vs_spy is not None:
                    rs_c = "text-green font-bold" if it.rs_vs_spy > 0 else ("text-red font-bold" if it.rs_vs_spy < 0 else "text-muted")
                    rs_str = f'<span class="{rs_c} font-mono">{it.rs_vs_spy:+.2f}%</span>'
                else:
                    rs_str = "&mdash;"

                # Readiness badge
                if it.readiness == "Ready":
                    readiness_badge = '<span class="badge badge-green font-bold">Ready</span>'
                elif it.readiness == "Watch":
                    readiness_badge = '<span class="badge badge-yellow font-bold">Watch</span>'
                else:
                    readiness_badge = '<span class="badge badge-neutral" style="color: var(--text-muted);">Not ready</span>'

                # Regime context badge
                if "Outlier Leader" in it.long_permitted or "Exception candidate" in it.long_permitted:
                    dir_badge = '<span class="badge badge-yellow font-bold" title="Outlier Leader (Defensive Regime)">Outlier Leader</span>'
                elif any(k in it.long_permitted for k in ("Defensive Regime", "BLOCKED", "INACTIVE")):
                    dir_badge = '<span class="badge badge-red font-bold">DEFENSIVE</span>'
                elif any(k in it.long_permitted for k in ("Mixed Regime", "SELECTIVE")):
                    dir_badge = '<span class="badge badge-yellow font-bold">SELECTIVE</span>'
                else:
                    dir_badge = '<span class="badge badge-green font-bold">SUPPORTIVE</span>'

                if it.next_earnings_date in ("none within 30d", "n/a (source unavailable)", "n/a"):
                    earn_text = "None (30d)" if it.next_earnings_date == "none within 30d" else it.next_earnings_date
                    earn_badge = f'<span class="text-muted font-mono" style="font-size: 12px;">{earn_text}</span>'
                else:
                    earn_badge = f'<span class="font-mono font-bold" style="font-size: 12px;">{it.next_earnings_date}</span>'

                wl_rows.append(f"""
                <tr>
                    <td><strong class="font-mono" style="color: var(--color-primary);">{it.ticker}</strong><br><span class="ticker-sub">{it.name}</span></td>
                    <td><span class="text-muted" style="font-size: 12px;">{it.sector}</span></td>
                    <td class="text-right font-mono">{c_str}</td>
                    <td class="text-right">{d20_str}</td>
                    <td class="text-right">{d50_str}</td>
                    <td class="text-right font-mono">{atr_str}</td>
                    <td class="text-right">{rs_str}</td>
                    <td class="text-center font-mono" style="font-size: 12px;">{it.sector_breadth_status}</td>
                    <td class="text-center">{readiness_badge}</td>
                    <td class="text-center">{earn_badge}</td>
                    <td class="text-center">{dir_badge}</td>
                    <td style="font-size: 12px; color: var(--text-muted); line-height: 1.35;">{getattr(it, 'why_flagged', '') or '&mdash;'}</td>
                </tr>
                """)

            exc_note = f", {wl.exception_count} Outlier Leaders" if getattr(wl, "exception_count", 0) > 0 else ""
            src_note = f" &bull; {wl.source_label}" if getattr(wl, "source_label", "") else ""
            monthly_wl_note = '<div style="font-size: 13px; color: var(--text-muted); margin-top: 6px;">💡 <em>Multi-Horizon Universe Context: Highlights equities exhibiting sustained 3-month relative-strength outperformance while holding above rising 50-day moving averages.</em></div>' if period.report_type == "monthly" else ""
            
            table_markup = f"""
            <div class="table-container">
                <table class="report-table compact-table watchlist-table">
                    <thead>
                        <tr>
                            <th>Asset</th>
                            <th>Sector</th>
                            <th class="text-right">Close</th>
                            <th class="text-right">&Delta; 20-DMA</th>
                            <th class="text-right">&Delta; 50-DMA</th>
                            <th class="text-right">14D ATR%</th>
                            <th class="text-right">RS vs SPY (3M)</th>
                            <th class="text-center">Sector Breadth (&gt;50-DMA)</th>
                            <th class="text-center">Technical Setup</th>
                            <th class="text-center">Next Earnings</th>
                            <th class="text-center">Regime Context</th>
                            <th>Why flagged</th>
                        </tr>
                    </thead>
                    <tbody>{''.join(wl_rows) if wl_rows else '<tr><td colspan="12" class="text-center text-muted">No watchlist focus items available.</td></tr>'}</tbody>
                </table>
            </div>
            """
            if period.report_type in ("weekly", "monthly"):
                table_markup = f"""
                <details style="margin-top: 16px;">
                    <summary>Focus Equity Screening Matrix &amp; Setups</summary>
                    {table_markup}
                </details>
                """

            body_html.append(f"""
            <div class="note-box" style="margin-bottom: 16px;">
                <strong>Focus Equity Universe:</strong> Tracking {wl.total_count} focus large-cap tickers{src_note}.<br>
                Regime Posture: <strong>{wl.gate_posture}</strong> &bull; Screen Backdrop: <strong>{wl.long_permitted_default}</strong>
                ({wl.blocked_count} Defensive, {wl.selective_count} Selective, {wl.permitted_count} Supportive{exc_note}).
                {monthly_wl_note}
            </div>
            {table_markup}
            """)

        # Section: S&P 500 Valuation vs Rates (CR v2.2 Stage 3 Task 3.1 & Stage 4 Task 4.2)
        elif isinstance(sec.data, ValuationData):
            v: ValuationData = sec.data
            if v.forward_pe is None:
                body_html.append("""
                <div style="padding: 12px 16px; background: var(--bg-card-subtle); border-radius: 8px; font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">
                    <span class="font-mono font-bold">n/a (source unavailable)</span> &mdash; Forward valuation multiples and consensus forward EPS estimates are currently unavailable from data provider. Equity Risk Premium metrics are omitted until verified forward earnings estimates are available.
                </div>
                """)
            else:
                pe_str = f"{v.forward_pe:.2f}x"
                ey_str = f"{v.earnings_yield_pct:.2f}%" if v.earnings_yield_pct is not None else '<span class="text-muted">n/a (source unavailable)</span>'
                nom_hurdle = f"10Y Treasury ({v.nominal_10y_yield:.2f}%)" if v.nominal_10y_yield is not None else "10Y Treasury (DGS10)"
                nom_erp_str = f"{v.nominal_erp_bps:+.1f} bps" if v.nominal_erp_bps is not None else "n/a (source unavailable)"
                nom_class = "text-green font-bold" if (v.nominal_erp_bps and v.nominal_erp_bps >= 100.0) else ("text-yellow font-bold" if v.nominal_erp_bps else "text-muted")
                real_hurdle = f"10Y TIPS ({v.real_10y_yield:.2f}%)" if v.real_10y_yield is not None else "10Y TIPS Real Yield (DFII10)"
                real_erp_str = f"{v.real_erp_bps:+.1f} bps" if v.real_erp_bps is not None else "n/a (source unavailable)"
                real_class = "text-green font-bold" if (v.real_erp_bps and v.real_erp_bps >= 200.0) else ("text-yellow font-bold" if v.real_erp_bps else "text-muted")

                body_html.append(f"""
                <div class="note-box" style="margin-bottom: 16px;">
                    <strong>Valuation &amp; ERP Context:</strong> {v.assessment}
                </div>
                <div class="table-container">
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th>Valuation Metric</th>
                                <th class="text-center">Current Level</th>
                                <th class="text-center">Benchmark / Hurdle</th>
                                <th class="text-center">Spread / Risk Premium</th>
                                <th>Sourcing &amp; Context</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Forward P/E</strong></td>
                                <td class="text-center font-mono font-bold">{pe_str}</td>
                                <td class="text-center">&mdash;</td>
                                <td class="text-center">&mdash;</td>
                                <td>Forward multiple on consensus 12M EPS ({v.source})</td>
                            </tr>
                            <tr>
                                <td><strong>Earnings Yield (1/PE)</strong></td>
                                <td class="text-center font-mono font-bold">{ey_str}</td>
                                <td class="text-center">&mdash;</td>
                                <td class="text-center">&mdash;</td>
                                <td>Equity cash generation yield (1 / Forward P/E)</td>
                            </tr>
                            <tr>
                                <td><strong>Nominal Equity Risk Premium (ERP)</strong></td>
                                <td class="text-center font-mono">{ey_str}</td>
                                <td class="text-center font-mono">{nom_hurdle}</td>
                                <td class="text-center font-mono {nom_class}">{nom_erp_str}</td>
                                <td>Earnings Yield minus 10Y Benchmark Yield</td>
                            </tr>
                            <tr>
                                <td><strong>Real Equity Risk Premium (Real ERP)</strong></td>
                                <td class="text-center font-mono">{ey_str}</td>
                                <td class="text-center font-mono">{real_hurdle}</td>
                                <td class="text-center font-mono {real_class}">{real_erp_str}</td>
                                <td>Earnings Yield minus 10Y TIPS Real Yield</td>
                            </tr>
                            <tr>
                                <td><strong>Forward EPS Revision Trend</strong></td>
                                <td class="text-center font-mono font-bold" colspan="3">{v.eps_revision_trend}</td>
                                <td>Revision momentum across S&amp;P 500 constituents</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """)

        # Section: Conclusion
        elif isinstance(sec.data, ConclusionData):
            cd: ConclusionData = sec.data
            v_low = cd.verdict.lower()
            v_class = "badge-bullish" if v_low in ("bullish", "risk-on", "constructive") else ("badge-bearish" if v_low in ("bearish", "risk-off", "cautious") else "badge-neutral")

            # 1. Category Breakdown Table (Requirement 2)
            cat_rows = []
            if cd.scorecard and cd.scorecard.categories:
                for cat in cd.scorecard.categories:
                    st_badge = '<span class="badge badge-green">ACTIVE</span>' if cat.status == "active" else '<span class="badge badge-muted">UNAVAILABLE (RENORMALIZED)</span>'
                    sc_class = "text-green font-bold" if cat.score > 0.1 else ("text-red font-bold" if cat.score < -0.1 else "text-muted")
                    w_sc_str = f"{cat.weighted_score:+.2f}" if cat.status == "active" else "—"
                    eff_w_str = f"{cat.effective_weight * 100:.1f}%" if cat.status == "active" else "— (dropped)"
                    cat_rows.append(f"""
                    <tr>
                        <td><strong>{cat.name}</strong></td>
                        <td class="text-center font-mono">{cat.weight * 100:.0f}%</td>
                        <td class="text-center font-mono font-bold">{eff_w_str}</td>
                        <td class="text-center font-mono {sc_class}">{cat.score:+.2f}</td>
                        <td class="text-center font-mono {sc_class}">{w_sc_str}</td>
                        <td class="text-center">{st_badge}</td>
                        <td><small>{cat.notes or f'{len(cat.components)} component indicators evaluated'}</small></td>
                    </tr>
                    """)

            # 2. Intraperiod Session Volatility Table (Requirement 6)
            intra_rows = []
            if cd.scorecard and cd.scorecard.intraperiod_context:
                for sess in cd.scorecard.intraperiod_context:
                    spx_c = "text-green font-bold" if sess.spx_change_pct > 0 else ("text-red font-bold" if sess.spx_change_pct < 0 else "text-muted")
                    ndx_c = "text-green font-bold" if sess.ndx_change_pct > 0 else ("text-red font-bold" if sess.ndx_change_pct < 0 else "text-muted")
                    alert_badge = '<span class="badge badge-red font-bold">⚠️ Large Move (&ge; 1.5%)</span>' if sess.is_large_move else '<span class="text-muted">Normal</span>'
                    intra_rows.append(f"""
                    <tr>
                        <td class="text-center font-mono">{sess.date}</td>
                        <td class="text-right font-mono">{sess.spx_close:,.2f}</td>
                        <td class="text-right font-mono {spx_c}">{sess.spx_change_pct:+.2f}%</td>
                        <td class="text-right font-mono">{sess.ndx_close:,.2f}</td>
                        <td class="text-right font-mono {ndx_c}">{sess.ndx_change_pct:+.2f}%</td>
                        <td class="text-center">{alert_badge}</td>
                    </tr>
                    """)

            # 3. Component Indicator Voting Rows
            vote_rows = []
            for v in (cd.scorecard.votes if (cd.scorecard and cd.scorecard.votes) else []):
                v_class_vote = "text-green font-bold" if v.vote > 0.05 else ("text-red font-bold" if v.vote < -0.05 else "text-muted")
                v_str = f"{v.vote:+.2f}"
                vote_rows.append(f"""
                <tr>
                    <td><strong>{v.rule_name}</strong></td>
                    <td class="text-center font-mono {v_class_vote}">{v_str}</td>
                    <td class="text-center font-mono">{v.weight:.1f}</td>
                    <td class="text-center font-mono">{v.weighted_vote:+.2f}</td>
                    <td><small>{v.condition}</small></td>
                    <td><em>{v.interpretation}</em></td>
                </tr>
                """)

            levels_html = []
            for al in cd.asset_levels:
                l_rows = []
                for kl in al.levels:
                    l_type_class = "text-green" if kl.level_type == "Support" else "text-red"
                    l_rows.append(f"""
                    <tr>
                        <td>{kl.label}</td>
                        <td class="text-right font-mono">{kl.price:,.2f}</td>
                        <td class="text-right font-mono">{_fmt_pct(kl.distance_pct)}</td>
                        <td class="text-center font-bold {l_type_class}">{kl.level_type}</td>
                    </tr>
                    """)
                levels_html.append(f"""
                <div class="levels-box">
                    <h4>{al.name} ({al.symbol}) &bull; Current: <span class="font-mono">{al.current_price:,.2f}</span></h4>
                    <table class="report-table compact-table">
                        <thead><tr><th>Level</th><th class="text-right">Price</th><th class="text-right">Distance</th><th class="text-center">Type</th></tr></thead>
                        <tbody>{''.join(l_rows)}</tbody>
                    </table>
                </div>
                """)

            forward_strat_html = ""
            if cd.regime_intelligence:
                intel = cd.regime_intelligence
                sec_sectors_link = sec_by_type.get(SectorReportData) or sec_by_type.get(MacroSectionData)
                sec_link_html = (
                    f' (see <a href="#section-{sec_sectors_link.section_num}" style="color: var(--primary); text-decoration: underline;">Section {sec_sectors_link.section_num}: {sec_sectors_link.title}</a> for sector momentum details)'
                    if sec_sectors_link else ""
                )
                forward_strat_html = f"""
                <div class="forward-strategy-card" style="margin-top: 24px; padding: 18px 22px; border-radius: 10px; background: var(--bg-card-subtle); border: 1px solid var(--border); border-left: 4px solid var(--primary);">
                    <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 8px; color: var(--primary); display: flex; align-items: center; gap: 8px;">
                        <span>🔭</span> Cross-Market Macro Alignment &amp; Market Regime
                    </h3>
                    <p style="font-size: 14px; line-height: 1.5; margin: 0; color: var(--text-main);">
                        <strong>Alignment:</strong> The technical trend stands at <strong>{cd.verdict.upper()}</strong> ({cd.normalized_score:+.2f}), operating within an active <strong>{intel.summary_badge}</strong> regime. Corporate credit remains <strong>{intel.credit.regime}</strong> ({intel.credit.spread_bps:.0f} bps OAS) with volatility term structure in <strong>{intel.vol_term.structure}</strong> (ratio {intel.vol_term.ratio:.3f}). Tactical sector rotation shows {intel.rotation.rotation_summary}{sec_link_html}.
                    </p>
                </div>
                """

            lenses_html = ""
            lenses = getattr(cd, "cross_strategy_lenses", None)
            if lenses:
                lens_cards = []
                for lens in lenses:
                    lens_cards.append(f"""
                    <div style="padding: 14px 16px; border-radius: 8px; background: var(--bg-card-subtle); border: 1px solid var(--border);">
                        <div style="display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 6px;">
                            <strong style="font-size: 13px; color: var(--text-main);">{lens.strategy}</strong>
                            <span class="badge badge-neutral" style="font-size: 11px;">{lens.regime_read}</span>
                        </div>
                        <p style="font-size: 13px; line-height: 1.45; margin: 0; color: var(--text-muted);">{lens.synthesis}</p>
                    </div>
                    """)
                lenses_html = f"""
                <div style="margin-top: 20px;">
                    <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 10px; color: var(--text-main);">Cross-Strategy Risk &amp; Regime Synthesis</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px;">
                        {''.join(lens_cards)}
                    </div>
                </div>
                """
                forward_strat_html = f"{forward_strat_html}{lenses_html}"

            conf_label = f'<div class="score-sub font-mono" style="margin-top: 4px; color: var(--text-muted); font-size: 11px;">Confidence: <strong>{cd.scorecard.confidence}</strong></div>' if cd.scorecard and cd.scorecard.confidence else ''

            sens_html = ""
            if cd.scorecard and getattr(cd.scorecard, "sensitivity", None):
                sens = cd.scorecard.sensitivity
                if period.report_type == "daily":
                    if sens.is_borderline:
                        sens_html = f"""
                        <div class="alert alert-warning" style="margin-top: 16px; margin-bottom: 12px;">
                            ⚠️ <strong>Regime Edge Warning:</strong> Current score <code>{sens.baseline_score:+.2f}</code> sits within 0.03 pts of regime threshold (<strong>{sens.borderline_label}</strong>).
                        </div>
                        """
                else:
                    sens_rows = []
                    for s_it in sens.items:
                        v_badge = "badge-green" if s_it.verdict_ex in ("Risk-on", "Constructive") else ("badge-red" if s_it.verdict_ex in ("Cautious", "Risk-off") else "badge-yellow")
                        d_color = "text-green" if s_it.delta_vs_baseline > 0 else ("text-red" if s_it.delta_vs_baseline < 0 else "text-muted")
                        sens_rows.append(f"""
                        <tr>
                            <td><strong>{s_it.excluded_category}</strong></td>
                            <td class="text-center font-mono">{s_it.score_ex:+.2f}</td>
                            <td class="text-center"><span class="badge {v_badge}">{s_it.verdict_ex}</span></td>
                            <td class="text-center font-mono {d_color}">{s_it.delta_vs_baseline:+.2f}</td>
                        </tr>
                        """)
                    borderline_alert = ""
                    if sens.is_borderline:
                        borderline_alert = f"""
                        <div class="alert alert-warning" style="margin-bottom: 12px;">
                            ⚠️ <strong>Regime Edge Warning:</strong> Current score <code>{sens.baseline_score:+.2f}</code> sits within 0.03 pts of regime threshold (<strong>{sens.borderline_label}</strong>).
                        </div>
                        """
                    sens_html = f"""
                    <h3 class="sub-title" style="margin-top: 24px;">Verdict Sensitivity &amp; Borderline Analysis</h3>
                    {borderline_alert}
                    <div class="table-container">
                        <table class="report-table compact-table">
                            <thead>
                                <tr>
                                    <th>Excluded Category</th>
                                    <th class="text-center">Score (ex-Category)</th>
                                    <th class="text-center">Implied Verdict</th>
                                    <th class="text-center">Delta vs Baseline</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(sens_rows)}</tbody>
                        </table>
                    </div>
                    """

            intra_html = ""
            if intra_rows and period.report_type != "daily":
                intra_html = f"""
                <h3 class="sub-title">Intraperiod Daily Session Volatility</h3>
                <div class="table-container">
                    <table class="report-table compact-table">
                        <thead>
                            <tr>
                                <th class="text-center">Session Date</th>
                                <th class="text-right">S&amp;P 500 Close</th>
                                <th class="text-right">S&amp;P 500 Return</th>
                                <th class="text-right">Nasdaq-100 Close</th>
                                <th class="text-right">Nasdaq-100 Return</th>
                                <th class="text-center">Single-Session Move (&ge; 1.5%)</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(intra_rows)}</tbody>
                    </table>
                </div>
                """

            voting_html = ""
            if period.report_type != "daily":
                voting_html = f"""
                <h3 class="sub-title">Explainable Component Indicator Voting</h3>
                <div class="table-container">
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th>Indicator Rule</th>
                                <th class="text-center">Gradient Vote</th>
                                <th class="text-center">Weight</th>
                                <th class="text-center">Weighted Vote</th>
                                <th>Metric Details</th>
                                <th>Interpretation</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(vote_rows)}</tbody>
                    </table>
                </div>
                """
            else:
                voting_html = """
                <div style="padding: 12px 16px; border-radius: 8px; background: var(--bg-card-subtle); margin-top: 16px; margin-bottom: 16px;">
                    <p style="font-size: 12px; margin: 0; color: var(--text-muted); font-style: italic;">Individual indicator voting rules, sensitivity analysis, and intraperiod trajectory are published in the Weekly &amp; Monthly strategic reports.</p>
                </div>
                """

            v_low = cd.verdict.lower()
            if v_low in ("bullish", "risk-on", "constructive"):
                v_class = "badge-green"
                banner_border = "var(--tropical-teal)"
            elif v_low in ("bearish", "risk-off", "cautious"):
                v_class = "badge-red"
                banner_border = "var(--rosy-copper)"
            else:
                v_class = "badge-amber"
                banner_border = "var(--amber-honey)"

            conf_html = f'<div style="font-size: 0.82rem; color: var(--text-muted); font-family: var(--font-mono); margin: -8px 0 20px 0;"><strong>Confidence:</strong> {cd.scorecard.confidence}</div>' if cd.scorecard and cd.scorecard.confidence else ''

            if period.report_type in ("weekly", "monthly"):
                scorecard_appendix_html = f"""
                <details style="margin-top: 18px;">
                    <summary>Regime Scorecard Category Breakdown, Sensitivity &amp; Indicator Voting Appendix</summary>
                    <h4 class="sub-title" style="margin-top: 14px;">Macro &amp; Micro Trend Scorecard by Category</h4>
                    <div class="table-container">
                        <table class="report-table">
                            <thead>
                                <tr>
                                    <th>Category</th>
                                    <th class="text-center">Weight Cap</th>
                                    <th class="text-center">Effective Weight</th>
                                    <th class="text-center">Category Score</th>
                                    <th class="text-center">Weighted Score</th>
                                    <th class="text-center">Status</th>
                                    <th>Diagnostic Notes / Drivers</th>
                                </tr>
                            </thead>
                            <tbody>{''.join(cat_rows)}</tbody>
                        </table>
                    </div>
                    {sens_html}
                    {intra_html}
                    {voting_html}
                </details>
                """
            else:
                scorecard_appendix_html = f"""
                <h3 class="sub-title">Macro &amp; Micro Trend Scorecard by Category</h3>
                <div class="table-container">
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th class="text-center">Weight Cap</th>
                                <th class="text-center">Effective Weight</th>
                                <th class="text-center">Category Score</th>
                                <th class="text-center">Weighted Score</th>
                                <th class="text-center">Status</th>
                                <th>Diagnostic Notes / Drivers</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(cat_rows)}</tbody>
                    </table>
                </div>

                {sens_html}

                {intra_html}

                {voting_html}
                """

            body_html.append(f"""
            <div class="verdict-banner" style="border-left-color: {banner_border}; margin-bottom: 14px;">
                <div class="verdict-box">
                    <span class="verdict-label">Final Regime Outlook</span>
                    <span class="verdict-badge {v_class}">{cd.verdict.upper()}</span>
                </div>
                <div class="score-box" style="margin-left: auto; text-align: right;">
                    <span class="score-label">Normalized Score</span>
                    <span class="score-val font-mono">{cd.normalized_score:+.2f}</span>
                    <span class="score-sub font-mono" style="color: rgba(255,255,255,0.65); font-size: 11px; margin-left: 8px;">(Scale: -1.0 to +1.0)</span>
                </div>
            </div>
            {conf_html}

            {scorecard_appendix_html}

            <div class="grid-2 drivers-grid" style="margin-top: 18px;">
                <div class="driver-card driver-bullish">
                    <h4 class="text-green">🟢 Top Bullish Factors</h4>
                    <ul>{''.join(f'<li>{bf}</li>' for bf in cd.top_bullish_factors)}</ul>
                </div>
                <div class="driver-card driver-bearish">
                    <h4 class="text-red">🔴 Top Bearish Factors</h4>
                    <ul>{''.join(f'<li>{br}</li>' for br in cd.top_bearish_factors)}</ul>
                </div>
            </div>

            {f'<div class="alert alert-warning">⚠️ <strong>Caution Flags:</strong> {" &bull; ".join(cd.caution_flags)}</div>' if cd.caution_flags else ''}

            <h3 class="sub-title">Key Benchmark Support & Resistance Levels</h3>
            <div class="grid-2">{''.join(levels_html)}</div>
            {forward_strat_html}
            """)

            # Substage 3.3: Historical Seasonality Summary (Complete Calendar Years Only)
            seas = getattr(cd, "seasonality", None)
            if seas is not None and getattr(seas, "complete_years_count", 0) > 0:
                c_win_cls = "badge-bullish" if seas.current_month_win_rate >= 55.0 else ("badge-bearish" if seas.current_month_win_rate <= 45.0 else "badge-neutral")
                n_win_cls = "badge-bullish" if seas.next_month_win_rate >= 55.0 else ("badge-bearish" if seas.next_month_win_rate <= 45.0 else "badge-neutral")
                body_html.append(f"""
                <h3 class="sub-title" style="margin-top: 24px;">S&P 500 Historical Monthly Seasonality (Complete Calendar Years Only)</h3>
                <div class="alert alert-info" style="margin-bottom: 12px;">
                    <strong>Historical Seasonality Context (N={seas.complete_years_count} Complete Calendar Years, {seas.history_start_year}–{seas.history_end_year}):</strong><br>
                    Current month (<strong>{seas.current_month_name}</strong>) historically averages {_fmt_pct(seas.current_month_avg_return)} ({seas.current_month_win_rate:.1f}% win rate, median {_fmt_pct(seas.current_month_median_return)}, IQR {seas.current_month_iqr_return:.2f}%).<br>
                    Next month (<strong>{seas.next_month_name}</strong>) historically averages {_fmt_pct(seas.next_month_avg_return)} ({seas.next_month_win_rate:.1f}% win rate, median {_fmt_pct(seas.next_month_median_return)}, IQR {seas.next_month_iqr_return:.2f}%).
                </div>
                <div class="table-container">
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th>Calendar Horizon</th>
                                <th>Month</th>
                                <th class="text-center">Historical Win %</th>
                                <th class="text-center">95% Wilson CI</th>
                                <th class="text-right">Historical Median</th>
                                <th class="text-right">Historical Mean</th>
                                <th class="text-right">IQR [P25, P75]</th>
                                <th class="text-center">Sample Size</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Current Month</strong></td>
                                <td>{seas.current_month_name}</td>
                                <td class="text-center"><span class="badge {c_win_cls}"><strong>{seas.current_month_win_rate:.1f}%</strong></span></td>
                                <td class="text-center font-mono text-muted">[{seas.current_month_win_rate_ci_low:.1f}%, {seas.current_month_win_rate_ci_high:.1f}%]</td>
                                <td class="text-right">{_fmt_pct(seas.current_month_median_return)}</td>
                                <td class="text-right">{_fmt_pct(seas.current_month_avg_return)}</td>
                                <td class="text-right font-mono">{seas.current_month_iqr_return:.2f}% <span class="text-muted">[{_fmt_pct(seas.current_month_p25_return)}, {_fmt_pct(seas.current_month_p75_return)}]</span></td>
                                <td class="text-center font-mono">N={seas.current_month_sample_size} complete yrs</td>
                            </tr>
                            <tr>
                                <td><strong>Next Month</strong></td>
                                <td>{seas.next_month_name}</td>
                                <td class="text-center"><span class="badge {n_win_cls}"><strong>{seas.next_month_win_rate:.1f}%</strong></span></td>
                                <td class="text-center font-mono text-muted">[{seas.next_month_win_rate_ci_low:.1f}%, {seas.next_month_win_rate_ci_high:.1f}%]</td>
                                <td class="text-right">{_fmt_pct(seas.next_month_median_return)}</td>
                                <td class="text-right">{_fmt_pct(seas.next_month_avg_return)}</td>
                                <td class="text-right font-mono">{seas.next_month_iqr_return:.2f}% <span class="text-muted">[{_fmt_pct(seas.next_month_p25_return)}, {_fmt_pct(seas.next_month_p75_return)}]</span></td>
                                <td class="text-center font-mono">N={seas.next_month_sample_size} complete yrs</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <p class="table-caption" style="margin-top: 6px;"><em>Note: Seasonality calculations evaluate strictly complete calendar years ({seas.history_start_year}–{seas.history_end_year}); the partial in-progress month ({seas.current_month_name} {period.session_date[:4]}) is excluded from historical sample distributions.</em></p>
                """)

            if charts and charts.get("seasonality_heatmap"):
                body_html.append(_render_chart_html(charts.get("seasonality_heatmap")))

        # Section: Footer
        elif isinstance(sec.data, dict) and "disclaimer" in sec.data:
            disclaimer = sec.data.get("disclaimer", DEFAULT_COMPLIANCE_DISCLAIMER)
            sources = sec.data.get("data_sources", [])
            overall_status = sec.data.get("overall_status", "PARTIAL")
            missing_inputs = sec.data.get("missing_inputs", [])
            sources_html = ""
            if sources:
                s_rows = []
                for s in sources:
                    status_badge = "badge-bullish" if "COMPLETE" in s.status.upper() else ("badge-neutral" if "PARTIAL" in s.status.upper() else "badge-muted")
                    s_rows.append(f"""
                    <tr>
                        <td><strong>{s.source_name}</strong></td>
                        <td>{s.coverage}</td>
                        <td class="font-mono text-center">{s.fetch_time}</td>
                        <td class="text-center"><span class="badge {status_badge}">{s.status.upper()}</span></td>
                        <td>{s.notes}</td>
                    </tr>
                    """)

                missing_html = ""
                if missing_inputs:
                    missing_str = "; ".join(missing_inputs)
                    missing_html = f'<div class="alert alert-warning" style="margin-top: 10px; text-align: left;">⚠️ <strong>Missing / Unavailable Inputs:</strong> {missing_str}</div>'

                overall_badge = "badge-bullish" if overall_status.upper() == "COMPLETE" else "badge-neutral"
                sources_html = f"""
                <h3 class="sub-title" style="margin-top: 24px;">Data Quality &amp; Provenance &bull; <span class="badge {overall_badge}">STATUS: {overall_status.upper()}</span></h3>
                {missing_html}
                <div class="table-container">
                    <table class="report-table compact-table">
                        <thead>
                            <tr>
                                <th>Source</th>
                                <th>Coverage</th>
                                <th class="text-center">Fetch / Window</th>
                                <th class="text-center">Status</th>
                                <th>Fallbacks &amp; Data Notes</th>
                            </tr>
                        </thead>
                        <tbody>{''.join(s_rows)}</tbody>
                    </table>
                </div>
                """

                if period.report_type in ("daily", "weekly"):
                    sources_html = f"""
                    <details class="methodology-note" style="margin-top: 18px;">
                        <summary>Data Quality &amp; Provenance Sources</summary>
                        {sources_html}
                    </details>
                    """

            body_html.append(f"""
            <footer class="report-footer">
                <p><strong>Disclaimer:</strong> {disclaimer}</p>
                {sources_html}
                <p class="footer-sub" style="margin-top: 14px;">Generated by Automated Market Report Engine &bull; System Time: {period.as_of_date}</p>
            </footer>
            """)

        body_html.append('</section>')

    # Complete HTML Document with Inline CSS and Theme Toggle
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{rep_title} - {period.session_date}</title>
    <style>
        html {{
            font-size: 20px;
        }}

        :root, body.theme-zen {{
            /* Tropical Earth Palette */
            --amber-honey: #eca400;
            --rosy-copper: #db504a;
            --dark-slate-grey: #2c514c;
            --tropical-teal: #56a3a6;
            --deep-walnut: #56351e;

            /* Mappings */
            --bg-page: #FDFBF7;
            --bg-sidebar: var(--dark-slate-grey);
            --bg-card: #FFFFFF;
            --bg-card-subtle: rgba(44, 81, 76, 0.05);
            --bg-nav: rgba(253, 251, 247, 0.94);
            
            --text-main: var(--deep-walnut);
            --text-muted: rgba(86, 53, 30, 0.65);
            --text-sidebar: #E0EAE9;
            
            --primary: var(--dark-slate-grey); 
            --border: rgba(86, 53, 30, 0.15);
            
            --accent-green-bg: rgba(86, 163, 166, 0.15);
            --accent-green-txt: #2c514c;
            --accent-red-bg: rgba(219, 80, 74, 0.12);
            --accent-red-txt: #b33934;
            --accent-amber-bg: rgba(236, 164, 0, 0.15);
            --accent-amber-txt: #8f5d00;
            
            --green: #2c514c;
            --red: #db504a;
            --amber: #eca400;
            --accent: var(--tropical-teal);
            
            --badge-bullish-bg: rgba(86, 163, 166, 0.15);
            --badge-bullish-text: #2c514c;
            --badge-bearish-bg: rgba(219, 80, 74, 0.12);
            --badge-bearish-text: #b33934;
            --badge-neutral-bg: rgba(236, 164, 0, 0.15);
            --badge-neutral-text: #8f5d00;
            
            --shadow-card: 0 4px 12px rgba(86, 53, 30, 0.04);
            --radius-card: 12px;
        }}

        html.dark, body.dark {{
            --bg-page: #152422;
            --bg-sidebar: #101c1a;
            --bg-card: #1c302d;
            --bg-card-subtle: rgba(86, 163, 166, 0.08);
            --bg-nav: rgba(21, 36, 34, 0.94);
            
            --text-main: #FDFBF7;
            --text-muted: rgba(253, 251, 247, 0.65);
            --text-sidebar: #E0EAE9;
            
            --primary: var(--tropical-teal);
            --border: rgba(253, 251, 247, 0.12);
            
            --accent-green-bg: rgba(86, 163, 166, 0.25);
            --accent-green-txt: #7cd3d5;
            --accent-red-bg: rgba(219, 80, 74, 0.25);
            --accent-red-txt: #ff7d78;
            --accent-amber-bg: rgba(236, 164, 0, 0.25);
            --accent-amber-txt: #ffc44d;
            
            --green: #7cd3d5;
            --red: #ff7d78;
            --amber: #ffc44d;
            --accent: var(--tropical-teal);
            
            --badge-bullish-bg: rgba(86, 163, 166, 0.25);
            --badge-bullish-text: #7cd3d5;
            --badge-bearish-bg: rgba(219, 80, 74, 0.25);
            --badge-bearish-text: #ff7d78;
            --badge-neutral-bg: rgba(236, 164, 0, 0.25);
            --badge-neutral-text: #ffc44d;
            
            --shadow-card: 0 4px 12px rgba(0, 0, 0, 0.25);
        }}

        @media (prefers-color-scheme: dark) {{
            :root:not(.light) {{
                --bg-page: #152422;
                --bg-sidebar: #101c1a;
                --bg-card: #1c302d;
                --bg-card-subtle: rgba(86, 163, 166, 0.08);
                --bg-nav: rgba(21, 36, 34, 0.94);
                
                --text-main: #FDFBF7;
                --text-muted: rgba(253, 251, 247, 0.65);
                --text-sidebar: #E0EAE9;
                
                --primary: var(--tropical-teal);
                --border: rgba(253, 251, 247, 0.12);
                
                --accent-green-bg: rgba(86, 163, 166, 0.25);
                --accent-green-txt: #7cd3d5;
                --accent-red-bg: rgba(219, 80, 74, 0.25);
                --accent-red-txt: #ff7d78;
                --accent-amber-bg: rgba(236, 164, 0, 0.25);
                --accent-amber-txt: #ffc44d;
                
                --green: #7cd3d5;
                --red: #ff7d78;
                --amber: #ffc44d;
                --accent: var(--tropical-teal);
                
                --badge-bullish-bg: rgba(86, 163, 166, 0.25);
                --badge-bullish-text: #7cd3d5;
                --badge-bearish-bg: rgba(219, 80, 74, 0.25);
                --badge-bearish-text: #ff7d78;
                --badge-neutral-bg: rgba(236, 164, 0, 0.25);
                --badge-neutral-text: #ffc44d;
                
                --shadow-card: 0 4px 12px rgba(0, 0, 0, 0.25);
            }}
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        html {{ scroll-behavior: smooth; font-size: 93.75%; }}
        
        body {{
            font-family: 'Founders Grotesk', 'Helvetica Neue', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-page);
            color: var(--text-main);
            line-height: 1.5;
            display: flex;
            min-height: 100vh;
            overflow-x: hidden;
        }}

        .font-mono {{
            font-family: 'Gt American Mono', 'JetBrains Mono', SFMono-Regular, Menlo, Monaco, Consolas, ui-monospace, monospace;
            letter-spacing: -0.02em;
        }}
        .font-bold {{ font-weight: 700; }}

        /* --- Sidebar --- */
        .sidebar {{
            width: 280px;
            background-color: var(--bg-sidebar);
            color: var(--text-sidebar);
            height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            flex-direction: column;
            padding: 22px 18px;
            border-right: 1px solid var(--border);
            transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            z-index: 100;
        }}
        
        .sidebar.collapsed {{
            transform: translateX(-100%);
        }}

        .brand {{ margin-bottom: 18px; }}
        .brand h1 {{
            font-size: 1.3rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            line-height: 1.2;
            margin-bottom: 4px;
            color: #FFFFFF;
        }}
        .brand p {{
            font-size: 0.88rem;
            color: var(--tropical-teal);
            font-weight: 500;
        }}

        .nav-menu {{
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            flex-grow: 1;
            margin-bottom: 16px;
            padding-right: 4px;
            scroll-target-group: auto;
        }}
        
        .nav-menu::-webkit-scrollbar {{ width: 4px; }}
        .nav-menu::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.2); border-radius: 4px; }}

        details > summary {{
            list-style: none;
            font-size: 0.96rem;
            font-weight: 700;
            color: var(--amber-honey);
            padding: 9px 10px;
            cursor: pointer;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background-color 0.2s;
            margin-top: 4px;
        }}
        details > summary::-webkit-details-marker {{ display: none; }}
        details > summary:hover {{ background-color: rgba(255, 255, 255, 0.05); }}
        details > summary::after {{ content: '+'; font-size: 1.15em; line-height: 0.5; color: var(--tropical-teal); }}
        details[open] > summary::after {{ content: '−'; }}

        .nav-link {{
            text-decoration: none;
            color: rgba(255,255,255,0.88);
            font-size: 0.9rem;
            font-weight: 400;
            padding: 6px 10px 6px 11px;
            border-left: 3px solid transparent;
            border-radius: 6px;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            line-height: 1.35;
            margin-bottom: 3px;
        }}
        .nav-link:hover {{
            background-color: rgba(255,255,255,0.1);
            color: #FFFFFF;
            transform: translateX(3px);
        }}
        .nav-link:is(:target-current, .target-current) {{
            color: var(--amber-honey);
            font-weight: 700;
            background-color: rgba(236, 164, 0, 0.14);
            border-left-color: var(--amber-honey);
        }}

        .sidebar-footer {{
            padding-top: 16px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}

        .theme-toggle-btn {{
            width: 100%;
            padding: 9px;
            background: var(--amber-honey);
            color: var(--deep-walnut);
            border: none;
            border-radius: 8px;
            font-family: inherit;
            font-weight: 700;
            font-size: 0.88rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 6px;
        }}
        .theme-toggle-btn:hover {{
            background-color: #f5b324;
        }}

        /* --- Main Content --- */
        .main-content {{
            margin-left: 280px;
            flex-grow: 1;
            padding: 24px 40px 48px 40px;
            max-width: 1380px;
            transition: margin-left 0.3s cubic-bezier(0.16, 1, 0.3, 1), max-width 0.3s;
        }}
        .main-content.expanded {{
            margin-left: 0;
            max-width: 100%;
        }}

        .top-bar {{
            display: flex;
            align-items: center;
            margin-bottom: 24px;
        }}

        .sticky-nav-bar {{
            position: sticky;
            top: 0;
            z-index: 90;
            background: var(--bg-nav);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 10px 0;
        }}
        .nav-inner {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
        }}
        .nav-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .toggle-sidebar-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--primary);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 700;
            font-family: inherit;
            font-size: 0.88rem;
            display: flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}
        .toggle-sidebar-btn:hover {{
            border-color: var(--tropical-teal);
        }}
        .nav-controls {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .nav-select {{
            background: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 0.84rem;
            font-weight: 600;
            cursor: pointer;
            outline: none;
            max-width: 280px;
        }}
        .nav-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 0.84rem;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
        }}
        .nav-btn:hover, .nav-select:hover {{
            border-color: var(--tropical-teal);
        }}

        @media (max-width: 1024px) {{
            .sidebar {{ transform: translateX(-100%); }}
            .sidebar.mobile-open {{ transform: translateX(0); }}
            .main-content {{ margin-left: 0; padding: 16px; }}
            .main-content.expanded {{ margin-left: 0; }}
        }}

        /* --- Cards & Headers --- */
        .section-card, .card {{
            background: var(--bg-card);
            border-radius: var(--radius-card);
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: var(--shadow-card);
            border: 1px solid var(--border);
            scroll-margin-top: 64px;
        }}

        .section-header {{
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }}
        .section-num {{
            background: var(--bg-sidebar);
            color: #FFF;
            width: 30px;
            height: 30px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 700;
        }}
        .section-title {{
            font-size: 1.3rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: var(--text-main);
            margin: 0;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            max-width: 100%;
            padding: 4px 9px;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 700;
            line-height: 1.25;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            text-align: center;
            white-space: normal;
            overflow-wrap: anywhere;
        }}
        .report-table .badge {{
            white-space: nowrap;
        }}
        .report-table td > .badge:only-child {{
            display: flex;
            width: 100%;
            box-sizing: border-box;
        }}
        .badge-primary {{ background: var(--bg-sidebar); color: #FFF; }}
        .badge-green, .badge-bullish {{ background: var(--accent-green-bg); color: var(--accent-green-txt); }}
        .badge-red, .badge-bearish {{ background: var(--accent-red-bg); color: var(--accent-red-txt); }}
        .badge-amber, .badge-neutral, .badge-yellow {{ background: var(--accent-amber-bg); color: var(--accent-amber-txt); }}
        .badge-muted {{ background: var(--bg-card-subtle); color: var(--text-muted); border: 1px solid var(--border); }}

        .text-green {{ color: var(--accent-green-txt); }}
        .text-red {{ color: var(--accent-red-txt); }}
        .text-muted {{ color: var(--text-muted); }}
        .text-right {{ text-align: right; }}
        .text-center {{ text-align: center; }}

        /* --- KPI Grid --- */
        .top10-ticker-strip {{ margin-bottom: 24px; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border-radius: var(--radius-card);
            padding: 16px;
            box-shadow: var(--shadow-card);
            display: flex;
            flex-direction: column;
            gap: 4px;
            border: 1px solid var(--border);
            text-decoration: none;
            color: inherit;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(86, 53, 30, 0.08);
        }}
        .kpi-label {{
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        .kpi-val {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-main);
        }}
        .kpi-sub {{
            font-size: 0.84rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .kpi-secondary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
        }}
        .kpi-secondary-card {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 2px;
            border: 1px solid var(--border);
            text-decoration: none;
            color: inherit;
            transition: background-color 0.15s ease;
        }}
        .kpi-secondary-card:hover {{
            background: var(--bg-card-subtle);
        }}
        .kpi-secondary-card .kpi-label {{ font-size: 0.74rem; }}
        .kpi-secondary-card .kpi-val {{ font-size: 1.1rem; }}
        .kpi-secondary-card .kpi-sub {{ font-size: 0.8rem; }}

        /* --- Tables --- */
        .table-container {{
            overflow-x: auto;
            margin-top: 12px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 1.0rem;
            text-align: left;
        }}
        .report-table th {{
            background: var(--bg-card-subtle);
            color: var(--text-main);
            font-size: 0.86rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
        }}
        .report-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            color: var(--text-main);
        }}
        .report-table tbody tr {{ transition: background-color 0.1s ease; }}
        .report-table tbody tr:nth-child(even) {{ background-color: var(--bg-card-subtle); }}
        .report-table tbody tr:hover {{ background-color: rgba(236, 164, 0, 0.08); }}
        .report-table tr:last-child td {{ border-bottom: none; }}
        .compact-table td, .compact-table th {{ padding: 8px 10px; font-size: 0.95rem; }}
        .watchlist-table td, .watchlist-table th {{ padding: 7px 9px; font-size: 0.85rem; vertical-align: middle; }}
        .watchlist-table th {{ font-size: 0.76rem; line-height: 1.2; }}
        .watchlist-table td:nth-child(3),
        .watchlist-table td:nth-child(4),
        .watchlist-table td:nth-child(5),
        .watchlist-table td:nth-child(6),
        .watchlist-table td:nth-child(7),
        .watchlist-table td:nth-child(10) {{ white-space: nowrap; }}

        .ticker-circle {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: var(--bg-sidebar);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.65rem;
            font-weight: 700;
            color: #FFF;
            margin-right: 8px;
        }}
        .flex-cell {{ display: flex; align-items: center; }}

        /* --- Verdict Banners --- */
        .verdict-banner {{
            display: flex;
            gap: 24px;
            background: var(--bg-sidebar);
            padding: 16px 24px;
            border-radius: 12px;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            border-left: 4px solid var(--amber-honey);
            color: #FFF;
        }}
        .verdict-box {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .score-box {{
            display: flex;
            align-items: baseline;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .verdict-label, .score-label {{
            font-size: 0.8rem;
            font-weight: 600;
            color: rgba(255,255,255,0.7);
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}
        .verdict-badge {{
            font-size: 1.0rem;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 600;
        }}
        .verdict-banner .badge-amber, .verdict-banner .badge-neutral {{
            background: var(--amber-honey);
            color: #1e332f;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .verdict-banner .badge-red, .verdict-banner .badge-bearish {{
            background: var(--rosy-copper);
            color: #FFFFFF;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .verdict-banner .badge-green, .verdict-banner .badge-bullish {{
            background: var(--tropical-teal);
            color: #FFFFFF;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .score-val {{
            font-size: 1.6rem;
            font-weight: 600;
            line-height: 1;
        }}

        .reasons-list {{ padding-left: 20px; display: flex; flex-direction: column; gap: 8px; }}
        .reasons-heading, .sub-title {{ font-size: 1rem; font-weight: 700; margin: 16px 0 10px 0; color: var(--text-main); }}

        .methodology-note {{
            margin: 16px 0;
            padding: 12px 16px;
            background: var(--bg-card-subtle);
            border: 1px dashed var(--border);
            border-radius: 8px;
            font-size: 0.88rem;
        }}
        .methodology-note summary {{
            cursor: pointer;
            font-weight: 700;
            color: var(--primary);
            user-select: none;
            outline: none;
        }}
        .methodology-note summary:hover {{ text-decoration: underline; }}
        .methodology-content {{
            margin-top: 10px;
            color: var(--text-muted);
            line-height: 1.6;
        }}
        .methodology-content strong {{ color: var(--text-main); }}

        .alert {{ padding: 12px 16px; border-radius: 8px; font-size: 0.92rem; margin: 12px 0; line-height: 1.5; }}
        .alert-info {{ background: var(--bg-card-subtle); border-left: 4px solid var(--tropical-teal); color: var(--text-main); }}
        .alert-warning {{ background: rgba(236, 164, 0, 0.12); color: var(--text-main); border-left: 4px solid var(--amber-honey); }}
        .alert-danger {{ background: rgba(219, 80, 74, 0.12); color: var(--text-main); border-left: 4px solid var(--rosy-copper); }}

        .chart-svg-container {{
            margin: 20px 0;
            text-align: center;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-card);
            padding: 16px 20px;
        }}
        .chart-svg-wrapper {{
            width: 100%;
            overflow-x: auto;
            display: flex;
            justify-content: center;
        }}
        .chart-svg-wrapper svg {{
            max-width: 100%;
            height: auto;
            display: block;
        }}
        .macro-chart-box {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; }}
        .macro-chart-svg {{ width: 100%; height: auto; display: block; }}
        .macro-charts-stack {{ display: flex; flex-direction: column; gap: 24px; margin: 20px 0 28px 0; }}

        .vix-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
        .vix-metric {{ background: var(--bg-card-subtle); padding: 14px; border-radius: 8px; display: flex; flex-direction: column; gap: 4px; border: 1px solid var(--border); }}
        .vix-header {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
        .vix-label {{ font-size: 0.74rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; }}
        .vix-val {{ font-size: 1.3rem; font-weight: 700; }}
        .vix-sub {{ font-size: 0.8rem; color: var(--text-muted); }}
        .vix-desc {{ font-size: 0.8rem; margin-top: 4px; color: var(--text-muted); }}

        .toggletip-btn {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 1px solid var(--border);
            background: var(--bg-card);
            color: var(--text-muted);
            font-size: 0.72rem;
            font-weight: 700;
            font-style: italic;
            font-family: Georgia, serif;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            flex-shrink: 0;
            transition: border-color 0.15s ease, color 0.15s ease, background-color 0.15s ease;
        }}
        .toggletip-btn:hover, .toggletip-btn:focus-visible {{
            border-color: var(--tropical-teal);
            color: var(--tropical-teal);
            outline: none;
        }}
        .toggletip-popover {{
            max-width: min(320px, 85vw);
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--bg-card);
            color: var(--text-main);
            font-size: 0.8rem;
            font-weight: 400;
            line-height: 1.45;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
            inset: auto;
            margin: 0;
        }}
        @supports (position-area: block-start) {{
            .toggletip-popover {{
                position-area: block-start;
                position-try-fallbacks: flip-block, flip-inline;
            }}
        }}

        .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-top: 10px; }}
        .tech-card {{
            background: var(--bg-card-subtle);
            border-radius: 10px;
            padding: 18px;
            margin-bottom: 16px;
            border: 1px solid var(--border);
        }}
        .tech-header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }}
        .tech-price {{ font-size: 1.3rem; font-weight: 700; }}
        .metric-group {{ display: flex; flex-direction: column; gap: 6px; }}
        .metric-title {{ font-size: 0.84rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }}
        .metric-row {{ display: flex; justify-content: space-between; font-size: 0.88rem; }}

        .mover-period-title {{ font-size: 1.05rem; font-weight: 700; margin: 20px 0 10px 0; color: var(--text-main); }}
        .movers-split {{ display: flex; flex-direction: column; gap: 20px; }}
        .mover-table-box {{ background: var(--bg-card-subtle); padding: 16px 18px; border-radius: 10px; overflow-x: auto; border: 1px solid var(--border); }}
        .mover-subheading {{ font-size: 0.92rem; font-weight: 700; margin-bottom: 8px; }}

        .drivers-grid {{ margin: 16px 0; }}
        .driver-card {{ padding: 16px; border-radius: 10px; }}
        .driver-card ul {{ padding-left: 20px; margin-top: 8px; display: flex; flex-direction: column; gap: 4px; font-size: 0.92rem; }}
        .driver-bullish {{ background: var(--accent-green-bg); color: var(--accent-green-txt); }}
        .driver-bearish {{ background: var(--accent-red-bg); color: var(--accent-red-txt); }}

        .levels-box {{ background: var(--bg-card-subtle); padding: 14px; border-radius: 8px; border: 1px solid var(--border); }}
        .levels-box h4 {{ font-size: 0.92rem; font-weight: 700; margin-bottom: 8px; }}

        .report-footer {{ border-top: 1px solid var(--border); padding-top: 16px; text-align: center; color: var(--text-muted); font-size: 0.88rem; }}
        .footer-sub {{ font-size: 0.78rem; margin-top: 4px; }}
        .table-caption {{ font-size: 0.84rem; color: var(--text-muted); margin-bottom: 8px; }}
    </style>
</head>
<body class="theme-zen">
    {sidebar_html}
    <main class="main-content" id="main-content">
        {top_bar_html}
        {header_html}
        {top10_html}
        {''.join(body_html)}
    </main>
    <script>
        function toggleSidebar() {{
            const sidebar = document.getElementById('sidebar');
            const main = document.getElementById('main-content');
            if (window.innerWidth > 1024) {{
                sidebar.classList.toggle('collapsed');
                main.classList.toggle('expanded');
            }} else {{
                sidebar.classList.toggle('mobile-open');
            }}
        }}
        function toggleTheme() {{
            const html = document.documentElement;
            const body = document.body;
            if (html.classList.contains('dark') || body.classList.contains('dark')) {{
                html.classList.remove('dark');
                body.classList.remove('dark');
                body.classList.add('theme-zen');
                localStorage.setItem('theme', 'light');
            }} else {{
                html.classList.add('dark');
                body.classList.add('dark');
                body.classList.remove('theme-zen');
                localStorage.setItem('theme', 'dark');
            }}
        }}
        (function() {{
            const saved = localStorage.getItem('theme');
            if (saved === 'dark') {{
                document.documentElement.classList.add('dark');
                document.body.classList.add('dark');
                document.body.classList.remove('theme-zen');
            }}

            const navLinks = Array.from(document.querySelectorAll('.nav-menu .nav-link'));
            const sections = Array.from(document.querySelectorAll('main section[id^="section-"]'));
            if (!navLinks.length || !sections.length) return;

            const revealInSidebar = (link) => {{
                if (!link) return;
                const details = link.closest('details');
                if (details && !details.open) details.open = true;
                const navMenu = link.closest('.nav-menu');
                if (!navMenu) return;
                const menuRect = navMenu.getBoundingClientRect();
                const linkRect = link.getBoundingClientRect();
                const pad = 12;
                if (linkRect.top < menuRect.top + pad) {{
                    navMenu.scrollBy({{ top: linkRect.top - menuRect.top - pad, behavior: 'smooth' }});
                }} else if (linkRect.bottom > menuRect.bottom - pad) {{
                    navMenu.scrollBy({{ top: linkRect.bottom - menuRect.bottom + pad, behavior: 'smooth' }});
                }}
            }};

            const isNearBottom = () =>
                window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 40;

            if (window.CSS && CSS.supports('scroll-target-group: auto')) {{
                let lastRevealed = null;
                const syncAriaCurrent = () => {{
                    let currentLink = document.querySelector('.nav-menu .nav-link:target-current');
                    if (!currentLink && isNearBottom()) {{
                        currentLink = navLinks[navLinks.length - 1];
                    }}
                    navLinks.forEach((link) => {{
                        const isActive = link === currentLink;
                        link.setAttribute('aria-current', isActive ? 'true' : 'false');
                        if (isActive && link !== lastRevealed) {{
                            lastRevealed = link;
                            revealInSidebar(link);
                        }}
                    }});
                }};
                let rafId = 0;
                const onScroll = () => {{
                    if (rafId) return;
                    rafId = requestAnimationFrame(() => {{
                        rafId = 0;
                        syncAriaCurrent();
                    }});
                }};
                syncAriaCurrent();
                document.addEventListener('scrollend', syncAriaCurrent);
                window.addEventListener('scroll', onScroll, {{ passive: true }});
            }} else if ('IntersectionObserver' in window) {{
                let lastActiveId = null;
                const setActive = (id) => {{
                    if (id === lastActiveId) return;
                    lastActiveId = id;
                    navLinks.forEach((link) => {{
                        const isActive = link.getAttribute('href') === `#${{id}}`;
                        link.classList.toggle('target-current', isActive);
                        link.setAttribute('aria-current', isActive ? 'true' : 'false');
                        if (isActive) revealInSidebar(link);
                    }});
                }};
                setActive(sections[0].id);
                const observer = new IntersectionObserver((entries) => {{
                    if (isNearBottom()) {{
                        setActive(sections[sections.length - 1].id);
                        return;
                    }}
                    entries.forEach((entry) => {{
                        if (entry.isIntersecting) {{
                            setActive(entry.target.id);
                        }}
                    }});
                }}, {{ rootMargin: '-50% 0px -50% 0px', threshold: 0 }});
                sections.forEach((section) => observer.observe(section));
                let rafId = 0;
                window.addEventListener('scroll', () => {{
                    if (rafId) return;
                    rafId = requestAnimationFrame(() => {{
                        rafId = 0;
                        if (isNearBottom()) {{
                            setActive(sections[sections.length - 1].id);
                        }}
                    }});
                }}, {{ passive: true }});
            }}

            // Toggletip support: aria-expanded sync, hint->auto fallback, and hover/position fallback
            const tipBtns = Array.from(document.querySelectorAll('.toggletip-btn[interestfor]'));
            if (tipBtns.length) {{
                const probe = document.createElement('div');
                probe.popover = 'hint';
                if (probe.popover !== 'hint') {{
                    document.addEventListener('beforetoggle', (e) => {{
                        const pop = e.target;
                        if (pop.getAttribute('popover') === 'hint' &&
                            document.querySelector(`[popovertarget="${{CSS.escape(pop.id)}}"]`)) {{
                            if (e.newState === 'open') pop.popover = 'auto';
                        }}
                    }}, true);
                }}
                document.addEventListener('toggle', (e) => {{
                    const pop = e.target;
                    if (!pop.id) return;
                    const isOpen = e.newState === 'open';
                    document.querySelectorAll(`.toggletip-btn[interestfor="${{CSS.escape(pop.id)}}"]`).forEach((btn) => {{
                        btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
                    }});
                }}, true);

                const hasInterest = 'interestForElement' in HTMLButtonElement.prototype;
                const hasAnchorPos = window.CSS && CSS.supports('position-area: block-start');
                const positionPopover = (btn, pop) => {{
                    if (hasAnchorPos) return;
                    const r = btn.getBoundingClientRect();
                    pop.style.position = 'fixed';
                    pop.style.left = `${{Math.max(12, Math.min(window.innerWidth - 332, r.right - 280))}}px`;
                    const topAbove = r.top - pop.offsetHeight - 8;
                    pop.style.top = `${{topAbove >= 12 ? topAbove : r.bottom + 8}}px`;
                }};

                tipBtns.forEach((btn) => {{
                    const pop = document.getElementById(btn.getAttribute('interestfor'));
                    if (!pop) return;
                    btn.setAttribute('aria-expanded', 'false');
                    pop.addEventListener('toggle', (e) => {{
                        if (e.newState === 'open') positionPopover(btn, pop);
                    }});
                    if (!hasInterest && typeof pop.showPopover === 'function') {{
                        let hideTimer = 0;
                        const openTip = () => {{
                            clearTimeout(hideTimer);
                            if (!pop.matches(':popover-open')) pop.showPopover();
                            positionPopover(btn, pop);
                        }};
                        const scheduleClose = () => {{
                            clearTimeout(hideTimer);
                            hideTimer = setTimeout(() => {{
                                if (pop.matches(':popover-open')) pop.hidePopover();
                            }}, 140);
                        }};
                        btn.addEventListener('mouseenter', openTip);
                        btn.addEventListener('focus', openTip);
                        btn.addEventListener('mouseleave', scheduleClose);
                        btn.addEventListener('blur', scheduleClose);
                        pop.addEventListener('mouseenter', () => clearTimeout(hideTimer));
                        pop.addEventListener('mouseleave', scheduleClose);
                    }}
                }});
            }}
        }})();
    </script>
</body>
</html>
"""
