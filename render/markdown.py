"""Markdown report rendering module."""

import re
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


def _fmt_chart_md(chart: Optional[Any]) -> str:
    """Format markdown image link for a chart."""
    if chart is None or not getattr(chart, "is_available", False) or not getattr(chart, "file_path", None):
        return ""
    # Use basename or relative path
    path = str(chart.file_path)
    return f"\n![{chart.title}]({path})\n*{chart.caption}*\n"


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


def _esc_md_dollar(text: Any) -> str:
    """Escape unescaped literal $ signs in Markdown so KaTeX math mode is not triggered."""
    if text is None:
        return ""
    return re.sub(r"(?<!\\)\$", r"\\$", str(text))


def _extract_hero_kpis(sections: List[SectionResult]) -> Dict[str, str]:
    """Extract top 6 Hero KPI metrics for the executive brief strip."""
    spx_str = "n/a"
    ndx_str = "n/a"
    y10_str = "n/a"
    vix_str = "n/a"
    br_str = "n/a"
    posture_str = "n/a"

    for s in sections:
        data = getattr(s, "data", None)
        if isinstance(data, IndexSnapshotData):
            for r in getattr(data, "rows", []):
                if r.symbol == "^GSPC":
                    spx_str = f"{r.close:,.2f} ({r.pct_change:+.2f}%)"
                elif r.symbol == "^NDX":
                    ndx_str = f"{r.close:,.2f} ({r.pct_change:+.2f}%)"
            if y10_str == "n/a" and getattr(data, "cross_asset_line", None):
                m_ca = re.search(r"10Y Yield\s+([0-9\.]+%)(?:\s*\(([^)]+)\))?", data.cross_asset_line)
                if m_ca:
                    val_part = m_ca.group(1)
                    d_part = f" ({m_ca.group(2)})" if m_ca.group(2) else ""
                    y10_str = f"{val_part}{d_part}"
        elif isinstance(data, SwingGateData):
            if posture_str == "n/a" and getattr(data, "posture", None):
                posture_str = f"**{data.posture}**"
            if vix_str == "n/a" and getattr(data, "vix_level", None) is not None:
                d_pts = getattr(data, "vix_5d_trend", None)
                d_str = f" ({d_pts:+.2f})" if d_pts is not None else ""
                vix_str = f"{data.vix_level:.2f}{d_str}"
            if br_str == "n/a" and getattr(data, "breadth_pct_above_50dma", None) is not None:
                br_str = f"{data.breadth_pct_above_50dma:.1f}%"
        elif isinstance(data, VolatilityData):
            d_pts = getattr(data, "change", None)
            d_str = f" ({d_pts:+.2f})" if d_pts is not None else ""
            vix_str = f"{data.current_level:.2f}{d_str}"
        elif isinstance(data, BreadthSectionData):
            p50 = getattr(data.result, "pct_above_50_dma", None)
            cur = getattr(p50, "current", None) if p50 is not None else None
            if cur is not None:
                br_str = f"{cur:.1f}%"
        elif isinstance(data, MacroSectionData):
            if getattr(data, "fed_rates", None):
                y = data.fed_rates.yield_10y
                dy = data.fed_rates.yield_10y_delta * 100.0
                y10_str = f"{y:.2f}% ({dy:+.1f} bps)"
        elif isinstance(data, WhatChangedData):
            if y10_str == "n/a":
                for d in getattr(data, "deltas", []):
                    if d.name == "10-Year Treasury Yield":
                        d_sign = "+" if (d.delta_str and not d.delta_str.startswith(("+", "-", "—"))) else ""
                        d_fmt = f" ({d_sign}{d.delta_str})" if d.delta_str and d.delta_str != "—" else ""
                        y10_str = f"{d.current_val}{d_fmt}"
        elif isinstance(data, ExecutiveSummaryData):
            if posture_str == "n/a" and getattr(data, "posture", None):
                posture_str = f"**{data.posture}**"

    return {
        "spx": spx_str,
        "ndx": ndx_str,
        "y10": y10_str,
        "vix": vix_str,
        "breadth": br_str,
        "posture": posture_str,
    }


def _render_scenario_matrix_md(mat: Any) -> List[str]:
    """Render a JPMorgan-style 3-branch Event Scenario Matrix in Markdown."""
    out: List[str] = []
    out.append(
        f"**{_esc_md_dollar(mat.event_title)}** — `{mat.event_date}` ({mat.event_time_et}) "
        f"| Impact: `{mat.importance}` | Prior: `{mat.prior_display}` | Cons.: `{mat.consensus_display}`"
    )
    out.append("| Scenario Branch | Condition Tier | Trigger / Condition | Options-Implied SPY Move | Rates & USD | Historically sensitive sectors |")
    out.append("| :--- | :---: | :--- | :---: | :--- | :--- |")
    for o in mat.outcomes:
        out.append(
            f"| **{o.scenario}** | `{o.probability_hint}` | {_esc_md_dollar(o.trigger_threshold)} | "
            f"`{_esc_md_dollar(o.spy_reaction)}` | {_esc_md_dollar(o.rates_usd_reaction)} | {_esc_md_dollar(o.sector_playbook)} |"
        )
    if getattr(mat, "desk_note", ""):
        out.append(f"*{_esc_md_dollar(mat.desk_note)}*")
    out.append("")
    return out


def _md_anchor(sec_num: int, title: str) -> str:
    """Build single-hyphen Markdown heading anchor matching rendered section heading."""
    raw = f"{sec_num} {title}".lower()
    return "#" + re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def render_markdown(
    period: PeriodInfo,
    sections: List[SectionResult],
    charts: Optional[Dict[str, Any]] = None,
) -> str:
    """Render complete report in standard GitHub-flavored Markdown."""
    lines: List[str] = []
    sec_by_type = {type(s.data): s for s in sections if s.is_available and s.data is not None}
    vix_period_label = (
        "1-Day Change" if period.report_type == "daily"
        else ("1-Week Change" if period.report_type == "weekly" else "1-Month Change")
    )

    # Title & Metadata
    rep_title = f"{period.report_type.capitalize()} US Market Report"
    lines.append(f"# {rep_title}")
    lines.append(f"**As of:** `{period.session_date}` | **Period Window:** `{period.current_start}` to `{period.current_end}` | **Evaluation:** US/Eastern")
    if period.is_partial:
        lines.append(f"> ℹ️ **Partial Period Notice:** {period.notes}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for sec in sections:
        lines.append(f"## {sec.section_num}. {sec.title}")

        if not sec.is_available:
            err = sec.error_message or "Data source temporarily unavailable."
            lines.append(f"> ⚠️ **Section Data Unavailable:** {err}\n")
            continue

        # Section: Executive Summary
        if isinstance(sec.data, ExecutiveSummaryData):
            d: ExecutiveSummaryData = sec.data
            v_badge = f"**{d.verdict.upper()}**"
            posture_str = f" | Stance: **{d.posture}**" if getattr(d, "posture", "") else ""
            conf_str = f" | {d.confidence}" if d.confidence else ""
            lines.append(f"### Verdict: {v_badge} | Trend Score: `{d.normalized_score:+.2f}`{posture_str}{conf_str}")
            lines.append(f"*Model Weight Normalization: Active category weight `{d.max_score:.2f}` on [-1.0, +1.0] scale.*\n")

            is_daily = (period.report_type == "daily")

            if is_daily:
                # Substage 4.1: Compact Hero KPI Strip in Daily Markdown
                kpis = _extract_hero_kpis(sections)
                lines.append("| S&P 500 | Nasdaq-100 | 10Y Yield | CBOE VIX | Breadth >50-DMA | Model Posture |")
                lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
                lines.append(f"| {kpis['spx']} | {kpis['ndx']} | {kpis['y10']} | {kpis['vix']} | {kpis['breadth']} | {kpis['posture']} |\n")

            brief = getattr(d, "brief", None)
            if brief is not None:
                if is_daily:
                    takeaway_lbl = "Morning Brief Takeaway"
                elif period.report_type == "weekly":
                    takeaway_lbl = "Weekly Desk Takeaway"
                else:
                    takeaway_lbl = "Executive Brief Takeaway"
                lines.append(f"> ⚡ **{takeaway_lbl}:** {_esc_md_dollar(brief.headline_takeaway)}\n")
                lines.append("#### 3 Things That Mattered")
                for item in brief.three_things:
                    lines.append(f"- {_esc_md_dollar(item)}")
                lines.append("")
                if is_daily:
                    lines.append("#### What Happened → What It Means → Model Read")
                    lines.append(f"- **What Happened:** {_esc_md_dollar(brief.what_happened)}")
                    lines.append(f"- **What It Means:** {_esc_md_dollar(brief.what_it_means)}")
                    lines.append(f"- **The Bottom Line (Model Read):** {_esc_md_dollar(brief.bottom_line)}")
                else:
                    lines.append("#### What Happened → What It Means → The Bottom Line")
                    lines.append(f"- **What Happened:** {_esc_md_dollar(brief.what_happened)}")
                    lines.append(f"- **What It Means:** {_esc_md_dollar(brief.what_it_means)}")
                    lines.append(f"- **The Bottom Line:** {_esc_md_dollar(brief.bottom_line)}")
                lines.append("")

            # Substage 4.2: Chart of the Day (Daily)
            cotd = getattr(d, "chart_of_the_day", None)
            if cotd is not None and is_daily:
                lines.append(f"### {cotd.title}")
                lines.append(f"![{cotd.title}]({cotd.chart_path})")
                lines.append(f"*{cotd.annotation}*\n")

            if period.report_type in ("daily", "weekly"):
                lines.append("<details><summary>Detail & Methodology</summary>\n")

            if getattr(d, "alignment_note", ""):
                lines.append(f"> ⚖️ **Regime vs. Posture Alignment:** {_esc_md_dollar(d.alignment_note)}\n")
            if d.forward_outlook:
                reg_name = d.regime_summary or "Active Macro Regime"
                lines.append(f"> 🧭 **Macro Regime & Forward Implications ({reg_name}):** {_esc_md_dollar(d.forward_outlook)}\n")
            lines.append("**Key Drivers & Rationale:**")
            for r in d.reasons:
                lines.append(f"- {_esc_md_dollar(r)}")
            lines.append("")
            h_matrix = getattr(d, "horizon_matrix", None)
            if h_matrix:
                lines.append("#### Multi-Horizon & Cross-Strategy Synthesis")
                lines.append("| Horizon | Primary Lens | Regime Read | Quantitative Anchor | Cross-Market Takeaway |")
                lines.append("| :--- | :--- | :--- | :--- | :--- |")
                for h in h_matrix:
                    lines.append(
                        f"| **{h.horizon}** | {h.audience} | `{_esc_md_dollar(h.state)}` | "
                        f"{_esc_md_dollar(h.key_metric)} | {_esc_md_dollar(h.implication)} |"
                    )
                lines.append("")

            if period.report_type in ("daily", "weekly"):
                lines.append("</details>\n")

        # Section: What Changed (Executive Deltas vs Prior Report)
        elif isinstance(sec.data, WhatChangedData):
            w_data: WhatChangedData = sec.data
            n_deltas = len(w_data.deltas)
            delta_word = f"{n_deltas} largest" if n_deltas > 0 else "key"
            lines.append(f"> ℹ️ **Executive Delta Tracking:** Highlights the {delta_word} macroeconomic and market shifts versus the prior report.\n")
            if w_data.single_biggest_risk:
                lines.append(f"> ⚠️ **Single Biggest Forward Risk:** {_esc_md_dollar(w_data.single_biggest_risk)}\n")
            lines.append("| Shifted Indicator | Current Level / Status | Prior Level | Period Δ | Strategic Context |")
            lines.append("| :--- | :---: | :---: | :---: | :--- |")
            for d in w_data.deltas:
                lines.append(f"| **{d.name}** | {_esc_md_dollar(d.current_val)} | {_esc_md_dollar(d.prior_val)} | `{_esc_md_dollar(d.delta_str)}` | {_esc_md_dollar(d.detail)} |")
            lines.append("")

        # Section: Market Regime & Risk Appetite
        elif isinstance(sec.data, SwingGateData):
            sg: SwingGateData = sec.data
            lines.append(f"### Market Regime & Risk Appetite: **{sg.posture}**")
            lines.append(f"> 🎯 **Triggered Rule:** {_esc_md_dollar(sg.triggered_rule)}\n")
            lines.append(f"> 🧭 **Model Regime Context:** {_esc_md_dollar(sg.execution_guidance)}\n")
            adh = getattr(sg, "range_adherence", None)
            if adh is not None and period.report_type == "weekly":
                lines.append(f"> 🛡️ **Prior Week Model Range Adherence:** {_esc_md_dollar(adh.summary_text)}\n")
            if period.report_type == "monthly":
                macro_sec = sec_by_type.get(MacroSectionData)
                macro_ref = f"Section {macro_sec.section_num}" if macro_sec else "macro regime signals"
                lines.append(f"> 💡 **Monthly Strategic Allocation Note:** This tactical regime gauge summarizes short-to-medium-term trend and volatility thresholds. For long-term multi-asset allocators, macro regime signals and credit spreads ({macro_ref}) take precedence over short-term moving-average signals.\n")
            if period.report_type in ("daily", "weekly"):
                lines.append("<details><summary>Technical Benchmark Moving Average Gates</summary>\n")
            lines.append("#### Technical Benchmark Moving Average Gates")
            lines.append("| Index Benchmark | Current Close | 20-Day EMA | Dist to 20-EMA | 20-EMA Status | 50-Day DMA | Dist to 50-DMA | 50-DMA Status |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            spy_20_stat = "🟢 PASS (Above)" if sg.spy_20ema_pass else "🔴 FAIL (Below)"
            spy_50_stat = "🟢 PASS (Above)" if sg.spy_50dma_pass else "🔴 FAIL (Below)"
            qqq_20_stat = "🟢 PASS (Above)" if sg.qqq_20ema_pass else "🔴 FAIL (Below)"
            qqq_50_stat = "🟢 PASS (Above)" if sg.qqq_50dma_pass else "🔴 FAIL (Below)"
            lines.append(f"| **S&P 500 ETF (SPY)** | \\${sg.spy_price:,.2f} | \\${sg.spy_20ema:,.2f} | `{sg.spy_20ema_dist_pct:+.2f}%` | {spy_20_stat} | \\${sg.spy_50dma:,.2f} | `{sg.spy_50dma_dist_pct:+.2f}%` | {spy_50_stat} |")
            lines.append(f"| **Nasdaq-100 ETF (QQQ)** | \\${sg.qqq_price:,.2f} | \\${sg.qqq_20ema:,.2f} | `{sg.qqq_20ema_dist_pct:+.2f}%` | {qqq_20_stat} | \\${sg.qqq_50dma:,.2f} | `{sg.qqq_50dma_dist_pct:+.2f}%` | {qqq_50_stat} |")
            lines.append("")
            if period.report_type in ("daily", "weekly"):
                lines.append("</details>\n")
            lines.append("#### Risk Appetite & Market Regime Diagnostics")
            lines.append(f"- **VIX Index Level:** `{sg.vix_level:.2f}` ({vix_period_label}: `{sg.vix_5d_trend:+.2f}` pts)")
            lines.append(f"- **VIX Term Structure:** `{sg.vix_term_structure}`")
            lines.append(f"- **S&P 500 Stocks > 50 DMA:** `{sg.breadth_pct_above_50dma:.1f}%`")
            lines.append(f"- **Advance / Decline Confirmation:** `{sg.breadth_ad_state}`")
            lines.append(f"- **52-Week Net New Highs:** `{sg.net_new_highs:+d}`")
            lines.append("")

            pt = getattr(sg, "posture_triggers", None)
            if pt is not None:
                if period.report_type in ("daily", "weekly"):
                    lines.append("<details><summary>Exposure Sizing & Regime Flip Triggers</summary>\n")
                lines.append("#### Exposure Sizing & Regime Flip Triggers")
                lines.append(f"- **Model Regime Tier:** `{pt.target_beta_band}`")
                lines.append(f"- **Model upgrade condition:** {_esc_md_dollar(pt.upgrade_trigger)}")
                lines.append(f"- **Model downgrade condition:** {_esc_md_dollar(pt.downgrade_trigger)}")
                lines.append("")
                if period.report_type in ("daily", "weekly"):
                    lines.append("</details>\n")

            rr_list = getattr(sg, "risk_ranges", None)
            if rr_list:
                if period.report_type in ("daily", "weekly"):
                    lines.append("<details><summary>Quantitative Risk Ranges™ & Duration Matrix (Trade / Trend / Tail)</summary>\n")
                lines.append("#### Quantitative Risk Ranges™ & Duration Matrix (Trade / Trend / Tail)")
                lines.append("| Asset | Last | Range Low (Support) | 20-EMA Pivot | Range High (Resistance) | Range % | 14D ATR% | Trade (1–3W) | Trend (3M) | Tail (1Y+) |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
                for rr in rr_list:
                    lines.append(
                        f"| **{rr.name}** (`{rr.symbol}`) | \\${rr.last_price:,.2f} | `\\${rr.range_low:,.2f}` | "
                        f"\\${rr.pivot_20ema:,.2f} | `\\${rr.range_high:,.2f}` | {rr.position_pct:.0f}% | "
                        f"{rr.atr_pct:.2f}% | `{rr.trade_dir}` | `{rr.trend_dir}` | `{rr.tail_dir}` |"
                    )
                lines.append("")
                if period.report_type in ("daily", "weekly"):
                    lines.append("</details>\n")

            pb = getattr(sg, "catalyst_playbook", None)
            if pb is not None:
                lines.append(f"#### Next 48H Catalyst Playbook (Through {pb.lookahead_date})")
                lines.append(f"> ⚡ **Key Tactical Levels:** {_esc_md_dollar(pb.spy_pivot_line)}\n")
                if pb.execution_takeaway:
                    lines.append(f"- **Desk Playbook Read:** {_esc_md_dollar(pb.execution_takeaway)}")
                if pb.macro_catalysts:
                    lines.append("\n**Imminent Scheduled Macro Releases**")
                    lines.append("| Date | Time (ET) | Event | Impact | Prior | Consensus |")
                    lines.append("| :---: | :---: | :--- | :---: | :---: | :---: |")
                    for ev in pb.macro_catalysts:
                        imp_tag = "🔴 HIGH" if ev.importance == "high" else ("🟡 MEDIUM" if ev.importance == "medium" else "⚪ LOW")
                        lines.append(
                            f"| `{ev.date}` | {ev.time_et} | **{ev.title}** | {imp_tag} | "
                            f"`{_fmt_ev_value(ev.previous, ev.unit)}` | `{_fmt_ev_value(ev.forecast, ev.unit)}` |"
                        )
                    lines.append("")
                if pb.earnings_tonight_tomorrow:
                    lines.append("**Notable S&P 500 Earnings (Tonight AMC / Tomorrow BMO)**")
                    lines.append("| Symbol | Company | Release Date | Timing |")
                    lines.append("| :--- | :--- | :---: | :---: |")
                    for er in pb.earnings_tonight_tomorrow:
                        lines.append(f"| **{er['symbol']}** | {er['name']} | `{er['date']}` | {er['timing']} |")
                    lines.append("")
                if pb.featured_scenario is not None:
                    if period.report_type == "daily":
                        lines.append("<details><summary>Featured Event Scenario Matrix (\"If-Then\" Desk Playbook)</summary>\n")
                    lines.append("**Featured Event Scenario Matrix (\"If-Then\" Desk Playbook)**\n")
                    lines.extend(_render_scenario_matrix_md(pb.featured_scenario))
                    if period.report_type == "daily":
                        lines.append("</details>\n")

        # Section: Index Snapshot
        elif isinstance(sec.data, IndexSnapshotData):
            snap_data: IndexSnapshotData = sec.data
            lines.append("| Index | Symbol | Close | Period Return | Vol vs 20d Avg | Prev Period Return | Period Δ |")
            lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
            for r in snap_data.rows:
                vol_str = r.volume_ratio_display if hasattr(r, "volume_ratio_display") and r.volume_ratio_display else (f"{r.volume_ratio:.2f}x" if r.volume_ratio > 0.01 else "—")
                lines.append(
                    f"| **{r.name}** | `{r.symbol}` | {r.close:,.2f} | **{r.pct_change:+.2f}%** | {vol_str} | {r.prev_period_change:+.2f}% | {r.delta_vs_prev_period:+.2f}% |"
                )
            lines.append("")
            if getattr(snap_data, "valuation", None) and snap_data.valuation.forward_pe is not None:
                v = snap_data.valuation
                pe_disp = f"{v.forward_pe:.2f}x"
                nom_disp = f"{v.nominal_erp_bps:+.1f} bps" if v.nominal_erp_bps is not None else "—"
                real_disp = f"{v.real_erp_bps:+.1f} bps" if v.real_erp_bps is not None else "—"
                trend_disp = f" | EPS Trend: `{v.eps_revision_trend}`" if v.eps_revision_trend and v.eps_revision_trend != "n/a (source unavailable)" else ""
                lines.append(f"> 💡 **S&P 500 Valuation vs. Rates:** Forward P/E: `{pe_disp}` | Nominal ERP: `{nom_disp}` | Real ERP: `{real_disp}`{trend_disp}\n")

            # Substage 4.4: Compact one-row Sector Line & Cross-Asset Line in Daily
            if getattr(snap_data, "sector_line", None):
                lines.append(f"{snap_data.sector_line}\n")
            if getattr(snap_data, "cross_asset_line", None):
                lines.append(f"{snap_data.cross_asset_line}\n")

        # Section: Technical Indicators
        elif isinstance(sec.data, TechnicalsData):
            t_data: TechnicalsData = sec.data
            for snap in t_data.snapshots:
                lines.append(f"### {snap.name} (`{snap.symbol}`)")
                lines.append(f"- **Current Close:** {snap.price:,.2f}")
                lines.append(f"- **20 DMA:** {snap.dma_20:,.2f} ({snap.price_vs_dma20_pct:+.2f}%) | **50 DMA:** {snap.dma_50:,.2f} ({snap.price_vs_dma50_pct:+.2f}%) | **200 DMA:** {snap.dma_200:,.2f} ({snap.price_vs_dma200_pct:+.2f}%)")
                lines.append(f"- **50 / 200 DMA Cross Status:** `{snap.cross_50_200}`")
                lines.append(f"- **RSI (14):** `{snap.rsi:.1f}` (Prev: {snap.prev_rsi:.1f}, Δ: {snap.delta_rsi:+.1f}) — *{snap.rsi_flag}*")
                if period.report_type == "weekly":
                    lines.append(f"<details><summary>{snap.name} MACD & ADX Trend Oscillators</summary>\n")
                lines.append(f"- **MACD (12,26,9):** Line `{snap.macd:.2f}` | Signal `{snap.macd_signal:.2f}` | Hist `{snap.macd_hist:+.2f}` (Prev MACD: {snap.prev_macd:.2f}, Δ: {snap.delta_macd:+.2f}) — *{snap.macd_status}*")
                lines.append(f"- **ADX (14):** `{snap.adx:.1f}` (+DI: {snap.plus_di:.1f}, -DI: {snap.minus_di:.1f}, Prev: {snap.prev_adx:.1f}, Δ: {snap.delta_adx:+.1f}) — *{snap.trend_strength}*")
                if period.report_type == "weekly":
                    lines.append("</details>\n")
                lines.append("")
            if charts:
                spx_c = _fmt_chart_md(charts.get("spx_technicals"))
                if spx_c:
                    lines.append(spx_c)
                ndx_c = _fmt_chart_md(charts.get("ndx_technicals"))
                if ndx_c:
                    lines.append(ndx_c)

        # Section: Volatility
        elif isinstance(sec.data, VolatilityData):
            vd: VolatilityData = sec.data
            vix_pctile_txt = f" · 1Y Rank: `{format_percentile_ordinal(getattr(vd, 'vix_pctile_1y', None))}`" if getattr(vd, "vix_pctile_1y", None) is not None else ""
            lines.append(f"- **VIX Level:** `{vd.current_level:.2f}` (Prev Period: {vd.prev_level:.2f}{vix_pctile_txt})")
            lines.append(f"- **Period Change:** `{vd.change:+.2f}` points ({vd.pct_change:+.2f}%)")
            lines.append(f"- **Implied Daily Move:** `±{vd.implied_daily_move_pct:.2f}%` (Rule of 16)")
            lines.append(f"- **Period Range:** Low `{vd.period_low:.2f}` — High `{vd.period_high:.2f}`")
            lines.append(f"- **Volatility Regime:** **{vd.regime}** — *{vd.regime_description}*")
            lines.append("")
            has_surface = any(
                getattr(vd, attr, None) is not None
                for attr in ("vix_9d", "vix_3m", "vix_6m", "vvix_level", "skew_level", "realized_vol_20d")
            )
            if has_surface:
                lines.append("### Volatility Term Structure & Tail Risk Surface")
                lines.append("| Surface Dimension | Current Reading | Spread / Context | Regime / Signal |")
                lines.append("| :--- | :---: | :---: | :--- |")
                if vd.vix_9d is not None or vd.vix_3m is not None or vd.vix_6m is not None:
                    tenor_parts = []
                    if vd.vix_9d is not None:
                        tenor_parts.append(f"9D `{vd.vix_9d:.2f}`")
                    tenor_parts.append(f"1M `{vd.current_level:.2f}`")
                    if vd.vix_3m is not None:
                        tenor_parts.append(f"3M `{vd.vix_3m:.2f}`")
                    if vd.vix_6m is not None:
                        tenor_parts.append(f"6M `{vd.vix_6m:.2f}`")
                    ratio_parts = []
                    if vd.ratio_9d_vix is not None:
                        ratio_parts.append(f"9D/1M: `{vd.ratio_9d_vix:.3f}`")
                    if vd.ratio_vix_3m is not None:
                        ratio_parts.append(f"1M/3M: `{vd.ratio_vix_3m:.3f}`")
                    if vd.term_slope_pts is not None:
                        ratio_parts.append(f"3M–1M: `{vd.term_slope_pts:+.2f}` pts")
                    lines.append(f"| **Implied Curve (`9D / 1M / 3M / 6M`)** | {' · '.join(tenor_parts)} | {' &bull; '.join(ratio_parts) or '—'} | `{vd.term_structure_label or 'Unavailable'}` |")
                if vd.vvix_level is not None:
                    vv_chg = f"Period Δ: `{vd.vvix_change:+.2f}` pts (Prev `{vd.vvix_prev:.2f}`)" if (vd.vvix_change is not None and vd.vvix_prev is not None) else "—"
                    if getattr(vd, "vvix_pctile_1y", None) is not None:
                        vv_chg += f" &bull; 1Y: `{format_percentile_ordinal(vd.vvix_pctile_1y)}`"
                    lines.append(f"| **CBOE VVIX Index (`^VVIX`)** | `{vd.vvix_level:.2f}` | {vv_chg} | `{vd.vvix_regime}` |")
                if vd.skew_level is not None:
                    sk_chg = f"Period Δ: `{vd.skew_change:+.2f}` pts (Prev `{vd.skew_prev:.2f}`)" if (vd.skew_change is not None and vd.skew_prev is not None) else "—"
                    if getattr(vd, "skew_pctile_1y", None) is not None:
                        sk_chg += f" &bull; 1Y: `{format_percentile_ordinal(vd.skew_pctile_1y)}`"
                    lines.append(f"| **CBOE SKEW Index (`^SKEW`)** | `{vd.skew_level:.2f}` | {sk_chg} | `{vd.skew_regime}` |")
                if vd.realized_vol_20d is not None:
                    rv_bits = []
                    if vd.realized_vol_10d is not None:
                        rv_bits.append(f"10D `{vd.realized_vol_10d:.2f}%`")
                    rv_bits.append(f"20D `{vd.realized_vol_20d:.2f}%`")
                    if vd.realized_vol_60d is not None:
                        rv_bits.append(f"60D `{vd.realized_vol_60d:.2f}%`")
                    vrp_ctx = f"20D VRP (`VIX - RV20`): `{vd.vrp_20d:+.2f}` pts" if vd.vrp_20d is not None else "—"
                    if getattr(vd, "vrp_pctile_1y", None) is not None and vd.vrp_20d is not None:
                        vrp_ctx += f" &bull; 1Y: `{format_percentile_ordinal(vd.vrp_pctile_1y)}`"
                    vrp_lbl = "Elevated Risk Premium" if (vd.vrp_20d is not None and vd.vrp_20d >= 3.0) else ("Inverted / Realized Stress" if (vd.vrp_20d is not None and vd.vrp_20d <= -2.0) else "Balanced Implied/Realized")
                    lines.append(f"| **Realized Volatility Cone (`^GSPC`)** | {' / '.join(rv_bits)} | {vrp_ctx} | `{vrp_lbl}` |")
                lines.append("")
                if getattr(vd, "surface_synthesis", ""):
                    lines.append(f"> 🧭 **Volatility Surface Synthesis:** {_esc_md_dollar(vd.surface_synthesis)}\n")
            if charts:
                vix_c = _fmt_chart_md(charts.get("vix_regime"))
                if vix_c:
                    lines.append(vix_c)

        # Section: Macro & Economic Events (Stage 2 & Stage 3: 3.1)
        elif isinstance(sec.data, MacroSectionData):
            m_data: MacroSectionData = sec.data

            if period.report_type == "weekly":
                if getattr(m_data, "cycle_dashboard", None):
                    lines.append("### 4-Pillar Macro Cycle Dashboard")
                    lines.append("> 🧭 **Macro Cycle Assessment:** Multi-pillar regime synthesis tracking structural trajectory across real growth, inflation, employment, and monetary plumbing.\n")
                    lines.append("| Macro Pillar | Key Anchor Metric | 3M Trajectory | Historical Context | Cycle Regime Signal |")
                    lines.append("| :--- | :--- | :---: | :---: | :--- |")
                    for cp in m_data.cycle_dashboard:
                        lines.append(f"| **{cp.pillar}** | {cp.anchor_metric} | {cp.trajectory_3m} | {cp.percentile_context} | {cp.regime_signal} |")
                    lines.append("")

                lines.append("<details><summary>Full Macro Indicator & Historical Appendix</summary>\n")

            # 1. Standing Macro Dashboard (latest values, Stage 3: 3.1)
            if getattr(m_data, "standing_dashboard", None):
                lines.append("### Macro Dashboard (latest values)")
                if period.report_type == "daily":
                    cpi_row = next((r for r in m_data.standing_dashboard if "CPI" in r.indicator and "YoY" in r.indicator), None)
                    pce_row = next((r for r in m_data.standing_dashboard if "Core PCE" in r.indicator), None)
                    pay_row = next((r for r in m_data.standing_dashboard if "Nonfarm" in r.indicator or "Payrolls" in r.indicator), None)
                    un_row = next((r for r in m_data.standing_dashboard if "Unemployment" in r.indicator), None)
                    gdp_row = next((r for r in m_data.standing_dashboard if "GDP" in r.indicator), None)

                    anchors = []
                    if cpi_row:
                        anchors.append(f"CPI YoY `{cpi_row.latest_value}`")
                    if pce_row:
                        anchors.append(f"Core PCE YoY `{pce_row.latest_value}`")
                    if pay_row:
                        anchors.append(f"Payrolls `{pay_row.latest_value}`")
                    if un_row:
                        anchors.append(f"Unemployment `{un_row.latest_value}`")
                    if gdp_row:
                        anchors.append(f"Real GDP `{gdp_row.latest_value}`")
                    anchor_str = " • ".join(anchors) if anchors else "Inflation and labor anchors stable"

                    lines.append(f"> ℹ️ **Standing Baseline:** No major macro prints released on this session. Key anchors: {anchor_str}. *The complete 16-indicator multi-month tracking matrix and release cadence are published in the Weekly & Monthly reports.*\n")
                else:
                    lines.append("> ℹ️ Standing macroeconomic indicators across Inflation, Labor, and Economic Activity with 3-month trajectories and scheduled catalyst cadence.\n")
                    lines.append("| Indicator | Category | Latest Value | Prior Value | 3M Direction | Signal Interpretation | Ref. Period | Release Date | Next Release | Source |")
                    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
                    for row in m_data.standing_dashboard:
                        lines.append(f"| **{row.indicator}** | {row.category} | `{row.latest_value}` | {row.prior_value} | {row.direction_3m} | `{row.plain_label}` | {row.reference_period} | `{row.release_date}` | `{row.next_release_date}` | {row.source} |")
                    lines.append("")

            # 2. Market-Based Indicators Table (Stage 3: 3.1)
            if getattr(m_data, "market_indicators", None):
                lines.append("### Market-Based Indicators")
                lines.append("> ℹ️ Daily financial series reflecting credit spreads, TIPS real yields, breakeven inflation, and term structure slopes.\n")
                lines.append("| Indicator | Series ID | Latest Level | Prior Level | Period Δ | Regime / Status | Source |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
                for row in m_data.market_indicators:
                    lines.append(
                        f"| **{row.indicator}** | `{row.series_id}` | `{_esc_md_dollar(row.latest_level)}` | "
                        f"{_esc_md_dollar(row.prior_level)} | `{_esc_md_dollar(row.period_change)}` | "
                        f"`{_esc_md_dollar(row.trend_3m)}` | {row.source} |"
                    )
                lines.append("")

            # 3. Key Economic Releases (this period)
            lines.append("### Key Economic Releases (this period)")
            if m_data.recent_releases:
                lines.append("| Event | Reference Period | Release Date | Actual | Forecast | Previous | Surprise |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
                for r in m_data.recent_releases[:10]:
                    act_str = _fmt_macro_val(r.actual, r.unit)
                    fc_str = _fmt_macro_val(r.forecast, r.unit) if r.forecast is not None else "—"
                    pr_str = _fmt_macro_val(r.previous, r.unit)
                    s_str = f"`{r.surprise_flag.upper()}`" if r.surprise_flag not in ("n/a", "n/a (no consensus)", "n/a (unpolled)") else "—"
                    ref_p = r.reference_period or r.date
                    rel_d = r.release_date or r.date
                    lines.append(f"| **{r.title}** | {ref_p} | `{rel_d}` | **{act_str}** | {fc_str} | {pr_str} | {s_str} |")
                lines.append("")
            else:
                lines.append("*No major economic releases published during this period.*\n")

            # 2. Fed Policy & Rates
            fr = m_data.fed_rates
            lines.append("### Federal Reserve & Treasury Yield Curve")
            lines.append(f"- **Effective Fed Funds Rate:** `{fr.fed_funds_rate:.2f}%` (Target Range: `{fr.target_range}`)")
            lines.append(f"- **FOMC Calendar:** Last Meeting `{fr.last_meeting_date}` | Next Meeting: `{fr.next_meeting_date}` ({fr.days_to_next} days away)")
            lines.append(f"- **2-Year Treasury Yield:** `{fr.yield_2y:.2f}%` (Period Δ: `{fr.yield_2y_delta * 100:+.1f}` bps)")
            lines.append(f"- **10-Year Treasury Yield:** `{fr.yield_10y:.2f}%` (Period Δ: `{fr.yield_10y_delta * 100:+.1f}` bps)")
            lines.append(f"- **2s10s Spread:** `{fr.spread_2s10s * 100:+.1f}` bps — **{fr.curve_status}**")
            lines.append("")
            if charts:
                rates_c = _fmt_chart_md(charts.get("rates_yield_curve"))
                if rates_c:
                    lines.append(rates_c)

            # Forward-Looking Financial Conditions & Regime Quadrant
            if m_data.intelligence:
                intel = m_data.intelligence
                has_mkt_ind = bool(getattr(m_data, "market_indicators", None))
                lines.append("### Forward-Looking Financial Conditions & Regime Quadrant")
                lines.append("| Forward Indicator | Current Level | Period Shift | Regime / Status | Strategic Assessment |")
                lines.append("| :--- | :---: | :---: | :---: | :--- |")
                if not has_mkt_ind:
                    lines.append(f"| **High-Yield Credit OAS** | `{intel.credit.spread_bps:.0f} bps` ({intel.credit.spread_pct:.2f}%) | `{intel.credit.delta_bps:+.1f} bps` | `{intel.credit.regime}` | {intel.credit.assessment} |")
                    lines.append(f"| **Real Cost of Capital (10Y TIPS)** | `{intel.rates.real_yield_10y:.2f}%` | `{intel.rates.real_yield_delta_bps:+.1f} bps` | `{intel.rates.driver_attribution}` | {intel.rates.assessment} |")
                    lines.append(f"| **10Y Breakeven Inflation** | `{intel.rates.breakeven_10y:.2f}%` | `{intel.rates.breakeven_delta_bps:+.1f} bps` | Anchored Expectations | 10-year inflation expectations priced by TIPS market. |")
                lines.append(f"| **VIX Term Structure (1M / 3M)** | `{intel.vol_term.ratio:.3f}` (`{intel.vol_term.vix_1m:.2f}` / `{intel.vol_term.vix_3m:.2f}`) | `{intel.vol_term.ratio_delta:+.3f}` | `{intel.vol_term.structure}` | {intel.vol_term.assessment} |")
                inflows_md = ", ".join(intel.rotation.top_inflows[:2]) if intel.rotation.top_inflows else "Defensives"
                outflows_md = ", ".join(intel.rotation.top_outflows[-2:]) if intel.rotation.top_outflows else "Growth"
                lines.append(f"| **Sector Rotation Spread (Cyc vs Def)** | `{intel.rotation.spread:+.2f}%` | Leading: {inflows_md} | `{intel.rotation.posture}` | Lagging: {outflows_md}. {intel.rotation.assessment} |")
                if not has_mkt_ind:
                    ig_md = f"`{intel.credit.ig_spread_bps:.0f} bps`" if intel.credit.ig_spread_bps is not None else "n/a (source unavailable)"
                    ig_d_md = f"`{intel.credit.ig_delta_bps:+.1f} bps`" if intel.credit.ig_delta_bps is not None else "n/a"
                    curve_md = f"`{intel.credit.spread_3m10y_bps:+.0f} bps`" if intel.credit.spread_3m10y_bps is not None else "n/a (source unavailable)"
                    lines.append(f"| **Investment-Grade Credit OAS** | {ig_md} | {ig_d_md} | `Tight / Low Spread` | ICE BofA US Investment Grade OAS (FRED BAMLC0A0CM). |")
                    lines.append(f"| **3M / 10Y Yield Curve Spread** | {curve_md} | Benchmark Spread | `Normal / Un-inverted` | 10-Year Treasury Yield minus 3-Month T-Bill (FRED T10Y3M). |")
                stress_md = f"`{intel.credit.financial_stress_index:.2f}`" if intel.credit.financial_stress_index is not None else "n/a (source unavailable)"
                lines.append(f"| **Financial Stress Index (STLFSI4)** | {stress_md} | `{intel.credit.financial_stress_label}` | `Low Stress` | St. Louis Fed Financial Stress Index (0 = average market stress). |")
                liq_obj = getattr(m_data, "liquidity", None)
                if liq_obj is not None:
                    imp4_md = f"{liq_obj.impulse_4w_b:+,.1f}B" if liq_obj.impulse_4w_b is not None else "n/a"
                    imp13_md = f"{liq_obj.impulse_13w_b:+,.1f}B" if liq_obj.impulse_13w_b is not None else "n/a"
                    lines.append(
                        f"| **Fed Net Liquidity Plumbing** | `\\${liq_obj.net_liquidity_t:.3f}T` | `4W: {imp4_md} \\| 13W: {imp13_md}` | "
                        f"`{liq_obj.regime}` | {_esc_md_dollar(liq_obj.assessment)} |"
                    )
                lines.append("")
                lines.append(f"#### Active Macro Regime Quadrant: **{intel.summary_badge}**")
                lines.append(f"- **Growth Backdrop:** `{intel.growth_signal}` | **Inflation Backdrop:** `{intel.inflation_signal}`")
                lines.append(f"- **Macro Positioning Synthesis:** Growth reflects `{intel.growth_signal}` alongside `{intel.inflation_signal}` inflation, setting an active {intel.rotation.posture.lower()} tone.")
                lines.append("")

            # S&P 500 Valuation vs. Rates & Equity Risk Premium (Stage 3 Task 3.1 & Stage 4 Task 4.2)
            if m_data.valuation and m_data.valuation.forward_pe is not None:
                v = m_data.valuation
                lines.append("### S&P 500 Valuation vs. Rates & Equity Risk Premium")
                if v.assessment:
                    lines.append(f"> 💡 **Valuation & ERP Context:** {v.assessment}\n")
                lines.append("| Valuation Metric | Current Level | Benchmark / Hurdle | Spread / Risk Premium | Sourcing & Context |")
                lines.append("| :--- | :---: | :---: | :---: | :--- |")
                pe_str = f"{v.forward_pe:.2f}x"
                ey_str = f"{v.earnings_yield_pct:.2f}%" if v.earnings_yield_pct is not None else "—"
                nom_hurdle = f"10Y Treasury ({v.nominal_10y_yield:.2f}%)" if v.nominal_10y_yield is not None else "10Y Treasury (DGS10)"
                nom_erp_str = f"`{v.nominal_erp_bps:+.1f} bps`" if v.nominal_erp_bps is not None else "—"
                real_hurdle = f"10Y TIPS ({v.real_10y_yield:.2f}%)" if v.real_10y_yield is not None else "10Y TIPS Real Yield (DFII10)"
                real_erp_str = f"`{v.real_erp_bps:+.1f} bps`" if v.real_erp_bps is not None else "—"
                lines.append(f"| **Forward P/E** | {pe_str} | — | — | Forward multiple on consensus 12M EPS ({v.source}) |")
                lines.append(f"| **Earnings Yield (1/PE)** | {ey_str} | — | — | Equity cash generation yield (1 / Forward P/E) |")
                lines.append(f"| **Nominal ERP** | {ey_str} | {nom_hurdle} | {nom_erp_str} | Earnings Yield minus 10Y Benchmark Yield |")
                lines.append(f"| **Real ERP** | {ey_str} | {real_hurdle} | {real_erp_str} | Earnings Yield minus 10Y TIPS Real Yield |")
                rev_trend = v.eps_revision_trend if v.eps_revision_trend != "n/a (source unavailable)" else "—"
                lines.append(f"| **Forward EPS Revision Trend** | `{rev_trend}` | Consensus Revisions | — | Revision momentum across S&P 500 constituents |")
                lines.append("")

            # Cross-Asset Intelligence Link (Stage 3: 3.6 De-duplicate; link to Section 10 & 12)
            comm_sec = sec_by_type.get(CommoditiesCrossAssetData)
            cal_sec = sec_by_type.get(ForwardCalendarData)
            cross_parts = []
            if comm_sec:
                c_anch = _md_anchor(comm_sec.section_num, comm_sec.title)
                cross_parts.append(f"For detailed commodity price performance, oil, gold, dollar, and bond cross-market alignments, see [Section {comm_sec.section_num}: {comm_sec.title}]({c_anch}).")
            if cal_sec:
                f_anch = _md_anchor(cal_sec.section_num, cal_sec.title)
                cross_parts.append(f"For the forward calendar and macro catalyst horizon, see [Section {cal_sec.section_num}: {cal_sec.title}]({f_anch}).")
            if cross_parts:
                lines.append(f"> 📊 *{' '.join(cross_parts)}*\n")

            # Historical Trends & Trajectory (Monthly Report)
            if m_data.historical_overview and m_data.historical_overview.indicators:
                lines.append("### Key Macroeconomic Indicators — Monthly & Yearly Historical Summary")
                lines.append("| Indicator | Category | Latest Period | Monthly Value | Yearly Value (YoY / 12M) | Prior MoM | Prior YoY | 1-Year Trend |")
                lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
                for ind in m_data.historical_overview.indicators:
                    lp = ind.latest_point
                    pp = ind.prior_point
                    if not lp:
                        continue

                    # Rate levels vs percentage changes
                    if "k" in ind.monthly_unit:
                        m_val_str = f"{lp.monthly_val:+.0f}k"
                        y_val_str = f"{lp.yearly_val:+.0f}k (12M)"
                        pr_m_str = f"{pp.monthly_val:+.0f}k" if pp else "n/a"
                        pr_y_str = f"{pp.yearly_val:+.0f}k (12M)" if pp else "n/a"
                    elif ind.category == "Rates & Policy" or ind.series_id == "UNRATE":
                        m_val_str = f"{lp.monthly_val:.2f}%"
                        y_val_str = f"{lp.yearly_val:+.2f}%"
                        pr_m_str = f"{pp.monthly_val:.2f}%" if pp else "n/a"
                        pr_y_str = f"{pp.yearly_val:+.2f}%" if pp else "n/a"
                    elif "index" in ind.monthly_unit:
                        m_val_str = f"{lp.monthly_val:.1f} index"
                        y_val_str = f"{lp.yearly_val:+.1f} pts"
                        pr_m_str = f"{pp.monthly_val:.1f} index" if pp else "n/a"
                        pr_y_str = f"{pp.yearly_val:+.1f} pts" if pp else "n/a"
                    else:
                        m_val_str = f"{lp.monthly_val:+.2f}{ind.monthly_unit}"
                        y_val_str = f"{lp.yearly_val:+.2f}{ind.yearly_unit}"
                        pr_m_str = f"{pp.monthly_val:+.2f}{ind.monthly_unit}" if pp else "n/a"
                        pr_y_str = f"{pp.yearly_val:+.2f}{ind.yearly_unit}" if pp else "n/a"

                    t_icon = "▲ Rising" if ind.trend_direction == "Rising" else ("▼ Falling" if ind.trend_direction == "Falling" else "▶ Steady")
                    lines.append(
                        f"| **{ind.name}** | {ind.category} | `{lp.label}` | **{m_val_str}** | **{y_val_str}** | {pr_m_str} | {pr_y_str} | `{t_icon}` |"
                    )
                lines.append("")

                # Trailing 6-Month Monthly Values Matrix (excluding quarterly series like GDP)
                anchor = next((ind for ind in m_data.historical_overview.indicators if len(ind.points) >= 6 and ind.series_id != "GDPC1"), None)
                if anchor:
                    matrix_dates = [p.label for p in anchor.points[-6:]]
                    lines.append("### Trailing 6-Month Macro Reporting Matrix")
                    header_cols = ["Indicator", "Metric"] + matrix_dates
                    lines.append("| " + " | ".join(header_cols) + " |")
                    lines.append("| " + " | ".join([":---", ":---:"] + [":---:"] * len(matrix_dates)) + " |")
                    for ind in m_data.historical_overview.indicators:
                        if ind.series_id == "GDPC1" or "Q" in getattr(ind.points[0], "label", ""):
                            continue
                        pt_map = {p.label: p for p in ind.points}
                        row_vals = []
                        for md in matrix_dates:
                            if md in pt_map:
                                p = pt_map[md]
                                if "k" in ind.monthly_unit:
                                    row_vals.append(f"{p.monthly_val:+.0f}k")
                                elif "index" in ind.monthly_unit:
                                    row_vals.append(f"{p.monthly_val:.1f}")
                                elif ind.category == "Rates & Policy" or ind.series_id == "UNRATE":
                                    row_vals.append(f"{p.monthly_val:.2f}%")
                                else:
                                    row_vals.append(f"{p.monthly_val:+.2f}%")
                            else:
                                row_vals.append("—")
                        lines.append(f"| **{ind.name}** | {ind.monthly_unit} | " + " | ".join(row_vals) + " |")
                    lines.append("")

            if period.report_type == "weekly":
                lines.append("</details>\n")

        # Section: Sector Report
        elif isinstance(sec.data, SectorReportData):
            s_data: SectorReportData = sec.data
            w_tag = "[1-Month Strategic View]" if period.report_type == "monthly" else ("[1-Week View]" if period.report_type == "weekly" else "[Daily Tactical View]")
            lines.append(f"### 11 SPDR Sector Relative Strength Matrix {w_tag}")
            lines.append(f"*Benchmark SPY Return: {s_data.spy_pct_change:+.2f}% (Prior: {s_data.spy_prev_pct_change:+.2f}%)*\n")
            lines.append("| Rank | Sector | Symbol | Close | Return | RS vs SPY | Prior Rank | Rank Δ |")
            lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
            for sr in s_data.rows:
                rank_delta_str = f"+{sr.rank_change}" if sr.rank_change > 0 else (str(sr.rank_change) if sr.rank_change < 0 else "0")
                lines.append(
                    f"| **#{sr.rank}** | {sr.name} | `{sr.symbol}` | \\${sr.close:,.2f} | **{sr.pct_change:+.2f}%** | {sr.relative_strength_vs_spy:+.2f}% | #{sr.prev_rank} | {rank_delta_str} |"
                )
            lines.append("")

            # Equity Style, Size & Factor Attribution (Phase 3C)
            style_attr = getattr(s_data, "style_attribution", None)
            if style_attr and getattr(style_attr, "items", None):
                if period.report_type == "weekly":
                    lines.append("<details><summary>Equity Style, Size & Factor Attribution</summary>\n")
                lines.append("### Equity Style, Size & Factor Attribution")
                lines.append("| Factor Pair | Period Spread | 1M Spread | 3M Spread | Style Regime |")
                lines.append("| :--- | :---: | :---: | :---: | :--- |")
                for sf in style_attr.items:
                    s1m = f"`{sf.ret_1m_spread_pct:+.2f}%`" if sf.ret_1m_spread_pct is not None else "—"
                    s3m = f"`{sf.ret_3m_spread_pct:+.2f}%`" if sf.ret_3m_spread_pct is not None else "—"
                    lines.append(f"| **{sf.factor_pair}** | **`{sf.period_spread_pct:+.2f}%`** | {s1m} | {s3m} | `{sf.regime_label}` |")
                lines.append("")
                if style_attr.synthesis:
                    lines.append(f"> 🧭 **Factor Rotation Synthesis (`{style_attr.dominant_style}`):** {_esc_md_dollar(style_attr.synthesis)}\n")
                if period.report_type == "weekly":
                    lines.append("</details>\n")

            # Sector Breadth (Stage 4: 4.2)
            if getattr(s_data, "sector_breadth", None):
                if period.report_type == "weekly":
                    lines.append("<details><summary>Sector Breadth & Moving Average Distribution</summary>\n")
                lines.append("### Sector Breadth & Moving Average Distribution")
                lines.append("| Sector Name | ETF Symbol | % > 50-DMA | % > 20-DMA | Net New Highs (52W) |")
                lines.append("| :--- | :---: | :---: | :---: | :---: |")
                for b_item in s_data.sector_breadth:
                    p50_str = f"`{b_item.pct_above_50dma:.1f}%`" if b_item.pct_above_50dma is not None else "n/a"
                    p20_str = f"`{b_item.pct_above_20dma:.1f}%`" if b_item.pct_above_20dma is not None else "n/a"
                    nh_str = f"`{b_item.net_new_highs:+d}`" if b_item.net_new_highs is not None else "n/a"
                    lines.append(f"| **{b_item.sector}** | `{b_item.symbol}` | {p50_str} | {p20_str} | {nh_str} |")
                lines.append("")
                if period.report_type == "weekly":
                    lines.append("</details>\n")

            # Stage 5 Substage 5.5: Sector Rotation Quadrants & Weekly Candle Structure
            if getattr(s_data, "rotation_quadrants", None):
                lines.append("### Sector Rotation Quadrants & Weekly Candle Structure")
                lines.append("> 🧭 **10-Week RS vs. Momentum Quadrant:** Categorizes SPDR sectors into Leading, Weakening, Lagging, and Improving regimes alongside weekly OHLC candle structure.\n")
                lines.append("| Sector | Symbol | 1W Return | 10W RS vs SPY | RS Momentum | Rotation Quadrant | Weekly Candle Structure |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
                for rq in s_data.rotation_quadrants:
                    q_badge = "🟢 Leading" if rq.quadrant == "Leading" else ("🟡 Weakening" if rq.quadrant == "Weakening" else ("🔴 Lagging" if rq.quadrant == "Lagging" else "🔵 Improving"))
                    lines.append(f"| **{rq.name}** | `{rq.symbol}` | `{rq.ret_1w_pct:+.2f}%` | `{rq.rs_10w_pct:+.2f}%` | `{rq.rs_momentum_pct:+.2f}%` | {q_badge} | {rq.candle_structure} |")
                lines.append("")

            # Sector RS Leaders (Stage 4: 4.2 & Stage 2: 2.2)
            if getattr(s_data, "sector_leaders", None):
                if period.report_type == "weekly":
                    lines.append("<details><summary>Top Relative Strength Leaders & Volume Breakdown</summary>\n")
                lines.append("### Top Relative Strength Leaders (Top 3 Leading Sectors)")
                if getattr(s_data, "defensive_exception_banner", None):
                    lines.append(f"> **{s_data.defensive_exception_banner}**\n")
                if period.report_type == "daily":
                    top_syms = []
                    for s_name, l_list in s_data.sector_leaders.items():
                        top_2 = [f"`{l.symbol}` ({l.name}, RS {l.composite_rs if l.composite_rs != 0.0 else l.rs_vs_spy_pct:+.1f}%)" for l in l_list[:2]]
                        if top_2:
                            top_syms.append(f"**{s_name}:** {', '.join(top_2)}")
                    summary_str = " • ".join(top_syms) if top_syms else "None"
                    lines.append(f"> 💡 **Tactical RS Leaders Snapshot:** {summary_str}. *The complete 21-stock screening matrix and multi-factor breakdown are published in the Weekly & Monthly reports.*\n")
                else:
                    lines.append("> *Ranked by Composite RS (3M & 6M vs. SPY). Screened for stocks holding above 50-DMA, within 15% of 52-week highs, and exceeding \\$20M daily dollar volume.*\n")
                    leader_ret_hdr = "1M Return" if period.report_type == "monthly" else "1W Return"
                    for sec_name, l_list in s_data.sector_leaders.items():
                        lines.append(f"#### Top Leaders: {sec_name}")
                        lines.append(f"| Rank | Ticker | Company Name | Close | Composite RS | {leader_ret_hdr} | Dist 52W High | 20D Vol Ratio | Avg \\$ Volume |")
                        lines.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
                        for rank_idx, leader in enumerate(l_list, 1):
                            if leader.avg_dollar_volume >= 1e9:
                                d_vol_str = f"\\${leader.avg_dollar_volume / 1e9:,.2f}B"
                            else:
                                d_vol_str = f"\\${leader.avg_dollar_volume / 1e6:,.1f}M"
                            crs_val = leader.composite_rs if leader.composite_rs != 0.0 else leader.rs_vs_spy_pct
                            leader_ret_val = leader.pct_change if period.report_type == "monthly" else leader.ret_1w
                            name_display = f"{leader.name} *(Outlier Leader (Defensive Regime))*" if getattr(leader, "is_exception_candidate", False) else leader.name
                            lines.append(f"| **#{rank_idx}** | `{leader.symbol}` | {_esc_md_dollar(name_display)} | \\${leader.close:,.2f} | **{crs_val:+.1f}%** | {leader_ret_val:+.2f}% | {leader.dist_52w_high_pct:+.1f}% | {leader.volume_ratio:.2f}x | {d_vol_str} |")
                        lines.append("")
                if period.report_type == "weekly":
                    lines.append("</details>\n")

            if charts:
                sec_hm = _fmt_chart_md(charts.get("sector_timeframe_heatmap"))
                if sec_hm:
                    lines.append(sec_hm)
                sec_c = _fmt_chart_md(charts.get("sector_performance"))
                if sec_c:
                    lines.append(sec_c)

        # Section: Market Breadth (Stage 3)
        elif isinstance(sec.data, BreadthSectionData):
            b_data: BreadthSectionData = sec.data
            r = b_data.result

            lines.append(f"> **Breadth Read:** *{r.breadth_read}*\n")
            lines.append(f"- **Universe Coverage:** `{r.included_constituents} of {r.total_constituents} S&P 500 constituents` ({r.coverage_pct:.1f}%)")
            if r.coverage_warning:
                lines.append(f"> ⚠️ **Coverage Warning:** {r.coverage_warning}\n")

            # 1. Moving Average Participation
            lines.append("### Moving Average Participation")
            lines.append("| Metric | Current | 1Y %ile | Prev Period | Period Δ | Trend Regime |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
            r20_regime = "Bullish Short-Term" if r.pct_above_20_dma.current >= 60 else ("Bearish Short-Term" if r.pct_above_20_dma.current <= 40 else "Neutral")
            r50_regime = "Bullish Medium-Term" if r.pct_above_50_dma.current >= 60 else ("Bearish Medium-Term" if r.pct_above_50_dma.current <= 40 else "Neutral")
            r200_regime = "Healthy Secular Trend" if r.pct_above_200_dma.current >= 60 else ("Deteriorated Secular Trend" if r.pct_above_200_dma.current <= 40 else "Neutral")
            p50_pctile_md = f"`{format_percentile_ordinal(getattr(r, 'pct_above_50_dma_pctile_1y', None))}`" if getattr(r, "pct_above_50_dma_pctile_1y", None) is not None else "—"
            p200_pctile_md = f"`{format_percentile_ordinal(getattr(r, 'pct_above_200_dma_pctile_1y', None))}`" if getattr(r, "pct_above_200_dma_pctile_1y", None) is not None else "—"
            lines.append(f"| **Stocks > 20 DMA** | **{r.pct_above_20_dma.current:.1f}%** | — | {r.pct_above_20_dma.previous:.1f}% | `{r.pct_above_20_dma.change:+.1f}%` | {r20_regime} |")
            lines.append(f"| **Stocks > 50 DMA** | **{r.pct_above_50_dma.current:.1f}%** | {p50_pctile_md} | {r.pct_above_50_dma.previous:.1f}% | `{r.pct_above_50_dma.change:+.1f}%` | {r50_regime} |")
            lines.append(f"| **Stocks > 200 DMA** | **{r.pct_above_200_dma.current:.1f}%** | {p200_pctile_md} | {r.pct_above_200_dma.previous:.1f}% | `{r.pct_above_200_dma.change:+.1f}%` | {r200_regime} |")
            lines.append("")

            # 2. Advance / Decline
            if period.report_type in ("daily", "weekly"):
                lines.append("<details><summary>Advance / Decline, Secondary Gauges & Historical Analog Study</summary>\n")

            lines.append("### Advance / Decline & Momentum")
            lines.append(f"- **Advancers vs Decliners:** 🟢 `{r.advancers} Advancers` | 🔴 `{r.decliners} Decliners` | ⚪ `{r.unchanged} Unchanged` (Net: `{r.net_advancers:+d}`)")
            lines.append(f"- **Advance/Decline Ratio:** `{r.ad_ratio:.2f}` (Prev: {r.prev_ad_ratio:.2f}, Δ: `{r.delta_ad_ratio:+.2f}`)")
            lines.append(f"- **Cumulative A/D Trend:** `{r.cum_ad_trend}` (Period Net Advances Δ: `{r.delta_cumulative_ad:+,.0f}`) — **{r.ad_confirmation}**")
            lines.append(f"- **Positive Period Return %:** `{r.pct_positive_return.current:.1f}%` of universe (Prev: {r.pct_positive_return.previous:.1f}%, Δ: `{r.pct_positive_return.change:+.1f}%`)")
            lines.append("")

            # 3. New 52-Week Highs vs Lows
            nh_pctile_txt = f" · 1Y Rank: `{format_percentile_ordinal(getattr(r, 'net_new_highs_pctile_1y', None))}`" if getattr(r, "net_new_highs_pctile_1y", None) is not None else ""
            lines.append("### New 52-Week Highs vs Lows")
            lines.append(f"- **New 52-Week Highs:** `{r.new_52w_highs}` | **New 52-Week Lows:** `{r.new_52w_lows}`")
            lines.append(f"- **Net New Highs:** `{r.net_new_highs:+d}` (Prev: {r.prev_net_new_highs:+d}, Δ: `{r.delta_net_new_highs:+d}`{nh_pctile_txt})")
            lines.append("")

            # 4. Equal-Weight vs Cap-Weight Leadership
            lines.append("### Equal-Weight vs Cap-Weight Divergence (RSP vs SPY)")
            lines.append(f"- **Cap-Weight SPY Return:** `{r.spy_return_pct:+.2f}%`")
            lines.append(f"- **Equal-Weight RSP Return:** `{r.rsp_return_pct:+.2f}%`")
            lines.append(f"- **Leadership Spread (SPY - RSP):** `{r.leadership_spread_pct:+.2f}%`")
            lines.append(f"- **Status:** **{r.leadership_status}**")
            lines.append("")

            # 5. McClellan Oscillator
            if r.mcclellan_oscillator is not None:
                mc_pctile_txt = f" · 1Y Rank: `{format_percentile_ordinal(getattr(r, 'mcclellan_pctile_1y', None))}`" if getattr(r, "mcclellan_pctile_1y", None) is not None else ""
                lines.append("### McClellan Oscillator")
                lines.append(f"- **Current Reading:** `{r.mcclellan_oscillator:+.2f}` (Prev: {r.prev_mcclellan_oscillator:+.2f}, Δ: `{r.delta_mcclellan_oscillator:+.2f}`{mc_pctile_txt})")
                lines.append(f"- **Oscillator Status:** **{r.mcclellan_status}**")
                lines.append("")

            # 6. Historical Analog Base-Rate Study (Stage B1 & Stage 3)
            analog = getattr(r, "analog_study", None)
            if analog is not None and getattr(analog, "horizons", None):
                lines.append("### Historical Analog Base-Rate Study")
                badge_part = f" {analog.small_sample_badge}" if getattr(analog, "small_sample_badge", "") else ""
                hist_yrs_str = f" / {analog.history_years}Y history" if getattr(analog, "history_years", 0) > 0 else ""
                lines.append(
                    f"> 📊 **Regime Match (`N={analog.sample_count}` independent setups{hist_yrs_str} / {analog.lookback_days} sessions):**{badge_part} "
                    f"*{analog.condition_summary}* — {_esc_md_dollar(analog.summary_text)}\n"
                )
                lines.append("| Forward Horizon | Analog Win Rate | 95% Wilson CI | Baseline Win % | Analog Median | IQR [P25, P75] | Baseline Median | Analog Mean | Worst Case |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
                for hz in analog.horizons:
                    if getattr(analog, "is_small_sample", False):
                        win_str = f"{hz.win_rate_pct:.1f}%"
                    else:
                        win_str = f"**{hz.win_rate_pct:.1f}%**"
                    ci_str = f"[{hz.win_rate_ci_low:.1f}%, {hz.win_rate_ci_high:.1f}%]" if getattr(hz, "win_rate_ci_high", 0.0) > 0 else "—"
                    iqr_str = f"`{hz.iqr_return:.2f}%` [`{hz.p25_return:+.2f}%`, `{hz.p75_return:+.2f}%`]" if hasattr(hz, "iqr_return") else "—"
                    lines.append(
                        f"| **+{hz.horizon_days} Trading Days ({analog.benchmark_symbol})** | "
                        f"{win_str} | {ci_str} | {hz.baseline_win_rate_pct:.1f}% | "
                        f"`{hz.median_return_pct:+.2f}%` | {iqr_str} | `{hz.baseline_median_pct:+.2f}%` | "
                        f"`{hz.mean_return_pct:+.2f}%` | `{hz.worst_return_pct:+.2f}%` |"
                    )
                if analog.analog_dates:
                    recent_dates = ", ".join(analog.analog_dates[-6:])
                    lines.append(f"\n*Recent non-overlapping analog dates (up to 6 most recent of {analog.sample_count}): {recent_dates}*")
                if getattr(analog, "breadth_cohort_note", ""):
                    lines.append(f"\n> ℹ️ *{_esc_md_dollar(analog.breadth_cohort_note)}*")
                if getattr(analog, "history_years", 0) >= 25:
                    lines.append(f"\n> ℹ️ *Historical lookback span: {analog.history_years} years ({analog.history_start_year}–present), {analog.lookback_days} evaluated sessions across long-history index regime.*")
                lines.append("")

            lines.append("> *Note: Market breadth calculations use the current S&P 500 constituent roster (subject to survivorship bias for historical comparisons).*\n")
            if period.report_type in ("daily", "weekly"):
                lines.append("</details>\n")
            if charts:
                sec_br = _fmt_chart_md(charts.get("sector_breadth_heatmap"))
                if sec_br:
                    lines.append(sec_br)
                br_c = _fmt_chart_md(charts.get("breadth_history"))
                if br_c:
                    lines.append(br_c)

        # Section: Theme & Industry Relative Strength
        elif isinstance(sec.data, ThemeRelativeStrengthData):
            tr: ThemeRelativeStrengthData = sec.data
            if tr.top_theme_summary:
                lines.append(f"> 🏆 **Theme Leadership:** {tr.top_theme_summary}\n")
            bench_text = getattr(tr, "primary_benchmark_label", f"SPY 1-Week ({tr.spy_1w_return:+.2f}%)")
            lines.append(f"*Benchmark: {bench_text}*\n")
            
            is_daily = getattr(tr, "primary_window", "1W") == "1D"
            is_monthly = getattr(tr, "primary_window", "1W") == "1M"
            if is_daily:
                w_label = " [Daily Tactical View]"
            elif is_monthly:
                w_label = " [1-Month Strategic View]"
            else:
                w_label = " [1-Week View]"
            lines.append(f"### Theme & Industry Performance Table{w_label}\n")
            
            if is_daily:
                lines.append("| Rank | Theme / Industry ETF | Symbol | Category | Close | 1D Return (Primary) | 1W Return | 1M Return | RS vs SPY | Rank Δ | 50-DMA Dist |")
            elif is_monthly:
                lines.append("| Rank | Theme / Industry ETF | Symbol | Category | Close | 1M Return (Primary) | 1W Return | 3M Return | RS vs SPY | Rank Δ | 50-DMA Dist |")
            else:
                lines.append("| Rank | Theme / Industry ETF | Symbol | Category | Close | 1W Return (Primary) | 1M Return | 3M Return | RS vs SPY | Rank Δ | 50-DMA Dist |")
            lines.append("| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for item in tr.items:
                if not item.is_available:
                    continue
                rd_str = f"+{item.rank_change}" if item.rank_change > 0 else (str(item.rank_change) if item.rank_change < 0 else "0")
                d50_badge = f"{item.dist_50dma_pct:+.2f}% (Above)" if item.above_50dma else f"{item.dist_50dma_pct:+.2f}% (Below)"
                if is_daily:
                    lines.append(f"| **#{item.rank}** | {item.name} | `{item.symbol}` | {item.category} | \\${item.close:,.2f} | **{item.ret_1d:+.2f}%** | {item.ret_1w:+.2f}% | {item.ret_1m:+.2f}% | {item.rs_vs_spy:+.2f}% | {rd_str} | {d50_badge} |")
                elif is_monthly:
                    lines.append(f"| **#{item.rank}** | {item.name} | `{item.symbol}` | {item.category} | \\${item.close:,.2f} | **{item.ret_1m:+.2f}%** | {item.ret_1w:+.2f}% | {item.ret_3m:+.2f}% | {item.rs_vs_spy:+.2f}% | {rd_str} | {d50_badge} |")
                else:
                    lines.append(f"| **#{item.rank}** | {item.name} | `{item.symbol}` | {item.category} | \\${item.close:,.2f} | **{item.ret_1w:+.2f}%** | {item.ret_1m:+.2f}% | {item.ret_3m:+.2f}% | {item.rs_vs_spy:+.2f}% | {rd_str} | {d50_badge} |")
            lines.append("")
            if tr.commentary:
                lines.append(f"> 💡 **Relative Strength Analysis:** {tr.commentary}\n")

        # Section: Commodities, Currencies & Cross-Asset
        elif isinstance(sec.data, CommoditiesCrossAssetData):
            ca_data: CommoditiesCrossAssetData = sec.data
            lines.append("### Cross-Asset Multi-Market Snapshot")
            lines.append("| Asset Name | Symbol | Current Price | Period Return | Prior Period Return | 3-Month Return |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            for it in ca_data.items:
                if not it.is_available:
                    continue
                pr_str = f"{it.prior_period_ret:+.2f}%" if it.prior_period_ret is not None else "n/a"
                is_fx = "=X" in it.symbol or "/" in it.name or "DXY" in it.name or "DX-Y" in it.symbol
                if is_fx:
                    if "JPY" in it.symbol or "JPY" in it.name:
                        price_display = f"{it.current_price:,.2f}"
                    elif it.current_price < 10.0:
                        price_display = f"{it.current_price:.4f}"
                    else:
                        price_display = f"{it.current_price:,.2f}"
                else:
                    price_display = f"\\${it.current_price:,.2f}"
                lines.append(f"| **{it.name}** | `{it.symbol}` | {price_display} | **{it.period_ret:+.2f}%** | {pr_str} | {it.ret_3m:+.2f}% |")
            lines.append("")
            if ca_data.energy_shock:
                eshk = ca_data.energy_shock
                status_icon = "⚠️ ACTIVE ENERGY SHOCK ALERT" if eshk.is_shock_active else "🟢 Inactive / Normal"
                lines.append("#### Energy Shock & Macro Inflation Pass-Through Indicator")
                lines.append(f"- **Shock Status:** `{status_icon}`")
                lines.append(f"- **Crude Oil 3-Month Trailing Return:** `{eshk.oil_3m_change_pct:+.2f}%`")
                lines.append(f"- **Headline CPI YoY:** `{eshk.headline_cpi_yoy:.2f}%` | **Core CPI YoY:** `{eshk.core_cpi_yoy:.2f}%` | **Gap (Headline - Core):** `{eshk.headline_core_gap_pct:+.2f}%`")
                lines.append(f"> 🧭 **Assessment:** {eshk.assessment}\n")
            if getattr(ca_data, "plumbing", None):
                pl = ca_data.plumbing
                if period.report_type == "weekly":
                    lines.append("<details><summary>Cross-Asset Macro Plumbing (Copper/Gold & Stock-Bond Correlation)</summary>\n")
                lines.append("#### Cross-Asset Macro Plumbing (Copper/Gold vs. 10Y & 60D Stock-Bond Correlation)")
                lines.append("| Cross-Asset Signal | Current Level | Shift / Context | Regime Classification |")
                lines.append("| :--- | :---: | :---: | :--- |")
                if pl.copper_gold_ratio is not None:
                    cg_chg = f"{pl.copper_gold_3m_chg_pct:+.2f}% (3M)" if pl.copper_gold_3m_chg_pct is not None else "n/a"
                    y10_ctx = (
                        f"10Y: {pl.nominal_10y_yield:.2f}% ({pl.yield_10y_13w_bps:+.0f} bps 13W)"
                        if (pl.nominal_10y_yield is not None and pl.yield_10y_13w_bps is not None)
                        else (f"10Y: {pl.nominal_10y_yield:.2f}%" if pl.nominal_10y_yield is not None else "n/a")
                    )
                    lines.append(f"| **Copper / Gold Ratio (x1000)** | `{pl.copper_gold_ratio:.3f}` | `{cg_chg}` &bull; {y10_ctx} | `{pl.copper_gold_regime}` |")
                if pl.stock_bond_corr_60d is not None:
                    prev_c = f"Prior 60D: {pl.stock_bond_corr_prev_60d:+.2f}" if pl.stock_bond_corr_prev_60d is not None else "Prior 60D: n/a"
                    tlt_c = f"TLT 3M: {pl.tlt_3m_ret:+.2f}%" if pl.tlt_3m_ret is not None else ""
                    ctx_sb = " &bull; ".join([p for p in (prev_c, tlt_c) if p])
                    lines.append(f"| **60D Stock-Bond Correlation (SPY / TLT)** | `{pl.stock_bond_corr_60d:+.2f}` | {ctx_sb} | `{pl.stock_bond_regime}` |")
                if pl.assessment:
                    lines.append(f"> 🧭 **Cross-Asset Plumbing Synthesis:** {_esc_md_dollar(pl.assessment)}\n")
                if period.report_type == "weekly":
                    lines.append("</details>\n")

        # Section: Federal Reserve Policy Path
        elif isinstance(sec.data, FedPathData):
            fp: FedPathData = sec.data
            lines.append("### Federal Reserve Monetary Stance & Policy Trajectory")
            hold_p = f"{fp.market_implied_prob_hold * 100:.1f}%" if fp.market_implied_prob_hold is not None else "n/a"
            hike_p = f"{fp.market_implied_prob_hike * 100:.1f}%" if getattr(fp, "market_implied_prob_hike", None) is not None else "0.0%"
            cut_p = f"{fp.market_implied_prob_cut * 100:.1f}%" if fp.market_implied_prob_cut is not None else "0.0%"
            prob_src = getattr(fp, "prob_source", "CBOT 30-Day Fed Funds Futures")

            if period.report_type == "daily" and fp.days_to_next_meeting > 1:
                lines.append(f"> 🏛️ **Fed Policy Stance:** Target Range `{fp.current_target_range}` (Effective: `{fp.effective_fed_funds:.2f}%`) • Next FOMC in **{fp.days_to_next_meeting} calendar days** (`{fp.next_meeting_date}`) • Probabilities: Hold `{hold_p}` | +25 bps Hike: `{hike_p}` | -25 bps Cut: `{cut_p}` (Source: `{prob_src}`) • 2Y Yield: `{fp.yield_2y:.2f}%` ({fp.spread_2y_vs_fed_funds_bps:+.1f} bps vs Fed Funds) • Real Policy Rate: `{fp.real_policy_rate:+.2f}%`.\n")
                lines.append(f"> 🧭 **Monetary Assessment:** {_esc_md_dollar(fp.assessment)}\n")
            elif period.report_type == "weekly":
                lines.append(f"> 🏛️ **Fed Policy Stance:** Target Range `{fp.current_target_range}` (Effective: `{fp.effective_fed_funds:.2f}%`) • Next FOMC in **{fp.days_to_next_meeting} calendar days** (`{fp.next_meeting_date}`) • Probabilities: Hold `{hold_p}` | +25 bps Hike: `{hike_p}` | -25 bps Cut: `{cut_p}` (Source: `{prob_src}`) • 2Y Yield: `{fp.yield_2y:.2f}%` ({fp.spread_2y_vs_fed_funds_bps:+.1f} bps vs Fed Funds) • Real Policy Rate: `{fp.real_policy_rate:+.2f}%`.\n")
                lines.append("<details><summary>Federal Reserve Policy Path & Net Liquidity Details</summary>\n")
                lines.append(f"- **Last FOMC Decision:** `{fp.last_action_date}` — {fp.last_action_description}")
                lines.append(f"- **Fed Funds Target Range:** `{fp.current_target_range}` (Effective Fed Funds Rate: `{fp.effective_fed_funds:.2f}%`)")
                lines.append(f"- **Next FOMC Meeting:** `{fp.next_meeting_date}` ({fp.days_to_next_meeting} calendar days away)")
                if fp.market_implied_prob_hold is not None:
                    lines.append(f"- **Market-Implied Probability for Next Decision:** Hold: `{hold_p}` | +25 bps Hike: `{hike_p}` | -25 bps Cut: `{cut_p}` (Source: `{prob_src}`)")
                else:
                    lines.append("- **Market-Implied Probability for Next Decision:** `n/a (source unavailable)`")
                lines.append(f"- **2-Year Treasury Yield vs. Fed Funds Spread:** `{fp.spread_2y_vs_fed_funds_bps:+.1f} bps` (2Y Yield: `{fp.yield_2y:.2f}%`)")
                lines.append(f"- **Real Policy Stance:** Real Fed Funds Rate `{fp.real_policy_rate:+.2f}%` (vs. {getattr(fp, 'real_rate_driver_measure', 'Core PCE')} `{fp.trailing_core_inflation:.2f}% YoY`)")
                fp_liq = getattr(fp, "liquidity", None)
                if fp_liq is not None:
                    imp4_fp = f"{fp_liq.impulse_4w_b:+,.1f}B" if fp_liq.impulse_4w_b is not None else "n/a"
                    imp13_fp = f"{fp_liq.impulse_13w_b:+,.1f}B" if fp_liq.impulse_13w_b is not None else "n/a"
                    lines.append(
                        f"- **Fed Net Liquidity Plumbing (WALCL - TGA - RRP):** `\\${fp_liq.net_liquidity_t:.3f}T` "
                        f"(Assets `\\${fp_liq.walcl_t:.3f}T` | TGA `\\${fp_liq.tga_b:,.1f}B` | ON RRP `\\${fp_liq.rrp_b:,.1f}B` | "
                        f"4W: `{imp4_fp}` | 13W: `{imp13_fp}` | Regime: `{fp_liq.regime}`)"
                    )
                lines.append(f"> 🧭 **Monetary Assessment:** {_esc_md_dollar(fp.assessment)}\n")
                lines.append("</details>\n")
            else:
                lines.append(f"- **Last FOMC Decision:** `{fp.last_action_date}` — {fp.last_action_description}")
                lines.append(f"- **Fed Funds Target Range:** `{fp.current_target_range}` (Effective Fed Funds Rate: `{fp.effective_fed_funds:.2f}%`)")
                lines.append(f"- **Next FOMC Meeting:** `{fp.next_meeting_date}` ({fp.days_to_next_meeting} calendar days away)")
                if fp.market_implied_prob_hold is not None:
                    lines.append(f"- **Market-Implied Probability for Next Decision:** Hold: `{hold_p}` | +25 bps Hike: `{hike_p}` | -25 bps Cut: `{cut_p}` (Source: `{prob_src}`)")
                else:
                    lines.append("- **Market-Implied Probability for Next Decision:** `n/a (source unavailable)`")
                lines.append(f"- **2-Year Treasury Yield vs. Fed Funds Spread:** `{fp.spread_2y_vs_fed_funds_bps:+.1f} bps` (2Y Yield: `{fp.yield_2y:.2f}%`)")
                lines.append(f"- **Real Policy Stance:** Real Fed Funds Rate `{fp.real_policy_rate:+.2f}%` (vs. {getattr(fp, 'real_rate_driver_measure', 'Core PCE')} `{fp.trailing_core_inflation:.2f}% YoY`)")
                fp_liq = getattr(fp, "liquidity", None)
                if fp_liq is not None:
                    imp4_fp = f"{fp_liq.impulse_4w_b:+,.1f}B" if fp_liq.impulse_4w_b is not None else "n/a"
                    imp13_fp = f"{fp_liq.impulse_13w_b:+,.1f}B" if fp_liq.impulse_13w_b is not None else "n/a"
                    lines.append(
                        f"- **Fed Net Liquidity Plumbing (WALCL - TGA - RRP):** `\\${fp_liq.net_liquidity_t:.3f}T` "
                        f"(Assets `\\${fp_liq.walcl_t:.3f}T` | TGA `\\${fp_liq.tga_b:,.1f}B` | ON RRP `\\${fp_liq.rrp_b:,.1f}B` | "
                        f"4W: `{imp4_fp}` | 13W: `{imp13_fp}` | Regime: `{fp_liq.regime}`)"
                    )
                lines.append(f"> 🧭 **Monetary Assessment:** {_esc_md_dollar(fp.assessment)}\n")

        # Section: Forward Catalyst Calendar
        elif isinstance(sec.data, ForwardCalendarData):
            fc: ForwardCalendarData = sec.data
            cal_label = "Imminent Catalysts (Next 24h–48h)" if period.report_type == "daily" else "Catalyst Horizon"
            lines.append(f"> 📅 **{cal_label}:** Tracking high-impact economic releases through `{fc.lookahead_date}` ({fc.high_impact_count} high-impact events identified).\n")

            if period.report_type == "weekly":
                lines.append("### The Week Ahead (Monday–Friday Catalyst Grid)")
                lines.append("> 📅 **Weekly Macro Catalyst Cadence:** Tracking scheduled releases, auctions, and Fed communication through Friday mapped to cross-asset sensitivity.\n")
                if getattr(fc, "weekly_ahead_events", None):
                    lines.append("| Day / Date | Time (ET) | Release | Impact | Prior | Market Most Sensitive |")
                    lines.append("| :---: | :---: | :--- | :---: | :---: | :--- |")
                    for wev in fc.weekly_ahead_events:
                        lines.append(f"| **{wev.day_date}** | {wev.time_et} | **{wev.release}** | {wev.impact} | {wev.prior} | {wev.market_sensitive} |")
                    lines.append("")
                else:
                    lines.append("*No major economic releases scheduled in the upcoming trading week.*\n")

                if getattr(fc, "weekly_ahead_earnings", None):
                    lines.append("### Earnings on Deck (Notable Large-Cap Reports)")
                    lines.append("> 📊 **Upcoming Corporate Earnings:** Key index constituents reporting during the week.\n")
                    lines.append("| Day / Date | Timing | Symbol | Company Name | Market Cap |")
                    lines.append("| :---: | :---: | :---: | :--- | :---: |")
                    for wea in fc.weekly_ahead_earnings:
                        lines.append(f"| **{wea.day_date}** | {wea.timing} | **`{wea.symbol}`** | {wea.name} | `{wea.market_cap}` |")
                    lines.append("")

                lines.append("<details><summary>Complete Forward Release Horizon & Scenario Matrices</summary>\n")

            if fc.events:
                lines.append("| Date | Time (ET) | Scheduled Catalyst Event | Impact Rating | Prior Value | Consensus Estimate | Schedule Source | Consensus Source |")
                lines.append("| :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |")
                for ev in fc.events:
                    is_speech = any(w in ev.title.lower() for w in ("remarks", "speech", "keynote", "discussion"))
                    is_chair_keynote = "chair keynote" in ev.title.lower()
                    if is_chair_keynote:
                        imp_badge = "🔴 **HIGH (KEYNOTE)**"
                    elif ev.importance == "high":
                        imp_badge = "🔴 **HIGH**"
                    elif ev.importance == "medium":
                        imp_badge = "🟡 MEDIUM"
                    else:
                        imp_badge = "⚪ LOW"

                    if is_speech:
                        pr_val = "—"
                        fc_val = "— (speaker)"
                        cons_src = "—"
                    else:
                        def _fmt_cal_val(val: Optional[float], unit: str) -> str:
                            if val is None:
                                return "—"
                            if not unit or unit == "index":
                                return f"`{val:.2f}`"
                            elif unit == "%":
                                return f"`{val:.2f}%`"
                            elif unit == "k":
                                return f"`{val:,.0f}k`"
                            else:
                                return f"`{val:.2f} {unit}`"

                        fc_val = _fmt_cal_val(ev.forecast, ev.unit) if ev.forecast is not None else "—"
                        pr_val = _fmt_cal_val(ev.previous, ev.unit)
                        raw_cons_src = getattr(ev, "consensus_source", "") or ""
                        cons_src = (
                            raw_cons_src if (ev.forecast is not None and raw_cons_src and raw_cons_src != "—")
                            else ("Official Survey" if ev.forecast is not None else "—")
                        )

                    sched_src = f"[{ev.source_name or ev.source}]({ev.source_url})" if getattr(ev, "source_url", None) else (ev.source or "Official U.S. Schedule")
                    lines.append(f"| `{ev.date}` | {ev.time_et} | **{ev.title}** | {imp_badge} | {pr_val} | {fc_val} | {sched_src} | {cons_src} |")
                lines.append("")
            else:
                lines.append("*No major economic releases scheduled in the upcoming window.*\n")

            if getattr(fc, "scenario_matrices", None):
                lines.append('### Event Scenario Matrix ("If-Then" Reaction Playbook)')
                for mat in fc.scenario_matrices:
                    lines.extend(_render_scenario_matrix_md(mat))

            if period.report_type == "weekly":
                lines.append("</details>\n")

        # Section: Corporate Earnings Layer
        elif isinstance(sec.data, EarningsLayerData):
            el: EarningsLayerData = sec.data
            pct_rep = f"{el.pct_reported:.1f}%" if el.pct_reported is not None else "n/a"
            eps_bt = f"{el.eps_beat_rate:.1f}%" if el.eps_beat_rate is not None else "n/a"
            rev_bt = f"{el.rev_beat_rate:.1f}%" if el.rev_beat_rate is not None else "n/a"
            med_sur = f"{el.median_surprise_pct:+.1f}%" if el.median_surprise_pct is not None else "n/a"

            if period.report_type == "daily":
                lines.append(f"### S&P 500 Corporate Earnings Layer ({el.season_name})")
                lines.append(f"> 📊 **Quarterly Progress:** `{pct_rep}` reported | EPS Beat Rate: `{eps_bt}` | Revenue Beat Rate: `{rev_bt}` | Median Surprise: `{med_sur}`\n")
                if el.status_note:
                    lines.append(f"> ℹ️ {el.status_note}\n")
                if el.reported_last_7_days:
                    lines.append("#### Recent Large-Cap Reports (Last 24h–48h)")
                    lines.append("| Ticker | Company Name | Sector | Report Date | Timing | EPS Surprise | Rev Surprise | Guidance | 1-Day Reaction | Dist 20-DMA | Dist 50-DMA |")
                    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
                    for r in el.reported_last_7_days:
                        eps_s = f"`{r.eps_surprise_pct:+.1f}%`" if r.eps_surprise_pct is not None else "n/a"
                        rev_s = f"`{r.rev_surprise_pct:+.1f}%`" if r.rev_surprise_pct is not None else "n/a"
                        react_s = f"**{r.reaction_1d_pct:+.2f}%**" if r.reaction_1d_pct is not None else "n/a"
                        d20_s = f"{r.dist_20dma_pct:+.2f}%" if r.dist_20dma_pct is not None else "n/a"
                        d50_s = f"{r.dist_50dma_pct:+.2f}%" if r.dist_50dma_pct is not None else "n/a"
                        lines.append(f"| **{r.symbol}** | {r.name} | {r.sector} | `{r.date}` | `{r.timing}` | {eps_s} | {rev_s} | **{r.guidance}** | {react_s} | {d20_s} | {d50_s} |")
                    lines.append("")
                if el.next_week_large_caps:
                    lines.append("#### Imminent Earnings (Tonight AMC & Tomorrow BMO)")
                    lines.append("| Ticker | Company Name | Sector | Market Cap | Scheduled Date | Timing |")
                    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: |")
                    for item in el.next_week_large_caps:
                        lines.append(f"| **{item.symbol}** | {_esc_md_dollar(item.name)} | {item.sector} | {_esc_md_dollar(item.market_cap_str)} | `{item.date}` | `{item.timing}` |")
                    lines.append("")
            else:
                lines.append(f"### S&P 500 Earnings Season Progress ({el.season_name})")
                lines.append(f"- **Progress:** `{pct_rep}` of S&P 500 constituents reported")
                lines.append(f"- **EPS Beat Rate:** `{eps_bt}` | **Revenue Beat Rate:** `{rev_bt}`")
                lines.append(f"- **Median Earnings Surprise:** `{med_sur}`")
                if el.status_note:
                    lines.append(f"> ℹ️ {_esc_md_dollar(el.status_note)}\n")
                recent_earn_hdr = (
                    "Recent Large-Cap Earnings Results (Trailing 30 Days)"
                    if period.report_type == "monthly"
                    else "Recent Large-Cap Earnings Results (Trailing 7 Days)"
                )
                upcoming_earn_hdr = (
                    "Next 30 Days Earnings: Top S&P 500 Large-Cap Catalysts"
                    if period.report_type == "monthly"
                    else "Upcoming Large-Cap Earnings (Top Index Constituents)"
                )
                if period.report_type == "weekly":
                    lines.append("<details><summary>Recent Large-Cap Earnings Results & Extended Horizon</summary>\n")
                if el.reported_last_7_days:
                    lines.append(f"#### {recent_earn_hdr}")
                    lines.append("| Ticker | Company Name | Sector | Report Date | Timing | EPS Surprise | Rev Surprise | Guidance | 1-Day Reaction | Dist 20-DMA | Dist 50-DMA |")
                    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
                    for r in el.reported_last_7_days:
                        eps_s = f"`{r.eps_surprise_pct:+.1f}%`" if r.eps_surprise_pct is not None else "n/a"
                        rev_s = f"`{r.rev_surprise_pct:+.1f}%`" if r.rev_surprise_pct is not None else "n/a"
                        react_s = f"**{r.reaction_1d_pct:+.2f}%**" if r.reaction_1d_pct is not None else "n/a"
                        d20_s = f"{r.dist_20dma_pct:+.2f}%" if r.dist_20dma_pct is not None else "n/a"
                        d50_s = f"{r.dist_50dma_pct:+.2f}%" if r.dist_50dma_pct is not None else "n/a"
                        lines.append(f"| **{r.symbol}** | {_esc_md_dollar(r.name)} | {r.sector} | `{r.date}` | `{r.timing}` | {eps_s} | {rev_s} | **{r.guidance}** | {react_s} | {d20_s} | {d50_s} |")
                    lines.append("")
                if el.next_week_large_caps:
                    lines.append(f"#### {upcoming_earn_hdr}")
                    lines.append("| Ticker | Company Name | Sector | Market Cap | Scheduled Date | Timing |")
                    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: |")
                    for item in el.next_week_large_caps:
                        lines.append(f"| **{item.symbol}** | {_esc_md_dollar(item.name)} | {item.sector} | {_esc_md_dollar(item.market_cap_str)} | `{item.date}` | `{item.timing}` |")
                    lines.append("")
                if period.report_type == "weekly":
                    lines.append("</details>\n")

        # Section: Top Movers
        elif isinstance(sec.data, TopMoversData):
            m_data: TopMoversData = sec.data
            # Display timeframe-appropriate mover sets to eliminate out-of-period clutter
            if period.report_type == "daily":
                display_periods = [(m_data.day, "Daily Movers")]
            elif period.report_type == "weekly":
                display_periods = [(m_data.week, "Weekly Movers")]
            else:  # monthly
                display_periods = [(m_data.month, "Monthly Movers")]

            for pm, title_label in display_periods:
                lines.append(f"### {title_label}")
                if pm.theme_summary:
                    lines.append(f"> 🏷️ **Movers by Theme:** {_esc_md_dollar(pm.theme_summary)}\n")
                all_movers_pm = list(pm.gainers or []) + list(pm.losers or [])
                has_move_causes = any(
                    bool(getattr(m, "catalyst_driver", ""))
                    or getattr(m, "move_cause_tag", None) not in (None, "", "Unknown", "—")
                    for m in all_movers_pm
                )
                lines.append("#### Top 5 Gainers")
                if has_move_causes:
                    lines.append("| Ticker | Company Name | Sector | Mkt Cap | Close | % Change | Vol vs 20d Avg | Earnings in Window? | Technical Context |")
                    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
                else:
                    lines.append("| Ticker | Company Name | Sector | Mkt Cap | Close | % Change | Vol vs 20d Avg | Earnings in Window? |")
                    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
                for g in pm.gainers:
                    earn_flag = "Yes" if g.has_earnings_in_window else "No"
                    mcap_s = _esc_md_dollar(g.market_cap_str or "n/a")
                    row_base = f"| **{g.symbol}** | {_esc_md_dollar(g.name)} | {g.sector} | {mcap_s} | \\${g.close:,.2f} | **{g.pct_change:+.2f}%** | {g.volume_ratio:.2f}x | {earn_flag} |"
                    if has_move_causes:
                        raw_g_cause = getattr(g, "catalyst_driver", "") or (g.move_cause_tag if g.move_cause_tag not in (None, "", "Unknown") else "—")
                        g_cause = _esc_md_dollar(raw_g_cause)
                        row_base += f" {g_cause} |"
                    lines.append(row_base)
                lines.append("")
                if period.report_type in ("daily", "weekly"):
                    lines.append("<details><summary>Top 5 Losers & Secondary Movers Details</summary>\n")
                lines.append("#### Top 5 Losers")
                if has_move_causes:
                    lines.append("| Ticker | Company Name | Sector | Mkt Cap | Close | % Change | Vol vs 20d Avg | Earnings in Window? | Technical Context |")
                    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
                else:
                    lines.append("| Ticker | Company Name | Sector | Mkt Cap | Close | % Change | Vol vs 20d Avg | Earnings in Window? |")
                    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
                for l in pm.losers:
                    earn_flag = "Yes" if l.has_earnings_in_window else "No"
                    mcap_s = _esc_md_dollar(l.market_cap_str or "n/a")
                    row_base = f"| **{l.symbol}** | {_esc_md_dollar(l.name)} | {l.sector} | {mcap_s} | \\${l.close:,.2f} | **{l.pct_change:+.2f}%** | {l.volume_ratio:.2f}x | {earn_flag} |"
                    if has_move_causes:
                        raw_l_cause = getattr(l, "catalyst_driver", "") or (l.move_cause_tag if l.move_cause_tag not in (None, "", "Unknown") else "—")
                        l_cause = _esc_md_dollar(raw_l_cause)
                        row_base += f" {l_cause} |"
                    lines.append(row_base)
                lines.append("")
                if period.report_type in ("daily", "weekly"):
                    lines.append("</details>\n")
            if charts:
                mov_c = _fmt_chart_md(charts.get("top_movers"))
                if mov_c:
                    lines.append(mov_c)

        # Section: Sentiment, Positioning & Volatility Risk Premium
        elif isinstance(sec.data, SentimentPositioningData):
            sp: SentimentPositioningData = sec.data

            if period.report_type == "weekly":
                if getattr(sp, "cot_positions", None):
                    lines.append("### CFTC COT Institutional Futures Positioning")
                    lines.append(f"*Source: {sp.cot_source}*\n")
                    lines.append("| Contract / Asset Class | Non-Commercial Net Position | 1W Net Change | 3-Year Percentile | Positioning Regime |")
                    lines.append("| :--- | :---: | :---: | :---: | :--- |")
                    for cp in sp.cot_positions:
                        net_str = f"`{cp.net_position:+d}`"
                        chg_str = f"`{cp.net_change_1w:+d}`"
                        lines.append(f"| **{cp.contract_name}** | {net_str} | {chg_str} | `{cp.percentile_label}` | {cp.regime} |")
                    lines.append("")

                if getattr(sp, "items", None):
                    lines.append("#### Investor Positioning & Survey Indicators")
                    lines.append("| Sentiment Metric | Current Reading | Status / Interpretation |")
                    lines.append("| :--- | :---: | :--- |")
                    for s_it in sp.items:
                        lines.append(f"| **{s_it.name}** | `{s_it.reading}` | {s_it.diagnostic} |")
                    lines.append("")

                lines.append("<details><summary>Volatility Risk Premium (VRP) & Options Surface Detail</summary>\n")
                lines.append(f"- **CBOE VIX Index (Implied Volatility):** `{sp.vix:.2f}`")
                lines.append(f"- **S&P 500 20-Day Realized Volatility:** `{sp.realized_vol_20d:.2f}%`")
                lines.append(f"- **Volatility Risk Premium (VRP = VIX - Realized Vol):** `{sp.vol_risk_premium:+.2f} pts`")
                lines.append(f"- **VRP Regime:** **{sp.vol_premium_regime}**")
                lines.append(f"> 🧭 **Options Pricing Context:** {sp.assessment}\n")
                lines.append("</details>\n")
            else:
                if getattr(sp, "cot_positions", None) and period.report_type == "monthly":
                    lines.append("### CFTC COT Institutional Futures Positioning")
                    lines.append(f"*Source: {sp.cot_source}*\n")
                    lines.append("| Contract / Asset Class | Non-Commercial Net Position | 1W Net Change | 3-Year Percentile | Positioning Regime |")
                    lines.append("| :--- | :---: | :---: | :---: | :--- |")
                    for cp in sp.cot_positions:
                        net_str = f"`{cp.net_position:+d}`"
                        chg_str = f"`{cp.net_change_1w:+d}`"
                        lines.append(f"| **{cp.contract_name}** | {net_str} | {chg_str} | `{cp.percentile_label}` | {cp.regime} |")
                    lines.append("")

                lines.append("### Volatility Risk Premium & Hedging Demand")
                lines.append(f"- **CBOE VIX Index (Implied Volatility):** `{sp.vix:.2f}`")
                lines.append(f"- **S&P 500 20-Day Realized Volatility:** `{sp.realized_vol_20d:.2f}%`")
                lines.append(f"- **Volatility Risk Premium (VRP = VIX - Realized Vol):** `{sp.vol_risk_premium:+.2f} pts`")
                lines.append(f"- **VRP Regime:** **{sp.vol_premium_regime}**")
                lines.append(f"> 🧭 **Options Pricing Context:** {sp.assessment}\n")
                if getattr(sp, "items", None):
                    lines.append("#### Investor Positioning & Sentiment Indicators")
                    lines.append("| Sentiment Metric | Current Reading | Status / Interpretation |")
                    lines.append("| :--- | :---: | :--- |")
                    for s_it in sp.items:
                        lines.append(f"| **{s_it.name}** | `{s_it.reading}` | {s_it.diagnostic} |")
                    lines.append("")

        # Section: Tactical Watchlist Panel (Stage 3: Task 3.2)
        elif isinstance(sec.data, WatchlistData):
            wl: WatchlistData = sec.data
            lines.append("### Focus Universe & Screener Matrix")
            src_note = f" &bull; {wl.source_label}" if getattr(wl, "source_label", "") else ""
            exc_note = f", {wl.exception_count} Outlier Leaders" if getattr(wl, "exception_count", 0) > 0 else ""
            lines.append(f"> 📋 **Focus Equity Universe:** Tracking {wl.total_count} focus large-cap tickers{src_note} | **Regime Posture:** `{wl.gate_posture}` ({wl.long_permitted_default})\n> ({wl.blocked_count} Defensive, {wl.selective_count} Selective, {wl.permitted_count} Supportive{exc_note})\n")
            if period.report_type == "monthly":
                lines.append("> 💡 **Multi-Horizon Universe Context:** Highlights equities exhibiting sustained 3-month relative-strength outperformance while holding above rising 50-day moving averages.\n")
            if wl.items:
                if period.report_type in ("daily", "weekly"):
                    lines.append(f"<details><summary>Complete Screener Matrix ({wl.total_count} Tickers)</summary>\n")
                lines.append("| Ticker | Company Name | Sector | Close | 20-DMA Δ | 50-DMA Δ | 14D ATR% | RS vs SPY (3M) | Sector Breadth | Technical Setup | Next Earnings | Regime Context | Why flagged |")
                lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |")
                for it in wl.items:
                    c_str = f"\\${it.close:,.2f}" if it.close > 0 else "n/a"
                    d20_str = f"{it.dist_20dma_pct:+.2f}%" if it.dist_20dma_pct is not None else "n/a"
                    d50_str = f"{it.dist_50dma_pct:+.2f}%" if it.dist_50dma_pct is not None else "n/a"
                    atr_str = f"{it.atr_pct:.2f}%" if it.atr_pct is not None else "n/a"
                    rs_str = f"**{it.rs_vs_spy:+.2f}%**" if it.rs_vs_spy is not None else "n/a"
                    why_str = getattr(it, "why_flagged", "") or "—"
                    lines.append(f"| **`{it.ticker}`** | {it.name} | {it.sector} | {c_str} | {d20_str} | {d50_str} | {atr_str} | {rs_str} | {it.sector_breadth_status} | `{it.readiness}` | `{it.next_earnings_date}` | **{it.long_permitted}** | {why_str} |")
                lines.append("")
                if period.report_type in ("daily", "weekly"):
                    lines.append("</details>\n")
            else:
                lines.append(f"> ℹ️ {wl.status_note or 'No watchlist candidates available.'}\n")

        # Section: Conclusion
        elif isinstance(sec.data, ConclusionData):
            cd: ConclusionData = sec.data
            conf_str = f" | Confidence: `{cd.scorecard.confidence}`" if cd.scorecard and cd.scorecard.confidence else ""
            lines.append(f"### Trend Verdict: **{cd.verdict.upper()}** (Score: `{cd.normalized_score:+.2f}`{conf_str})\n")

            # 1. Category Breakdown Table
            if cd.scorecard and cd.scorecard.categories:
                if period.report_type == "weekly":
                    lines.append("<details><summary>Multi-Factor Indicator Scorecard & Factor Weights</summary>\n")
                lines.append("#### Macro & Micro Scorecard by Category")
                lines.append("| Category | Weight Cap | Effective Weight | Category Score | Weighted Score | Status | Diagnostic Notes / Drivers |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
                for cat in cd.scorecard.categories:
                    w_str = f"{cat.weight * 100:.0f}%"
                    eff_w_str = f"{cat.effective_weight * 100:.1f}%" if cat.status == "active" else "— (dropped)"
                    w_sc_str = f"`{cat.weighted_score:+.2f}`" if cat.status == "active" else "—"
                    st_str = "Active" if cat.status == "active" else "Unavailable (renormalized)"
                    lines.append(f"| **{cat.name}** | {w_str} | **{eff_w_str}** | `{cat.score:+.2f}` | {w_sc_str} | {st_str} | {cat.notes or f'{len(cat.components)} component indicators evaluated'} |")
                lines.append("")

            # Verdict Sensitivity & Borderline Assessment (Stage 1: 1.3)
            if cd.scorecard and getattr(cd.scorecard, "sensitivity", None):
                sens = cd.scorecard.sensitivity
                if period.report_type == "daily":
                    if sens.is_borderline:
                        lines.append(f"> ⚠️ **Regime Edge Notice:** Current score `{sens.baseline_score:+.2f}` sits within 0.03 pts of regime threshold (**{sens.borderline_label}**).\n")
                else:
                    lines.append("#### Verdict Sensitivity & Borderline Analysis")
                    if sens.is_borderline:
                        lines.append(f"> ⚠️ **Regime Edge Notice:** Current score `{sens.baseline_score:+.2f}` sits within 0.03 pts of regime threshold (**{sens.borderline_label}**).\n")
                    lines.append("| Excluded Category | Score (ex-Category) | Implied Verdict | Delta vs Baseline |")
                    lines.append("| :--- | :---: | :---: | :---: |")
                    for s_it in sens.items:
                        lines.append(f"| **{s_it.excluded_category}** | `{s_it.score_ex:+.2f}` | **{s_it.verdict_ex}** | `{s_it.delta_vs_baseline:+.2f}` |")
                    lines.append("")

            # 2. Intraperiod Session Volatility Table (Requirement 6)
            if cd.scorecard and cd.scorecard.intraperiod_context and period.report_type != "daily":
                lines.append("#### Intraperiod Daily Session Volatility")
                lines.append("| Session Date | S&P 500 Close | S&P 500 Return | Nasdaq-100 Close | Nasdaq-100 Return | Single-Session Move (≥ 1.5%) |")
                lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
                for sess in cd.scorecard.intraperiod_context:
                    alert_str = "⚠️ **Large Move (≥ 1.5%)**" if sess.is_large_move else "Normal"
                    lines.append(f"| `{sess.date}` | {sess.spx_close:,.2f} | `{sess.spx_change_pct:+.2f}%` | {sess.ndx_close:,.2f} | `{sess.ndx_change_pct:+.2f}%` | {alert_str} |")
                lines.append("")

            # 3. Component Indicator Voting Scorecard
            if period.report_type != "daily":
                lines.append("#### Explainable Component Indicator Voting")
                lines.append("| Indicator Rule | Gradient Vote | Weight | Weighted Vote | Metric Details | Interpretation |")
                lines.append("| :--- | :---: | :---: | :---: | :--- | :--- |")
                for v in cd.scorecard.votes:
                    lines.append(f"| **{v.rule_name}** | `{v.vote:+.2f}` | {v.weight:.1f} | `{v.weighted_vote:+.2f}` | {_esc_md_dollar(v.condition)} | {_esc_md_dollar(v.interpretation)} |")
                lines.append("")
                if period.report_type == "weekly":
                    lines.append("</details>\n")
            else:
                lines.append("> ℹ️ *Individual indicator voting rules, sensitivity analysis, and intraperiod session trajectories are published in the Weekly & Monthly strategic reports.*\n")

            lines.append("#### Key Bullish Drivers")
            for bf in cd.top_bullish_factors:
                lines.append(f"- 🟢 {_esc_md_dollar(bf)}")
            lines.append("")

            lines.append("#### Key Bearish Drivers")
            for br in cd.top_bearish_factors:
                lines.append(f"- 🔴 {_esc_md_dollar(br)}")
            lines.append("")

            if cd.caution_flags:
                lines.append("#### Caution Warnings")
                for cf in cd.caution_flags:
                    lines.append(f"- ⚠️ {_esc_md_dollar(cf)}")
                lines.append("")

            if period.report_type == "weekly":
                lines.append("<details><summary>Key Technical Levels to Watch (S&P 500 & Nasdaq-100)</summary>\n")
            lines.append("#### Key Technical Levels to Watch")
            for al in cd.asset_levels:
                lines.append(f"**{al.name} (`{al.symbol}`) — Current Price: {al.current_price:,.2f}**")
                lines.append("| Level | Price | Distance (%) | Type |")
                lines.append("| :--- | :---: | :---: | :---: |")
                for kl in al.levels:
                    lines.append(f"| {kl.label} | {kl.price:,.2f} | {kl.distance_pct:+.2f}% | **{kl.level_type}** |")
                lines.append("")
            if period.report_type == "weekly":
                lines.append("</details>\n")

            if cd.regime_intelligence:
                intel = cd.regime_intelligence
                sec_sectors_link = sec_by_type.get(SectorReportData) or sec_by_type.get(MacroSectionData)
                if sec_sectors_link:
                    sec_anch = _md_anchor(sec_sectors_link.section_num, sec_sectors_link.title)
                    sec_ref_str = f" (see [Section {sec_sectors_link.section_num}: {sec_sectors_link.title}]({sec_anch}) for sector momentum details)"
                else:
                    sec_ref_str = ""
                lines.append("#### Cross-Market Macro Alignment & Market Regime")
                lines.append(f"- **Alignment:** Technical Trend is **{cd.verdict.upper()}** ({cd.normalized_score:+.2f}), operating within a **{intel.summary_badge}** regime. Credit risk is `{intel.credit.regime}` ({intel.credit.spread_bps:.0f} bps OAS) while volatility term structure reflects `{intel.vol_term.structure}` (ratio {intel.vol_term.ratio:.3f}). Tactical sector rotation shows {_esc_md_dollar(intel.rotation.rotation_summary)}{sec_ref_str}.")
                lines.append("")

            lenses = getattr(cd, "cross_strategy_lenses", None)
            if lenses:
                if period.report_type == "weekly":
                    lines.append("<details><summary>Cross-Strategy Risk & Regime Synthesis</summary>\n")
                lines.append("#### Cross-Strategy Risk & Regime Synthesis")
                lines.append("| Strategy / Horizon | Regime Read | Key Analytical Synthesis |")
                lines.append("| :--- | :--- | :--- |")
                for lens in lenses:
                    lines.append(
                        f"| **{lens.strategy}** | `{_esc_md_dollar(lens.regime_read)}` | {_esc_md_dollar(lens.synthesis)} |"
                    )
                lines.append("")
                if period.report_type == "weekly":
                    lines.append("</details>\n")

            # Substage 3.3: Historical Seasonality Summary (Complete Calendar Years Only)
            seas = getattr(sec.data, "seasonality", None) if hasattr(sec, "data") else None
            if seas is not None and getattr(seas, "complete_years_count", 0) > 0:
                if period.report_type == "weekly":
                    lines.append("<details><summary>S&P 500 Historical Monthly Seasonality Analysis</summary>\n")
                lines.append("### S&P 500 Historical Monthly Seasonality (Complete Calendar Years Only)")
                lines.append(
                    f"> 📅 **Historical Seasonality Context (`N={seas.complete_years_count}` Complete Calendar Years, {seas.history_start_year}–{seas.history_end_year}):** "
                    f"Current month (**{seas.current_month_name}**) historically averages `{seas.current_month_avg_return:+.2f}%` "
                    f"({seas.current_month_win_rate:.1f}% win rate, median `{seas.current_month_median_return:+.2f}%`, IQR `{seas.current_month_iqr_return:.2f}%`). "
                    f"Next month (**{seas.next_month_name}**) historically averages `{seas.next_month_avg_return:+.2f}%` "
                    f"({seas.next_month_win_rate:.1f}% win rate, median `{seas.next_month_median_return:+.2f}%`, IQR `{seas.next_month_iqr_return:.2f}%`).\n"
                )
                lines.append("| Calendar Horizon | Month | Historical Win Rate | 95% Wilson CI | Historical Median | Historical Mean | IQR [P25, P75] | Sample Size (N) |")
                lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
                lines.append(
                    f"| **Current Month** | {seas.current_month_name} | **{seas.current_month_win_rate:.1f}%** | "
                    f"[{seas.current_month_win_rate_ci_low:.1f}%, {seas.current_month_win_rate_ci_high:.1f}%] | "
                    f"`{seas.current_month_median_return:+.2f}%` | `{seas.current_month_avg_return:+.2f}%` | "
                    f"`{seas.current_month_iqr_return:.2f}%` [`{seas.current_month_p25_return:+.2f}%`, `{seas.current_month_p75_return:+.2f}%`] | "
                    f"N={seas.current_month_sample_size} complete years |"
                )
                lines.append(
                    f"| **Next Month** | {seas.next_month_name} | **{seas.next_month_win_rate:.1f}%** | "
                    f"[{seas.next_month_win_rate_ci_low:.1f}%, {seas.next_month_win_rate_ci_high:.1f}%] | "
                    f"`{seas.next_month_median_return:+.2f}%` | `{seas.next_month_avg_return:+.2f}%` | "
                    f"`{seas.next_month_iqr_return:.2f}%` [`{seas.next_month_p25_return:+.2f}%`, `{seas.next_month_p75_return:+.2f}%`] | "
                    f"N={seas.next_month_sample_size} complete years |"
                )
                lines.append(
                    f"\n*Note: Seasonality calculations evaluate strictly complete calendar years ({seas.history_start_year}–{seas.history_end_year}); "
                    f"the partial in-progress month ({seas.current_month_name} {period.session_date[:4]}) is excluded from historical sample distributions.*\n"
                )

                if charts and charts.get("seasonality_heatmap"):
                    seas_c = _fmt_chart_md(charts.get("seasonality_heatmap"))
                    if seas_c:
                        lines.append(seas_c)
                if period.report_type == "weekly":
                    lines.append("</details>\n")

        # Section: S&P 500 Valuation vs. Rates & ERP (Stage 3 Task 3.1 & Stage 4 Task 4.2)
        elif isinstance(sec.data, ValuationData):
            v: ValuationData = sec.data
            lines.append("### S&P 500 Valuation vs. Rates & Equity Risk Premium")
            if v.forward_pe is None:
                lines.append("`n/a (source unavailable)` — Forward valuation multiples and consensus forward EPS estimates are currently unavailable. Equity Risk Premium metrics are omitted until verified forward earnings estimates are available.\n")
            else:
                if v.assessment:
                    lines.append(f"> 💡 **Valuation & ERP Context:** {v.assessment}\n")
                lines.append("| Valuation Metric | Current Level | Benchmark / Hurdle | Spread / Risk Premium | Sourcing & Context |")
                lines.append("| :--- | :---: | :---: | :---: | :--- |")
                pe_str = f"{v.forward_pe:.2f}x"
                ey_str = f"{v.earnings_yield_pct:.2f}%" if v.earnings_yield_pct is not None else "n/a (source unavailable)"
                nom_hurdle = f"10Y Treasury ({v.nominal_10y_yield:.2f}%)" if v.nominal_10y_yield is not None else "10Y Treasury (DGS10)"
                nom_erp_str = f"`{v.nominal_erp_bps:+.1f} bps`" if v.nominal_erp_bps is not None else "n/a (source unavailable)"
                real_hurdle = f"10Y TIPS ({v.real_10y_yield:.2f}%)" if v.real_10y_yield is not None else "10Y TIPS Real Yield (DFII10)"
                real_erp_str = f"`{v.real_erp_bps:+.1f} bps`" if v.real_erp_bps is not None else "n/a (source unavailable)"
                lines.append(f"| **Forward P/E** | {pe_str} | — | — | Forward multiple on consensus 12M EPS ({v.source}) |")
                lines.append(f"| **Earnings Yield (1/PE)** | {ey_str} | — | — | Equity cash generation yield (1 / Forward P/E) |")
                lines.append(f"| **Nominal ERP** | {ey_str} | {nom_hurdle} | {nom_erp_str} | Earnings Yield minus 10Y Benchmark Yield |")
                lines.append(f"| **Real ERP** | {ey_str} | {real_hurdle} | {real_erp_str} | Earnings Yield minus 10Y TIPS Real Yield |")
                lines.append(f"| **Forward EPS Revision Trend** | `{v.eps_revision_trend}` | Consensus Revisions | — | Revision momentum across S&P 500 constituents |")
                lines.append("")

        # Section: Footer
        elif sec.section_num == len(sections) or (isinstance(sec.data, dict) and "disclaimer" in sec.data):
            disclaimer = sec.data.get("disclaimer", DEFAULT_COMPLIANCE_DISCLAIMER) if isinstance(sec.data, dict) else DEFAULT_COMPLIANCE_DISCLAIMER
            lines.append(f"> **Notice:** {disclaimer}\n")
            if isinstance(sec.data, dict) and "data_sources" in sec.data and sec.data["data_sources"]:
                if period.report_type in ("daily", "weekly"):
                    lines.append("<details><summary>Data Quality & Provenance Sources</summary>\n")
                overall_status = sec.data.get("overall_status", "PARTIAL")
                missing_inputs = sec.data.get("missing_inputs", [])
                lines.append(f"### Data Quality & Provenance (Status: **{overall_status}**)")
                if missing_inputs:
                    lines.append(f"> ⚠️ **Missing / Unavailable Inputs:** {'; '.join(missing_inputs)}\n")
                lines.append("| Source | Coverage | Fetch / Window | Status | Fallbacks & Data Notes |")
                lines.append("| :--- | :--- | :---: | :---: | :--- |")
                for s in sec.data["data_sources"]:
                    lines.append(f"| **{s.source_name}** | {s.coverage} | `{s.fetch_time}` | `{s.status}` | {s.notes} |")
                lines.append("")
                if period.report_type in ("daily", "weekly"):
                    lines.append("</details>\n")

    return "\n".join(lines)
