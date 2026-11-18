"""Section 1: Executive Summary builder."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from data.calendar import PeriodInfo
from scoring.engine import Scorecard
from sections.base import SectionResult


@dataclass
class HorizonPerspective:
    horizon: str
    audience: str
    state: str
    key_metric: str
    implication: str


@dataclass
class ExecutiveBrief:
    headline_takeaway: str
    three_things: List[str]
    what_happened: str
    what_it_means: str
    bottom_line: str


@dataclass
class ChartOfTheDay:
    title: str
    metric_name: str
    current_value_str: str
    percentile_1y: float
    chart_path: str
    annotation: str


@dataclass
class ExecutiveSummaryData:
    verdict: str
    normalized_score: float
    raw_score: float
    max_score: float
    reasons: List[str]
    report_type: str
    as_of_date: str
    session_date: str
    is_partial: bool
    notes: str
    regime_summary: Optional[str] = None
    forward_outlook: Optional[str] = None
    confidence: str = ""
    alignment_note: str = ""
    scorecard: Optional[Scorecard] = None
    horizon_matrix: List[HorizonPerspective] = field(default_factory=list)
    brief: Optional[ExecutiveBrief] = None
    posture: str = ""
    chart_of_the_day: Optional[ChartOfTheDay] = None
    vrp_20d: Optional[float] = None
    wti_price: Optional[float] = None
    wti_3m: Optional[float] = None


def _cat_by_key(scorecard: Scorecard) -> Dict[str, Any]:
    return {c.category_key: c for c in getattr(scorecard, "categories", []) if getattr(c, "status", "") == "active"}


def _build_horizon_matrix(
    scorecard: Scorecard,
    swing_data: Optional[Any] = None,
    regime_intelligence: Optional[Any] = None,
) -> List[HorizonPerspective]:
    """Derive a multi-horizon, strategy-agnostic regime matrix from live scorecard and macro data."""
    cats = _cat_by_key(scorecard)
    cat_trend = cats.get("trend")
    cat_breadth = cats.get("breadth")
    cat_rates = cats.get("rates_macro")

    # 1. Tactical / Short-Term (1–5 Sessions)
    if swing_data is not None:
        posture_upper = str(getattr(swing_data, "posture", "")).upper()
        if posture_upper == "FULL":
            tac_state = "Constructive Expansion"
            tac_imp = "SPY and QQQ hold above 20-EMA and 50-DMA; short-term trend alignment reflects constructive momentum structure."
        elif posture_upper == "REDUCED":
            tac_state = "Measured Consolidation"
            tac_imp = "Near-term pullback below 20-EMA while holding above 50-DMA support; historically associated with range-bound consolidation rather than breakout extension."
        elif posture_upper == "SELECTIVE":
            tac_state = "Divergent / Selective"
            tac_imp = "Index-level dispersion highlights divergence between relative-strength leaders and broad index averages."
        else:
            tac_state = "Defensive / Below EMAs"
            tac_imp = "Benchmarks trade below primary moving averages; historically coincides with elevated volatility and wider intraday trading ranges."

        spy20 = getattr(swing_data, "spy_20ema_dist_pct", None)
        qqq20 = getattr(swing_data, "qqq_20ema_dist_pct", None)
        vix_lvl = getattr(swing_data, "vix_level", None)
        parts = []
        if spy20 is not None:
            parts.append(f"SPY 20-EMA {spy20:+.2f}%")
        if qqq20 is not None:
            parts.append(f"QQQ 20-EMA {qqq20:+.2f}%")
        if vix_lvl is not None:
            parts.append(f"VIX {vix_lvl:.2f}")
        tac_metric = ", ".join(parts) if parts else f"Trend score {cat_trend.score:+.2f}" if cat_trend else "See Section 3"
    else:
        t_score = cat_trend.score if cat_trend else scorecard.normalized_score
        tac_state = "Constructive" if t_score >= 0.2 else ("Defensive" if t_score <= -0.2 else "Mixed / Range-Bound")
        tac_metric = f"Trend Category: {t_score:+.2f}"
        tac_imp = "Reflects short-term index moving average and momentum alignment relative to benchmark trend lines."

    # 2. Intermediate / Breadth & Sector (2–8 Weeks)
    b_score = cat_breadth.score if cat_breadth else 0.0
    rot_obj = getattr(regime_intelligence, "rotation", None) if regime_intelligence is not None else None
    rsp_spread = getattr(rot_obj, "rsp_vs_spy_spread", 0.0) if rot_obj is not None else 0.0
    if cat_breadth is None:
        int_state = "Limited Breadth Feed"
    elif b_score >= 0.25 and rsp_spread < -0.25:
        int_state = "Healthy Base / Narrow Session"
    elif b_score >= 0.25:
        int_state = "Broadening Participation"
    elif b_score <= -0.25:
        int_state = "Narrowing / Concentrated"
    else:
        int_state = "Rotational / Mixed"

    b_parts = []
    if swing_data is not None and getattr(swing_data, "breadth_pct_above_50dma", None) is not None:
        b_parts.append(f"{swing_data.breadth_pct_above_50dma:.1f}% > 50-DMA")
    if regime_intelligence is not None and getattr(regime_intelligence, "rotation", None) is not None:
        rot = regime_intelligence.rotation
        b_parts.append(f"RSP vs SPY {rot.rsp_vs_spy_spread:+.2f}%")
        rot_sum = getattr(rot, "rotation_summary", "").rstrip(".")
        int_imp = f"Sector rotation shows {rot_sum}; breadth participation guides factor and relative-strength positioning." if rot_sum else "Breadth vs index cap-weight spread guides intermediate sector and swing positioning."
    else:
        int_imp = "Underlying constituent participation above 50-DMA guides sector rotation and medium-term equity exposure."
    if not b_parts and cat_breadth is not None:
        b_parts.append(f"Breadth Score {b_score:+.2f}")
    int_metric = ", ".join(b_parts) if b_parts else "Participation gauge"

    # 3. Structural / Macro & Multi-Asset (1–6+ Months)
    if regime_intelligence is not None:
        str_state = getattr(regime_intelligence, "quadrant", "Macro Neutral")
        cr = getattr(regime_intelligence, "credit", None)
        rt = getattr(regime_intelligence, "rates", None)
        m_parts = []
        if cr is not None:
            m_parts.append(f"HY OAS {cr.spread_bps:.0f} bps ({cr.regime})")
        if rt is not None:
            m_parts.append(f"10Y Real Yield {rt.real_yield_10y:.2f}%")
        str_metric = ", ".join(m_parts) if m_parts else (f"Rates & Macro {cat_rates.score:+.2f}" if cat_rates else "Cross-asset backdrop")
        if cr is not None and rt is not None:
            solvency_note = "benign corporate credit spreads" if cr.spread_bps < 400 else "widening corporate credit spreads"
            rate_note = "restrictive real yields capping valuation expansion" if rt.real_yield_10y >= 2.0 else "accommodative real yields supporting valuations"
            str_imp = f"Combines {solvency_note} with {rate_note}; informs strategic asset allocation, duration, and factor tilts."
        else:
            str_imp = "Macro growth, inflation, and discount-rate trends anchor multi-month portfolio risk budgeting."
    else:
        r_score = cat_rates.score if cat_rates else 0.0
        str_state = "Supportive" if r_score >= 0.2 else ("Restrictive" if r_score <= -0.2 else "Balanced")
        str_metric = f"Rates & Macro Score: {r_score:+.2f}"
        str_imp = "Tracks structural discount rates, credit conditions, and inflation impulses for long-horizon allocators."

    return [
        HorizonPerspective(
            horizon="Tactical (1–5 Sessions)",
            audience="Intraday & Momentum",
            state=tac_state,
            key_metric=tac_metric,
            implication=tac_imp,
        ),
        HorizonPerspective(
            horizon="Intermediate (2–8 Weeks)",
            audience="Sector Rotation & Swing",
            state=int_state,
            key_metric=int_metric,
            implication=int_imp,
        ),
        HorizonPerspective(
            horizon="Structural (1–6+ Months)",
            audience="Multi-Asset & Core",
            state=str_state,
            key_metric=str_metric,
            implication=str_imp,
        ),
    ]


def _build_executive_brief(
    scorecard: Scorecard,
    period_info: PeriodInfo,
    regime_intelligence: Optional[Any] = None,
    swing_data: Optional[Any] = None,
    spx_snap: Optional[Any] = None,
    ndx_snap: Optional[Any] = None,
    spx_period_ret: Optional[float] = None,
    ndx_period_ret: Optional[float] = None,
    breadth_data: Optional[Any] = None,
) -> ExecutiveBrief:
    """Build Sevens-style 30-second editorial brief (What Happened -> What It Means -> The Bottom Line)."""
    rt = period_info.report_type.lower()
    window_noun = "Session" if rt == "daily" else ("Week" if rt == "weekly" else "Month")

    # Gather factual tape numbers
    spx_close = getattr(spx_snap, "close", None)
    ndx_close = getattr(ndx_snap, "close", None)
    spx_ret_str = f"{spx_period_ret:+.2f}%" if spx_period_ret is not None else "flat"
    ndx_ret_str = f"{ndx_period_ret:+.2f}%" if ndx_period_ret is not None else "n/a"

    adv = getattr(breadth_data, "advancers", None)
    dec = getattr(breadth_data, "decliners", None)
    ad_ratio = getattr(breadth_data, "ad_ratio", None)
    p50_obj = getattr(breadth_data, "pct_above_50_dma", None)
    p200_obj = getattr(breadth_data, "pct_above_200_dma", None)
    p50 = getattr(p50_obj, "current", None) if p50_obj is not None else getattr(swing_data, "breadth_pct_above_50dma", None)
    p200 = getattr(p200_obj, "current", None) if p200_obj is not None else None

    rot = getattr(regime_intelligence, "rotation", None) if regime_intelligence else None
    credit = getattr(regime_intelligence, "credit", None) if regime_intelligence else None
    rates = getattr(regime_intelligence, "rates", None) if regime_intelligence else None
    vol_term = getattr(regime_intelligence, "vol_term", None) if regime_intelligence else None

    rsp_spread = getattr(rot, "rsp_vs_spy_spread", 0.0) if rot else 0.0
    leaders_2 = ", ".join(rot.top_inflows[:2]) if (rot and rot.top_inflows) else "leading sectors"
    laggards_2 = ", ".join(rot.top_outflows[-2:]) if (rot and rot.top_outflows) else "lagging sectors"

    def _is_def_sector(lbl: str) -> bool:
        low = lbl.lower()
        return any(k in low for k in ("xlp", "xlu", "xlv", "staples", "utilities", "health care", "defensive"))

    top2_inflows = rot.top_inflows[:2] if (rot and getattr(rot, "top_inflows", None)) else []
    def_lead_cnt = sum(1 for lbl in top2_inflows if _is_def_sector(lbl))
    is_defensive_led = bool(
        rot is not None
        and (def_lead_cnt >= 2 or (def_lead_cnt >= 1 and rot.spread <= 0.0) or rot.spread <= -0.5)
    )

    # Characterize tape accurately
    is_narrow_bounce = (
        spx_period_ret is not None
        and spx_period_ret > 0
        and ((ad_ratio is not None and ad_ratio < 0.85) or rsp_spread < -0.25)
    )
    if is_narrow_bounce:
        tape_label = "Narrow cap-weighted bounce into soft underlying breadth"
    elif is_defensive_led:
        if rsp_spread > 0.35:
            tape_label = "Defensive and equal-weight rotation outperforming mega-cap tech"
        else:
            tape_label = "Defensive sector rotation outperforming cyclical beta"
    elif spx_period_ret is not None and spx_period_ret > 0.25 and (ad_ratio is None or ad_ratio >= 1.25):
        tape_label = "Broad-based risk-on advance with confirming internal participation"
    elif spx_period_ret is not None and spx_period_ret < -0.25 and (ad_ratio is None or ad_ratio <= 0.75):
        tape_label = "Broad selling pressure across index and constituent breadth"
    elif rsp_spread > 0.35:
        tape_label = "Equal-weight and cyclical rotation outperforming mega-cap tech"
    else:
        tape_label = "Mixed, rotational tape with selective sector leadership"

    # 1. Headline Takeaway
    posture = str(getattr(swing_data, "posture", "") or scorecard.verdict).upper()
    spy_50 = getattr(swing_data, "spy_50dma", None)
    spy_50_pass = getattr(swing_data, "spy_50dma_pass", None)
    if swing_data is not None and spy_50 and spy_50 > 0:
        if posture == "DEFENSIVE":
            trigger_short = f"model posture remains Defensive below SPY ${spy_50:,.2f} (50-DMA)"
        elif posture in ("SELECTIVE", "REDUCED"):
            trigger_short = f"model posture remains Selective while indices churn near SPY ${spy_50:,.2f} (50-DMA)"
        else:
            trigger_short = f"trend posture remains Constructive above SPY ${spy_50:,.2f} (50-DMA)"
    else:
        trigger_short = f"regime posture is {posture} (composite score {scorecard.normalized_score:+.2f})"
    headline_takeaway = f"{tape_label}; {trigger_short}."

    # 2. Three Things That Mattered
    ad_txt = f"{adv} advancers vs {dec} decliners (A/D {ad_ratio:.2f})" if (adv is not None and dec is not None and ad_ratio is not None) else ""
    p50_txt = f"{p50:.1f}% > 50-DMA" if p50 is not None else ""
    p200_txt = f", {p200:.1f}% > 200-DMA" if p200 is not None else ""
    breadth_clause = "; ".join([x for x in [ad_txt, f"{p50_txt}{p200_txt}".strip()] if x])

    spx_lvl_txt = f" ({spx_close:,.2f})" if spx_close else ""
    ndx_lvl_txt = f" ({ndx_close:,.2f})" if ndx_close else ""
    thing_1 = f"**Tape & Breadth:** S&P 500 moved {spx_ret_str}{spx_lvl_txt} and Nasdaq-100 {ndx_ret_str}{ndx_lvl_txt} while Equal-weight RSP vs SPY printed {rsp_spread:+.2f}%."
    if breadth_clause:
        thing_1 = thing_1[:-1] + f" — {breadth_clause}."

    macro_bits = []
    if rates is not None:
        macro_bits.append(f"10Y Treasury yield {rates.nominal_10y:.2f}% (period {rates.nominal_delta_bps:+.1f} bps, real yield {rates.real_yield_10y:.2f}%)")
    if credit is not None:
        macro_bits.append(f"HY OAS {credit.spread_bps:.0f} bps ({credit.regime})")
    vix_lvl = getattr(swing_data, "vix_level", None)
    vix_ts = getattr(swing_data, "vix_term_structure", None)
    if vix_lvl is not None:
        ts_note = f" in {vix_ts}" if vix_ts else ""
        macro_bits.append(f"VIX {vix_lvl:.2f}{ts_note}")
    thing_2 = "**Rates, Credit & Vol:** " + ("; ".join(macro_bits) + "." if macro_bits else "Cross-asset rates and credit conditions remain primary valuation anchors.")

    if rot is not None:
        lead_prefix = "Defensive rotation led by" if is_defensive_led else "Gains led by"
        thing_3 = f"**Sector Leadership & Risk:** {lead_prefix} {leaders_2} while {laggards_2} lagged."
    else:
        thing_3 = f"**Model Consensus & Risk:** {scorecard.verdict} composite ({scorecard.normalized_score:+.2f}); {scorecard.confidence or 'multi-factor weight of the evidence'}."
    if swing_data is not None:
        spy_d50 = getattr(swing_data, "spy_50dma_dist_pct", None)
        qqq_d50 = getattr(swing_data, "qqq_50dma_dist_pct", None)
        if spy_d50 is not None and qqq_d50 is not None:
            thing_3 = f"{thing_3} Benchmarks sit at SPY {spy_d50:+.2f}% and QQQ {qqq_d50:+.2f}% vs 50-DMA."

    # 3. What Happened -> What It Means -> The Bottom Line
    wh_parts = [f"Over the {window_noun.lower()}, the S&P 500 returned {spx_ret_str} (Nasdaq-100 {ndx_ret_str})"]
    if rot is not None:
        wh_parts.append(f"with leadership concentrated in {leaders_2} against weakness in {laggards_2}")
    if ad_txt:
        wh_parts.append(f"on {ad_txt}")
    what_happened = ", ".join(wh_parts) + "."

    wim_parts = []
    if credit is not None and vol_term is not None:
        if credit.spread_bps < 400 and vol_term.ratio <= 1.0:
            wim_parts.append(f"Tight credit spreads ({credit.spread_bps:.0f} bps OAS) and VIX contango (ratio {vol_term.ratio:.3f}) are consistent with low priced-in systemic funding stress")
        else:
            wim_parts.append(f"Credit spreads ({credit.spread_bps:.0f} bps OAS) and VIX term ratio ({vol_term.ratio:.3f}) require active tail-risk monitoring")
    if is_narrow_bounce:
        wim_parts.append("however, sub-1.0 A/D breadth and RSP trailing SPY show buyers are crowding into mega-caps rather than accumulating the broader index")
    elif p50 is not None and p50 >= 60.0 and spy_50_pass is False:
        wim_parts.append(f"meanwhile, {p50:.1f}% of constituents still holding above their 50-DMA indicates underlying breadth is stronger than cap-weighted index averages")
    if rates is not None and rates.real_yield_10y >= 2.0:
        wim_parts.append(f"while restrictive {rates.real_yield_10y:.2f}% real yields keep a valuation ceiling on long-duration multiples")
    what_it_means = ("; ".join(wim_parts) + ".") if wim_parts else "Cross-asset signals point to balanced risk-reward until trend momentum resolves key moving averages."

    if swing_data is not None and spy_50 and spy_50 > 0:
        spy_px = getattr(swing_data, "spy_price", 0.0)
        spy_20 = getattr(swing_data, "spy_20ema", 0.0)
        qqq_px = getattr(swing_data, "qqq_price", 0.0)
        qqq_50 = getattr(swing_data, "qqq_50dma", 0.0)
        if posture == "DEFENSIVE":
            bottom_line = (
                f"Model posture: Defensive (Tier 4). Historically, this tier coincides with benchmarks trading below their 50-DMAs "
                f"(SPY ${spy_px:,.2f} vs ${spy_50:,.2f}; QQQ ${qqq_px:,.2f} vs ${qqq_50:,.2f}) and elevated distribution risk. "
                f"Model upgrade condition: daily close back above SPY ${spy_20:,.2f} (20-EMA) and ${spy_50:,.2f} (50-DMA) with positive A/D confirmation, "
                f"alongside QQQ reclaiming ${qqq_50:,.2f} (50-DMA)."
            )
        elif posture == "SELECTIVE":
            bottom_line = (
                f"Model posture: Selective (Tier 3). Historically, this tier reflects benchmark divergence around 50-day moving averages. "
                f"Model upgrade condition: both SPY (> ${spy_50:,.2f}) and QQQ (> ${qqq_50:,.2f}) holding above 20-EMA and 50-DMA thresholds with VIX < 19.5."
            )
        elif posture == "REDUCED":
            bottom_line = (
                f"Model posture: Measured (Tier 2) above SPY 50-DMA (${spy_50:,.2f}). "
                f"Model upgrade condition: reclaim of both SPY 20-EMA (${spy_20:,.2f}) and QQQ 20-EMA (${getattr(swing_data, 'qqq_20ema', 0.0):,.2f})."
            )
        else:
            bottom_line = (
                f"Model posture: Constructive Expansion (Tier 1) while SPY holds above ${spy_20:,.2f} (20-EMA) and ${spy_50:,.2f} (50-DMA). "
                f"Model downgrade condition: daily close below ${spy_20:,.2f} signals initial momentum cooling."
            )
    else:
        bottom_line = f"Model posture: {posture.title()} regime state aligned with the {scorecard.verdict} composite signal ({scorecard.normalized_score:+.2f})."

    return ExecutiveBrief(
        headline_takeaway=headline_takeaway,
        three_things=[thing_1, thing_2, thing_3],
        what_happened=what_happened,
        what_it_means=what_it_means,
        bottom_line=bottom_line,
    )


def build_executive_summary(
    scorecard: Optional[Scorecard],
    period_info: PeriodInfo,
    regime_intelligence: Optional[Any] = None,
    swing_data: Optional[Any] = None,
    spx_snap: Optional[Any] = None,
    ndx_snap: Optional[Any] = None,
    spx_period_ret: Optional[float] = None,
    ndx_period_ret: Optional[float] = None,
    breadth_data: Optional[Any] = None,
    chart_of_the_day: Optional[ChartOfTheDay] = None,
) -> SectionResult:
    """Construct Executive Summary section result."""
    if not scorecard:
        return SectionResult(
            section_num=1,
            title="Executive Summary",
            status="unavailable",
            error_message="Trend scorecard calculation failed or was omitted.",
        )

    regime_summary = None
    forward_outlook = None
    if regime_intelligence is not None:
        regime_summary = getattr(regime_intelligence, "summary_badge", None)
        forward_outlook = getattr(regime_intelligence, "forward_outlook", None)

    # Generate Regime vs Posture Alignment Explanation (Fix 1.3)
    alignment_note = ""
    if swing_data is not None:
        verdict = scorecard.verdict
        posture = getattr(swing_data, "posture", "DEFENSIVE").capitalize()
        triggered = getattr(swing_data, "triggered_rule", "benchmark moving averages")
        if " [Event-Risk Overlay:" in triggered:
            triggered_clean = triggered.split(" [Event-Risk Overlay:")[0].rstrip(".") + " (see Section 3 for catalyst overlay)"
        else:
            triggered_clean = triggered

        active_cats = [c for c in getattr(scorecard, "categories", []) if c.status == "active"]
        pos_cats = [c for c in active_cats if c.weighted_score > 0]
        neg_cats = [c for c in active_cats if c.weighted_score < 0]
        top_pos = max(pos_cats, key=lambda c: c.weighted_score) if pos_cats else None
        top_neg = min(neg_cats, key=lambda c: c.weighted_score) if neg_cats else None

        if top_pos and top_neg:
            v_reason = f"cross-asset signals are conflicted: supportive {top_pos.name} ({top_pos.score:+.2f}) is offset by restrictive {top_neg.name} ({top_neg.score:+.2f})"
        elif top_pos:
            v_reason = f"driven primarily by supportive {top_pos.name} ({top_pos.score:+.2f})"
        elif top_neg:
            v_reason = f"pressured primarily by restrictive {top_neg.name} ({top_neg.score:+.2f})"
        else:
            v_reason = "category signals remain balanced around neutral baselines"

        alignment_note = f"Regime is {verdict} because {v_reason}; posture is {posture} because {triggered_clean}"

    horizon_matrix = _build_horizon_matrix(
        scorecard=scorecard,
        swing_data=swing_data,
        regime_intelligence=regime_intelligence,
    )
    brief = _build_executive_brief(
        scorecard=scorecard,
        period_info=period_info,
        regime_intelligence=regime_intelligence,
        swing_data=swing_data,
        spx_snap=spx_snap,
        ndx_snap=ndx_snap,
        spx_period_ret=spx_period_ret,
        ndx_period_ret=ndx_period_ret,
        breadth_data=breadth_data,
    )
    posture_str = str(getattr(swing_data, "posture", "") or scorecard.verdict).upper()

    data = ExecutiveSummaryData(
        verdict=scorecard.verdict,
        normalized_score=scorecard.normalized_score,
        raw_score=scorecard.raw_score,
        max_score=scorecard.max_score,
        reasons=scorecard.summary_reasons,
        report_type=period_info.report_type.capitalize(),
        as_of_date=period_info.as_of_date,
        session_date=period_info.session_date,
        is_partial=period_info.is_partial,
        notes=period_info.notes,
        regime_summary=regime_summary,
        forward_outlook=forward_outlook,
        confidence=scorecard.confidence,
        alignment_note=alignment_note,
        scorecard=scorecard,
        horizon_matrix=horizon_matrix,
        brief=brief,
        posture=posture_str,
        chart_of_the_day=chart_of_the_day,
    )

    return SectionResult(
        section_num=1,
        title="Executive Summary",
        status="available",
        data=data,
    )


def select_chart_of_the_day(
    period_info: PeriodInfo,
    vol_data: Optional[Any] = None,
    breadth_data: Optional[Any] = None,
    charts: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Any] = None,
) -> Optional[ChartOfTheDay]:
    """Deterministically select the most extreme 1Y percentile reading of the day (Stage 4: Substage 4.2).

    Candidate metrics (in exact order for deterministic tie-breaking):
    1. VIX 1Y %ile
    2. VRP 1Y %ile
    3. VVIX 1Y %ile
    4. SKEW 1Y %ile
    5. Breadth >50-DMA 1Y %ile
    6. Breadth >200-DMA 1Y %ile
    7. Net New Highs 1Y %ile
    """
    candidates = []
    sess_date = period_info.session_date

    def _chart_path(key: str, default_name: str) -> str:
        if charts and key in charts and getattr(charts[key], "file_path", None):
            return str(charts[key].file_path)
        return f"{sess_date}_charts/{default_name}"

    from indicators.breadth import format_percentile_ordinal

    # 1. VIX 1Y %ile
    vix_pct = getattr(vol_data, "vix_pctile_1y", None) if vol_data else None
    if vix_pct is not None:
        cur_lvl = getattr(vol_data, "current_level", None)
        lvl_str = f"{cur_lvl:.2f}" if cur_lvl is not None else "n/a"
        ord_str = format_percentile_ordinal(vix_pct)
        candidates.append({
            "metric_name": "VIX 1Y %ile",
            "pctile": float(vix_pct),
            "title": "Chart of the Day: CBOE Volatility Index (VIX)",
            "current_value_str": lvl_str,
            "chart_key": "vix_regime",
            "default_file": "vix_regime.png",
            "annotation": f"Spot VIX at {lvl_str} trades at the {ord_str} (1Y).",
        })

    # 2. VRP 1Y %ile
    vrp_pct = getattr(vol_data, "vrp_pctile_1y", None) if vol_data else None
    if vrp_pct is not None:
        vrp_val = getattr(vol_data, "vrp_20d", None)
        vrp_str = f"{vrp_val:+.2f} pts" if vrp_val is not None else "n/a"
        ord_str = format_percentile_ordinal(vrp_pct)
        candidates.append({
            "metric_name": "VRP 1Y %ile",
            "pctile": float(vrp_pct),
            "title": "Chart of the Day: CBOE Volatility & Realized Vol (VRP)",
            "current_value_str": vrp_str,
            "chart_key": "vix_regime",
            "default_file": "vix_regime.png",
            "annotation": f"20-day Volatility Risk Premium ({vrp_str}) ranks at the {ord_str} (1Y).",
        })

    # 3. VVIX 1Y %ile
    vvix_pct = getattr(vol_data, "vvix_pctile_1y", None) if vol_data else None
    if vvix_pct is not None:
        vvix_val = getattr(vol_data, "vvix_level", None)
        vvix_str = f"{vvix_val:.1f}" if vvix_val is not None else "n/a"
        ord_str = format_percentile_ordinal(vvix_pct)
        candidates.append({
            "metric_name": "VVIX 1Y %ile",
            "pctile": float(vvix_pct),
            "title": "Chart of the Day: Volatility of Volatility (VVIX)",
            "current_value_str": vvix_str,
            "chart_key": "vix_regime",
            "default_file": "vix_regime.png",
            "annotation": f"VVIX index at {vvix_str} ranks at the {ord_str} (1Y).",
        })

    # 4. SKEW 1Y %ile
    skew_pct = getattr(vol_data, "skew_pctile_1y", None) if vol_data else None
    if skew_pct is not None:
        skew_val = getattr(vol_data, "skew_level", None)
        skew_str = f"{skew_val:.1f}" if skew_val is not None else "n/a"
        ord_str = format_percentile_ordinal(skew_pct)
        candidates.append({
            "metric_name": "SKEW 1Y %ile",
            "pctile": float(skew_pct),
            "title": "Chart of the Day: Tail-Risk Index (SKEW)",
            "current_value_str": skew_str,
            "chart_key": "vix_regime",
            "default_file": "vix_regime.png",
            "annotation": f"CBOE SKEW at {skew_str} ranks at the {ord_str} (1Y).",
        })

    # Helpers for breadth metrics (handling BreadthSectionData or MarketBreadthResult)
    br_obj = getattr(breadth_data, "result", breadth_data) if breadth_data else None

    # 5. Breadth >50-DMA 1Y %ile
    b50_pct = getattr(br_obj, "pct_above_50_dma_pctile_1y", None) if br_obj else None
    if b50_pct is not None:
        cur_m = getattr(br_obj, "pct_above_50_dma", None)
        cur_v = getattr(cur_m, "current", cur_m) if cur_m is not None else None
        b50_str = f"{cur_v:.1f}%" if cur_v is not None else "n/a"
        ord_str = format_percentile_ordinal(b50_pct)
        candidates.append({
            "metric_name": "Breadth >50-DMA 1Y %ile",
            "pctile": float(b50_pct),
            "title": "Chart of the Day: S&P 500 Constituent Breadth (>50-DMA)",
            "current_value_str": b50_str,
            "chart_key": "breadth_history",
            "default_file": "breadth_history.png",
            "annotation": f"S&P 500 constituents above 50-DMA ({b50_str}) rank at the {ord_str} (1Y).",
        })

    # 6. Breadth >200-DMA 1Y %ile
    b200_pct = getattr(br_obj, "pct_above_200_dma_pctile_1y", None) if br_obj else None
    if b200_pct is not None:
        cur_m = getattr(br_obj, "pct_above_200_dma", None)
        cur_v = getattr(cur_m, "current", cur_m) if cur_m is not None else None
        b200_str = f"{cur_v:.1f}%" if cur_v is not None else "n/a"
        ord_str = format_percentile_ordinal(b200_pct)
        candidates.append({
            "metric_name": "Breadth >200-DMA 1Y %ile",
            "pctile": float(b200_pct),
            "title": "Chart of the Day: S&P 500 Secular Breadth (>200-DMA)",
            "current_value_str": b200_str,
            "chart_key": "breadth_history",
            "default_file": "breadth_history.png",
            "annotation": f"S&P 500 constituents above 200-DMA ({b200_str}) rank at the {ord_str} (1Y).",
        })

    # 7. Net New Highs 1Y %ile
    nnh_pct = getattr(br_obj, "net_new_highs_pctile_1y", None) if br_obj else None
    if nnh_pct is not None:
        nnh_v = getattr(br_obj, "net_new_highs", None)
        nnh_str = f"{nnh_v:+d}" if nnh_v is not None else "n/a"
        ord_str = format_percentile_ordinal(nnh_pct)
        candidates.append({
            "metric_name": "Net New Highs 1Y %ile",
            "pctile": float(nnh_pct),
            "title": "Chart of the Day: 52-Week Net New Highs",
            "current_value_str": nnh_str,
            "chart_key": "breadth_history",
            "default_file": "breadth_history.png",
            "annotation": f"Net 52-week new highs ({nnh_str}) rank at the {ord_str} (1Y).",
        })

    if not candidates:
        return None

    # Deterministic selection: largest abs(pctile - 50.0). Stable max breaks ties by initial metric order!
    winner = max(candidates, key=lambda c: abs(c["pctile"] - 50.0))
    resolved_path = _chart_path(winner["chart_key"], winner["default_file"])

    return ChartOfTheDay(
        title=winner["title"],
        metric_name=winner["metric_name"],
        current_value_str=winner["current_value_str"],
        percentile_1y=winner["pctile"],
        chart_path=resolved_path,
        annotation=winner["annotation"],
    )
