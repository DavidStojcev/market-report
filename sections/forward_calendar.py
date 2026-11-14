"""Section: Forward Calendar & Catalyst Horizon (next 5-10 trading days)."""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from data.macro_base import EconCalendarProvider, MacroEvent
from sections.base import SectionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventScenarioOutcome:
    """Single conditional reaction branch in a JPMorgan-style Event Scenario Matrix."""
    scenario: str
    probability_hint: str
    trigger_threshold: str
    spy_reaction: str
    rates_usd_reaction: str
    sector_playbook: str


@dataclass
class EventScenarioMatrix:
    """Three-branch conditional reaction matrix for an upcoming macro catalyst."""
    event_title: str
    event_date: str
    event_time_et: str
    importance: str
    consensus_display: str
    prior_display: str
    implied_1d_move_pct: float
    outcomes: List[EventScenarioOutcome] = field(default_factory=list)
    desk_note: str = ""


@dataclass
class DailyCatalystPlaybook:
    """Next 24h-48h macro releases, BMO/AMC earnings, and key SPY tactical levels for Daily brief."""
    as_of_date: str
    lookahead_date: str
    implied_1d_move_pct: float
    spy_pivot_line: str
    macro_catalysts: List[MacroEvent] = field(default_factory=list)
    earnings_tonight_tomorrow: List[Dict[str, str]] = field(default_factory=list)
    featured_scenario: Optional[EventScenarioMatrix] = None
    execution_takeaway: str = ""


@dataclass
class WeeklyAheadEvent:
    """Monday-Friday catalyst release for Weekly report (Stage 5: 5.2)."""
    day_date: str          # "Mon Nov 16"
    time_et: str           # "08:30 AM"
    release: str           # "Retail Sales MoM"
    impact: str            # "🔴 HIGH", "🟡 MEDIUM"
    prior: str             # "0.40%" or "—"
    market_sensitive: str  # "2Y/10Y Yields & Growth", "USD & Cyclicals", etc.


@dataclass
class WeeklyAheadEarnings:
    """Top scheduled earnings on deck for Weekly report (Stage 5: 5.2)."""
    day_date: str          # "Tue Nov 17"
    timing: str            # "BMO (Pre-Open)", "AMC (After Close)"
    symbol: str            # "WMT"
    name: str              # "Walmart Inc."
    market_cap: str        # "$680B" or sector


def classify_macro_sensitivity(title: str) -> str:
    """Map macro release to the most sensitive asset class / market cross-current."""
    t = (title or "").lower()
    if any(k in t for k in ("cpi", "pce", "inflation", "ppi", "import price")):
        return "2Y/10Y Yields & Growth"
    elif any(k in t for k in ("fomc", "rate decision", "powell", "chair keynote", "fed governor", "fed vice chair", "fed minutes")):
        return "Broad Market, Duration & USD"
    elif any(k in t for k in ("nonfarm", "payroll", "unemployment", "jobless claims", "adp", "jolts")):
        return "Fed Path, USD & Cyclicals"
    elif any(k in t for k in ("gdp", "retail sales", "consumer sentiment", "umich", "personal income")):
        return "Cyclicals & Consumer Discretionary"
    elif any(k in t for k in ("auction", "refunding", "treasury")):
        return "Long-End Treasuries (TLT) & Duration"
    elif any(k in t for k in ("ism", "pmi", "manufacturing", "durable goods", "factory orders", "industrial production")):
        return "Cyclicals, Industrials & Growth"
    elif any(k in t for k in ("housing", "home sales", "building permits")):
        return "Real Estate (XLRE) & Rates"
    elif any(k in t for k in ("crude", "oil", "petroleum", "eia", "natural gas")):
        return "Energy (XLE) & WTI Crude"
    return "Broad Equities & Fixed Income"


@dataclass
class ForwardCalendarData:
    as_of_date: str
    lookahead_date: str
    events: List[MacroEvent] = field(default_factory=list)
    high_impact_count: int = 0
    data_source: str = "Official Schedules (Federal Reserve FOMC, BLS, BEA, DOL, U.S. Census)"
    notes: str = ""
    implied_1d_move_pct: Optional[float] = None
    scenario_matrices: List[EventScenarioMatrix] = field(default_factory=list)
    weekly_ahead_events: List[WeeklyAheadEvent] = field(default_factory=list)
    weekly_ahead_earnings: List[WeeklyAheadEarnings] = field(default_factory=list)


def _fmt_ev_value(val: Optional[float], unit: str) -> str:
    """Format consensus/prior numeric value according to its unit."""
    if val is None:
        return "—"
    if not unit or unit == "index":
        return f"{val:.2f}"
    if unit == "%":
        return f"{val:.2f}%"
    if unit == "k":
        return f"{val:,.0f}k"
    return f"{val:.2f} {unit}"


def _fmt_spy_band(lo_pct: float, hi_pct: float, spy_price: Optional[float]) -> str:
    """Format expected SPY % band with VIX-calibrated dollar price targets."""
    if abs(lo_pct + hi_pct) < 0.05 and lo_pct < 0 < hi_pct:
        pct_txt = f"±{abs(hi_pct):.1f}%"
    else:
        pct_txt = f"{lo_pct:+.1f}% to {hi_pct:+.1f}%"
    if spy_price is not None and spy_price > 0:
        px_lo = spy_price * (1.0 + lo_pct / 100.0)
        px_hi = spy_price * (1.0 + hi_pct / 100.0)
        lo_px, hi_px = min(px_lo, px_hi), max(px_lo, px_hi)
        return f"{pct_txt} (${lo_px:,.0f}–${hi_px:,.0f})"
    return pct_txt


def _catalyst_priority(ev: MacroEvent) -> tuple[int, int, str]:
    """Rank upcoming events by Tier-1 macro scenario importance, then chronologically."""
    t = (ev.title or "").lower()
    imp_rank = 0 if ev.importance == "high" else (1 if ev.importance == "medium" else 2)
    if any(k in t for k in ("fomc", "rate decision", "nonfarm", "employment situation", "consumer price", "cpi", "pce")):
        tier = 0
    elif any(k in t for k in ("chair keynote", "chair speech", "chair press", "treasury quarterly refunding", "quarterly refunding", "gdp", "retail sales", "ppi")):
        tier = 1
    elif any(k in t for k in ("election", "ism ", "adp", "jobless claims")):
        tier = 2
    else:
        tier = 3
    return (imp_rank, tier, ev.date or "9999-12-31")


def _build_single_scenario_matrix(
    ev: MacroEvent,
    implied_1d: float,
    spy_price: Optional[float],
) -> EventScenarioMatrix:
    """Generate a VIX-calibrated 3-branch JPMorgan-style scenario matrix for a macro catalyst."""
    t = (ev.title or "").lower()
    cons_disp = _fmt_ev_value(ev.forecast, ev.unit)
    prior_disp = _fmt_ev_value(ev.previous, ev.unit)
    ref_val = ev.forecast if ev.forecast is not None else ev.previous
    ref_label = "consensus" if ev.forecast is not None else "prior"

    if any(k in t for k in ("cpi", "consumer price", "pce", "ppi")):
        step = 0.2 if ev.unit == "%" else 0.1
        hot_thr = f"Prints ≥ {ref_val + step:.1f}% vs. {cons_disp} cons." if (ref_val is not None and ev.unit == "%") else "Above consensus (sticky inflation)"
        inl_thr = f"Within ±0.1 pp of {_fmt_ev_value(ref_val, ev.unit)}" if (ref_val is not None and ev.unit == "%") else f"In line with {ref_label}"
        cool_thr = f"Prints ≤ {max(0.0, ref_val - step):.1f}% vs. {cons_disp} cons." if (ref_val is not None and ev.unit == "%") else "Below consensus (disinflation beat)"
        outcomes = [
            EventScenarioOutcome(
                scenario="Hot / Sticky Print",
                probability_hint="Print > Ref + 0.1pp (Sticky)",
                trigger_threshold=hot_thr,
                spy_reaction=_fmt_spy_band(-1.8 * implied_1d, -1.1 * implied_1d, spy_price),
                rates_usd_reaction="2Y/10Y yields +8 to +14 bps; USD firms",
                sector_playbook="Energy (XLE), Consumer Staples (XLP) vs. Long-Duration Tech (XLK)",
            ),
            EventScenarioOutcome(
                scenario="In-Line / Benign Print",
                probability_hint="Print Within ±0.1pp of Ref",
                trigger_threshold=inl_thr,
                spy_reaction=_fmt_spy_band(-0.4 * implied_1d, 0.7 * implied_1d, spy_price),
                rates_usd_reaction="Yields steady (±3 bps); post-print vol compression",
                sector_playbook="Broad Index Benchmarks (SPY, RSP) and Industrials (XLI)",
            ),
            EventScenarioOutcome(
                scenario="Cool / Disinflation Surprise",
                probability_hint="Print < Ref - 0.1pp (Cooling)",
                trigger_threshold=cool_thr,
                spy_reaction=_fmt_spy_band(1.0 * implied_1d, 1.7 * implied_1d, spy_price),
                rates_usd_reaction="10Y yield -8 to -13 bps; rate-relief USD dip",
                sector_playbook="Growth / Tech (XLK), Small Caps (IWM), Homebuilders (XHB)",
            ),
        ]
    elif any(k in t for k in ("nonfarm", "payroll", "employment situation", "adp")):
        step = 35.0 if (ref_val is not None and ev.unit == "k") else 1.0
        hot_thr = f"> {ref_val + step:,.0f}k (strong labor beat)" if (ref_val is not None and ev.unit == "k") else "Payrolls beat consensus & UER tightens"
        inl_thr = f"{max(0.0, ref_val - step):,.0f}k–{ref_val + step:,.0f}k goldilocks" if (ref_val is not None and ev.unit == "k") else "Payrolls near consensus; UER steady"
        weak_thr = f"< {max(0.0, ref_val - 40.0):,.0f}k or UER uptick" if (ref_val is not None and ev.unit == "k") else "Soft payroll miss & rising unemployment"
        outcomes = [
            EventScenarioOutcome(
                scenario="Hot Labor / Rate Pressure",
                probability_hint="Print > Ref + 35k (Firm Labor)",
                trigger_threshold=hot_thr,
                spy_reaction=_fmt_spy_band(-0.9 * implied_1d, 0.5 * implied_1d, spy_price),
                rates_usd_reaction="2Y/10Y +7 to +12 bps; Fed cuts priced later",
                sector_playbook="Industrials (XLI), Financials (XLF) vs. Utilities (XLU), REITs (XLRE)",
            ),
            EventScenarioOutcome(
                scenario="Goldilocks / In-Line",
                probability_hint="Print Within ±35k of Ref",
                trigger_threshold=inl_thr,
                spy_reaction=_fmt_spy_band(0.6 * implied_1d, 1.3 * implied_1d, spy_price),
                rates_usd_reaction="10Y steady (±4 bps); VIX event crush lower",
                sector_playbook="Equal-Weight S&P 500 (RSP) and Cyclicals",
            ),
            EventScenarioOutcome(
                scenario="Weak / Growth Scare",
                probability_hint="Print < Ref - 40k (Softening)",
                trigger_threshold=weak_thr,
                spy_reaction=_fmt_spy_band(-1.9 * implied_1d, -1.0 * implied_1d, spy_price),
                rates_usd_reaction="10Y yield -10 to -16 bps; Treasuries rally",
                sector_playbook="Defensives (XLP, XLU, XLV) and Long-Duration Treasuries (TLT)",
            ),
        ]
    elif "jobless claims" in t:
        step = 15.0
        tight_thr = f"< {max(150.0, ref_val - step):,.0f}k (tight labor)" if ref_val is not None else "Claims drop sharply below consensus"
        inl_thr = f"{max(150.0, ref_val - step):,.0f}k–{ref_val + step:,.0f}k" if ref_val is not None else f"Within ±15k of {cons_disp}"
        soft_thr = f"> {ref_val + step:,.0f}k (cooling signal)" if ref_val is not None else "Claims spike above consensus band"
        outcomes = [
            EventScenarioOutcome(
                scenario="Sub-Consensus Tight Labor",
                probability_hint="Print < Ref - 15k (Tight)",
                trigger_threshold=tight_thr,
                spy_reaction=_fmt_spy_band(-0.6 * implied_1d, 0.4 * implied_1d, spy_price),
                rates_usd_reaction="Front-end 2Y +4 to +7 bps; USD ticks up",
                sector_playbook="Financials (XLF) and Cyclicals vs. Long Duration (TLT)",
            ),
            EventScenarioOutcome(
                scenario="In-Line Benign Claims",
                probability_hint="Print Within ±15k of Ref",
                trigger_threshold=inl_thr,
                spy_reaction=_fmt_spy_band(-0.3 * implied_1d, 0.5 * implied_1d, spy_price),
                rates_usd_reaction="Rates range-bound (±2 bps); calm intraday tape",
                sector_playbook="Broad Equities (SPY, QQQ) and Secular Leaders",
            ),
            EventScenarioOutcome(
                scenario="Soft / Rising Claims",
                probability_hint="Print > Ref + 15k (Rising)",
                trigger_threshold=soft_thr,
                spy_reaction=_fmt_spy_band(-1.2 * implied_1d, -0.4 * implied_1d, spy_price),
                rates_usd_reaction="10Y yield -5 to -9 bps; defensive bond bid",
                sector_playbook="Health Care (XLV), Consumer Staples (XLP), Gold (GC=F) vs. Small Caps (IWM)",
            ),
        ]
    elif any(k in t for k in ("fomc", "rate decision", "chair keynote", "chair speech", "chair press", "minutes")):
        outcomes = [
            EventScenarioOutcome(
                scenario="Hawkish Hold / Higher-for-Longer",
                probability_hint="Hawkish Tone / Tightening Bias",
                trigger_threshold="Resists near-term easing; highlights inflation persistence",
                spy_reaction=_fmt_spy_band(-1.6 * implied_1d, -0.9 * implied_1d, spy_price),
                rates_usd_reaction="2Y yield +8 to +13 bps; curve bear-flattens",
                sector_playbook="Energy (XLE) and Low-Leverage Large Caps vs. Small Caps (IWM)",
            ),
            EventScenarioOutcome(
                scenario="Balanced / Data-Dependent",
                probability_hint="Unchanged Guidance / Neutral",
                trigger_threshold="Reiterates meeting-by-meeting path; no policy surprise",
                spy_reaction=_fmt_spy_band(-0.5 * implied_1d, 0.6 * implied_1d, spy_price),
                rates_usd_reaction="Yields whip ±4 bps then settle; implied vol drops",
                sector_playbook="Low-Dispersion Benchmark Averages (SPY, QQQ)",
            ),
            EventScenarioOutcome(
                scenario="Dovish Pivot / Easing Bias",
                probability_hint="Dovish Tone / Easing Bias",
                trigger_threshold="Signals confidence on disinflation or slower QT",
                spy_reaction=_fmt_spy_band(0.9 * implied_1d, 1.7 * implied_1d, spy_price),
                rates_usd_reaction="2Y/10Y yields -8 to -14 bps; USD weakens",
                sector_playbook="Rate-Sensitive Growth (QQQ), Small Caps (IWM), Financials (XLF)",
            ),
        ]
    elif "refunding" in t or "auction" in t:
        outcomes = [
            EventScenarioOutcome(
                scenario="Heavier Coupon Supply / Weak Bid",
                probability_hint="Coupon Supply > Expectations",
                trigger_threshold="Larger 10Y/30Y issuance sizes or >2 bps tail",
                spy_reaction=_fmt_spy_band(-1.3 * implied_1d, -0.6 * implied_1d, spy_price),
                rates_usd_reaction="10Y/30Y yields +6 to +11 bps; curve bear-steepens",
                sector_playbook="Financials (XLF), Energy (XLE) vs. REITs (XLRE), Tech (XLK)",
            ),
            EventScenarioOutcome(
                scenario="Unchanged Schedule / Solid Absorption",
                probability_hint="Issuance In-Line with Schedule",
                trigger_threshold="Coupon mix matches dealer expectations cleanly",
                spy_reaction=_fmt_spy_band(-0.3 * implied_1d, 0.5 * implied_1d, spy_price),
                rates_usd_reaction="Long-end yields stabilize (±3 bps); calm duration",
                sector_playbook="Broad Index Benchmarks (SPY) and Intermediate Duration",
            ),
            EventScenarioOutcome(
                scenario="Lighter Duration Supply / Strong Bid",
                probability_hint="Lower Coupon Supply / Strong Bid",
                trigger_threshold="Shifts issuance toward bills or high bid-to-cover",
                spy_reaction=_fmt_spy_band(0.6 * implied_1d, 1.2 * implied_1d, spy_price),
                rates_usd_reaction="10Y/30Y yields -6 to -10 bps; TLT bounces",
                sector_playbook="Long-Duration Growth (XLK), Real Estate (XLRE), Treasuries (TLT)",
            ),
        ]
    elif "election" in t:
        outcomes = [
            EventScenarioOutcome(
                scenario="Single-Party Sweep (Fiscal / Tariff Shift)",
                probability_hint="Unified Control Outcome",
                trigger_threshold="Unified Congress & executive majority outcome",
                spy_reaction=_fmt_spy_band(-1.2 * implied_1d, 1.4 * implied_1d, spy_price),
                rates_usd_reaction="10Y yields +7 to +14 bps on fiscal supply",
                sector_playbook="Financials (XLF), Domestic Small Caps (IWM) vs. Long Duration",
            ),
            EventScenarioOutcome(
                scenario="Divided Government (Policy Gridlock)",
                probability_hint="Divided Control Outcome",
                trigger_threshold="Split control caps major tax or spending shifts",
                spy_reaction=_fmt_spy_band(0.5 * implied_1d, 1.3 * implied_1d, spy_price),
                rates_usd_reaction="Yields ease -4 to -8 bps; post-event VIX crush",
                sector_playbook="Technology (XLK), Industrials (XLI), Broad Benchmarks",
            ),
            EventScenarioOutcome(
                scenario="Delayed / Contested Count",
                probability_hint="Prolonged Count Uncertainty",
                trigger_threshold="Undecided key races prolong policy uncertainty",
                spy_reaction=_fmt_spy_band(-1.6 * implied_1d, -0.7 * implied_1d, spy_price),
                rates_usd_reaction="Flight to quality bids Gold (GC=F) & short Treasuries",
                sector_playbook="Consumer Staples (XLP), Utilities (XLU), Gold (GC=F), Short Treasuries",
            ),
        ]
    else:
        outcomes = [
            EventScenarioOutcome(
                scenario="Upside Growth / Activity Beat",
                probability_hint="Print > Prior / Consensus",
                trigger_threshold=f"Prints meaningfully above {cons_disp} consensus",
                spy_reaction=_fmt_spy_band(0.5 * implied_1d, 1.2 * implied_1d, spy_price),
                rates_usd_reaction="10Y yield +3 to +7 bps on firmer growth tone",
                sector_playbook="Consumer Discretionary (XLY), Industrials (XLI), Financials (XLF)",
            ),
            EventScenarioOutcome(
                scenario="In-Line / Steady Print",
                probability_hint="Print Near Prior / Consensus",
                trigger_threshold=f"Tracks near {cons_disp} consensus expectations",
                spy_reaction=_fmt_spy_band(-0.4 * implied_1d, 0.4 * implied_1d, spy_price),
                rates_usd_reaction="Yields unchanged (±2 bps); focus stays on Fed/earnings",
                sector_playbook="Broad Cap-Weighted Benchmarks (SPY, QQQ)",
            ),
            EventScenarioOutcome(
                scenario="Downside Miss / Slowdown Signal",
                probability_hint="Print < Prior / Consensus",
                trigger_threshold=f"Falls meaningfully below {cons_disp} consensus",
                spy_reaction=_fmt_spy_band(-1.3 * implied_1d, -0.6 * implied_1d, spy_price),
                rates_usd_reaction="10Y yield -5 to -9 bps on safety bid",
                sector_playbook="Consumer Staples (XLP), Health Care (XLV), Treasuries (TLT)",
            ),
        ]

    desk_note = (
        f"Options-implied move calibrated for {ev.title} ({ev.date}) using current VIX 1-day implied 1σ move of ±{implied_1d:.2f}% "
        f"(Prior: {prior_disp}; Consensus: {cons_disp})."
    )
    return EventScenarioMatrix(
        event_title=ev.title,
        event_date=ev.date,
        event_time_et=ev.time_et,
        importance=ev.importance.upper(),
        consensus_display=cons_disp,
        prior_display=prior_disp,
        implied_1d_move_pct=round(implied_1d, 2),
        outcomes=outcomes,
        desk_note=desk_note,
    )


def build_event_scenario_matrices(
    events: List[MacroEvent],
    vix_current: Optional[float] = None,
    spy_price: Optional[float] = None,
    max_matrices: int = 2,
) -> List[EventScenarioMatrix]:
    """Select top upcoming catalysts and build VIX-calibrated JPMorgan-style reaction matrices."""
    if not events or max_matrices <= 0:
        return []

    vix_val = float(vix_current) if (vix_current is not None and vix_current > 0) else 16.0
    implied_1d = max(0.65, vix_val / 15.8745)

    ranked = sorted(events, key=_catalyst_priority)
    selected: List[MacroEvent] = []
    seen_titles: set[str] = set()
    for ev in ranked:
        if ev.importance != "high" and any(e.importance == "high" for e in ranked):
            continue
        t_key = ev.title.strip().lower()
        if t_key in seen_titles:
            continue
        seen_titles.add(t_key)
        selected.append(ev)
        if len(selected) >= max_matrices:
            break

    # If no HIGH events existed and ranked fell through, allow up to 1 MEDIUM print
    if not selected and ranked:
        selected = [ranked[0]]

    return [_build_single_scenario_matrix(ev, implied_1d, spy_price) for ev in selected]


def build_daily_catalyst_playbook(
    as_of_date: str,
    lookahead_date: str,
    events: List[MacroEvent],
    earnings_events: Optional[List[Any]] = None,
    scenario_matrices: Optional[List[EventScenarioMatrix]] = None,
    spy_price: Optional[float] = None,
    spy_20ema: Optional[float] = None,
    spy_range: Optional[Any] = None,
    vix_current: Optional[float] = None,
    posture: str = "DEFENSIVE",
) -> DailyCatalystPlaybook:
    """Construct the next 24h-48h tactical catalyst playbook for the Daily report."""
    vix_val = float(vix_current) if (vix_current is not None and vix_current > 0) else 16.0
    implied_1d = round(vix_val / 15.8745, 2)

    pivot_parts: List[str] = []
    if spy_price is not None and spy_price > 0:
        pivot_parts.append(f"SPY ${spy_price:,.2f}")
    if spy_20ema is not None and spy_20ema > 0:
        pivot_parts.append(f"20-EMA Pivot ${spy_20ema:,.2f}")
    if spy_range is not None:
        pivot_parts.append(f"Risk Range ${spy_range.range_low:,.2f}–${spy_range.range_high:,.2f}")
    pivot_parts.append(f"Options-Implied 1D Move ±{implied_1d:.2f}%")
    spy_pivot_line = " · ".join(pivot_parts)

    # Top upcoming S&P 500 earnings tonight (AMC) / next morning (BMO)
    earn_rows: List[Dict[str, str]] = []
    if earnings_events:
        dedup: Dict[str, Any] = {}
        for ev in earnings_events:
            sym = getattr(ev, "symbol", "")
            ev_date = getattr(ev, "date", "")
            if not sym or not ev_date or ev_date < as_of_date:
                continue
            prev = dedup.get(sym)
            if prev is None or float(getattr(ev, "market_cap", 0.0) or 0.0) > float(getattr(prev, "market_cap", 0.0) or 0.0):
                dedup[sym] = ev
        sorted_earn = sorted(
            dedup.values(),
            key=lambda e: (getattr(e, "date", "9999-12-31"), -float(getattr(e, "market_cap", 0.0) or 0.0)),
        )
        for ev in sorted_earn[:8]:
            hr_raw = str(getattr(ev, "hour", "") or "").strip().lower()
            hr_label = "BMO (Pre-Open)" if hr_raw == "bmo" else ("AMC (After Close)" if hr_raw == "amc" else (hr_raw.upper() or "TBD"))
            earn_rows.append({
                "symbol": str(ev.symbol),
                "name": str(getattr(ev, "name", "") or ev.symbol),
                "date": str(ev.date),
                "timing": hr_label,
            })

    featured: Optional[EventScenarioMatrix] = None
    if scenario_matrices:
        featured = scenario_matrices[0]
    elif events:
        mats = build_event_scenario_matrices(events, vix_current=vix_current, spy_price=spy_price, max_matrices=1)
        featured = mats[0] if mats else None

    hi_cnt = sum(1 for e in events if getattr(e, "importance", "") == "high")
    earn_cnt = len(earn_rows)
    if featured is not None:
        execution_takeaway = (
            f"Daily desk posture {posture} into {featured.event_title} ({featured.event_date}): "
            f"{hi_cnt} HIGH-impact macro release(s) and {earn_cnt} notable corporate earnings scheduled through {lookahead_date}."
        )
    else:
        execution_takeaway = (
            f"No Tier-1 macro print inside {as_of_date} to {lookahead_date}; "
            f"posture remains {posture} with {earn_cnt} scheduled reporter(s) and price discovery anchored at SPY pivots."
        )

    return DailyCatalystPlaybook(
        as_of_date=as_of_date,
        lookahead_date=lookahead_date,
        implied_1d_move_pct=implied_1d,
        spy_pivot_line=spy_pivot_line,
        macro_catalysts=list(events),
        earnings_tonight_tomorrow=earn_rows,
        featured_scenario=featured,
        execution_takeaway=execution_takeaway,
    )


def build_forward_calendar(
    calendar_provider: Optional[EconCalendarProvider],
    as_of_date: str,
    cfg: Any,
    lookahead_days: int = 14,
    macro_data: Optional[Any] = None,
    vix_current: Optional[float] = None,
    spy_price: Optional[float] = None,
    earnings_events: Optional[List[Any]] = None,
) -> SectionResult:
    """Build forward-looking economic release and catalyst calendar for next 5-10 trading days."""
    try:
        as_of_dt = pd.to_datetime(as_of_date)
        lookahead_dt = as_of_dt + pd.Timedelta(days=lookahead_days)
        lookahead_date_str = lookahead_dt.strftime("%Y-%m-%d")

        high_impact_rules = []
        if hasattr(cfg, "forward_calendar") and hasattr(cfg.forward_calendar, "high_impact_rules"):
            high_impact_rules = [r.lower() for r in cfg.forward_calendar.high_impact_rules]

        events: List[MacroEvent] = []
        metrics = getattr(macro_data, "metrics", {}) if macro_data else {}

        if calendar_provider is not None:
            raw_events = calendar_provider.get_upcoming_events(
                start_date=as_of_date,
                end_date=lookahead_date_str,
                min_importance="medium",
            )
            for ev in raw_events:
                # Guarantee no past or on-period events are ever shown as upcoming
                if ev.date <= as_of_date:
                    continue

                # Synchronize event Prior with official latest actuals if available (Bug 1.5)
                t_lower = ev.title.lower()
                if "jobless claims" in t_lower and "claims" in metrics:
                    ev.previous = metrics["claims"].headline_value
                elif ("nonfarm" in t_lower or "payrolls" in t_lower) and "payrolls" in metrics:
                    ev.previous = metrics["payrolls"].headline_value
                elif "unemployment" in t_lower and "unemployment" in metrics:
                    ev.previous = metrics["unemployment"].headline_value
                elif "cpi" in t_lower and "cpi" in metrics:
                    ev.previous = metrics["cpi"].headline_value

                # Apply fixed high-impact and medium-impact rule sets from config (Stage 1: 1.7)
                title_lower = ev.title.lower()
                is_high = any(rule in title_lower for rule in high_impact_rules)
                if is_high:
                    ev.importance = "high"
                elif ev.importance == "medium" or any(rule in title_lower for rule in [
                    "ism", "adp", "producer price index", "ppi", "retail sales",
                    "gross domestic product", "gdp", "jobless claims", "auction",
                    "fed governor", "fed vice chair", "minutes"
                ]):
                    ev.importance = "medium"
                else:
                    ev.importance = "low"

                # Ensure source provenance fields are populated
                if not ev.source_name:
                    ev.source_name = ev.source or "Official U.S. Schedule"
                if not ev.source_url:
                    ev.source_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
                if not ev.retrieval_timestamp:
                    ev.retrieval_timestamp = "2026-10-30T16:00:00Z"

                events.append(ev)

        # Sort chronologically by date and time
        events.sort(key=lambda x: (x.date, x.time_et))

        high_count = sum(1 for e in events if e.importance == "high")
        scenario_matrices = build_event_scenario_matrices(
            events=events,
            vix_current=vix_current,
            spy_price=spy_price,
            max_matrices=2,
        )
        imp_1d = round(float(vix_current) / 15.8745, 2) if (vix_current is not None and vix_current > 0) else None
        has_finnhub_cons = any(
            getattr(e, "forecast", None) is not None and getattr(e, "consensus_source", "") == "Finnhub Economic API"
            for e in events
        )
        cal_data_source = (
            "Official Schedules (Federal Reserve FOMC, BLS, BEA, DOL) & Finnhub"
            if has_finnhub_cons
            else "Official Federal Schedules (FOMC, BLS, BEA, DOL, U.S. Census)"
        )

        # Build Weekly Ahead Grid (Monday-Friday) and Earnings on Deck (Stage 5: 5.2)
        weekly_ahead_events: List[WeeklyAheadEvent] = []
        for ev in events:
            ev_dt = pd.to_datetime(ev.date)
            if (ev_dt - as_of_dt).days <= 7:
                day_str = ev_dt.strftime("%a %b %d")
                imp_str = "🔴 HIGH" if ev.importance == "high" else ("🟡 MEDIUM" if ev.importance == "medium" else "⚪ LOW")
                pr_str = _fmt_ev_value(ev.previous, ev.unit)
                sens_str = classify_macro_sensitivity(ev.title)
                weekly_ahead_events.append(
                    WeeklyAheadEvent(
                        day_date=day_str,
                        time_et=ev.time_et,
                        release=ev.title,
                        impact=imp_str,
                        prior=pr_str,
                        market_sensitive=sens_str,
                    )
                )

        weekly_ahead_earnings: List[WeeklyAheadEarnings] = []
        if earnings_events:
            dedup_earn: Dict[str, Any] = {}
            for ev in earnings_events:
                sym = getattr(ev, "symbol", "")
                ev_date = getattr(ev, "date", "")
                if not sym or not ev_date or ev_date <= as_of_date:
                    continue
                ev_dt = pd.to_datetime(ev_date)
                if (ev_dt - as_of_dt).days > 7:
                    continue
                prev = dedup_earn.get(sym)
                if prev is None or float(getattr(ev, "market_cap", 0.0) or 0.0) > float(getattr(prev, "market_cap", 0.0) or 0.0):
                    dedup_earn[sym] = ev
            sorted_earn = sorted(
                dedup_earn.values(),
                key=lambda e: (getattr(e, "date", "9999-12-31"), -float(getattr(e, "market_cap", 0.0) or 0.0)),
            )
            for ev in sorted_earn[:10]:
                e_dt = pd.to_datetime(ev.date)
                hr_raw = str(getattr(ev, "hour", "") or "").strip().lower()
                timing_lbl = "BMO (Pre-Open)" if hr_raw == "bmo" else ("AMC (After Close)" if hr_raw == "amc" else (hr_raw.upper() or "TBD"))
                mcap_val = float(getattr(ev, "market_cap", 0.0) or 0.0)
                if mcap_val >= 1e12:
                    mcap_str = f"${mcap_val / 1e12:.2f}T"
                elif mcap_val >= 1e9:
                    mcap_str = f"${mcap_val / 1e9:,.1f}B"
                elif mcap_val > 0:
                    mcap_str = f"${mcap_val / 1e6:,.0f}M"
                else:
                    mcap_str = "—"
                weekly_ahead_earnings.append(
                    WeeklyAheadEarnings(
                        day_date=e_dt.strftime("%a %b %d"),
                        timing=timing_lbl,
                        symbol=str(ev.symbol),
                        name=str(getattr(ev, "name", "") or ev.symbol),
                        market_cap=mcap_str,
                    )
                )

        data = ForwardCalendarData(
            as_of_date=as_of_date,
            lookahead_date=lookahead_date_str,
            events=events,
            high_impact_count=high_count,
            data_source=cal_data_source,
            notes=f"Displaying {len(events)} scheduled catalysts over the next 10 trading sessions ({high_count} marked High Impact).",
            implied_1d_move_pct=imp_1d,
            scenario_matrices=scenario_matrices,
            weekly_ahead_events=weekly_ahead_events,
            weekly_ahead_earnings=weekly_ahead_earnings,
        )

        return SectionResult(
            section_num=10,
            title="Forward Calendar & Catalyst Horizon",
            status="available",
            data=data,
        )

    except Exception as e:
        logger.error("Failed building forward calendar section: %s", e)
        return SectionResult(
            section_num=10,
            title="Forward Calendar & Catalyst Horizon",
            status="available",
            data=ForwardCalendarData(
                as_of_date=as_of_date,
                lookahead_date=as_of_date,
                events=[],
                high_impact_count=0,
                notes="Forward calendar temporarily unavailable (source error).",
            ),
        )
