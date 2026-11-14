"""Section: Corporate Earnings Layer & Next-Week Large-Cap Catalysts."""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from data.constituents import Constituent
from data.macro_base import EconCalendarProvider, EarningsEvent
from sections.base import SectionResult

logger = logging.getLogger(__name__)


@dataclass
class UpcomingEarningsItem:
    symbol: str
    name: str
    market_cap: float
    market_cap_str: str
    date: str
    timing: str           # 'Before Market Open', 'After Market Close', 'Time Unspecified'
    sector: str = ""


@dataclass
class ReportedEarningsItem:
    """Large cap that reported in the trailing 7 days (Stage 4: 4.1)."""
    symbol: str
    name: str
    sector: str
    date: str
    timing: str           # 'BMO' or 'AMC'
    market_cap: float
    market_cap_str: str
    eps_surprise_pct: Optional[float]
    rev_surprise_pct: Optional[float]
    guidance: str         # 'Raised', 'Lowered', 'Maintained', 'Unspecified'
    reaction_1d_pct: Optional[float]
    dist_20dma_pct: Optional[float]
    dist_50dma_pct: Optional[float]


@dataclass
class EarningsLayerData:
    season_name: str
    pct_reported: Optional[float]
    eps_beat_rate: Optional[float]
    rev_beat_rate: Optional[float]
    median_surprise_pct: Optional[float]
    next_week_large_caps: List[UpcomingEarningsItem] = field(default_factory=list)
    reported_last_7_days: List[ReportedEarningsItem] = field(default_factory=list)
    movers_by_theme: str = ""
    status_note: str = ""


def _fmt_mcap(mcap: float) -> str:
    """Format market cap in billions or trillions."""
    if mcap >= 1e12:
        return f"\\${mcap / 1e12:.2f}T"
    elif mcap >= 1e9:
        return f"\\${mcap / 1e9:.1f}B"
    elif mcap > 0:
        return f"\\${mcap / 1e6:.0f}M"
    return "n/a"


def build_earnings_layer(
    calendar_provider: Optional[EconCalendarProvider],
    constituents: List[Constituent],
    ticker_infos: Dict[str, dict],
    as_of_date: str,
    cfg: Any,
    movers_theme_summary: str = "",
    bars: Optional[Dict[str, pd.DataFrame]] = None,
    report_type: str = "daily",
    lookahead_days: Optional[int] = None,
) -> SectionResult:
    """Build Earnings Season Layer, Reported Large-Cap Performance, and Next-Week Catalysts."""
    try:
        as_of_dt = pd.to_datetime(as_of_date)
        is_daily = (report_type.lower() == "daily")
        is_monthly = (report_type.lower() == "monthly")

        if lookahead_days is None:
            if is_daily:
                lookahead_days = 2
            elif is_monthly:
                lookahead_days = 30
            else:
                lookahead_days = max(getattr(cfg.earnings, "lookahead_days", 7) if hasattr(cfg, "earnings") else 7, 8)

        next_week_end = (as_of_dt + pd.Timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
        next_week_start = as_of_dt.strftime("%Y-%m-%d") if is_daily else (as_of_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        trailing_days = 2 if is_daily else 7
        trailing_start = (as_of_dt - pd.Timedelta(days=trailing_days)).strftime("%Y-%m-%d")

        # 1. Map index constituents
        const_with_caps = []
        for c in constituents:
            info = ticker_infos.get(c.symbol, {})
            mcap = float(info.get("marketCap", 0) or 0)
            const_with_caps.append((c, mcap))

        const_with_caps.sort(key=lambda x: x[1], reverse=True)
        top_map = {x[0].symbol: (x[0], x[1]) for x in const_with_caps}

        sec_map = {
            "AAPL": "Information Technology", "PLTR": "Information Technology",
            "AMD": "Information Technology", "AMGN": "Health Care",
            "ANET": "Information Technology", "QCOM": "Information Technology",
            "BRK-B": "Financials", "MSFT": "Information Technology",
            "GOOGL": "Communication Services", "META": "Communication Services",
            "AMZN": "Consumer Discretionary", "NOW": "Information Technology",
            "INTC": "Information Technology", "RDDT": "Communication Services",
            "FTNT": "Information Technology", "VRT": "Industrials",
            "CVNA": "Consumer Discretionary", "PNR": "Industrials",
            "GEN": "Information Technology", "GPN": "Financials",
        }

        # 2. Query upcoming next-week releases
        upcoming_items: List[UpcomingEarningsItem] = []
        if calendar_provider is not None:
            raw_earnings = calendar_provider.get_upcoming_earnings(
                start_date=next_week_start,
                end_date=next_week_end,
                symbols=None,
            )
            for ev in raw_earnings:
                if is_daily and ev.date == as_of_date and ev.hour.lower() != "amc":
                    continue
                c_item, mcap = top_map.get(ev.symbol, (None, ev.market_cap))
                if (mcap is None or mcap <= 0.0) and ev.market_cap > 0.0:
                    mcap = ev.market_cap
                is_saturday = ev.hour.lower() in ("sat", "saturday") or pd.to_datetime(ev.date).day_of_week == 5
                if is_saturday:
                    timing_str = "Saturday"
                elif ev.hour.lower() == "bmo":
                    timing_str = "Before Market Open"
                elif ev.hour.lower() == "amc":
                    timing_str = "After Market Close"
                else:
                    timing_str = "Time Unspecified"
                sec_name = c_item.sector if c_item else sec_map.get(ev.symbol, "General")
                upcoming_items.append(
                    UpcomingEarningsItem(
                        symbol=ev.symbol,
                        name=c_item.name if c_item else ev.name,
                        market_cap=mcap,
                        market_cap_str=_fmt_mcap(mcap),
                        date=ev.date,
                        timing=timing_str,
                        sector=sec_name,
                    )
                )

        upcoming_items.sort(key=lambda x: (x.date, -x.market_cap))

        # 3. Query trailing 7-day reported earnings (Stage 4: 4.1)
        reported_items: List[ReportedEarningsItem] = []
        if calendar_provider is not None:
            raw_reported = calendar_provider.get_recent_reported_earnings(
                start_date=trailing_start,
                end_date=as_of_date,
                symbols=None,
            )
            for rep in raw_reported:
                c_item, mcap = top_map.get(rep.symbol, (None, rep.market_cap))
                if (mcap is None or mcap <= 0.0) and rep.market_cap > 0.0:
                    mcap = rep.market_cap
                sec_name = c_item.sector if c_item else sec_map.get(rep.symbol, "General")
                timing_label = rep.hour.upper() if rep.hour in ("bmo", "amc") else "AMC"

                # Calculate price reaction & DMA distances from bars if available
                react_1d: Optional[float] = None
                d_20dma: Optional[float] = None
                d_50dma: Optional[float] = None

                if bars and rep.symbol in bars:
                    df = bars[rep.symbol]
                    if df is not None and not df.empty and "Close" in df.columns:
                        sub = df[df.index <= as_of_dt]
                        if not sub.empty:
                            c_curr = float(sub["Close"].iloc[-1])
                            if len(sub) >= 20:
                                m20 = float(sub["Close"].rolling(20).mean().iloc[-1])
                                if m20 > 0:
                                    d_20dma = ((c_curr - m20) / m20) * 100.0
                            if len(sub) >= 50:
                                m50 = float(sub["Close"].rolling(50).mean().iloc[-1])
                                if m50 > 0:
                                    d_50dma = ((c_curr - m50) / m50) * 100.0

                            # 1-day reaction around report date
                            rep_dt = pd.to_datetime(rep.date)
                            exact_or_post = df[df.index >= rep_dt]
                            pre = df[df.index < rep_dt]
                            if rep.hour.lower() == "amc" and len(exact_or_post) >= 2:
                                c_base = float(exact_or_post["Close"].iloc[0])
                                c_post = float(exact_or_post["Close"].iloc[1])
                                if c_base > 0:
                                    react_1d = round(((c_post - c_base) / c_base) * 100.0, 2)
                            elif rep.hour.lower() == "bmo" and not pre.empty and not exact_or_post.empty:
                                c_base = float(pre["Close"].iloc[-1])
                                c_post = float(exact_or_post["Close"].iloc[0])
                                if c_base > 0:
                                    react_1d = round(((c_post - c_base) / c_base) * 100.0, 2)
                            elif len(sub) >= 2 and (rep_dt.date() - pd.to_datetime(sub.index[-1]).date()).days <= 5:
                                c_base = float(sub["Close"].iloc[-2])
                                c_post = float(sub["Close"].iloc[-1])
                                if c_base > 0:
                                    react_1d = round(((c_post - c_base) / c_base) * 100.0, 2)

                reported_items.append(
                    ReportedEarningsItem(
                        symbol=rep.symbol,
                        name=c_item.name if c_item else rep.name,
                        sector=sec_name,
                        date=rep.date,
                        timing=timing_label,
                        market_cap=mcap,
                        market_cap_str=_fmt_mcap(mcap),
                        eps_surprise_pct=rep.eps_surprise_pct,
                        rev_surprise_pct=rep.rev_surprise_pct,
                        guidance=rep.guidance,
                        reaction_1d_pct=react_1d,
                        dist_20dma_pct=d_20dma,
                        dist_50dma_pct=d_50dma,
                    )
                )

        reported_items.sort(key=lambda x: (x.date, -x.market_cap), reverse=True)

        upcoming_items.sort(key=lambda x: (x.date, -x.market_cap))

        # 3. Determine earnings season metrics dynamically from reported season data
        # For late October / November, it is Q3 reporting season
        month = as_of_dt.month
        quarter_str = "Q3" if month in (10, 11, 12) else ("Q4" if month in (1, 2, 3) else ("Q1" if month in (4, 5, 6) else "Q2"))
        season_name = f"{quarter_str} {as_of_dt.year} Earnings Season"

        season_reported: List[Any] = []
        if calendar_provider is not None and hasattr(calendar_provider, "get_recent_reported_earnings"):
            season_start = (as_of_dt - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
            season_reported = calendar_provider.get_recent_reported_earnings(
                start_date=season_start,
                end_date=as_of_date,
                symbols=None,
            ) or []

        eps_surprises = [
            float(r.eps_surprise_pct)
            for r in season_reported
            if getattr(r, "eps_surprise_pct", None) is not None
        ]
        rev_surprises = [
            float(r.rev_surprise_pct)
            for r in season_reported
            if getattr(r, "rev_surprise_pct", None) is not None
        ]

        if eps_surprises:
            eps_beat_rate: Optional[float] = round(100.0 * sum(1 for x in eps_surprises if x > 0) / len(eps_surprises), 1)
            median_surprise_pct: Optional[float] = round(float(np.median(eps_surprises)), 2)
            rev_beat_rate: Optional[float] = (
                round(100.0 * sum(1 for x in rev_surprises if x > 0) / len(rev_surprises), 1)
                if rev_surprises
                else None
            )
            denom = len(constituents) if len(constituents) >= 20 else max(len(season_reported), 20)
            pct_reported: Optional[float] = round(min(100.0, 100.0 * len(season_reported) / denom), 1)
            rev_txt = f" | {rev_beat_rate:.1f}% revenue beat rate" if rev_beat_rate is not None else ""
            status_note = (
                f"Tracked large-cap sample ({len(season_reported)} reported in trailing 45d): "
                f"{eps_beat_rate:.1f}% EPS beat rate, median surprise {median_surprise_pct:+.1f}%{rev_txt}."
            )
        else:
            pct_reported = None
            eps_beat_rate = None
            rev_beat_rate = None
            median_surprise_pct = None
            status_note = "Aggregate S&P 500 beat rates: n/a (source unavailable)."

        data = EarningsLayerData(
            season_name=season_name,
            pct_reported=pct_reported,
            eps_beat_rate=eps_beat_rate,
            rev_beat_rate=rev_beat_rate,
            median_surprise_pct=median_surprise_pct,
            next_week_large_caps=upcoming_items,
            reported_last_7_days=reported_items,
            movers_by_theme=movers_theme_summary,
            status_note=status_note,
        )

        return SectionResult(
            section_num=11,
            title="Earnings Season Progress & Upcoming Large-Cap Releases",
            status="available",
            data=data,
        )

    except Exception as e:
        logger.error("Failed building earnings layer section: %s", e)
        return SectionResult(
            section_num=11,
            title="Earnings Season Progress & Upcoming Large-Cap Releases",
            status="available",
            data=EarningsLayerData(
                season_name="Corporate Earnings",
                pct_reported=None,
                eps_beat_rate=None,
                rev_beat_rate=None,
                median_surprise_pct=None,
                next_week_large_caps=[],
                reported_last_7_days=[],
                movers_by_theme="",
                status_note="Earnings layer temporarily unavailable.",
            ),
        )
