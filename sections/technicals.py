"""Section 3: Technical Indicators builder for S&P 500 and Nasdaq-100."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd

from config.settings import IndicatorsConfig, SymbolItem
from data.calendar import PeriodInfo
from indicators.engine import TechnicalSnapshot, compute_technicals
from sections.base import SectionResult


@dataclass
class SecularTechnicalsItem:
    """Monthly-bar secular moving average technicals (10M & 12M SMA) (Substage 6.4)."""
    symbol: str
    name: str
    close: float
    sma_10m: float
    dist_10m_pct: float
    sma_12m: float
    dist_12m_pct: float
    secular_trend: str

    @property
    def price_vs_10m_pct(self) -> float:
        return self.dist_10m_pct

    @property
    def price_vs_12m_pct(self) -> float:
        return self.dist_12m_pct

    @property
    def trend_status(self) -> str:
        return self.secular_trend


@dataclass
class MonthlySecularTechnicalsData:
    as_of_date: str
    items: List[SecularTechnicalsItem] = field(default_factory=list)
    synthesis: str = ""

    @property
    def summary_text(self) -> str:
        return self.synthesis


@dataclass
class TechnicalsData:
    snapshots: List[TechnicalSnapshot]
    secular_technicals: Optional[MonthlySecularTechnicalsData] = None


def compute_monthly_secular_technicals(
    bars: Dict[str, pd.DataFrame],
    as_of_date: str,
    symbols: Optional[List[Tuple[str, str]]] = None,
) -> MonthlySecularTechnicalsData:
    """Compute 10-Month and 12-Month SMA secular trend technicals from month-end bars (Substage 6.4)."""
    if symbols is None:
        symbols = [
            ("^GSPC", "S&P 500 Index"),
            ("^NDX", "Nasdaq-100 Index"),
            ("^RUT", "Russell 2000 Index"),
            ("SPY", "SPDR S&P 500 ETF"),
            ("QQQ", "Invesco QQQ Trust"),
        ]

    items: List[SecularTechnicalsItem] = []
    as_of_dt = pd.to_datetime(as_of_date)

    for sym, name in symbols:
        df = bars.get(sym)
        if df is None or df.empty or "Close" not in df.columns:
            continue

        sub = df[df.index <= as_of_dt]["Close"].dropna()
        if len(sub) < 30:
            continue

        monthly_closes = sub.resample("ME").last().dropna()
        if not monthly_closes.empty and monthly_closes.index[-1].date() != sub.index[-1].date():
            monthly_closes = pd.concat([monthly_closes, pd.Series([sub.iloc[-1]], index=[sub.index[-1]])])

        if len(monthly_closes) >= 5:
            sma_10 = float(monthly_closes.rolling(10, min_periods=5).mean().iloc[-1])
            sma_12 = float(monthly_closes.rolling(12, min_periods=6).mean().iloc[-1])
            curr_px = float(sub.iloc[-1])
            d10 = round(((curr_px - sma_10) / sma_10) * 100.0, 2)
            d12 = round(((curr_px - sma_12) / sma_12) * 100.0, 2)
            trend = "Secular Uptrend (Holding > 10M SMA)" if curr_px >= sma_10 else "Below 10M SMA (Secular Caution)"

            items.append(
                SecularTechnicalsItem(
                    symbol=sym,
                    name=name,
                    close=round(curr_px, 2),
                    sma_10m=round(sma_10, 2),
                    dist_10m_pct=d10,
                    sma_12m=round(sma_12, 2),
                    dist_12m_pct=d12,
                    secular_trend=trend,
                )
            )

    uptrend_count = sum(1 for it in items if "Uptrend" in it.secular_trend)
    synthesis = (
        f"{uptrend_count} of {len(items)} tracked benchmark indices and core ETFs hold above their 10-month moving averages, "
        f"{'confirming sustained secular bull market structural integrity' if uptrend_count >= max(1, len(items) - 1) else 'reflecting multi-month secular consolidation and divergence below key long-term averages'}."
    )

    return MonthlySecularTechnicalsData(
        as_of_date=as_of_date,
        items=items,
        synthesis=synthesis,
    )


def build_technicals(
    bars: Dict[str, pd.DataFrame],
    tech_indices: List[SymbolItem],
    period: PeriodInfo,
    ind_cfg: IndicatorsConfig,
) -> SectionResult:
    """Build technical indicator tables for key benchmarks (SPX and NDX)."""
    snapshots: List[TechnicalSnapshot] = []

    for item in tech_indices:
        df = bars.get(item.symbol)
        if df is None or df.empty:
            continue

        snap = compute_technicals(
            df=df,
            symbol=item.symbol,
            name=item.name,
            as_of_date=period.session_date,
            prev_date=period.prev_period_end,
            dma_fast=ind_cfg.dma_fast,
            dma_medium=ind_cfg.dma_medium,
            dma_slow=ind_cfg.dma_slow,
            rsi_period=ind_cfg.rsi_period,
            rsi_overbought=ind_cfg.rsi_overbought,
            rsi_oversold=ind_cfg.rsi_oversold,
            macd_fast=ind_cfg.macd_fast,
            macd_slow=ind_cfg.macd_slow,
            macd_signal=ind_cfg.macd_signal,
            adx_period=ind_cfg.adx_period,
        )
        if snap:
            snapshots.append(snap)

    if not snapshots:
        return SectionResult(
            section_num=3,
            title="Technical Indicators",
            status="unavailable",
            error_message="Technical calculations could not be completed for benchmark indices.",
        )

    secular = None
    if period.report_type == "monthly":
        secular = compute_monthly_secular_technicals(bars=bars, as_of_date=period.session_date)

    return SectionResult(
        section_num=3,
        title="Technical Indicators",
        status="available",
        data=TechnicalsData(snapshots=snapshots, secular_technicals=secular),
    )
