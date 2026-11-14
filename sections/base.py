"""Base definitions and protocols for report sections."""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class CanonicalFactsRegistry:
    """Single canonical source of truth for report-wide cross-section metrics."""
    as_of_date: str
    baseline_date: Optional[str] = None
    report_type: str = "daily"
    spx_close: Optional[float] = None
    spx_period_ret_pct: Optional[float] = None
    spy_close: Optional[float] = None
    qqq_close: Optional[float] = None
    ndx_close: Optional[float] = None
    ndx_period_ret_pct: Optional[float] = None
    vix_current: Optional[float] = None
    vix_previous: Optional[float] = None
    vix_delta: Optional[float] = None
    vix_term_ratio: Optional[float] = None
    yield_10y: Optional[float] = None
    yield_10y_delta_bps: Optional[float] = None
    real_yield_10y: Optional[float] = None
    hy_oas_bps: Optional[float] = None
    pct_above_50dma: Optional[float] = None
    pct_above_200dma: Optional[float] = None
    net_new_highs: Optional[int] = None
    net_new_highs_pctile_1y: Optional[float] = None
    macro_quadrant: str = ""
    inflation_signal: str = ""
    growth_signal: str = ""
    high_impact_events: Tuple[str, ...] = ()


@dataclass
class SectionResult:
    """Standardized result wrapper for each report section."""
    section_num: int
    title: str
    status: str = "available"  # 'available' or 'unavailable'
    data: Optional[Any] = None
    error_message: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self.status == "available"
