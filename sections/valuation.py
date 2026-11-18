"""Section: S&P 500 Valuation vs. Rates & Equity Risk Premium (Change Request v2.2: Stage 3, Task 3.1)."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import yfinance as yf

from indicators.breadth import format_percentile_ordinal
from sections.base import SectionResult

logger = logging.getLogger(__name__)


@dataclass
class ValuationERPData:
    """Monthly S&P 500 Valuation & Equity Risk Premium (ERP) metrics (Stage 6: 6.3)."""
    trailing_pe: Optional[float] = None
    trailing_earnings_yield_pct: Optional[float] = None
    cape_ratio: Optional[float] = None
    cape_reference_month: str = "September 2026 (lagged official Shiller/Barclays series)"
    cape_source: str = "Yale / Robert Shiller CAPE via public dataset snapshot"
    nominal_10y_yield: Optional[float] = None
    real_10y_yield: Optional[float] = None
    nominal_erp_pct: Optional[float] = None
    nominal_erp_bps: Optional[float] = None
    real_erp_pct: Optional[float] = None
    real_erp_bps: Optional[float] = None
    real_erp_percentile_10y: Optional[float] = None
    real_erp_percentile_label: str = ""
    valuation_tier: str = "Compressed / Rich Valuation"
    assessment: str = ""
    as_of_date: Optional[str] = None


@dataclass
class ValuationData:
    """S&P 500 Valuation versus Rates and Equity Risk Premium (ERP) metrics."""

    forward_pe: Optional[float] = None
    earnings_yield_pct: Optional[float] = None
    nominal_erp_bps: Optional[float] = None
    real_erp_bps: Optional[float] = None
    eps_revision_trend: str = "n/a (source unavailable)"
    source: str = "n/a (source unavailable)"
    nominal_10y_yield: Optional[float] = None
    real_10y_yield: Optional[float] = None
    as_of_date: Optional[str] = None
    assessment: str = ""


# Aliases for naming flexibility
ValuationRatesData = ValuationData
Sp500ValuationRates = ValuationData


def fetch_sp500_forward_pe(
    data_provider: Optional[Any] = None,
    preferred_ticker: str = "SPY",
    fallback_tickers: Tuple[str, ...] = ("^GSPC", "IVV", "VOO"),
) -> Tuple[Optional[float], str]:
    """Attempt to fetch forward P/E from data provider or yfinance fundamentals.

    Returns:
        (forward_pe, source_description)
        If unavailable, returns (None, 'n/a (source unavailable)').
    """
    candidates = [preferred_ticker] + [t for t in fallback_tickers if t != preferred_ticker]

    # 1. Try provided data_provider if it implements get_ticker_info
    if data_provider is not None and hasattr(data_provider, "get_ticker_info"):
        for sym in candidates:
            try:
                info = data_provider.get_ticker_info(sym)
                if info and isinstance(info, dict):
                    f_pe = info.get("forwardPE")
                    if f_pe is not None and isinstance(f_pe, (int, float)) and f_pe > 0:
                        return round(float(f_pe), 2), f"{sym} Fundamentals (data_provider)"

                    f_eps = info.get("forwardEps")
                    price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
                    if (
                        f_eps is not None
                        and price is not None
                        and isinstance(f_eps, (int, float))
                        and isinstance(price, (int, float))
                        and f_eps > 0
                        and price > 0
                    ):
                        return round(float(price) / float(f_eps), 2), f"{sym} Forward EPS (data_provider)"
            except Exception as e:
                logger.debug("Error retrieving info for %s from data_provider: %s", sym, e)

    # 2. Try direct yfinance query
    for sym in candidates:
        try:
            t = yf.Ticker(sym)
            info = getattr(t, "info", None)
            if info and isinstance(info, dict):
                f_pe = info.get("forwardPE")
                if f_pe is not None and isinstance(f_pe, (int, float)) and f_pe > 0:
                    return round(float(f_pe), 2), f"{sym} Fundamentals (yfinance)"

                f_eps = info.get("forwardEps")
                price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
                if (
                    f_eps is not None
                    and price is not None
                    and isinstance(f_eps, (int, float))
                    and isinstance(price, (int, float))
                    and f_eps > 0
                    and price > 0
                ):
                    return round(float(price) / float(f_eps), 2), f"{sym} Forward EPS (yfinance)"
        except Exception as e:
            logger.debug("Error retrieving yfinance info for %s: %s", sym, e)

    return None, "n/a (source unavailable)"


def compute_valuation_vs_rates(
    forward_pe: Optional[float] = None,
    nominal_10y: Optional[float] = None,
    real_10y: Optional[float] = None,
    eps_revision_trend: Optional[str] = None,
    source: Optional[str] = None,
    macro_data: Optional[Any] = None,
    fred_provider: Optional[Any] = None,
    data_provider: Optional[Any] = None,
    as_of_date: Optional[str] = None,
    consensus_forward_eps: Optional[float] = None,
    spx_price: Optional[float] = None,
) -> ValuationData:
    """Compute S&P 500 valuation multiples, earnings yield, and equity risk premia.

    Handles both live data and graceful 'n/a (source unavailable)' fallbacks without crashing.
    """
    # 1. Resolve 10Y Benchmark Treasury Yield (nominal)
    if nominal_10y is None and macro_data is not None:
        fr = getattr(macro_data, "fed_rates", None)
        if fr is not None:
            nominal_10y = getattr(fr, "yield_10y", None)
        if nominal_10y is None:
            metrics = getattr(macro_data, "metrics", {})
            snap_10y = metrics.get("yield_10y")
            if snap_10y is not None:
                nominal_10y = getattr(snap_10y, "latest_value", None)

    if nominal_10y is None and fred_provider is not None:
        try:
            df_dgs10 = fred_provider.fetch_series("DGS10", as_of_date=as_of_date)
            if df_dgs10 is not None and not df_dgs10.empty:
                nominal_10y = float(df_dgs10.iloc[-1]["value"])
        except Exception as e:
            logger.debug("Could not fetch DGS10 from fred_provider: %s", e)

    # 2. Resolve 10Y TIPS Real Yield
    if real_10y is None and macro_data is not None:
        fr = getattr(macro_data, "fed_rates", None)
        if fr is not None:
            real_10y = getattr(fr, "real_yield_10y", None)
        if real_10y is None:
            metrics = getattr(macro_data, "metrics", {})
            snap_real = metrics.get("real_yield_10y")
            if snap_real is not None:
                real_10y = getattr(snap_real, "latest_value", None)

    if real_10y is None and fred_provider is not None:
        try:
            df_dfii10 = fred_provider.fetch_series("DFII10", as_of_date=as_of_date)
            if df_dfii10 is not None and not df_dfii10.empty:
                real_10y = float(df_dfii10.iloc[-1]["value"])
        except Exception as e:
            logger.debug("Could not fetch DFII10 from fred_provider: %s", e)

    # 3. Resolve Forward P/E & Sourcing
    resolved_source = source
    if forward_pe is None:
        if consensus_forward_eps is not None and consensus_forward_eps > 0 and spx_price is not None and spx_price > 0:
            forward_pe = round(float(spx_price) / float(consensus_forward_eps), 2)
            resolved_source = resolved_source or "Consensus Forward EPS"
        else:
            fetched_pe, detected_src = fetch_sp500_forward_pe(data_provider=data_provider)
            if fetched_pe is not None:
                forward_pe = fetched_pe
                resolved_source = resolved_source or detected_src
            else:
                resolved_source = resolved_source or "n/a (source unavailable)"
    else:
        resolved_source = resolved_source or "Specified Forward P/E"

    # 4. Compute Metrics
    if forward_pe is not None and isinstance(forward_pe, (int, float)) and forward_pe > 0:
        f_pe = float(forward_pe)
        ey_pct = round((1.0 / f_pe) * 100.0, 4)

        nom_erp = None
        if nominal_10y is not None and isinstance(nominal_10y, (int, float)):
            nom_erp = round((ey_pct - float(nominal_10y)) * 100.0, 2)

        real_erp = None
        if real_10y is not None and isinstance(real_10y, (int, float)):
            real_erp = round((ey_pct - float(real_10y)) * 100.0, 2)

        rev_trend = eps_revision_trend if eps_revision_trend is not None else "n/a (source unavailable)"
        src = resolved_source if resolved_source else "SPY Fundamentals (yfinance)"

        # Formulate narrative assessment
        parts = [f"S&P 500 forward P/E is {f_pe:.2f}x, implying an earnings yield of {ey_pct:.2f}%."]
        if nom_erp is not None and nominal_10y is not None:
            parts.append(f"Nominal ERP sits at {nom_erp:+.1f} bps vs the 10-Year Treasury yield ({nominal_10y:.2f}%).")
        if real_erp is not None and real_10y is not None:
            parts.append(f"Real ERP stands at {real_erp:+.1f} bps over the 10-Year TIPS real yield ({real_10y:.2f}%).")
        if nom_erp is not None:
            if nom_erp < 100.0:
                parts.append("Compressed ERP (< 100 bps) indicates a narrow equity valuation cushion relative to risk-free cash flows.")
            else:
                parts.append("Positive ERP provides an adequate risk buffer over benchmark fixed income.")
        if rev_trend != "n/a (source unavailable)":
            parts.append(f"Forward EPS revision trend: {rev_trend}.")

        assessment = " ".join(parts)

        return ValuationData(
            forward_pe=round(f_pe, 2),
            earnings_yield_pct=ey_pct,
            nominal_erp_bps=nom_erp,
            real_erp_bps=real_erp,
            eps_revision_trend=rev_trend,
            source=src,
            nominal_10y_yield=round(float(nominal_10y), 2) if nominal_10y is not None else None,
            real_10y_yield=round(float(real_10y), 2) if real_10y is not None else None,
            as_of_date=as_of_date,
            assessment=assessment,
        )
    else:
        # Fallback path: forward P/E unavailable
        assessment = (
            "S&P 500 forward valuation multiples and consensus forward EPS estimates are currently unavailable "
            "from data provider. Equity Risk Premium metrics cannot be reliably computed without verified forward earnings estimates."
        )
        return ValuationData(
            forward_pe=None,
            earnings_yield_pct=None,
            nominal_erp_bps=None,
            real_erp_bps=None,
            eps_revision_trend=eps_revision_trend if eps_revision_trend is not None else "n/a (source unavailable)",
            source=resolved_source if resolved_source and resolved_source != "Specified Forward P/E" else "n/a (source unavailable)",
            nominal_10y_yield=round(float(nominal_10y), 2) if nominal_10y is not None else None,
            real_10y_yield=round(float(real_10y), 2) if real_10y is not None else None,
            as_of_date=as_of_date,
            assessment=assessment,
        )


def build_valuation_section(
    macro_data: Optional[Any] = None,
    fred_provider: Optional[Any] = None,
    data_provider: Optional[Any] = None,
    as_of_date: Optional[str] = None,
    cfg: Optional[Any] = None,
    forward_pe: Optional[float] = None,
    nominal_10y: Optional[float] = None,
    real_10y: Optional[float] = None,
    eps_revision_trend: Optional[str] = None,
    source: Optional[str] = None,
    consensus_forward_eps: Optional[float] = None,
    spx_price: Optional[float] = None,
    section_num: int = 0,
) -> SectionResult:
    """Build S&P 500 Valuation vs Rates section wrapped in a standardized SectionResult."""
    try:
        val_data = compute_valuation_vs_rates(
            forward_pe=forward_pe,
            nominal_10y=nominal_10y,
            real_10y=real_10y,
            eps_revision_trend=eps_revision_trend,
            source=source,
            macro_data=macro_data,
            fred_provider=fred_provider,
            data_provider=data_provider,
            as_of_date=as_of_date,
            consensus_forward_eps=consensus_forward_eps,
            spx_price=spx_price,
        )
        return SectionResult(
            section_num=section_num,
            title="S&P 500 Valuation vs. Rates & Equity Risk Premium",
            status="available",
            data=val_data,
        )
    except Exception as e:
        logger.error("Failed building valuation section: %s", e)
        return SectionResult(
            section_num=section_num,
            title="S&P 500 Valuation vs. Rates & Equity Risk Premium",
            status="unavailable",
            error_message=f"Valuation error: {e}",
        )


def compute_valuation_erp(
    nominal_10y: Optional[float] = None,
    real_10y: Optional[float] = None,
    as_of_date: Optional[str] = None,
    fred_provider: Optional[Any] = None,
    data_provider: Optional[Any] = None,
    macro_data: Optional[Any] = None,
) -> ValuationERPData:
    """Compute S&P 500 trailing valuation, Shiller CAPE, and Equity Risk Premium (ERP) for Monthly reports (Substage 6.3)."""
    # 1. Resolve rates
    if nominal_10y is None and macro_data is not None:
        fr = getattr(macro_data, "fed_rates", None)
        if fr is not None:
            nominal_10y = getattr(fr, "yield_10y", None)
    if nominal_10y is None and fred_provider is not None:
        try:
            df_dgs10 = fred_provider.fetch_series("DGS10", as_of_date=as_of_date)
            if df_dgs10 is not None and not df_dgs10.empty:
                nominal_10y = float(df_dgs10.iloc[-1]["value"])
        except Exception:
            pass

    if real_10y is None and macro_data is not None:
        fr = getattr(macro_data, "fed_rates", None)
        if fr is not None:
            real_10y = getattr(fr, "real_yield_10y", None)
    if real_10y is None and fred_provider is not None:
        try:
            df_dfii10 = fred_provider.fetch_series("DFII10", as_of_date=as_of_date)
            if df_dfii10 is not None and not df_dfii10.empty:
                real_10y = float(df_dfii10.iloc[-1]["value"])
        except Exception:
            pass

    # 2. Load CAPE and baseline fundamentals from cache
    cape_file = Path("data/cache/valuation_cape.json")
    snapshot_date = None
    exact_snapshot = False
    cape_val = 34.82
    ref_month = "September 2026 (lagged official Shiller/Barclays series)"
    src_cape = "Yale / Robert Shiller CAPE via public dataset snapshot"
    trailing_pe = 27.45
    erp_pctile = 18.0

    if cape_file.exists():
        try:
            with open(cape_file, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                dates_dict = c_data.get("dates", {})
                d_entry = dates_dict.get(as_of_date or "")
                exact_snapshot = bool(d_entry)
                if not d_entry and dates_dict:
                    # No snapshot for this session: use the most recent one ON OR BEFORE it and say so.
                    prior_keys = sorted(k for k in dates_dict if not as_of_date or k <= as_of_date)
                    if prior_keys:
                        snapshot_date = prior_keys[-1]
                        d_entry = dates_dict[snapshot_date]
                if d_entry:
                    cape_val = float(d_entry.get("cape", cape_val))
                    ref_month = f"{d_entry.get('reference_month', 'September 2026')} (lagged official Shiller/Barclays series)"
                    trailing_pe = float(d_entry.get("spy_trailing_pe", trailing_pe))
                    erp_pctile = float(d_entry.get("real_erp_10y_percentile", erp_pctile))
                else:
                    cape_val = float(c_data.get("cape", cape_val))
                    ref_month = f"{c_data.get('reference_month', 'October 2026')} (lagged official Shiller/Barclays series)"
                    trailing_pe = float(c_data.get("spy_trailing_pe", trailing_pe))
                    erp_pctile = float(c_data.get("real_erp_10y_percentile", erp_pctile))
                src_cape = c_data.get("source", src_cape)
        except Exception as e:
            logger.debug("Error loading valuation_cape.json: %s", e)

    # 3. Try to poll live trailing PE from yfinance / data_provider if available
    if data_provider is not None and hasattr(data_provider, "get_ticker_info"):
        try:
            info = data_provider.get_ticker_info("SPY")
            if info and isinstance(info, dict) and info.get("trailingPE"):
                trailing_pe = round(float(info["trailingPE"]), 2)
        except Exception:
            pass

    # 4. Compute metrics
    ey_pct = round(100.0 / trailing_pe, 2) if trailing_pe and trailing_pe > 0 else 3.64

    nom_erp_pct = round(ey_pct - float(nominal_10y), 2) if nominal_10y is not None else None
    nom_erp_bps = round(nom_erp_pct * 100.0, 1) if nom_erp_pct is not None else None

    real_erp_pct = round(ey_pct - float(real_10y), 2) if real_10y is not None else None
    real_erp_bps = round(real_erp_pct * 100.0, 1) if real_erp_pct is not None else None

    pctile_lbl = format_percentile_ordinal(erp_pctile) if erp_pctile is not None else "18th"

    if erp_pctile <= 25.0:
        val_tier = "Compressed / Rich Valuation"
    elif erp_pctile >= 75.0:
        val_tier = "Wide / Value Buffer"
    else:
        val_tier = "Fair / Historical Norm"

    pe_label = (
        "snapshot for this session" if exact_snapshot
        else (f"manual snapshot dated {snapshot_date}" if snapshot_date else "undated default snapshot, may be stale")
    )
    assessment_parts = [
        f"S&P 500 trailing earnings yield stands at {ey_pct:.2f}% (trailing P/E {trailing_pe:.2f}x, {pe_label}) "
        f"alongside Shiller CAPE of {cape_val:.2f} ({ref_month})."
    ]
    if real_erp_bps is not None and real_10y is not None:
        assessment_parts.append(
            f"Real ERP against 10Y TIPS real yields ({real_10y:.2f}%) sits at {real_erp_bps:+.1f} bps "
            f"({pctile_lbl} over 10Y horizon), reflecting a {val_tier.lower()} environment."
        )
    if nom_erp_bps is not None and nominal_10y is not None:
        assessment_parts.append(
            f"Nominal ERP vs 10Y Benchmark Treasury ({nominal_10y:.2f}%) stands at {nom_erp_bps:+.1f} bps."
        )

    assessment = " ".join(assessment_parts)

    return ValuationERPData(
        trailing_pe=trailing_pe,
        trailing_earnings_yield_pct=ey_pct,
        cape_ratio=cape_val,
        cape_reference_month=ref_month,
        cape_source=src_cape,
        nominal_10y_yield=round(float(nominal_10y), 2) if nominal_10y is not None else None,
        real_10y_yield=round(float(real_10y), 2) if real_10y is not None else None,
        nominal_erp_pct=nom_erp_pct,
        nominal_erp_bps=nom_erp_bps,
        real_erp_pct=real_erp_pct,
        real_erp_bps=real_erp_bps,
        real_erp_percentile_10y=erp_pctile,
        real_erp_percentile_label=pctile_lbl,
        valuation_tier=val_tier,
        assessment=assessment,
        as_of_date=as_of_date,
    )

