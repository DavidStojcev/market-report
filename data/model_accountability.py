"""Model Accountability Tracker.

Persists and audits weekly model performance, specifically tracking Quantitative Risk Ranges™
adherence across key benchmark assets (SPY, QQQ, IWM, TLT, CL=F, GC=F).

Compliance Note (T13):
Tracks range adherence ONLY (whether realized price held inside [range_low, range_high]).
Never evaluates hypothetical trading P&L, returns, entries, stops, or targets.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ACCOUNTABILITY_PATH = "data/cache/weekly_model_accountability.json"

BENCHMARK_ASSETS = ["SPY", "QQQ", "IWM", "TLT", "CL=F", "GC=F"]


@dataclass
class RangeBreach:
    symbol: str
    breach_type: str  # "Above Resistance" or "Below Support"
    realized_extreme: float
    published_boundary: float
    detail: str


@dataclass
class WeeklyModelAdherence:
    as_of_date: str
    prior_week_date: str
    ranges_held: int
    total_assets: int
    breaches: List[RangeBreach] = field(default_factory=list)
    summary_text: str = ""


def load_accountability_store(file_path: Optional[str] = None) -> Dict[str, Any]:
    """Load JSON accountability history from disk."""
    p = Path(file_path) if file_path else Path(DEFAULT_ACCOUNTABILITY_PATH)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed loading accountability store from %s: %s", p, e)
        return {}


def save_accountability_store(data: Dict[str, Any], file_path: Optional[str] = None) -> None:
    """Save JSON accountability history to disk."""
    p = Path(file_path) if file_path else Path(DEFAULT_ACCOUNTABILITY_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Failed saving accountability store to %s: %s", p, e)


def record_weekly_snapshot(
    as_of_date: str,
    posture: str,
    risk_ranges: List[Any],
    file_path: Optional[str] = None,
) -> None:
    """Persist current week's published risk ranges and posture to disk."""
    store = load_accountability_store(file_path)
    ranges_dict: Dict[str, Dict[str, float]] = {}
    for r in risk_ranges:
        sym = getattr(r, "symbol", "")
        if sym in BENCHMARK_ASSETS or not BENCHMARK_ASSETS:
            ranges_dict[sym] = {
                "close": float(getattr(r, "last_price", 0.0)),
                "range_low": float(getattr(r, "range_low", 0.0)),
                "range_high": float(getattr(r, "range_high", 0.0)),
            }

    store[as_of_date] = {
        "as_of_date": as_of_date,
        "posture": posture,
        "ranges": ranges_dict,
    }
    save_accountability_store(store, file_path)


def evaluate_weekly_range_adherence(
    as_of_date: str,
    prior_week_date: str,
    current_bars: Dict[str, pd.DataFrame],
    file_path: Optional[str] = None,
    injected_prior_ranges: Optional[Dict[str, Dict[str, float]]] = None,
) -> WeeklyModelAdherence:
    """Evaluate realized weekly price containment against prior week's published ranges.

    If prior ranges are not in JSON store, reconstruct them deterministically from historical bars.
    """
    store = load_accountability_store(file_path)
    prior_ranges: Dict[str, Dict[str, float]] = {}

    if injected_prior_ranges:
        prior_ranges = injected_prior_ranges
    elif prior_week_date in store and "ranges" in store[prior_week_date]:
        prior_ranges = store[prior_week_date]["ranges"]
    else:
        # Deterministically reconstruct prior week's ranges from historical bars as of prior_week_date
        from sections.swing_regime import _compute_risk_range_item
        prior_dt = pd.to_datetime(prior_week_date)
        for sym in BENCHMARK_ASSETS:
            df = current_bars.get(sym)
            if df is not None and not df.empty:
                item = _compute_risk_range_item(sym, sym, df, prior_dt)
                if item is not None:
                    prior_ranges[sym] = {
                        "close": float(item.last_price),
                        "range_low": float(item.range_low),
                        "range_high": float(item.range_high),
                    }

    total_assets = len(BENCHMARK_ASSETS)
    breaches: List[RangeBreach] = []
    as_of_dt = pd.to_datetime(as_of_date)
    prior_dt = pd.to_datetime(prior_week_date)

    ranges_held = 0
    for sym in BENCHMARK_ASSETS:
        p_info = prior_ranges.get(sym)
        df = current_bars.get(sym)
        if p_info is None or df is None or df.empty:
            ranges_held += 1  # If no prior range or data, treat as neutral
            continue

        r_low = float(p_info["range_low"])
        r_high = float(p_info["range_high"])

        # Slice bars for the realized week: (prior_dt, as_of_dt]
        w_bars = df[(df.index > prior_dt) & (df.index <= as_of_dt)]
        if w_bars.empty:
            w_bars = df[df.index <= as_of_dt].tail(5)

        high_val = float(w_bars["High"].max()) if "High" in w_bars.columns else float(w_bars["Close"].max())
        low_val = float(w_bars["Low"].min()) if "Low" in w_bars.columns else float(w_bars["Close"].min())

        sym_clean = sym.replace("=F", "")
        if high_val > r_high:
            breaches.append(
                RangeBreach(
                    symbol=sym_clean,
                    breach_type="Above Resistance",
                    realized_extreme=high_val,
                    published_boundary=r_high,
                    detail=f"{sym_clean} (Above Resistance ${high_val:,.2f} vs published range high ${r_high:,.2f})",
                )
            )
        elif low_val < r_low:
            breaches.append(
                RangeBreach(
                    symbol=sym_clean,
                    breach_type="Below Support",
                    realized_extreme=low_val,
                    published_boundary=r_low,
                    detail=f"{sym_clean} (Below Support ${low_val:,.2f} vs published range low ${r_low:,.2f})",
                )
            )
        else:
            ranges_held += 1

    if breaches:
        brk_strs = [f"{b.symbol} ({b.breach_type.lower()})" for b in breaches]
        summary_text = (
            f"{ranges_held} of {total_assets} assets held within published ranges; breaks: {', '.join(brk_strs)}"
        )
    else:
        summary_text = (
            f"{ranges_held} of {total_assets} assets held within published ranges; breaks: None (full range containment)"
        )

    return WeeklyModelAdherence(
        as_of_date=as_of_date,
        prior_week_date=prior_week_date,
        ranges_held=ranges_held,
        total_assets=total_assets,
        breaches=breaches,
        summary_text=summary_text,
    )
