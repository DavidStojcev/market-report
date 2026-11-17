"""Local caching layer using Parquet for market data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STANDARD_SPLIT_FACTORS = (2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 0.5, 1.0 / 3.0, 0.25, 0.2, 0.1)


def _repair_split_discontinuities(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Heuristic split back-adjustment from price jumps (NOT used by the cache anymore).

    A ~50% one-day crash or ~100% gain is indistinguishable from a 2:1 split by price alone, so
    this rewrote genuine moves.  Kept for callers/tests that use it explicitly on known-unadjusted
    data; the cache uses `_rebase_to_provider_adjustment` instead.
    """
    if df is None or df.empty or len(df) < 2 or "Close" not in df.columns:
        return df

    # Indices, futures (=F), and FX (=X) do not undergo corporate share splits
    t_upper = (ticker or "").upper()
    if t_upper.startswith("^") or t_upper.startswith("IDX_") or "=" in t_upper:
        return df

    repaired = df.sort_index().copy()
    closes = pd.to_numeric(repaired["Close"], errors="coerce").to_numpy(dtype=float)
    price_cols = [c for c in ("Open", "High", "Low", "Close", "Adj Close") if c in repaired.columns]
    has_vol = "Volume" in repaired.columns

    for i in range(1, len(closes)):
        c_prev = closes[i - 1]
        c_curr = closes[i]
        if not (np.isfinite(c_prev) and np.isfinite(c_curr) and c_prev > 0 and c_curr > 0):
            continue
        raw_ratio = c_prev / c_curr
        if abs(raw_ratio - 1.0) <= 0.35:
            continue
        best_s = None
        best_err = 1e9
        for s in STANDARD_SPLIT_FACTORS:
            err = abs(raw_ratio - s) / s
            if err <= 0.08 and err < best_err:
                best_err = err
                best_s = s
        if best_s is not None:
            logger.info(
                "Back-adjusting unadjusted split discontinuity in %s at %s (ratio %.3f -> split %.2fx)",
                ticker or "<df>",
                str(repaired.index[i])[:10],
                raw_ratio,
                best_s,
            )
            for col in price_cols:
                repaired.iloc[:i, repaired.columns.get_loc(col)] = (
                    pd.to_numeric(repaired[col].iloc[:i], errors="coerce") / best_s
                )
            if has_vol:
                repaired.iloc[:i, repaired.columns.get_loc("Volume")] = (
                    pd.to_numeric(repaired["Volume"].iloc[:i], errors="coerce") * best_s
                )
            closes = pd.to_numeric(repaired["Close"], errors="coerce").to_numpy(dtype=float)

    return repaired


def _rebase_to_provider_adjustment(existing: pd.DataFrame, fresh: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Rescale cached bars when the provider has re-adjusted history (split / reverse split).

    Yahoo returns split-adjusted closes for whatever window is downloaded, so after a split the
    freshly downloaded bars are on a different scale from bars cached before the split.  Bars
    that exist in BOTH frames give the exact adjustment factor; it is applied to the cached bars
    only.  Unlike guessing splits from price jumps, this never rewrites a genuine large move.
    """
    if existing is None or existing.empty or fresh is None or fresh.empty:
        return existing
    common = existing.index.intersection(fresh.index)
    if len(common) == 0 or "Close" not in existing.columns or "Close" not in fresh.columns:
        return existing
    old_c = pd.to_numeric(existing.loc[common, "Close"], errors="coerce")
    new_c = pd.to_numeric(fresh.loc[common, "Close"], errors="coerce")
    ratio = (old_c / new_c).replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio > 0]
    if ratio.empty:
        return existing
    factor = float(ratio.median())
    # Ignore tiny differences (revisions, rounding); require consistency across the overlap.
    if abs(factor - 1.0) <= 0.02 or float((ratio / factor - 1.0).abs().max()) > 0.02:
        return existing
    logger.info("Rescaling cached bars for %s by 1/%.4f to match provider's re-adjusted history (split)", ticker or "<df>", factor)
    out = existing.copy()
    for col in ("Open", "High", "Low", "Close", "Adj Close"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") / factor
    if "Volume" in out.columns:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce") * factor
    return out


class ParquetCache:
    """Manages disk caching of ticker bar data in Parquet format."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.bars_dir = self.cache_dir / "bars"
        self.bars_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, ticker: str, interval: str = "1d") -> Path:
        """Sanitize ticker symbol for safe file path."""
        safe_ticker = ticker.replace("^", "IDX_").replace("/", "_").replace(".", "_")
        return self.bars_dir / f"{safe_ticker}_{interval}.parquet"

    def read_bars(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Read cached bars for a ticker, optionally filtering by date range."""
        path = self._get_path(ticker, interval)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if df.empty:
                return None

            # Ensure DatetimeIndex normalized to midnight tz-naive
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None).normalize()
            else:
                df.index = df.index.normalize()

            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]

            return df if not df.empty else None
        except Exception as e:
            logger.warning("Error reading cache for %s: %s", ticker, e)
            return None

    def write_bars(
        self,
        ticker: str,
        df: pd.DataFrame,
        interval: str = "1d",
    ) -> None:
        """Write or update cached bars for a ticker."""
        if df.empty:
            return

        path = self._get_path(ticker, interval)
        try:
            # Normalize index
            clean_df = df.copy()
            if not isinstance(clean_df.index, pd.DatetimeIndex):
                clean_df.index = pd.to_datetime(clean_df.index)
            if clean_df.index.tz is not None:
                clean_df.index = clean_df.index.tz_convert(None).normalize()
            else:
                clean_df.index = clean_df.index.normalize()

            # Merge with existing if present
            if path.exists():
                try:
                    existing = pd.read_parquet(path)
                    if not isinstance(existing.index, pd.DatetimeIndex):
                        existing.index = pd.to_datetime(existing.index)
                    if existing.index.tz is not None:
                        existing.index = existing.index.tz_convert(None).normalize()
                    else:
                        existing.index = existing.index.normalize()

                    existing = _rebase_to_provider_adjustment(existing, clean_df, ticker=ticker)
                    clean_df = pd.concat([existing, clean_df])
                    clean_df = clean_df[~clean_df.index.duplicated(keep="last")]
                    clean_df = clean_df.sort_index()
                except Exception as ex:
                    logger.debug("Could not merge existing parquet cache for %s: %s", ticker, ex)

            clean_df.to_parquet(path, engine="pyarrow", index=True)
        except Exception as e:
            logger.warning("Error writing cache for %s: %s", ticker, e)
