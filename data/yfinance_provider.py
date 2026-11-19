"""YFinance implementation of DataProvider with rate-limiting, retries, and parquet caching."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yfinance as yf

# Suppress benign yfinance pandas deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

from data.base import DataProvider
from data.cache import ParquetCache

logger = logging.getLogger(__name__)


THROTTLE_PATTERNS = ("rate limit", "rate-limit", "too many requests", "yfratelimiterror", " 429")


class _YFLogCapture(logging.Handler):
    """Collects yfinance's own log messages: yf.download swallows per-ticker errors (including
    rate limiting) and only logs them, so this is the only way to see why a batch came back empty."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:
            pass


class YFinanceDataProvider(DataProvider):
    """Yahoo Finance provider built to stay under Yahoo's (unpublished) rate limits.

    - Concurrency is capped (``download_threads``) instead of one thread per ticker in a batch.
    - Batches are paced (``request_delay`` seconds between downloads).
    - An empty batch is retried only after a real backoff, and only when it looks like throttling.
    - Circuit breaker: once Yahoo throttles despite backoff, no further requests are made in this
      run; every remaining symbol is served from the cache (and flagged stale by the report).
    - Requests that cannot succeed are not repeated every run: symbols with no data are skipped
      for a few days, and a full-history request is remembered even when the symbol's history
      starts later than requested (e.g. SPY from 1990 -> 1993).
    - --no-cache refreshes are resumable: symbols fully re-downloaded in the last few days are
      not downloaded again, so a throttled refresh continues where it stopped on the next run.
    """

    DOWNLOAD_THREADS = 4             # concurrent requests inside one yf.download call
    THROTTLE_WAITS = (20.0, 60.0)    # seconds to wait before retrying a throttled batch
    THROTTLE_MIN_BATCH = 5           # an all-empty batch at least this big is treated as throttling
    NO_DATA_SKIP_DAYS = 3            # skip symbols that returned no data (new/delisted) for N days
    FULL_REFRESH_SKIP_DAYS = 3       # --no-cache: symbols replaced within N days are not refetched

    def __init__(
        self,
        cache_dir: str = ".cache",
        batch_size: int = 50,
        request_delay: float = 1.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        force_refresh: bool = False,
    ):
        self.cache = ParquetCache(cache_dir=cache_dir)
        # --no-cache: re-download every ticker's full window once per run and REPLACE its cache file
        # (merging would keep bars written by older code, e.g. ones rescaled by the old split heuristic).
        self.force_refresh = force_refresh
        self._refreshed: set = set()
        self.batch_size = batch_size
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._sleep = time.sleep          # injectable for tests
        self._today = pd.Timestamp.now().normalize()

        # Circuit-breaker state, read by the report to disclose degraded data.
        self.throttled = False
        self.throttle_note = ""
        self.network_requests = 0

        self.state_path = Path(cache_dir) / "fetch_state.json"
        self._state: Dict[str, Dict[str, str]] = self._load_state()

        self.info_cache_path = Path(cache_dir) / "ticker_info.json"
        self._info_cache: Dict[str, Dict[str, Any]] = self._load_info_cache()

    # ---------- persistent fetch state ----------
    def _load_state(self) -> Dict[str, Dict[str, str]]:
        base = {"no_data": {}, "history_from": {}, "full_refresh": {}, "daily_once": {}}
        try:
            if self.state_path.exists():
                with open(self.state_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for k in base:
                    if isinstance(loaded.get(k), dict):
                        base[k] = loaded[k]
        except Exception as e:
            logger.debug("Failed to read fetch state: %s", e)
        return base

    def _save_state(self) -> None:
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=1, sort_keys=True)
        except Exception as e:
            logger.debug("Failed to save fetch state: %s", e)

    def _recent(self, bucket: str, key: str, days: int) -> bool:
        val = self._state[bucket].get(key)
        if not val:
            return False
        try:
            return (self._today - pd.Timestamp(val)).days < days
        except Exception:
            return False

    def once_per_day(self, key: str) -> bool:
        """True the first time `key` is asked for on a given day (and network is usable).

        For best-effort lookups that rarely succeed (e.g. forward P/E), so they cost at most one
        attempt per day instead of one per run.
        """
        if self.throttled or self._recent("daily_once", key, 1):
            return False
        self._state["daily_once"][key] = self._today.strftime("%Y-%m-%d")
        self._save_state()
        return True

    def _trip_breaker(self, reason: str) -> None:
        if not self.throttled:
            self.throttled = True
            self.throttle_note = reason
            logger.warning("Yahoo throttling detected (%s): no further downloads this run; using cached data.", reason)

    def _load_info_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.info_cache_path.exists():
            try:
                with open(self.info_cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.debug("Failed to read ticker_info cache: %s", e)
        return {}

    def _save_info_cache(self) -> None:
        try:
            with open(self.info_cache_path, "w", encoding="utf-8") as f:
                json.dump(self._info_cache, f, indent=2)
        except Exception as e:
            logger.debug("Failed to save ticker_info cache: %s", e)

    def _extract_ticker_df(self, raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Extract a single ticker's DataFrame with clean ['Open', 'High', 'Low', 'Close', 'Volume']."""
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        df = pd.DataFrame()
        if isinstance(raw_df.columns, pd.MultiIndex):
            lvl0 = raw_df.columns.get_level_values(0)
            lvl1 = raw_df.columns.get_level_values(1)

            if ticker in lvl0:
                df = raw_df[ticker].copy()
            elif ticker in lvl1:
                sub = {}
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if (col, ticker) in raw_df.columns:
                        sub[col] = raw_df[(col, ticker)]
                df = pd.DataFrame(sub, index=raw_df.index)
        else:
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw_df.columns]
            df = raw_df[cols].copy()

        if df.empty:
            return pd.DataFrame()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = float("nan")

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        if df.empty:
            return pd.DataFrame()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None).normalize()
        else:
            df.index = df.index.normalize()

        df = df.sort_index()
        return df

    CANARY = "SPY"

    def _canary_ok(self, start_date: str, end_str: str, interval: str) -> bool:
        """One request for a symbol that always has data: tells 'these symbols are dead' apart
        from 'Yahoo is returning nothing to us'."""
        try:
            self.network_requests += 1
            d = yf.download(tickers=[self.CANARY], start=start_date, end=end_str, interval=interval,
                            auto_adjust=False, progress=False, threads=False)
            return d is not None and not d.empty
        except Exception:
            return False

    def _fetch_batch_with_retry(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download one batch politely; returns an empty frame if throttled or nothing came back."""
        if self.throttled or not tickers:
            return pd.DataFrame()
        end_str = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        attempts = 1 + min(len(self.THROTTLE_WAITS), max(0, self.max_retries - 1))
        data = pd.DataFrame()
        for attempt in range(attempts):
            if self.request_delay > 0:
                self._sleep(self.request_delay + random.uniform(0.0, 0.25))
            capture = _YFLogCapture()
            yf_logger = logging.getLogger("yfinance")
            yf_logger.addHandler(capture)
            rate_limited = False
            try:
                self.network_requests += len(tickers)
                data = yf.download(
                    tickers=tickers,
                    start=start_date,
                    end=end_str,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=min(self.DOWNLOAD_THREADS, len(tickers)),
                )
            except Exception as e:
                data = pd.DataFrame()
                rate_limited = "ratelimit" in type(e).__name__.lower() or any(
                    p in str(e).lower() for p in THROTTLE_PATTERNS
                )
                if not rate_limited:
                    logger.warning("Download failed for %d tickers: %s", len(tickers), e)
            finally:
                yf_logger.removeHandler(capture)
            rate_limited = rate_limited or any(
                p in m.lower() for m in capture.messages for p in THROTTLE_PATTERNS
            )
            got_any = data is not None and not data.empty and any(
                not self._extract_ticker_df(data, t).empty for t in tickers
            )
            looks_throttled = rate_limited or (
                not got_any and len(tickers) >= self.THROTTLE_MIN_BATCH and not self._canary_ok(start_date, end_str, interval)
            )
            if got_any and not rate_limited:
                return data
            if not looks_throttled:
                # Small batch with no data and no rate-limit message: most likely new/delisted symbols.
                return data if data is not None else pd.DataFrame()
            if attempt < attempts - 1:
                wait = self.THROTTLE_WAITS[attempt] + random.uniform(0.0, 3.0)
                logger.warning(
                    "Batch of %d tickers came back %s; waiting %.0fs before one more try",
                    len(tickers), "rate-limited" if rate_limited else "empty", wait,
                )
                self._sleep(wait)
                continue
            self._trip_breaker(
                f"{'rate-limit response' if rate_limited else 'empty batch'} for {len(tickers)} tickers "
                f"after {attempts} attempts"
            )
        return data if data is not None else pd.DataFrame()

    def get_history(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV bars with cache-first lookup and chunked network downloads."""
        results: Dict[str, pd.DataFrame] = {}
        missing_tickers: List[str] = []

        target_start = pd.to_datetime(start_date)
        target_end = pd.to_datetime(end_date)

        def _covers_start(df: pd.DataFrame, t: str) -> bool:
            if df.index.min() <= target_start + pd.Timedelta(days=15):
                return True
            # A full download from an earlier start already showed history begins later (SPY: 1993).
            known = self._state["history_from"].get(t)
            return bool(known) and pd.Timestamp(known) <= target_start

        def _cached_window(t: str) -> pd.DataFrame:
            c = self.cache.read_bars(t, start_date=start_date, end_date=end_date, interval=interval)
            return c if c is not None else pd.DataFrame()

        for t in tickers:
            cached_df = self.cache.read_bars(t, start_date=None, end_date=None, interval=interval)
            has_cache = cached_df is not None and not cached_df.empty
            if self.force_refresh and t not in self._refreshed:
                if has_cache and self._recent("full_refresh", t, self.FULL_REFRESH_SKIP_DAYS) and _covers_start(cached_df, t):
                    self._refreshed.add(t)  # replaced by a recent --no-cache run; resume with the rest
                else:
                    missing_tickers.append(t)
                    continue
            if has_cache:
                cached_max = cached_df.index.max()
                if cached_max >= target_end and _covers_start(cached_df, t):
                    sub = cached_df[(cached_df.index >= target_start) & (cached_df.index <= target_end)]
                    results[t] = sub if not sub.empty else cached_df
                    continue
            elif self._recent("no_data", t, self.NO_DATA_SKIP_DAYS):
                results[t] = pd.DataFrame()  # returned nothing recently (new/delisted): don't ask again yet
                continue
            missing_tickers.append(t)

        if not missing_tickers:
            return results
        if self.throttled:
            for t in missing_tickers:
                results[t] = _cached_window(t)
            return results

        # Separate into stale tickers (cover start date, only need recent delta) vs fresh tickers (need full history)
        stale_tickers: List[str] = []
        fresh_tickers: List[str] = []
        for t in missing_tickers:
            if self.force_refresh and t not in self._refreshed:
                fresh_tickers.append(t)
                continue
            c_df = self.cache.read_bars(t, start_date=None, end_date=None, interval=interval)
            if c_df is not None and not c_df.empty and _covers_start(c_df, t):
                stale_tickers.append(t)
            else:
                fresh_tickers.append(t)

        # 1. Fetch fresh tickers needing full history
        if fresh_tickers:
            logger.info("Fetching full history for %d new/uninitialized tickers (%s to %s)", len(fresh_tickers), start_date, end_date)
            for i in range(0, len(fresh_tickers), self.batch_size):
                chunk = fresh_tickers[i : i + self.batch_size]
                was_throttled = self.throttled
                raw_data = self._fetch_batch_with_retry(chunk, start_date, end_date, interval)
                for t in chunk:
                    ticker_df = self._extract_ticker_df(raw_data, t)
                    if not ticker_df.empty:
                        replace = self.force_refresh and t not in self._refreshed
                        if replace:
                            # Never trade a long cached history for a shorter download (e.g. the
                            # 800-day core window replacing ^GSPC's 1990+ history): merge instead,
                            # where the overlap check still rescales for splits.
                            prev = self.cache.read_bars(t, interval=interval)
                            if prev is not None and not prev.empty and ticker_df.index.min() > prev.index.min() + pd.Timedelta(days=15):
                                replace = False
                        self.cache.write_bars(t, ticker_df, interval=interval, replace=replace)
                        self._refreshed.add(t)
                        if replace:
                            self._state["full_refresh"][t] = self._today.strftime("%Y-%m-%d")
                        if ticker_df.index.min() > target_start + pd.Timedelta(days=15):
                            self._state["history_from"][t] = start_date
                        self._state["no_data"].pop(t, None)
                        full_df = self.cache.read_bars(t, start_date=start_date, end_date=end_date, interval=interval)
                        results[t] = full_df if full_df is not None and not full_df.empty else ticker_df
                    else:
                        # Download failed: fall back to whatever is cached (staleness is flagged by the report).
                        c_df = _cached_window(t)
                        results[t] = c_df
                        if c_df.empty and not self.throttled and not was_throttled:
                            self._state["no_data"][t] = self._today.strftime("%Y-%m-%d")
            self._save_state()

        # 2. Fetch incremental delta for stale tickers
        if stale_tickers:
            # Start the incremental window inside the cached range so (a) there is no gap when the
            # cache is more than 15 days old and (b) overlapping bars reveal split re-adjustments.
            oldest_cached_max = min(
                (self.cache.read_bars(t, interval=interval).index.max() for t in stale_tickers),
                default=target_end,
            )
            oldest_cached_max = max(oldest_cached_max, target_end - pd.Timedelta(days=120))
            recent_start_ts = min(target_end - pd.Timedelta(days=15), oldest_cached_max - pd.Timedelta(days=7))
            recent_start = recent_start_ts.strftime("%Y-%m-%d")
            logger.info("Fetching incremental update for %d cached tickers (%s to %s)", len(stale_tickers), recent_start, end_date)
            for i in range(0, len(stale_tickers), self.batch_size):
                chunk = stale_tickers[i : i + self.batch_size]
                if self.throttled:
                    for t in chunk:
                        results[t] = _cached_window(t)
                    continue
                raw_data = self._fetch_batch_with_retry(chunk, recent_start, end_date, interval)
                for t in chunk:
                    ticker_df = self._extract_ticker_df(raw_data, t)
                    if not ticker_df.empty:
                        self.cache.write_bars(t, ticker_df, interval=interval)
                        full_df = self.cache.read_bars(t, start_date=start_date, end_date=end_date, interval=interval)
                        results[t] = full_df if full_df is not None and not full_df.empty else ticker_df
                    else:
                        c_df = self.cache.read_bars(t, start_date=start_date, end_date=end_date, interval=interval)
                        results[t] = c_df if c_df is not None else pd.DataFrame()

        return results

    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch ticker metadata with local disk caching."""
        if ticker in self._info_cache:
            return self._info_cache[ticker]
        if self.throttled:
            return {"symbol": ticker, "shortName": ticker, "sector": "Unknown", "marketCap": 0, "regularMarketPrice": 0.0}

        info_dict: Dict[str, Any] = {
            "symbol": ticker,
            "shortName": ticker,
            "sector": "Unknown",
            "marketCap": 0,  # unknown -> shown as n/a; never a fabricated $10B
            "regularMarketPrice": 0.0,
        }

        try:
            t = yf.Ticker(ticker)
            fast = getattr(t, "fast_info", None)
            if fast:
                mcap = getattr(fast, "market_cap", None)
                if mcap:
                    info_dict["marketCap"] = mcap
                last_p = getattr(fast, "last_price", None)
                if last_p:
                    info_dict["regularMarketPrice"] = last_p

            self._info_cache[ticker] = info_dict
        except Exception as e:
            logger.debug("Failed to fetch info for %s: %s", ticker, e)

        return info_dict

    def get_bulk_ticker_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch ticker metadata concurrently with thread pool."""
        res: Dict[str, Dict[str, Any]] = {}
        missing = [t for t in tickers if t not in self._info_cache]

        for t in tickers:
            if t in self._info_cache:
                res[t] = self._info_cache[t]

        if missing:
            logger.info("Fetching metadata for %d tickers in background threadpool", len(missing))
            with ThreadPoolExecutor(max_workers=self.DOWNLOAD_THREADS) as executor:
                futures = {executor.submit(self.get_ticker_info, t): t for t in missing}
                for fut in as_completed(futures):
                    t = futures[fut]
                    try:
                        res[t] = fut.result()
                    except Exception:
                        res[t] = {"symbol": t, "shortName": t, "sector": "Unknown", "marketCap": 0}
            self._save_info_cache()

        return res
