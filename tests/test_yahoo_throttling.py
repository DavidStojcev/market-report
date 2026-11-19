"""Yahoo rate-limit safety: concurrency cap, backoff, circuit breaker, and no pointless re-requests.

yf.download is replaced by a fake, so these tests never touch the network.
"""
import logging

import numpy as np
import pandas as pd
import pytest

import data.yfinance_provider as yp
from data.cache import ParquetCache
from data.yfinance_provider import YFinanceDataProvider

END = "2026-11-13"


def _bars(start, end=END, base=100.0):
    idx = pd.bdate_range(start, end)
    c = np.linspace(base, base * 1.1, len(idx))
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6}, index=idx)


class FakeYahoo:
    """Records every call; `mode` decides what Yahoo 'returns'."""

    def __init__(self, mode="ok", dead=(), history_start=None):
        self.mode, self.dead, self.history_start = mode, set(dead), history_start
        self.calls = []

    def __call__(self, tickers, start, end, interval="1d", auto_adjust=False, progress=False, threads=True):
        tickers = list(tickers)
        self.calls.append({"tickers": tickers, "start": start, "threads": threads})
        if self.mode == "throttled":
            return pd.DataFrame()
        if self.mode == "ratelimit_log":
            logging.getLogger("yfinance").error("['X']: YFRateLimitError('Too Many Requests. Rate limited.')")
        frames = {}
        for t in tickers:
            if t in self.dead:
                continue
            s = max(pd.Timestamp(start), pd.Timestamp(self.history_start)) if self.history_start else pd.Timestamp(start)
            frames[t] = _bars(s, min(pd.Timestamp(end) - pd.Timedelta(days=1), pd.Timestamp(END)))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    def symbols_requested(self):
        return [t for c in self.calls for t in c["tickers"]]


def _provider(tmp_path, monkeypatch, fake, **kw):
    monkeypatch.setattr(yp.yf, "download", fake)
    p = YFinanceDataProvider(cache_dir=str(tmp_path), request_delay=0.0, batch_size=kw.pop("batch_size", 10), **kw)
    p._sleep = lambda s: None
    return p


def test_concurrency_is_capped(tmp_path, monkeypatch):
    fake = FakeYahoo()
    p = _provider(tmp_path, monkeypatch, fake, batch_size=50)
    p.get_history([f"T{i}" for i in range(30)], "2026-06-01", END)
    assert fake.calls and all(c["threads"] <= YFinanceDataProvider.DOWNLOAD_THREADS for c in fake.calls)


def test_throttle_trips_breaker_and_stops_all_further_requests(tmp_path, monkeypatch):
    fake = FakeYahoo(mode="throttled")
    waits = []
    p = _provider(tmp_path, monkeypatch, fake)
    p._sleep = waits.append
    out = p.get_history([f"T{i}" for i in range(30)], "2026-06-01", END)
    assert p.throttled and "empty batch" in p.throttle_note
    assert waits[:2] == pytest.approx(list(YFinanceDataProvider.THROTTLE_WAITS), abs=3.1)  # real backoff, not instant retries
    n_calls = len(fake.calls)
    # first batch: 3 attempts + 1 canary each time it was ambiguous; then nothing else for the run
    assert n_calls <= 6
    p.get_history(["SPY", "QQQ"], "2026-06-01", END)
    assert len(fake.calls) == n_calls
    assert all(df.empty for df in out.values())


def test_rate_limit_message_is_detected_even_with_partial_data(tmp_path, monkeypatch):
    fake = FakeYahoo(mode="ratelimit_log")
    p = _provider(tmp_path, monkeypatch, fake)
    p.get_history([f"T{i}" for i in range(3)], "2026-06-01", END)
    assert p.throttled and "rate-limit" in p.throttle_note


def test_throttled_run_serves_cached_bars(tmp_path, monkeypatch):
    ParquetCache(str(tmp_path)).write_bars("AAA", _bars("2026-01-02", "2026-11-06"))
    p = _provider(tmp_path, monkeypatch, FakeYahoo(mode="throttled"))
    p._trip_breaker("test")
    out = p.get_history(["AAA"], "2026-06-01", END)
    assert out["AAA"].index.max() == pd.Timestamp("2026-11-06")  # stale but present (report flags it)


def test_dead_symbols_do_not_trip_breaker_and_are_not_re_requested(tmp_path, monkeypatch):
    dead = [f"DEAD{i}" for i in range(6)]
    fake = FakeYahoo(dead=dead)
    p = _provider(tmp_path, monkeypatch, fake)
    out = p.get_history(dead, "2026-06-01", END)
    assert not p.throttled  # canary (SPY) returned data, so these are just symbols without data
    assert all(out[t].empty for t in dead)
    fake2 = FakeYahoo(dead=dead)
    p2 = _provider(tmp_path, monkeypatch, fake2)
    p2.get_history(dead + ["LIVE"], "2026-06-01", END)
    assert fake2.symbols_requested() == ["LIVE"]


def test_late_inception_history_is_not_redownloaded_every_run(tmp_path, monkeypatch):
    fake = FakeYahoo(history_start="1993-01-29")  # SPY: asking from 1990 returns data from 1993
    p = _provider(tmp_path, monkeypatch, fake)
    p.get_history(["SPY"], "1990-01-01", END)
    assert len(fake.calls) == 1
    fake2 = FakeYahoo(history_start="1993-01-29")
    p2 = _provider(tmp_path, monkeypatch, fake2)
    out = p2.get_history(["SPY"], "1990-01-01", END)
    assert fake2.calls == [] and out["SPY"].index.min() == pd.Timestamp("1993-01-29")


def test_no_cache_refresh_resumes_after_throttling(tmp_path, monkeypatch):
    tickers = [f"T{i}" for i in range(20)]
    cache = ParquetCache(str(tmp_path))
    for t in tickers:
        cache.write_bars(t, _bars("2025-06-02", "2026-11-06"))

    class FirstBatchOnly(FakeYahoo):
        def __call__(self, tickers, *a, **k):
            if len(self.calls) >= 1:
                self.calls.append({"tickers": list(tickers), "start": a[0] if a else None, "threads": k.get("threads")})
                return pd.DataFrame()
            return super().__call__(tickers, *a, **k)

    fake = FirstBatchOnly()
    p = _provider(tmp_path, monkeypatch, fake, force_refresh=True)
    p.get_history(tickers, "2025-06-02", END)
    assert p.throttled
    done_first = set(tickers[:10])

    fake2 = FakeYahoo()
    p2 = _provider(tmp_path, monkeypatch, fake2, force_refresh=True)
    p2.get_history(tickers, "2025-06-02", END)
    requested = set(fake2.symbols_requested()) - {"SPY"}
    assert requested == set(tickers) - done_first  # only the ones the throttled run could not finish


def test_no_cache_never_shortens_a_long_cached_history(tmp_path, monkeypatch):
    ParquetCache(str(tmp_path)).write_bars("^GSPC", _bars("1990-01-02", "2026-11-06"))
    p = _provider(tmp_path, monkeypatch, FakeYahoo(), force_refresh=True)
    p.get_history(["^GSPC"], "2024-09-01", END)  # short core window first
    kept = ParquetCache(str(tmp_path)).read_bars("^GSPC")
    assert kept.index.min() == pd.Timestamp("1990-01-02") and kept.index.max() == pd.Timestamp(END)
