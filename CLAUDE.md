# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt && pip install -e .   # setup (run.sh / run.bat do this automatically)
python report.py -t daily -O                          # generate report (-t daily|weekly|monthly, -d YYYY-MM-DD, -O opens HTML)
python report.py --tickers NVDA,AAPL -O              # ad-hoc watchlist (or -w watchlist.csv)
python report.py --no-cache -q                        # bypass price cache / quiet console
python report.py --dry-run-email                      # writes .eml without sending (--email actually sends via SMTP)
pytest                                                # full suite (pytest.ini: pythonpath=., testpaths=tests)
pytest tests/test_scoring.py::test_name -v            # single test
```

Windows: `run.bat`; Unix: `run.sh`. Running `report.py` with no args starts an interactive wizard.

## Architecture

`report.py` (`generate_market_report`, ~1000 lines) is the orchestrator: it fetches data, runs every stage, builds all sections, renders, then runs QA. Pipeline:

1. **data/** – `YFinanceDataProvider` (parquet OHLCV cache in `.cache/`, chunking/backoff) is the primary source, no API keys needed. Optional FRED (`fred_provider`), Finnhub (`calendar_provider`), COT (`cot_provider`) feeds degrade gracefully when keys are absent. `calendar.py` wraps `exchange_calendars` and defines `PeriodInfo` (the current vs. baseline period for daily/weekly/monthly). `data/cache/*.json` are committed caches (CAPE valuation, COT, model accountability).
2. **indicators/** – pure computation (SMA, RSI, MACD/ADX, breadth). `engine.compute_technicals` returns a frozen `TechnicalSnapshot`.
3. **scoring/** – `rules.py` casts +1/0/-1 votes in three categories (technical, macro, breadth); `engine.TrendScoringEngine` applies weights from `config/config.yaml` and normalizes to [-1, 1] → Bullish/Neutral/Bearish (±0.30).
4. **sections/** – one module per report section, each returning a `SectionResult` (`sections/base.py`) with `status` "available"/"unavailable" so a failing section never aborts the report. `CanonicalFactsRegistry` (frozen, in `sections/base.py`) is the single source of truth for cross-section numbers (SPX close, VIX, yields, breadth); sections should read shared figures from it rather than recompute, so the report stays internally consistent.
5. **render/** – `markdown.py`, `html.py` (standalone, base64-embedded charts, light/dark), `charts.py` (matplotlib PNGs), `svg_charts.py`. New section data must be wired into both markdown and HTML renderers.
6. **qa_checks.py** – post-generation harness (checks T1–T12) that parses the produced HTML/section data for cross-section contradictions and writes `qa_report.md`; a FAIL fails the build. It imports each section's `*Data` class, so renaming those breaks QA.
7. **storage/snapshots.py** – SQLite store of prior-period state used for "what changed" deltas; **delivery/email_sender.py** – SMTP with inline CID charts and `.eml` export.

Config: `config/config.yaml` (symbols, weights, thresholds, filters) loaded into typed `AppConfig` via `config/settings.py`. Secrets/optional keys come from `.env` (see `.env.example`). When adding a new top-level package, also add it to `[tool.setuptools] packages` in `pyproject.toml`.

## Notes

- `tests/` is organized largely by change-request stage (`test_stage*_v2x.py`, `test_stage_a/b/c_*`), not by module; grep for the section/function name to find relevant tests. Tests should not hit the network (see `tests/conftest.py`).
- Generated output (`reports/`, `.cache/`, `logs/`) is git-ignored. CI (`.github/workflows/market_report.yml`) runs at 21:00 UTC Mon–Fri with `--if-scheduled-day` and deploys to GitHub Pages.
