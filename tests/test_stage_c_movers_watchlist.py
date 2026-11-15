"""Tests for Stage C: Top Movers 'Why It Moved' Catalyst Synthesis & Impersonal Watchlist Analytics."""

from datetime import date
from pathlib import Path
import pandas as pd

from config.settings import load_config
from data.calendar import MarketCalendar
from data.constituents import ConstituentManager
from sections.base import SectionResult
from sections.movers import MoverItem, PeriodMovers, TopMoversData, build_movers
from sections.watchlist import WatchlistData, WatchlistItem, build_watchlist_section
from render.markdown import render_markdown
from render.html import render_html


def test_mover_item_catalyst_driver_synthesis_and_backward_compatibility():
    """Verify MoverItem preserves move_cause_tag compatibility while populating rich catalyst_driver."""
    cal = MarketCalendar()
    cfg = load_config()
    cm = ConstituentManager()
    constituents = cm.get_universe()

    dates = pd.date_range("2025-11-03", "2026-10-30", freq="B")
    n = len(dates)

    # NVDA: 52W high breakout on 2.1x volume without earnings
    df_nvda = pd.DataFrame(
        {"Close": [120.0] * (n - 1) + [135.0], "Volume": [1_000_000.0] * (n - 1) + [2_100_000.0]},
        index=dates,
    )
    # AMD: Semiconductor cohort sympathy move (+6%) on 1.8x volume
    df_amd = pd.DataFrame(
        {"Close": [150.0] * (n - 5) + [140.0, 141.0, 142.0, 140.0, 148.4], "Volume": [1_000_000.0] * (n - 1) + [1_800_000.0]},
        index=dates,
    )
    # INTC: Earnings selloff (-15%) on 3.2x volume
    df_intc = pd.DataFrame(
        {"Close": [40.0] * (n - 1) + [34.0], "Volume": [1_000_000.0] * (n - 1) + [3_200_000.0]},
        index=dates,
    )
    # XOM: 52W low breakdown (-8%) on 1.9x volume
    df_xom = pd.DataFrame(
        {"Close": [110.0] * (n - 1) + [101.2], "Volume": [1_000_000.0] * (n - 1) + [1_900_000.0]},
        index=dates,
    )
    # CVX: Energy drop (-5%)
    df_cvx = pd.DataFrame(
        {"Close": [150.0] * (n - 1) + [142.5], "Volume": [1_000_000.0] * (n - 1) + [1_600_000.0]},
        index=dates,
    )
    # LLY: Earnings beat (+9%) on 2.5x volume
    df_lly = pd.DataFrame(
        {"Close": [800.0] * (n - 1) + [872.0], "Volume": [1_000_000.0] * (n - 1) + [2_500_000.0]},
        index=dates,
    )

    bars = {
        "NVDA": df_nvda,
        "AMD": df_amd,
        "INTC": df_intc,
        "XOM": df_xom,
        "CVX": df_cvx,
        "LLY": df_lly,
    }
    earnings_syms = {"INTC", "LLY"}

    sec = build_movers(
        bars=bars,
        constituents=constituents,
        ticker_infos={},
        calendar=cal,
        as_of_session="2026-10-30",
        movers_cfg=cfg.movers,
        earnings_symbols=earnings_syms,
    )
    assert sec.is_available
    assert isinstance(sec.data, TopMoversData)
    day_movers = sec.data.day

    gainer_map = {g.symbol: g for g in day_movers.gainers}
    loser_map = {l.symbol: l for l in day_movers.losers}

    assert gainer_map["LLY"].move_cause_tag == "Earnings"
    assert "Post-earnings reaction / upside volume" in gainer_map["LLY"].catalyst_driver

    assert loser_map["INTC"].move_cause_tag == "Earnings"
    assert "Post-earnings reaction / downside volume" in loser_map["INTC"].catalyst_driver

    assert "52W high breakout" in gainer_map["NVDA"].catalyst_driver
    assert "cohort sympathy" in gainer_map["NVDA"].catalyst_driver
    assert "cohort sympathy" in gainer_map["AMD"].catalyst_driver

    assert "52W low breakdown" in loser_map["XOM"].catalyst_driver
    assert "cohort sympathy" in loser_map["XOM"].catalyst_driver


def test_watchlist_impersonal_analytics_only(tmp_path: Path):
    """Verify WatchlistItem contains only impersonal analytics and no prescriptive trade levels."""
    w_file = tmp_path / "watchlist.csv"
    w_file.write_text("ticker\nAAPL\nNVDA\n", encoding="utf-8")

    dates = pd.date_range("2025-11-03", "2026-10-30", freq="B")
    n = len(dates)

    closes_aapl = [200.0 + i * 0.25 for i in range(n)]
    df_aapl = pd.DataFrame(
        {
            "Open": closes_aapl,
            "Close": closes_aapl,
            "High": [c + 2.5 for c in closes_aapl],
            "Low": [c - 2.5 for c in closes_aapl],
            "Volume": [50_000_000.0] * n,
        },
        index=dates,
    )
    closes_nvda = [140.0] * (n - 10) + [132.0] * 10
    df_nvda = pd.DataFrame(
        {
            "Open": closes_nvda,
            "Close": closes_nvda,
            "High": [c + 3.0 for c in closes_nvda],
            "Low": [c - 3.0 for c in closes_nvda],
            "Volume": [60_000_000.0] * n,
        },
        index=dates,
    )
    closes_spy = [500.0 + i * 0.3 for i in range(n)]
    df_spy = pd.DataFrame(
        {"Open": closes_spy, "Close": closes_spy, "High": closes_spy, "Low": closes_spy, "Volume": [80_000_000.0] * n},
        index=dates,
    )

    sec = build_watchlist_section(
        watchlist_path=w_file,
        bars={"SPY": df_spy},
        equity_bars={"AAPL": df_aapl, "NVDA": df_nvda},
        gate_posture="CONSTRUCTIVE",
        as_of_date="2026-10-30",
    )
    assert sec.is_available
    assert isinstance(sec.data, WatchlistData)
    assert len(sec.data.items) == 2

    for item in sec.data.items:
        assert not hasattr(item, "entry_zone")
        assert not hasattr(item, "stop_loss")
        assert not hasattr(item, "target_price")
        assert not hasattr(item, "reward_risk")
        assert not hasattr(item, "trade_trigger")


def test_stage_c_markdown_and_html_rendering():
    """Verify Markdown and HTML render Catalyst / Driver and impersonal Watchlist analytics cleanly."""
    cal = MarketCalendar()
    period = cal.resolve_period("daily", date(2026, 11, 3))

    movers_data = TopMoversData(
        day=PeriodMovers(
            period_name="Daily",
            gainers=[
                MoverItem(
                    symbol="NVDA",
                    name="NVIDIA Corp.",
                    sector="Technology",
                    pct_change=6.4,
                    close=138.50,
                    volume=85_000_000,
                    avg_volume_20d=40_000_000,
                    volume_ratio=2.13,
                    market_cap=3.2e12,
                    has_earnings_in_window=False,
                    move_cause_tag="Unknown",
                    catalyst_driver="52W high breakout in AI & Semiconductors (2.1x vol)",
                    market_cap_str="$3.20T",
                )
            ],
            losers=[
                MoverItem(
                    symbol="INTC",
                    name="Intel Corp.",
                    sector="Technology",
                    pct_change=-11.2,
                    close=22.40,
                    volume=95_000_000,
                    avg_volume_20d=30_000_000,
                    volume_ratio=3.17,
                    market_cap=95e9,
                    has_earnings_in_window=True,
                    move_cause_tag="Earnings",
                    catalyst_driver="Post-earnings reaction / downside volume (3.2x vol, AI & Semiconductors)",
                    market_cap_str="$95.0B",
                )
            ],
            theme_summary="Gainers led by AI & Semiconductors; losers driven by post-earnings downside volume.",
        ),
        week=PeriodMovers("Weekly", [], []),
        month=PeriodMovers("Monthly", [], []),
    )

    wl_data = WatchlistData(
        items=[
            WatchlistItem(
                ticker="AAPL",
                name="Apple Inc.",
                sector="Technology",
                close=230.00,
                dist_20dma_pct=1.2,
                dist_50dma_pct=3.5,
                dist_52w_high_pct=-1.0,
                atr_pct=1.8,
                rs_vs_spy=4.2,
                next_earnings_date="none within 30d",
                sector_breadth_status="65.0%",
                long_permitted="Supportive Regime",
                readiness="Ready",
            )
        ],
        gate_posture="Constructive",
        long_permitted_default="Supportive Regime",
        total_count=1,
        permitted_count=1,
        as_of_date="2026-11-03",
    )

    sections = [
        SectionResult(section_num=6, title="Top Movers", data=movers_data),
        SectionResult(section_num=7, title="Watchlist", data=wl_data),
    ]

    md_out = render_markdown(period=period, sections=sections)
    html_out = render_html(period=period, sections=sections)

    assert "Technical Context" in md_out
    assert "52W high breakout in AI & Semiconductors (2.1x vol)" in md_out
    assert "Post-earnings reaction / downside volume (3.2x vol, AI & Semiconductors)" in md_out
    assert "Technical Context" in html_out
    assert "52W high breakout in AI & Semiconductors (2.1x vol)" in html_out

    for forbidden in ("Entry Zone", "Stop (1.5 ATR)", "Target (T1)"):
        assert forbidden not in md_out
        assert forbidden not in html_out
    assert "\\$" not in html_out
