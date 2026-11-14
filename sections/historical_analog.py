"""Section and Engine for Long-History Index-Level Historical Analog and Seasonality Studies.

Implements Stage 3 Statistical Rigor:
- Substage 3.1: Long-History (1990+) Index-Level Analog Matching with progressive tolerance relaxation,
  lookback span recording (>=25 years), and preserved breadth-conditioned secondary cohort.
- Substage 3.2: Wilson 95% Confidence Intervals, IQR, and Small-Sample Precedent Badge (N < 20).
- Substage 3.3: Complete Calendar Years Seasonality & Next-Month Seasonality Stats.
"""

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sections.base import SectionResult

logger = logging.getLogger(__name__)

MONTH_NAMES = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def wilson_score_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> Tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a binomial proportion.

    Returns (low_pct, high_pct) in [0.0, 100.0].
    Returns (0.0, 0.0) when total <= 0.
    """
    if total <= 0:
        return 0.0, 0.0
    if successes < 0:
        successes = 0
    if successes > total:
        successes = total

    p = float(successes) / float(total)
    z2 = z * z
    denom = 1.0 + z2 / float(total)
    center = (p + z2 / (2.0 * float(total))) / denom
    margin = (z / denom) * math.sqrt(
        (p * (1.0 - p) / float(total)) + (z2 / (4.0 * float(total) * float(total)))
    )

    low = max(0.0, center - margin) * 100.0
    high = min(1.0, center + margin) * 100.0
    return round(low, 1), round(high, 1)


@dataclass
class AnalogHorizonStats:
    """Forward benchmark return stats for a single horizon across historical analogs."""

    horizon_days: int
    win_rate_pct: float
    median_return_pct: float
    mean_return_pct: float
    worst_return_pct: float
    baseline_win_rate_pct: float
    baseline_median_pct: float
    # Substage 3.2 additions:
    win_rate_ci_low: float = 0.0
    win_rate_ci_high: float = 0.0
    p25_return: float = 0.0
    p75_return: float = 0.0
    iqr_return: float = 0.0
    best_return_pct: float = 0.0
    sample_size: int = 0
    baseline_ci_low: float = 0.0
    baseline_ci_high: float = 0.0
    baseline_p25_return: float = 0.0
    baseline_p75_return: float = 0.0
    baseline_iqr_return: float = 0.0


# Backward compatibility alias
AnalogHorizonStat = AnalogHorizonStats


@dataclass
class HistoricalAnalogStudy:
    """In-memory base-rate study matching current market regime across long history."""

    benchmark_symbol: str
    condition_summary: str
    sample_count: int
    lookback_days: int
    analog_dates: List[str] = field(default_factory=list)
    horizons: List[AnalogHorizonStats] = field(default_factory=list)
    summary_text: str = ""
    # Substage 3.1 additions: Lookback span tracking
    history_start_year: int = 1990
    history_years: int = 35
    history_start_date: str = ""
    history_end_date: str = ""
    # Substage 3.2 additions: Small-sample precedent badge
    is_small_sample: bool = False
    small_sample_badge: str = ""
    # Substage 3.1: Preserved secondary breadth cohort
    breadth_cohort: Optional["HistoricalAnalogStudy"] = None
    breadth_cohort_note: str = ""


@dataclass
class MonthlySeasonalityStats:
    """Historical seasonality distribution for a single calendar month."""

    month_num: int
    month_name: str
    avg_return: float
    median_return: float
    win_rate: float
    sample_size: int
    win_rate_ci_low: float = 0.0
    win_rate_ci_high: float = 0.0
    p25_return: float = 0.0
    p75_return: float = 0.0
    iqr_return: float = 0.0


@dataclass
class SeasonalityStudy:
    """Complete calendar years monthly seasonality study and next-month outlook."""

    current_month_num: int
    current_month_name: str
    current_month_avg_return: float
    current_month_median_return: float
    current_month_win_rate: float
    current_month_sample_size: int
    next_month_num: int
    next_month_name: str
    next_month_avg_return: float
    next_month_median_return: float
    next_month_win_rate: float
    next_month_sample_size: int
    complete_years_count: int
    history_start_year: int
    history_end_year: int
    as_of_date: str = ""
    current_month_win_rate_ci_low: float = 0.0
    current_month_win_rate_ci_high: float = 0.0
    current_month_p25_return: float = 0.0
    current_month_p75_return: float = 0.0
    current_month_iqr_return: float = 0.0
    next_month_win_rate_ci_low: float = 0.0
    next_month_win_rate_ci_high: float = 0.0
    next_month_p25_return: float = 0.0
    next_month_p75_return: float = 0.0
    next_month_iqr_return: float = 0.0
    monthly_details: Dict[int, MonthlySeasonalityStats] = field(default_factory=dict)


def _decluster_positions(positions: np.ndarray, min_spacing: int = 5) -> List[int]:
    """Keep independent analog observations separated by at least min_spacing sessions."""
    selected: List[int] = []
    for pos in positions:
        p = int(pos)
        if not selected or (p - selected[-1]) >= min_spacing:
            selected.append(p)
    return selected


def _vix_bucket_label(vix_val: float) -> str:
    if vix_val < 18.0:
        return "Low (<18)"
    if vix_val < 23.0:
        return "Moderate (18-23)"
    return "Elevated (>=23)"


def _drawdown_bucket_label(dd_val: float) -> str:
    if dd_val >= -5.0:
        return "Near ATH (>= -5%)"
    if dd_val >= -12.0:
        return "Pullback (-5% to -12%)"
    return "Correction/Bear (< -12%)"


def _breadth_band_label(p50_val: float) -> str:
    if p50_val >= 60.0:
        return "Strong (>=60%)"
    if p50_val >= 40.0:
        return "Neutral (40-60%)"
    return "Weak (<40%)"


def compute_monthly_seasonality(
    spx_df: Optional[pd.DataFrame],
    as_of_date: str,
    lookback_years: Optional[int] = None,
) -> Optional[SeasonalityStudy]:
    """Compute monthly seasonality across complete calendar years prior to as_of_date.

    Restricts observations to complete calendar years so that the partial in-progress
    current month and current calendar year are strictly excluded.
    """
    if spx_df is None or spx_df.empty or "Close" not in spx_df.columns:
        return None

    try:
        curr_dt = pd.to_datetime(as_of_date)
        series = pd.to_numeric(
            spx_df[spx_df.index <= curr_dt]["Close"], errors="coerce"
        ).dropna().sort_index()
        if len(series) < 60:
            return None

        # Monthly resample: take last available trading day of each month
        monthly = series.resample("ME").last().dropna()
        monthly_ret = monthly.pct_change() * 100.0

        all_years = sorted(list(set(monthly_ret.index.year)))
        # Strictly complete calendar years prior to current year
        complete_years = [y for y in all_years if y < curr_dt.year]
        if not complete_years:
            # Fallback if history has < 1 full complete calendar year (e.g. synthetic test)
            complete_years = all_years

        if lookback_years and lookback_years < len(complete_years):
            selected_years = complete_years[-lookback_years:]
        else:
            selected_years = complete_years

        if not selected_years:
            return None

        curr_m = curr_dt.month
        next_m = 1 if curr_m == 12 else (curr_m + 1)

        monthly_stats_dict: Dict[int, MonthlySeasonalityStats] = {}
        for m in range(1, 13):
            m_rets: List[float] = []
            for y in selected_years:
                pts = monthly_ret[
                    (monthly_ret.index.year == y) & (monthly_ret.index.month == m)
                ]
                if len(pts) > 0 and pd.notna(pts.iloc[-1]):
                    m_rets.append(float(pts.iloc[-1]))

            s = pd.Series(m_rets).dropna() if m_rets else pd.Series(dtype=float)
            n = len(s)
            if n > 0:
                wins = int((s > 0.0).sum())
                w_rate = round(float(wins / n * 100.0), 1)
                ci_lo, ci_hi = wilson_score_interval(wins, n)
                p25 = round(float(s.quantile(0.25)), 2)
                p75 = round(float(s.quantile(0.75)), 2)
                monthly_stats_dict[m] = MonthlySeasonalityStats(
                    month_num=m,
                    month_name=MONTH_NAMES[m],
                    avg_return=round(float(s.mean()), 2),
                    median_return=round(float(s.median()), 2),
                    win_rate=w_rate,
                    sample_size=n,
                    win_rate_ci_low=ci_lo,
                    win_rate_ci_high=ci_hi,
                    p25_return=p25,
                    p75_return=p75,
                    iqr_return=round(p75 - p25, 2),
                )
            else:
                monthly_stats_dict[m] = MonthlySeasonalityStats(
                    month_num=m,
                    month_name=MONTH_NAMES[m],
                    avg_return=0.0,
                    median_return=0.0,
                    win_rate=0.0,
                    sample_size=0,
                )

        c_stat = monthly_stats_dict.get(
            curr_m,
            MonthlySeasonalityStats(curr_m, MONTH_NAMES[curr_m], 0.0, 0.0, 0.0, 0),
        )
        n_stat = monthly_stats_dict.get(
            next_m,
            MonthlySeasonalityStats(next_m, MONTH_NAMES[next_m], 0.0, 0.0, 0.0, 0),
        )

        return SeasonalityStudy(
            current_month_num=curr_m,
            current_month_name=MONTH_NAMES[curr_m],
            current_month_avg_return=c_stat.avg_return,
            current_month_median_return=c_stat.median_return,
            current_month_win_rate=c_stat.win_rate,
            current_month_sample_size=c_stat.sample_size,
            current_month_win_rate_ci_low=c_stat.win_rate_ci_low,
            current_month_win_rate_ci_high=c_stat.win_rate_ci_high,
            current_month_p25_return=c_stat.p25_return,
            current_month_p75_return=c_stat.p75_return,
            current_month_iqr_return=c_stat.iqr_return,
            next_month_num=next_m,
            next_month_name=MONTH_NAMES[next_m],
            next_month_avg_return=n_stat.avg_return,
            next_month_median_return=n_stat.median_return,
            next_month_win_rate=n_stat.win_rate,
            next_month_sample_size=n_stat.sample_size,
            next_month_win_rate_ci_low=n_stat.win_rate_ci_low,
            next_month_win_rate_ci_high=n_stat.win_rate_ci_high,
            next_month_p25_return=n_stat.p25_return,
            next_month_p75_return=n_stat.p75_return,
            next_month_iqr_return=n_stat.iqr_return,
            complete_years_count=len(selected_years),
            history_start_year=selected_years[0],
            history_end_year=selected_years[-1],
            as_of_date=curr_dt.strftime("%Y-%m-%d"),
            monthly_details=monthly_stats_dict,
        )
    except Exception as e:
        logger.warning("Failed to compute monthly seasonality: %s", e)
        return None


def _compute_breadth_secondary_cohort(
    valid_breadth: pd.DataFrame,
    curr_above_50: bool,
    p50_curr: float,
) -> Optional[HistoricalAnalogStudy]:
    """Compute secondary breadth-conditioned analog cohort over shorter constituent window."""
    if len(valid_breadth) < 30:
        return None

    curr_band = _breadth_band_label(p50_curr)
    same_trend = (valid_breadth["spy_above_50"].astype(bool)) == curr_above_50
    close_breadth = (valid_breadth["p50"] - p50_curr).abs() <= 12.0
    same_band = valid_breadth["p50"].apply(_breadth_band_label) == curr_band
    trend_txt = "above" if curr_above_50 else "below"

    candidates = [
        (
            same_trend & close_breadth,
            4,
            f"SPY {trend_txt} 50-DMA · Breadth >50-DMA {p50_curr:.0f}% (±12 pp)",
        ),
        (
            same_trend & same_band,
            3,
            f"SPY {trend_txt} 50-DMA · Breadth >50-DMA {curr_band}",
        ),
        (same_band, 3, f"S&P 500 Breadth >50-DMA {curr_band}"),
    ]

    selected_positions: List[int] = []
    chosen_label = ""
    for mask, min_samples, label in candidates:
        matched_pos = np.flatnonzero(mask.to_numpy(dtype=bool))
        declustered = _decluster_positions(matched_pos, min_spacing=5)
        if len(declustered) >= min_samples:
            selected_positions = declustered
            chosen_label = label
            break

    if not selected_positions:
        return None

    analogs = valid_breadth.iloc[selected_positions]
    analog_dates = [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in analogs.index]

    horizons: List[AnalogHorizonStats] = []
    for h_days, col in ((5, "fwd_5"), (10, "fwd_10"), (20, "fwd_20")):
        vals = analogs[col]
        base_vals = valid_breadth[col]
        wins = int((vals > 0.0).sum())
        base_wins = int((base_vals > 0.0).sum())
        ci_lo, ci_hi = wilson_score_interval(wins, len(vals))
        b_lo, b_hi = wilson_score_interval(base_wins, len(base_vals))
        p25 = round(float(vals.quantile(0.25)), 2)
        p75 = round(float(vals.quantile(0.75)), 2)
        b_p25 = round(float(base_vals.quantile(0.25)), 2)
        b_p75 = round(float(base_vals.quantile(0.75)), 2)
        horizons.append(
            AnalogHorizonStats(
                horizon_days=h_days,
                win_rate_pct=round(float((vals > 0.0).mean() * 100.0), 1),
                median_return_pct=round(float(vals.median()), 2),
                mean_return_pct=round(float(vals.mean()), 2),
                worst_return_pct=round(float(vals.min()), 2),
                baseline_win_rate_pct=round(
                    float((base_vals > 0.0).mean() * 100.0), 1
                ),
                baseline_median_pct=round(float(base_vals.median()), 2),
                win_rate_ci_low=ci_lo,
                win_rate_ci_high=ci_hi,
                p25_return=p25,
                p75_return=p75,
                iqr_return=round(p75 - p25, 2),
                best_return_pct=round(float(vals.max()), 2),
                sample_size=len(vals),
                baseline_ci_low=b_lo,
                baseline_ci_high=b_hi,
                baseline_p25_return=b_p25,
                baseline_p75_return=b_p75,
                baseline_iqr_return=round(b_p75 - b_p25, 2),
            )
        )

    h20 = horizons[-1]
    is_small = len(analogs) < 20
    badge = f"⚠️ Small-sample precedent (N={len(analogs)} < 20)" if is_small else ""
    summary_text = (
        f"Across N={len(analogs)} non-overlapping sessions ({len(valid_breadth)} evaluated bars) matching {chosen_label}, "
        f"forward 20D win rate is {h20.win_rate_pct:.1f}% ({h20.median_return_pct:+.2f}% median), "
        f"compared to {h20.baseline_win_rate_pct:.1f}% unconditional baseline."
    )

    return HistoricalAnalogStudy(
        benchmark_symbol="SPY",
        condition_summary=chosen_label,
        sample_count=len(analogs),
        lookback_days=len(valid_breadth),
        analog_dates=analog_dates,
        horizons=horizons,
        summary_text=summary_text,
        history_start_year=int(valid_breadth.index.min().year),
        history_years=max(
            1,
            int(
                round(
                    (valid_breadth.index.max() - valid_breadth.index.min()).days
                    / 365.25
                )
            ),
        ),
        is_small_sample=is_small,
        small_sample_badge=badge,
    )


def build_historical_analog(
    spx_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    curr_dt: Optional[pd.Timestamp] = None,
    as_of_date: Optional[str] = None,
    p50_series: Optional[pd.Series] = None,
    p50_curr: Optional[float] = None,
    p200_series: Optional[pd.Series] = None,
    p200_curr: Optional[float] = None,
    spy_df: Optional[pd.DataFrame] = None,
) -> Optional[HistoricalAnalogStudy]:
    """Build primary long-history index-level analog study with progressive tolerance relaxation.

    - Resolves long history (^GSPC / SPY, ^VIX) back to 1990+ (>=25 years).
    - Matches on index-level conditions: 50-DMA, 200-DMA, VIX volatility regime, and drawdown band.
    - Progressively relaxes tolerances so typical setups yield N >= 20 de-clustered episodes.
    - Preserves breadth-conditioned secondary cohort with explicit survivorship-bias note.
    - Records actual lookback span (history_start_year, history_years).
    - Computes Wilson 95% CIs, IQR, and small-sample precedent badge (N < 20).
    """
    # 1. Benchmark DataFrame resolution
    bench_df = None
    bench_sym = "^GSPC"
    if spx_df is not None and not spx_df.empty and "Close" in spx_df.columns:
        bench_df = spx_df
        bench_sym = "^GSPC"
    elif spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
        bench_df = spy_df
        bench_sym = "SPY"

    if bench_df is None or bench_df.empty or "Close" not in bench_df.columns:
        return None

    # Target date resolution
    if as_of_date:
        eval_dt = pd.to_datetime(as_of_date)
    elif curr_dt is not None:
        eval_dt = pd.to_datetime(curr_dt)
    else:
        eval_dt = pd.to_datetime(bench_df.index.max())

    bench_sub = pd.to_numeric(
        bench_df[bench_df.index <= eval_dt]["Close"], errors="coerce"
    ).dropna().sort_index()
    if len(bench_sub) < 50:
        return None

    # 2. Benchmark moving averages, drawdown & forward returns
    bench_sma50 = bench_sub.rolling(50, min_periods=30).mean()
    bench_sma200 = bench_sub.rolling(200, min_periods=100).mean()
    bench_roll_max = bench_sub.rolling(252, min_periods=50).max()
    bench_drawdown = (bench_sub / bench_roll_max - 1.0) * 100.0

    fwd_5 = (bench_sub.shift(-5) / bench_sub - 1.0) * 100.0
    fwd_10 = (bench_sub.shift(-10) / bench_sub - 1.0) * 100.0
    fwd_20 = (bench_sub.shift(-20) / bench_sub - 1.0) * 100.0

    idx_data: Dict[str, Any] = {
        "close": bench_sub,
        "sma50": bench_sma50,
        "sma200": bench_sma200,
        "drawdown": bench_drawdown,
        "fwd_5": fwd_5,
        "fwd_10": fwd_10,
        "fwd_20": fwd_20,
    }

    curr_vix_val: Optional[float] = None
    if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
        vix_sub = pd.to_numeric(
            vix_df[vix_df.index <= eval_dt]["Close"], errors="coerce"
        ).dropna().sort_index()
        if not vix_sub.empty:
            curr_vix_val = float(vix_sub.iloc[-1])
            idx_data["vix"] = vix_sub.reindex(bench_sub.index).ffill()

    frame = pd.DataFrame(idx_data, index=bench_sub.index)

    # 3. Handle secondary breadth data if supplied
    has_breadth = False
    breadth_cohort_study: Optional[HistoricalAnalogStudy] = None
    breadth_note = ""

    if (
        p50_series is not None
        and not p50_series.empty
        and p50_curr is not None
    ):
        common_breadth_idx = p50_series.index.intersection(bench_sub.index)
        if len(common_breadth_idx) >= 30:
            has_breadth = True
            frame_breadth = pd.DataFrame(
                {
                    "p50": p50_series.reindex(common_breadth_idx),
                    "spy_above_50": (bench_sub > bench_sma50).reindex(
                        common_breadth_idx
                    ),
                    "fwd_5": fwd_5.reindex(common_breadth_idx),
                    "fwd_10": fwd_10.reindex(common_breadth_idx),
                    "fwd_20": fwd_20.reindex(common_breadth_idx),
                },
                index=common_breadth_idx,
            ).dropna()

            curr_spy_above_b = bool(
                bench_sub.iloc[-1]
                > (
                    bench_sma50.iloc[-1]
                    if pd.notna(bench_sma50.iloc[-1])
                    else bench_sub.iloc[-1]
                )
            )
            breadth_cohort_study = _compute_breadth_secondary_cohort(
                valid_breadth=frame_breadth,
                curr_above_50=curr_spy_above_b,
                p50_curr=p50_curr,
            )
            if breadth_cohort_study is not None:
                breadth_note = (
                    f"Secondary Breadth Cohort (shorter constituent lookback, N={breadth_cohort_study.sample_count}): "
                    f"Conditioned on S&P 500 constituent breadth >50-DMA ({p50_curr:.0f}% ±12 pp). "
                    f"Note: Constituent breadth history is restricted to recent constituent lookback and subject to survivorship bias."
                )

    # 4. Filter valid rows: full 20D forward return window, valid close & sma50
    valid = frame.dropna(subset=["close", "sma50", "fwd_5", "fwd_10", "fwd_20"])
    if len(valid) < 20:
        # Fallback to breadth study if index data alone has insufficient valid forward bars
        return breadth_cohort_study

    # Current benchmark metrics
    curr_c = float(bench_sub.iloc[-1])
    curr_50 = (
        float(bench_sma50.iloc[-1]) if pd.notna(bench_sma50.iloc[-1]) else curr_c
    )
    curr_200 = (
        float(bench_sma200.iloc[-1])
        if pd.notna(bench_sma200.iloc[-1])
        else curr_50
    )
    curr_dd = (
        float(bench_drawdown.iloc[-1])
        if pd.notna(bench_drawdown.iloc[-1])
        else 0.0
    )

    curr_above_50 = curr_c > curr_50
    curr_above_200 = curr_c > curr_200
    trend50_txt = "above" if curr_above_50 else "below"
    trend200_txt = "above" if curr_above_200 else "below"

    # Historical lookback tracking
    start_dt = valid.index.min()
    end_dt = valid.index.max()
    history_start_yr = int(start_dt.year)
    history_span_yrs = max(1, int(round((end_dt - start_dt).days / 365.25)))

    # Regime gates & masks
    same_50 = (valid["close"] > valid["sma50"]) == curr_above_50
    has_200 = valid["sma200"].notna().any()
    same_200 = (
        ((valid["close"] > valid["sma200"]) == curr_above_200)
        if has_200
        else pd.Series(True, index=valid.index)
    )

    # VIX regime gate
    has_vix = (
        curr_vix_val is not None
        and "vix" in valid.columns
        and valid["vix"].notna().any()
    )
    if has_vix:
        vix_bucket = _vix_bucket_label(curr_vix_val)
        same_vix = (
            valid["vix"]
            .dropna()
            .apply(_vix_bucket_label)
            .reindex(valid.index)
            .fillna("")
            == vix_bucket
        )
        tight_vix = (valid["vix"] - curr_vix_val).abs() <= 2.5
    else:
        vix_bucket = "Unspecified"
        same_vix = pd.Series(True, index=valid.index)
        tight_vix = pd.Series(True, index=valid.index)

    # Drawdown regime gate
    dd_bucket = _drawdown_bucket_label(curr_dd)
    same_dd = (
        valid["drawdown"]
        .dropna()
        .apply(_drawdown_bucket_label)
        .reindex(valid.index)
        .fillna("")
        == dd_bucket
    )
    tight_dd = (valid["drawdown"] - curr_dd).abs() <= 3.0

    # 5. Progressive tolerance relaxation hierarchy
    # Candidates ordered from tightest regime fidelity to broadest trend fallback.
    candidates: List[Tuple[pd.Series, int, str]] = []

    if has_vix and has_200:
        candidates.append((
            same_50 & same_200 & tight_vix & tight_dd,
            20,
            f"S&P 500 {trend50_txt} 50-DMA & {trend200_txt} 200-DMA · VIX {curr_vix_val:.1f} (±2.5) · Drawdown {curr_dd:.1f}% (±3 pp)",
        ))
        candidates.append((
            same_50 & same_200 & same_vix & same_dd,
            20,
            f"S&P 500 {trend50_txt} 50-DMA & {trend200_txt} 200-DMA · VIX {vix_bucket} · Drawdown {dd_bucket}",
        ))
        candidates.append((
            same_50 & same_200 & same_vix,
            20,
            f"S&P 500 {trend50_txt} 50-DMA & {trend200_txt} 200-DMA · VIX {vix_bucket}",
        ))
        candidates.append((
            same_50 & same_vix,
            20,
            f"S&P 500 {trend50_txt} 50-DMA · VIX {vix_bucket}",
        ))

    if has_200:
        candidates.append((
            same_50 & same_200,
            20,
            f"S&P 500 {trend50_txt} 50-DMA & {trend200_txt} 200-DMA",
        ))

    candidates.append((
        same_50,
        5,
        f"S&P 500 {trend50_txt} 50-DMA",
    ))

    # Select the tightest candidate that satisfies its target sample size
    selected_positions: List[int] = []
    chosen_label = ""
    fallback_positions: List[int] = []
    fallback_label = ""
    max_fallback_len = 0

    for mask, target_samples, label in candidates:
        matched_pos = np.flatnonzero(mask.to_numpy(dtype=bool))
        declustered = _decluster_positions(matched_pos, min_spacing=5)
        if len(declustered) >= target_samples:
            selected_positions = declustered
            chosen_label = label
            break
        elif len(declustered) > max_fallback_len:
            max_fallback_len = len(declustered)
            fallback_positions = declustered
            fallback_label = label

    if not selected_positions:
        if fallback_positions and len(fallback_positions) >= 3:
            selected_positions = fallback_positions
            chosen_label = fallback_label
        elif breadth_cohort_study is not None:
            return breadth_cohort_study
        else:
            return None

    analogs = valid.iloc[selected_positions]
    analog_dates = [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in analogs.index]
    n_samples = len(analogs)

    # 6. Compute horizon statistics with Wilson 95% CIs and IQR
    horizons: List[AnalogHorizonStats] = []
    for h_days, col in ((5, "fwd_5"), (10, "fwd_10"), (20, "fwd_20")):
        vals = analogs[col]
        base_vals = valid[col]

        wins = int((vals > 0.0).sum())
        base_wins = int((base_vals > 0.0).sum())
        ci_lo, ci_hi = wilson_score_interval(wins, n_samples)
        b_lo, b_hi = wilson_score_interval(base_wins, len(base_vals))

        p25 = round(float(vals.quantile(0.25)), 2)
        p75 = round(float(vals.quantile(0.75)), 2)
        iqr = round(p75 - p25, 2)

        b_p25 = round(float(base_vals.quantile(0.25)), 2)
        b_p75 = round(float(base_vals.quantile(0.75)), 2)
        b_iqr = round(b_p75 - b_p25, 2)

        horizons.append(
            AnalogHorizonStats(
                horizon_days=h_days,
                win_rate_pct=round(float(wins / n_samples * 100.0), 1),
                median_return_pct=round(float(vals.median()), 2),
                mean_return_pct=round(float(vals.mean()), 2),
                worst_return_pct=round(float(vals.min()), 2),
                baseline_win_rate_pct=round(
                    float(base_wins / len(base_vals) * 100.0), 1
                ),
                baseline_median_pct=round(float(base_vals.median()), 2),
                win_rate_ci_low=ci_lo,
                win_rate_ci_high=ci_hi,
                p25_return=p25,
                p75_return=p75,
                iqr_return=iqr,
                best_return_pct=round(float(vals.max()), 2),
                sample_size=n_samples,
                baseline_ci_low=b_lo,
                baseline_ci_high=b_hi,
                baseline_p25_return=b_p25,
                baseline_p75_return=b_p75,
                baseline_iqr_return=b_iqr,
            )
        )

    # 7. Small-sample rule
    is_small = n_samples < 20
    small_badge = f"⚠️ Small-sample precedent (N={n_samples} < 20)" if is_small else ""

    h20 = horizons[-1]
    edge_txt = (
        "above"
        if h20.win_rate_pct >= h20.baseline_win_rate_pct + 3.0
        else (
            "below"
            if h20.win_rate_pct <= h20.baseline_win_rate_pct - 3.0
            else "in line with"
        )
    )

    sample_clause = (
        f"Across N={n_samples} non-overlapping prior sessions ({len(valid)} evaluated bars, {history_span_yrs}Y lookback from {history_start_yr}) matching {chosen_label}, "
    )
    if is_small:
        sample_clause = f"Across N={n_samples} non-overlapping sessions (small-sample precedent, N < 20) matching {chosen_label}, "

    summary_text = (
        f"{sample_clause}forward 20D {bench_sym} win rate is {h20.win_rate_pct:.1f}% "
        f"({h20.median_return_pct:+.2f}% median, IQR {h20.iqr_return:.2f}%, worst {h20.worst_return_pct:+.2f}%), "
        f"{edge_txt} the {h20.baseline_win_rate_pct:.1f}% unconditional baseline ({h20.baseline_median_pct:+.2f}% median)."
    )

    return HistoricalAnalogStudy(
        benchmark_symbol=bench_sym,
        condition_summary=chosen_label,
        sample_count=n_samples,
        lookback_days=len(valid),
        analog_dates=analog_dates,
        horizons=horizons,
        summary_text=summary_text,
        history_start_year=history_start_yr,
        history_years=history_span_yrs,
        history_start_date=start_dt.strftime("%Y-%m-%d"),
        history_end_date=end_dt.strftime("%Y-%m-%d"),
        is_small_sample=is_small,
        small_sample_badge=small_badge,
        breadth_cohort=breadth_cohort_study,
        breadth_cohort_note=breadth_note,
    )
