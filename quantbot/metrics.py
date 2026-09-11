"""Performance statistics, including ones designed to *deflate* optimism.

Sharpe ratio alone is close to useless for judging a strategy you found by
searching. If you try 50 configurations and report the best, its Sharpe is
biased upward by the search itself. The Probabilistic and Deflated Sharpe
Ratios (Bailey & López de Prado) correct for exactly that, plus for the
non-normality of returns. A backtest Sharpe of 1.5 that a DSR knocks below 0.5
is telling you something important: you probably found noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from quantbot.scipy_shim import norm_cdf


@dataclass
class Metrics:
    n_bars: int
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    turnover: float
    exposure: float
    skew: float
    kurtosis: float
    psr: float
    dsr: float
    var_95: float
    cvar_95: float

    def to_dict(self) -> dict:
        return asdict(self)


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def _degenerate(std: float, mean: float) -> bool:
    """True when a return stream is constant to within floating-point noise.

    The standard deviation of 500 identical values is not 0 but ~4e-19, which
    would divide out to a Sharpe of 2e17. A strategy parked in cash, or one
    whose positions never change, lands here -- and must report 0, not infinity.
    """
    return std <= max(abs(mean), 1.0) * 1e-12


def zero_metrics() -> "Metrics":
    """An all-zero result, for empty inputs. Built from the field list so it
    cannot drift out of sync when a metric is added."""
    import dataclasses
    return Metrics(**{f.name: 0 for f in dataclasses.fields(Metrics)})


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    return _safe(drawdown_series(equity).min())


def sharpe_ratio(returns: pd.Series, bars_per_year: float, rf: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / bars_per_year
    sd = float(excess.std(ddof=1))
    if _degenerate(sd, float(excess.mean())):
        return 0.0
    return _safe(excess.mean() / sd * math.sqrt(bars_per_year))


def sortino_ratio(returns: pd.Series, bars_per_year: float) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if len(downside) < 2:
        return 0.0
    dd = math.sqrt((downside**2).mean())
    if _degenerate(dd, float(r.mean())):
        return 0.0
    return _safe(r.mean() / dd * math.sqrt(bars_per_year))


def probabilistic_sharpe(observed_sr: float, returns: pd.Series,
                         bars_per_year: float, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark), adjusting for skew, kurtosis and sample size.

    Negatively skewed, fat-tailed return streams -- the signature of strategies
    that sell volatility or hold through drawdowns -- get penalised here, which
    is exactly right: their Sharpe overstates how safe they are.
    """
    r = returns.dropna()
    n = len(r)
    if n < 20:
        return 0.0
    sr = observed_sr / math.sqrt(bars_per_year)   # de-annualise
    bench = benchmark_sr / math.sqrt(bars_per_year)
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurt()) + 3.0          # pandas kurt is excess
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2
    if denom <= 0:
        return 0.0
    z = (sr - bench) * math.sqrt(n - 1) / math.sqrt(denom)
    return _safe(norm_cdf(z))


def deflated_sharpe(observed_sr: float, returns: pd.Series, bars_per_year: float,
                    n_trials: int, trial_sr_std: float | None = None) -> float:
    """PSR against the Sharpe you'd expect from the *best of n_trials* noise runs.

    If you tested 100 variants, the luckiest pure-noise variant would still post
    a respectable Sharpe. DSR asks whether yours beats that bar.
    """
    if n_trials <= 1:
        return probabilistic_sharpe(observed_sr, returns, bars_per_year)
    r = returns.dropna()
    if len(r) < 20:
        return 0.0
    sr_std = trial_sr_std if trial_sr_std and trial_sr_std > 0 else max(
        observed_sr / math.sqrt(bars_per_year), 1e-6) * 0.5

    euler = 0.5772156649015329
    # Expected maximum of n_trials draws from N(0, sr_std^2).
    q1 = norm_inv(1.0 - 1.0 / n_trials)
    q2 = norm_inv(1.0 - 1.0 / (n_trials * math.e))
    expected_max = sr_std * ((1 - euler) * q1 + euler * q2)
    return probabilistic_sharpe(observed_sr, returns, bars_per_year,
                                benchmark_sr=expected_max * math.sqrt(bars_per_year))


def norm_inv(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def compute(returns: pd.Series, equity: pd.Series, bars_per_year: float,
            weights: pd.DataFrame | None = None, n_trials: int = 1) -> Metrics:
    r = returns.dropna()
    n = len(r)
    if n == 0 or len(equity) == 0:
        return zero_metrics()

    years = n / bars_per_year
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = _safe((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else 0.0
    ann_vol = _safe(r.std(ddof=1) * math.sqrt(bars_per_year))
    sr = sharpe_ratio(r, bars_per_year)
    mdd = max_drawdown(equity)

    if weights is not None and len(weights) > 1:
        turnover = _safe(weights.diff().abs().sum(axis=1).mean() * bars_per_year)
        exposure = _safe((weights.abs().sum(axis=1) > 1e-9).mean())
    else:
        turnover, exposure = 0.0, 0.0

    nonzero = r[r != 0]
    return Metrics(
        n_bars=n,
        total_return=total,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=sr,
        sortino=sortino_ratio(r, bars_per_year),
        max_drawdown=mdd,
        calmar=_safe(cagr / abs(mdd)) if mdd < -1e-9 else 0.0,
        hit_rate=_safe((nonzero > 0).mean()) if len(nonzero) else 0.0,
        turnover=turnover,
        exposure=exposure,
        skew=_safe(r.skew()),
        kurtosis=_safe(r.kurt()),
        psr=probabilistic_sharpe(sr, r, bars_per_year),
        dsr=deflated_sharpe(sr, r, bars_per_year, n_trials),
        var_95=_safe(r.quantile(0.05)),
        cvar_95=_safe(r[r <= r.quantile(0.05)].mean()) if n > 20 else 0.0,
    )


def compare_to_benchmark(strategy: Metrics, benchmark: Metrics) -> tuple[bool, str]:
    """Judge a strategy against buy-and-hold. Returns (is_better, explanation).

    Sharpe alone is NOT sufficient here, and getting this wrong is dangerous.
    When both return streams are negative the Sharpe ordering inverts: losing
    less money with lower volatility can score a *worse* Sharpe than losing
    more with higher volatility, because dividing a negative mean by a smaller
    standard deviation makes it more negative. A naive comparison then tells
    the user to hold an asset that halved, which is the opposite of the truth.

    So: when both are losing, rank by capital preserved and drawdown endured.
    When returns are positive, Sharpe is the right risk-adjusted measure.
    """
    s_ret, b_ret = strategy.total_return, benchmark.total_return
    s_dd, b_dd = strategy.max_drawdown, benchmark.max_drawdown

    facts = (f"Strategy {s_ret:+.1%} vs buy & hold {b_ret:+.1%}"
             f"  ·  max drawdown {s_dd:.1%} vs {b_dd:.1%}"
             f"  ·  Sharpe {strategy.sharpe:.2f} vs {benchmark.sharpe:.2f}")

    if s_ret <= 0 and b_ret <= 0:
        # Both lost. The only meaningful question is which lost less, and hurt
        # less on the way. Sharpe is not trustworthy in this regime.
        better = s_ret > b_ret and s_dd >= b_dd
        verdict = ("Lost less than buy & hold, with shallower drawdowns - but "
                   "both lost money" if better else
                   "Did not beat buy & hold")
        return better, f"{verdict}.\n{facts}"

    if s_ret > 0 and b_ret <= 0:
        return True, f"Made money while buy & hold lost.\n{facts}"

    if s_ret <= 0 and b_ret > 0:
        return False, f"Lost money while buy & hold gained.\n{facts}"

    better = strategy.sharpe > benchmark.sharpe
    verdict = ("Beats buy & hold on risk-adjusted return" if better
               else "Does NOT beat buy & hold - holding the benchmark was better")
    return better, f"{verdict}.\n{facts}"


def format_report(m: Metrics, title: str = "Strategy") -> str:
    pct = lambda x: f"{x * 100:,.2f}%"
    return "\n".join([
        f"--- {title} ---",
        f"  Bars              {m.n_bars:,}",
        f"  Total return      {pct(m.total_return)}",
        f"  CAGR              {pct(m.cagr)}",
        f"  Annual vol        {pct(m.ann_vol)}",
        f"  Sharpe            {m.sharpe:.2f}",
        f"  Sortino           {m.sortino:.2f}",
        f"  Max drawdown      {pct(m.max_drawdown)}",
        f"  Calmar            {m.calmar:.2f}",
        f"  Hit rate          {pct(m.hit_rate)}",
        f"  Ann. turnover     {m.turnover:.1f}x",
        f"  Time in market    {pct(m.exposure)}",
        f"  Skew / Kurtosis   {m.skew:.2f} / {m.kurtosis:.2f}",
        f"  VaR95 / CVaR95    {pct(m.var_95)} / {pct(m.cvar_95)}",
        f"  PSR               {m.psr:.3f}   P(true Sharpe > 0)",
        f"  DSR               {m.dsr:.3f}   after multiple-testing haircut",
    ])
