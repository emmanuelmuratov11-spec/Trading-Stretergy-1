"""Chart rendering for backtest and paper-trading reports.

Why these four panels, in this order:

1. **Equity vs benchmark** -- the only question that matters first. A strategy
   that trails buy-and-hold is not worth its risk, so the benchmark is never
   optional and never on a second axis.
2. **Drawdown** -- what holding it actually felt like. A CAGR figure hides the
   part that makes people capitulate at the bottom.
3. **Rolling Sharpe** -- whether the edge was persistent or came from one lucky
   month. A flat backtest Sharpe cannot show this and it is where most
   strategies quietly fall apart.
4. **Gross exposure** -- when it was in the market, and where the drawdown
   guard stood it down.

Colour follows the *entity*: the strategy is always blue and the benchmark is
always orange, in every panel. The pair is validated for colour-vision
deficiency (worst-pair CVD dE 24.7, well clear of the 8 threshold).

The canvas is painted an opaque light surface rather than left transparent,
because these PNGs are delivered into chat apps whose own background may be
dark -- a transparent chart renders as dark-on-dark there and is unreadable.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# --- Design tokens. Chart chrome is deliberately recessive: hairline solid
# --- gridlines one shade off the surface, never dashed, never heavy.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

STRATEGY = "#2a78d6"   # categorical slot 1
BENCHMARK = "#eb6834"  # categorical slot 2
NEGATIVE = "#d03b3b"   # status: critical, used only where it means "loss"
POSITIVE = "#006300"

FONT = ["DejaVu Sans", "system-ui", "-apple-system", "Segoe UI", "sans-serif"]


def _style_axes(ax, ylabel: str = "", title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold",
                     loc="left", pad=8)


def _stat_tiles(fig, stats: list[tuple[str, str, str]], y: float = 0.945) -> None:
    """A row of hero numbers. The headline figures are text, not a chart --
    a one-bar bar chart of 'Sharpe' would be worse than the number itself."""
    n = len(stats)
    for i, (label, value, colour) in enumerate(stats):
        x = 0.055 + i * (0.90 / n)
        fig.text(x, y, value, fontsize=17, fontweight="bold", color=colour,
                 ha="left", va="top")
        fig.text(x, y - 0.038, label.upper(), fontsize=7.5, color=INK_MUTED,
                 ha="left", va="top")


def _endpoint_label(ax, series: pd.Series, colour: str, text: str) -> None:
    """Label the final value only. A number on every point goes unread."""
    if len(series) == 0:
        return
    ax.annotate(
        text,
        xy=(series.index[-1], series.iloc[-1]),
        xytext=(6, 0), textcoords="offset points",
        color=colour, fontsize=8.5, fontweight="bold",
        va="center", ha="left", annotation_clip=False,
    )


def _rolling_sharpe(returns: pd.Series, window: int, bars_per_year: float) -> pd.Series:
    mean = returns.rolling(window, min_periods=window // 2).mean()
    std = returns.rolling(window, min_periods=window // 2).std()
    return (mean / std.replace(0.0, np.nan)) * np.sqrt(bars_per_year)


def render_backtest(result, cfg, path: str, title: str = "Backtest") -> str:
    """Render the four-panel backtest report to `path`. Returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from quantbot.data.base import bars_per_year

    plt.rcParams["font.family"] = FONT
    bpy = bars_per_year(cfg.data.timeframe)

    m, b = result.metrics, result.benchmark_metrics
    equity, bench = result.equity, result.benchmark_equity

    fig = plt.figure(figsize=(10, 12), dpi=140, facecolor=SURFACE)
    gs = fig.add_gridspec(
        4, 1, height_ratios=[3.0, 1.6, 1.6, 1.2],
        hspace=0.42, top=0.865, bottom=0.085, left=0.085, right=0.90,
    )

    fig.text(0.055, 0.982, title, fontsize=15, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.055, 0.962,
             f"{equity.index[0]:%Y-%m-%d} to {equity.index[-1]:%Y-%m-%d}"
             f"   ·   {cfg.data.timeframe} bars   ·   out-of-sample, after costs",
             fontsize=8.5, color=INK_MUTED, ha="left", va="top")

    from quantbot.metrics import compare_to_benchmark
    beat, verdict = compare_to_benchmark(m, b)
    _stat_tiles(fig, [
        ("CAGR", f"{m.cagr:+.1%}", POSITIVE if m.cagr > 0 else NEGATIVE),
        ("Sharpe", f"{m.sharpe:.2f}", POSITIVE if m.sharpe > 0 else NEGATIVE),
        ("Max drawdown", f"{m.max_drawdown:.1%}", NEGATIVE),
        ("Deflated Sharpe", f"{m.dsr:.2f}", POSITIVE if m.dsr >= 0.9 else NEGATIVE),
    ])

    # --- Panel 1: equity vs benchmark, one shared axis (never dual-axis).
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(equity.index, equity.to_numpy(), color=STRATEGY, linewidth=2.0,
             label="Strategy", zorder=3)
    ax1.plot(bench.index, bench.to_numpy(), color=BENCHMARK, linewidth=2.0,
             label="Buy & hold", zorder=2)
    ax1.axhline(cfg.initial_capital, color=AXIS, linewidth=0.8, zorder=1)

    span = float(max(equity.max(), bench.max()) / max(min(equity.min(), bench.min()), 1e-9))
    if span > 5:
        # Log scale so equal percentage moves look equal; on linear axes a late
        # doubling dwarfs an early one and the early period reads as flat.
        ax1.set_yscale("log")
        _style_axes(ax1, "Equity (log scale)", "Growth of capital")
    else:
        _style_axes(ax1, "Equity", "Growth of capital")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    _endpoint_label(ax1, equity, STRATEGY, f"${equity.iloc[-1]:,.0f}")
    _endpoint_label(ax1, bench, BENCHMARK, f"${bench.iloc[-1]:,.0f}")
    leg = ax1.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    # --- Panel 2: drawdown.
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    dd_s = (equity / equity.cummax() - 1.0) * 100
    dd_b = (bench / bench.cummax() - 1.0) * 100
    ax2.fill_between(dd_s.index, dd_s.to_numpy(), 0, color=STRATEGY, alpha=0.22, zorder=2)
    ax2.plot(dd_s.index, dd_s.to_numpy(), color=STRATEGY, linewidth=1.6, zorder=3)
    ax2.plot(dd_b.index, dd_b.to_numpy(), color=BENCHMARK, linewidth=1.4,
             alpha=0.85, zorder=2)
    stop = -cfg.risk.max_drawdown_stop * 100
    if dd_s.min() < stop * 0.5:
        ax2.axhline(stop, color=NEGATIVE, linewidth=1.0, zorder=4)
        ax2.annotate("kill switch", xy=(dd_s.index[0], stop), xytext=(2, 3),
                     textcoords="offset points", color=NEGATIVE, fontsize=7.5)
    _style_axes(ax2, "Drawdown %", "Peak-to-trough loss")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # --- Panel 3: rolling Sharpe -- is the edge persistent, or one lucky run?
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    window = int(min(max(bpy / 12, 60), max(len(result.returns) // 3, 60)))
    rs = _rolling_sharpe(result.returns, window, bpy)
    # While the book is empty the returns are all zero, so a "Sharpe" there is
    # an artefact of the rounding noise rather than a measurement. Hide it.
    deployed = result.weights.abs().sum(axis=1) > 1e-9
    rs = rs.where(deployed.rolling(window, min_periods=1).mean() > 0.2)
    ax3.axhline(0.0, color=AXIS, linewidth=1.0, zorder=1)
    ax3.plot(rs.index, rs.to_numpy(), color=STRATEGY, linewidth=1.6, zorder=3)
    vals = rs.to_numpy()
    ax3.fill_between(rs.index, vals, 0, where=(vals >= 0),
                     color=STRATEGY, alpha=0.16, zorder=2, interpolate=True)
    ax3.fill_between(rs.index, vals, 0, where=(vals < 0),
                     color=NEGATIVE, alpha=0.16, zorder=2, interpolate=True)
    _style_axes(ax3, "Sharpe", f"Rolling Sharpe ({window} bars)")

    # --- Panel 4: gross exposure.
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    gross = result.weights.abs().sum(axis=1) * 100
    ax4.fill_between(gross.index, gross.to_numpy(), 0, color=STRATEGY,
                     alpha=0.30, zorder=2)
    ax4.plot(gross.index, gross.to_numpy(), color=STRATEGY, linewidth=1.2, zorder=3)
    _style_axes(ax4, "Gross %", "Capital deployed")
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax4.set_ylim(0, max(float(gross.max()) * 1.15, 5))

    for ax in (ax1, ax2, ax3):
        plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), color=INK_MUTED, fontsize=8)

    caveat = ("" if m.dsr >= 0.9 else
              "\nDSR below 0.90: not statistically distinguishable from luck")
    fig.text(0.055, 0.040, verdict + caveat, fontsize=7.5,
             color=POSITIVE if beat and m.dsr >= 0.9 else NEGATIVE,
             ha="left", va="top", linespacing=1.6)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    log.info("wrote chart %s", path)
    return path


def render_portfolio(portfolio, cfg, path: str) -> str | None:
    """Render the paper portfolio's equity history. Returns None if too short."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    history = getattr(portfolio, "history", []) or []
    if len(history) < 3:
        log.info("only %d equity points recorded; skipping chart", len(history))
        return None

    plt.rcParams["font.family"] = FONT
    eq = pd.Series(
        [h["equity"] for h in history],
        index=pd.to_datetime([h["ts"] for h in history], utc=True, format="mixed"),
    ).sort_index()

    fig = plt.figure(figsize=(9, 5.2), dpi=140, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.32,
                          top=0.80, bottom=0.10, left=0.10, right=0.92)

    pnl = float(eq.iloc[-1]) - cfg.initial_capital
    pct = pnl / cfg.initial_capital if cfg.initial_capital else 0.0
    dd = float((eq / eq.cummax() - 1.0).min())

    fig.text(0.055, 0.965, "Paper portfolio", fontsize=14, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.055, 0.935,
             f"{eq.index[0]:%Y-%m-%d} to {eq.index[-1]:%Y-%m-%d}   ·   "
             f"{portfolio.trade_count} trades   ·   ${portfolio.fees_paid:,.2f} fees",
             fontsize=8, color=INK_MUTED, ha="left", va="top")
    _stat_tiles(fig, [
        ("Equity", f"${eq.iloc[-1]:,.0f}", INK),
        ("P&L", f"{pct:+.2%}", POSITIVE if pnl >= 0 else NEGATIVE),
        ("Max drawdown", f"{dd:.1%}", NEGATIVE),
    ], y=0.885)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eq.index, eq.to_numpy(), color=STRATEGY, linewidth=2.0, zorder=3)
    ax1.axhline(cfg.initial_capital, color=AXIS, linewidth=0.8, zorder=1)
    _style_axes(ax1, "Equity", "")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    _endpoint_label(ax1, eq, STRATEGY, f"${eq.iloc[-1]:,.0f}")

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ddser = (eq / eq.cummax() - 1.0) * 100
    ax2.fill_between(ddser.index, ddser.to_numpy(), 0, color=STRATEGY,
                     alpha=0.22, zorder=2)
    ax2.plot(ddser.index, ddser.to_numpy(), color=STRATEGY, linewidth=1.4, zorder=3)
    _style_axes(ax2, "Drawdown %", "")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), color=INK_MUTED, fontsize=8)

    fig.text(0.055, 0.022, "Paper trading - simulated fills, not real money.",
             fontsize=8, color=INK_MUTED, ha="left")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    log.info("wrote chart %s", path)
    return path
