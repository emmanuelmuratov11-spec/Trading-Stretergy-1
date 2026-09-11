"""Command line interface.

    python -m quantbot selftest    # verify the install end to end, offline
    python -m quantbot backtest    # walk-forward backtest on real data
    python -m quantbot signal      # alert only: tells you what to trade
    python -m quantbot paper       # alert + update the paper portfolio
    python -m quantbot report      # show current paper portfolio
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from quantbot.config import Config
from quantbot.data import load_universe
from quantbot.data.loader import align
from quantbot.notify import build_notifiers, send_all
from quantbot.portfolio import Portfolio
from quantbot.risk import DrawdownGuard

log = logging.getLogger("quantbot")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(cfg: Config, use_cache: bool = True):
    frames = align(load_universe(cfg.data, use_cache=use_cache))
    bars = {s: len(d) for s, d in frames.items()}
    log.info("loaded %d symbols: %s", len(frames), bars)
    return frames


def cmd_backtest(cfg: Config, args) -> int:
    from quantbot import backtest
    frames = _load(cfg, use_cache=not args.no_cache)
    res = backtest.run(cfg, frames, n_trials=args.trials)
    print()
    print(res.summary())

    if args.json_out:
        payload = {
            "strategy": res.metrics.to_dict(),
            "benchmark": res.benchmark_metrics.to_dict(),
            "diagnostics": res.diagnostics,
            "halted_bars": res.halted_bars,
        }
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        log.info("wrote %s", args.json_out)

    if args.equity_out:
        os.makedirs(os.path.dirname(args.equity_out) or ".", exist_ok=True)
        res.equity.to_frame("equity").join(
            res.benchmark_equity.to_frame("benchmark")).to_csv(args.equity_out)
        log.info("wrote %s", args.equity_out)

    # A strategy that does not beat buy-and-hold is not worth its risk.
    if res.metrics.sharpe <= res.benchmark_metrics.sharpe:
        log.warning("strategy Sharpe (%.2f) does NOT beat buy & hold (%.2f)",
                    res.metrics.sharpe, res.benchmark_metrics.sharpe)
    if res.metrics.dsr < 0.90:
        log.warning("deflated Sharpe %.3f is below 0.90 -- this result is NOT "
                    "statistically distinguishable from luck", res.metrics.dsr)
    return 0


def _run_live(cfg: Config, args, mode: str) -> int:
    from quantbot import signals

    frames = _load(cfg, use_cache=not args.no_cache)
    sig = signals.generate(cfg, frames)

    state_path = os.path.join(cfg.state_dir, "portfolio.json")
    pf = Portfolio.load(state_path, cfg.initial_capital)
    pf.mark(sig.prices)

    guard = DrawdownGuard(cfg.risk)
    guard.peak = pf.peak_equity
    guard.halted = pf.halted
    allowed = guard.update(pf.equity(sig.prices))
    pf.halted = guard.halted

    targets = sig.targets if allowed else {s: 0.0 for s in sig.targets}
    orders = pf.plan_rebalance(
        targets, sig.prices,
        min_notional=args.min_notional,
        band=cfg.alerts.min_weight_change,
    )

    if mode == "paper" and orders:
        fee = (cfg.costs.taker_fee_bps + cfg.costs.half_spread_bps) / 10_000.0
        pf.apply(orders, fee_rate=fee)
        pf.mark(sig.prices)

    alert = signals.format_alert(sig, orders, pf, cfg, mode=mode, halted=not allowed)

    if orders or not allowed or cfg.alerts.send_heartbeat or args.force_send:
        notifiers = build_notifiers(cfg.alerts.channels)
        results = send_all(notifiers, alert)
        log.info("alert delivery: %s", results)
        if not any(results.values()):
            log.error("every alert channel failed")
    else:
        log.info("nothing material changed; no alert sent")

    if mode == "paper" and not args.dry_run:
        pf.save(state_path)
        log.info("saved portfolio state to %s", state_path)
    return 0


def cmd_signal(cfg: Config, args) -> int:
    return _run_live(cfg, args, mode="alert")


def cmd_paper(cfg: Config, args) -> int:
    return _run_live(cfg, args, mode="paper")


def cmd_report(cfg: Config, args) -> int:
    state_path = os.path.join(cfg.state_dir, "portfolio.json")
    if not os.path.exists(state_path):
        print("No paper portfolio yet. Run: python -m quantbot paper")
        return 1
    pf = Portfolio.load(state_path, cfg.initial_capital)
    last_eq = pf.history[-1]["equity"] if pf.history else pf.cash
    pnl = last_eq - cfg.initial_capital
    print(f"Equity      ${last_eq:,.2f}")
    print(f"P&L         ${pnl:+,.2f} ({pnl / cfg.initial_capital:+.2%})")
    print(f"Cash        ${pf.cash:,.2f}")
    print(f"Peak        ${pf.peak_equity:,.2f}")
    print(f"Trades      {pf.trade_count}   fees ${pf.fees_paid:,.2f}")
    print(f"Halted      {pf.halted}")
    print(f"Updated     {pf.updated_at}")
    for s, p in sorted(pf.positions.items()):
        if abs(p.qty) > 1e-10:
            print(f"  {s:12s} {p.qty:.6g} @ {p.avg_price:,.2f}")
    return 0


def cmd_selftest(cfg: Config, args) -> int:
    """Prove the whole pipeline runs offline, on synthetic data."""
    from quantbot import backtest

    cfg.data.source = "synthetic"
    cfg.data.lookback_days = 300
    cfg.model.train_bars = 1500
    cfg.model.min_train_bars = 600
    cfg.model.retrain_every = 480
    cfg.model.n_seeds = 2
    cfg.alerts.channels = ["console"]

    frames = align(load_universe(cfg.data, use_cache=False))
    res = backtest.run(cfg, frames, n_trials=1)
    print(res.summary())
    print("\nNOTE: these numbers come from SYNTHETIC data. They prove the code "
          "runs; they say nothing whatsoever about real profitability.\n")

    from quantbot import signals
    sig = signals.generate(cfg, frames)
    pf = Portfolio(cash=cfg.initial_capital, peak_equity=cfg.initial_capital)
    orders = pf.plan_rebalance(sig.targets, sig.prices)
    alert = signals.format_alert(sig, orders, pf, cfg, mode="paper")
    send_all(build_notifiers(["console"]), alert)
    print("SELFTEST PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="quantbot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["backtest", "signal", "paper", "report", "selftest"])
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="ignore the local bar cache")
    ap.add_argument("--dry-run", action="store_true", help="do not persist portfolio state")
    ap.add_argument("--force-send", action="store_true", help="send an alert even if nothing changed")
    ap.add_argument("--trials", type=int, default=1,
                    help="number of configurations you have tested; inflates the "
                         "deflated-Sharpe haircut. Be honest with this.")
    ap.add_argument("--min-notional", type=float, default=10.0)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--equity-out", default="")
    args = ap.parse_args(argv)

    _setup_logging(args.verbose)
    try:
        cfg = Config.from_yaml(args.config)
    except Exception as exc:
        log.error("bad config: %s", exc)
        return 2

    handlers = {
        "backtest": cmd_backtest, "signal": cmd_signal, "paper": cmd_paper,
        "report": cmd_report, "selftest": cmd_selftest,
    }
    try:
        return handlers[args.command](cfg, args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log.error("%s failed: %s", args.command, exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
