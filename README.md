# quantbot

A self-retraining crypto trading system: it learns from rolling windows of
market history, sizes positions by risk rather than hunch, paper-trades a
portfolio, and messages your phone when the book should change.

---

## Read this before anything else

**This will not print money, and nothing can.** If a strategy reliably printed
money, the people who own it would run it with their own capital rather than
publish it. What follows is the honest version of what this repository is.

* **Markets are close to efficient.** Any edge in liquid crypto majors is
  small, decays as others find it, and is frequently smaller than trading
  costs. Costs are certain; the edge is not.
* **A good backtest is weak evidence.** It is trivially easy to produce a
  Sharpe of 3 in a backtest and lose money live. This repo fights that with
  purged walk-forward validation, deliberately pessimistic costs, and a
  deflated Sharpe ratio that penalises you for the number of variants you have
  tried. Those tools do not make a backtest proof — they only make it less of
  a lie.
* **You should expect this to lose money at first**, and quite possibly
  always. The default configuration has never been validated on real market
  data (see "Current status"), because the environment it was built in had no
  market-data access.
* **Only risk money you can lose entirely.** Not rent, not savings, not
  borrowed money. Leverage is capped at 1.0 by default; raising it is how
  accounts go to zero.
* This is not financial advice. Crypto is unregulated in most jurisdictions,
  and you may owe tax on every trade.

The single most valuable thing here is not the model. It is the **measurement
apparatus** — the part that tells you the truth about whether an edge exists.

---

## What it actually does

```
market data → causal features → triple-barrier labels
           → walk-forward model (refit weekly, never sees its own test data)
           → calibrated probability → fractional-Kelly conviction
           → risk-parity tilt → portfolio volatility target → leverage caps
           → drawdown kill switch → orders → your phone
```

**Self-learning** means the model is refit from scratch on a rolling window as
new bars arrive, so it tracks the current regime instead of a frozen snapshot
of 2021. It does not mean the system becomes reliably profitable over time.

### The parts that matter

| Component | What it is for |
|---|---|
| `features.py` | ~46 strictly backward-looking features. Any lookahead here invalidates everything downstream. |
| `labels.py` | Triple-barrier labels: was the profit target hit *before* the stop? A fixed-horizon label ignores the path and flatters the result. |
| `validation.py` | Purged + embargoed walk-forward splits. Overlapping labels leak the future; this removes them. |
| `model.py` | Boosted-tree ensemble, recency-weighted, probability-calibrated on a chronological holdout. |
| `risk.py` | Fractional Kelly, risk parity, portfolio vol targeting on the full covariance matrix, drawdown kill switch. |
| `backtest.py` | Walk-forward simulation with fees, spread and volatility-scaled slippage. |
| `metrics.py` | Sharpe, Sortino, Calmar — plus PSR and DSR, which haircut your Sharpe for luck and for multiple testing. |
| `signals.py` | Live signals, sharing one implementation with the backtest so the two cannot drift apart. |

### Design decisions worth knowing about

* **Ambiguous bars resolve pessimistically.** When a single bar touches both
  the profit target and the stop, we cannot know which came first, so the label
  assumes the stop. Optimism here is worth a large fake Sharpe.
* **Recency via weighted bootstrap, not `sample_weight`.** Passing
  `sample_weight` to `HistGradientBoostingClassifier` is ~20x slower because it
  disables sklearn's fast path. Each ensemble member instead draws a
  recency-weighted bootstrap, which also decorrelates the ensemble.
* **Turnover is the silent killer.** An unsmoothed vol scalar retrades every
  bar. Smoothing plus a no-trade band cut annual turnover by roughly a third in
  testing, and every unit of turnover is a guaranteed cost against an uncertain
  edge.
* **Coinbase is the default feed.** `api.binance.com` returns HTTP 451 to US
  IPs, and GitHub Actions runners are US-based.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Verify everything works, fully offline, on synthetic data.
python -m quantbot selftest

# 2. Backtest on real market data. --trials matters: see below.
python -m quantbot backtest --trials 1

# 3. See what it would do right now (no state changes).
python -m quantbot signal

# 4. Paper trade: same, but updates state/portfolio.json.
python -m quantbot paper
python -m quantbot report
```

`--trials` should be the number of configurations you have tried in total. It
drives the deflated-Sharpe haircut. Lying to it only lies to you.

---

## Phone alerts

Telegram is recommended: free, instant, no phone number shared, works
everywhere.

1. Message **@BotFather** on Telegram, send `/newbot`, copy the token.
2. Message your new bot once — a bot cannot open a conversation with you.
3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy
   `result[0].message.chat.id`.
4. In your repo: **Settings → Secrets and variables → Actions → New secret**
   * `TELEGRAM_BOT_TOKEN`
   * `TELEGRAM_CHAT_ID`

Locally, export the same two variables. Discord (`DISCORD_WEBHOOK_URL`), Slack
(`SLACK_WEBHOOK_URL`) and ntfy (`NTFY_TOPIC`) work the same way. Unconfigured
channels are skipped rather than failing the run. Secrets are read only from
the environment, never from `config.yaml`, so they cannot be committed by
accident.

An alert looks like this:

```
2 order(s) - BUY BTC, SELL ETH

ACTION REQUIRED
  BUY  0.0412 BTC/USD  @ ~67,240.11
       $2,770.29   (target +27.7% vs current +0.0%)
       suggested stop: 65,102.44

SIGNALS
  BTC/USD    p(up)=0.612 [UP strong]   target=+27.7%  vol=52%
  ETH/USD    p(up)=0.478 [DOWN weak]   target=+0.0%   vol=61%

PORTFOLIO
  Equity      $10,412.88
  P&L         $+412.88 (+4.13%)
  Drawdown    -1.20%
```

## Running it on a schedule

`.github/workflows/signals.yml` runs hourly, generates signals, alerts you, and
commits the updated paper portfolio back to the repo (Actions containers are
ephemeral, so state has to round-trip through git).

**Scheduled workflows only run from the repository's default branch.** Until
this branch is merged, trigger it manually from the Actions tab
("Run workflow"). If a run fails, you get a Telegram message saying so —
silence should never be ambiguous.

---

## How to evaluate this honestly

Run the backtest and look at these, in order:

1. **Does it beat buy-and-hold?** The report prints the benchmark next to the
   strategy. Most crypto strategies lose to simply holding BTC, after costs. If
   it does not beat the benchmark, the correct action is to hold the benchmark.
2. **What is the DSR?** Below ~0.90, the result is not statistically
   distinguishable from luck. The CLI warns you when this happens.
3. **Is turnover plausible?** 50x annual turnover at 8bps round-trip is ~4% a
   year in costs, straight off the top.
4. **Does it survive worse assumptions?** Double the fees and the slippage
   coefficient. An edge that evaporates was never an edge.
5. **Then paper trade it for months.** Not days. This is the step everyone
   skips and the only one that reflects reality.

### If you eventually trade this for real

Start with an amount whose total loss would not change your life. Keep
`max_gross_leverage` at 1.0. Do not disable the drawdown guard. Turn off
`allow_shorts` unless you understand that a short's loss is unbounded. Compare
live fills against the simulated prices — persistent slippage beyond the model
means the edge is smaller than you think.

---

## Current status

* 58 tests pass, including a leak test that is itself verified: introducing a
  deliberate lookahead bug makes it fail and names the offending feature.
* The full pipeline is exercised end to end offline via `selftest`.
* **The strategy has never been run on real market data.** The sandbox this was
  built in blocked every exchange host at the network-policy level, so the only
  backtests so far are on synthetic data — which proves the plumbing works and
  proves nothing at all about profitability. Running
  `python -m quantbot backtest --trials 1` on your own machine or in Actions is
  the first real test, and its result should be treated as the first genuine
  evidence either way.

## Layout

```
quantbot/
  config.py      data/          features.py   labels.py
  validation.py  model.py       risk.py       backtest.py
  metrics.py     signals.py     portfolio.py  notify/      cli.py
tests/           58 tests, including lookahead and leakage checks
.github/workflows/
  signals.yml    hourly signal run + alert
  tests.yml      CI on every push
```
