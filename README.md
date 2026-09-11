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
* **The default configuration loses money on real data.** That is measured,
  not feared: −23.3% against −8.2% for buy-and-hold over 18 months of BTC/ETH/SOL.
  See "Current status" for the full table and the diagnosis.
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
| `plotting.py` | Four-panel visual report, delivered to your phone as an image. |

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

# 3. Same, with a chart.
python -m quantbot backtest --plot reports/backtest.png

# 4. See what it would do right now (no state changes).
python -m quantbot signal

# 5. Paper trade: same, but updates state/portfolio.json.
#    --plot also sends an equity chart to your phone.
python -m quantbot paper --plot
python -m quantbot report --plot
```

`--trials` should be the number of configurations you have tried in total. It
drives the deflated-Sharpe haircut. Lying to it only lies to you.

---

## The visual report

`--plot` renders a four-panel report, chosen to answer the questions that
actually decide whether a strategy is worth running:

1. **Growth of capital** — strategy against buy-and-hold, on one shared axis.
   Log scale past a 5x range, so equal percentage moves look equal instead of
   letting a late doubling dwarf an early one.
2. **Peak-to-trough loss** — what holding it would have *felt* like. A CAGR
   number hides the part that makes people capitulate at the bottom.
3. **Rolling Sharpe** — whether the edge persisted or came from one lucky
   stretch. This is where most strategies quietly fall apart, and a single
   headline Sharpe cannot show it.
4. **Capital deployed** — when it was in the market, and where the drawdown
   guard stood it down.

The strategy is blue and the benchmark is orange in every panel, so colour
always identifies the same entity. The pair is validated for colour-vision
deficiency (worst-pair ΔE 24.7, against a threshold of 8). Charts render on an
opaque light background because a transparent PNG renders dark-on-dark in chat
apps set to dark mode.

The footer states the comparison in plain numbers rather than a single verdict
word, because **Sharpe ranking inverts when both returns are negative** — losing
4% with shallow drawdowns scores a *worse* Sharpe than losing 27% with violent
ones, and ranking on Sharpe alone would advise holding the asset that halved.

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

* 66 tests pass, including a leak test that is itself verified: introducing a
  deliberate lookahead bug makes it fail and names the offending feature.
* The full pipeline is exercised end to end offline via `selftest`.

### It has now been run on real data, and it lost money

First real backtest: **BTC/ETH/SOL, hourly, 13,021 bars (~18 months) of
Coinbase data**, walk-forward, out-of-sample, after costs.

| | Strategy | Buy & hold |
|---|---|---|
| Total return | **−23.3%** | −8.2% |
| CAGR | −16.3% | −5.6% |
| Sharpe | **−1.98** | 0.07 |
| Max drawdown | −25.1% | −53.9% |
| Hit rate | 48.2% | — |
| Annual turnover | **112.5x** | 0x |
| Deflated Sharpe | **0.007** | 0.535 |

**The strategy as configured does not work.** It lost roughly three times what
simply holding the basket lost. The drawdown guard fired and stood the book
down for 2,427 bars, which is the one part that behaved exactly as designed.

The diagnosis is mostly in one number: **112.5x annual turnover**. At the
configured ~11.5bps per unit of turnover that is ~12.9% a year in costs, or
about **19 of the 23 percentage points lost**. The remaining ~4 points are
consistent with a 48.2% hit rate — a model with no edge, or a slightly
negative one. So this is not primarily a bad predictor; it is a mediocre
predictor being taxed to death by trading too often.

The reporting did its job: DSR of 0.007 correctly refuses to call this
anything but noise, and the run warned on both counts without being asked.

**Do not read this table as "needs tuning until it goes green."** Every
re-test on the same data is another draw from the multiple-testing urn, which
is exactly what DSR exists to punish — raise `--trials` honestly each time you
retest. Reducing turnover is a legitimate fix, because it plugs a known cost
leak rather than fitting returns. Raising leverage or loosening the deadband
until the curve points up is not.

Reproduce or re-run it yourself from the Actions tab
(`.github/workflows/backtest.yml`), or locally with
`python -m quantbot backtest --trials 1 --plot`.

## Layout

```
quantbot/
  config.py      data/          features.py   labels.py
  validation.py  model.py       risk.py       backtest.py
  metrics.py     signals.py     portfolio.py  notify/      cli.py
tests/           66 tests, including lookahead and leakage checks
.github/workflows/
  signals.yml    hourly signal run + alert
  tests.yml      CI on every push
```
