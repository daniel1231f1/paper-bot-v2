# paper-bot

An intraday **paper-trading** bot with a live dashboard. It starts with $100 of simulated capital,
trades crypto majors on 5-minute bars between 07:00 and 23:00 IST for exactly 10 days, then freezes
and issues a PASS / MARGINAL / FAIL verdict against criteria fixed in advance.

> **Paper only.** There is no broker client, no API key, and no order-placement code anywhere in
> this repository. The only network calls are public candle downloads (and an optional Telegram message).

> **Read [BACKTEST.md](BACKTEST.md) first.** The walk-forward backtest says this strategy has **no
> demonstrated edge after costs**. The live experiment is an honest out-of-sample check of that
> finding, not a money machine.

Dashboard: `https://<owner>.github.io/<this-repo>/` once Pages is enabled (add `?demo` to preview the layout with clearly labelled fake data).

This is the successor to [paper-bot](https://github.com/daniel1231f1/paper-bot) (v1), which keeps running separately.

## How it runs with your laptop off

```
GitHub Actions (one job at a time, ~5.5 h sessions, each queues the next; cron as backup)
   restore state  <- `state` branch (single commit)
   loop: on every closed 5m bar -> fetch candles -> engine -> save state atomically
   publish        -> `state` branch: state.json + dashboard.json + index.html
GitHub Pages serves the `state` branch -> static dashboard polls dashboard.json
```

| Concern | How it is handled |
|---|---|
| GitHub's cron is late or never fires | The bot does not depend on it. Each job runs ~5.5 h, naps through the night with a heartbeat every 30 min, and **queues its own successor** when it ends. The cron is only a backup that restarts the chain if a job is lost. |
| A run is missed anyway | The next run replays every candle since the last processed bar, in order. Stops, targets and the 23:00 flatten are applied exactly as they would have been. |
| Never act on stale data | A signal may open a trade only if its bar closed less than 150 s ago. Catch-up bars can close positions but never open them. If the newest candle is older than 15 min the bot reports "Data stale, not trading". |
| Never double-trade | A bar at or before `last_bar_ts` is ignored. One job at a time (`concurrency`), and every publish is a compare-and-swap on the `state` branch: if anything else moved it, the run stops instead of overwriting. |
| Crashed or cancelled run | State is written to a temp file, fsynced and renamed. The next run resumes from the last published state. |
| API outage / rate limit | Retry with exponential backoff and jitter, then skip the pass safely. |
| Scheduled workflows auto-disabled after 60 days idle | `keepalive.yml` re-enables them weekly through the API. |
| IST schedule | All schedule logic goes through `bot/clock.py` using `zoneinfo` (`Asia/Kolkata`, UTC+05:30, no DST). |

Why GitHub Actions and not an always-on VM: it is free for public repos, needs no server to patch,
and the v1 bot already used it. The cost is timer jitter, which the catch-up design absorbs.

## The 10-day clock and the config lock

* The clock starts with the first bar that closes inside the trading window, provided at least 8
  hours of that session are left; otherwise the bot stays "Armed" and starts at the next 07:00 IST.
  Day 1 is the IST calendar date of the start; the experiment ends at 23:00 IST on day 10.
* At that moment every position is flattened, the engine freezes, and the final report is written
  once. Later runs only refresh the heartbeat.
* When the experiment starts the bot stores a snapshot of `config.yaml` (everything except `ops`)
  in its state and shows the SHA-256 on the dashboard. **Editing `config.yaml` mid-experiment has no
  effect**; the dashboard shows a warning instead. A fingerprint of the trading code (`bot/*.py`) is
  recorded too, and a later change to it is flagged. Dashboard, docs and Outlook changes are not.
* To start over with a new config: Actions → paper-bot → Run workflow → type `RESET`. The old state
  is archived to `archive/state-<timestamp>` first.

## Strategy

1. **Features** (`bot/features.py`): EMA 20/50/200, RSI, Bollinger z-score, rolling VWAP deviation,
   ADX, ATR and its 7-day percentile, Donchian channel, volume ratio, plus hourly EMA/ADX/ATR that
   only update when the hour has closed. Every value at bar *i* uses bars ≤ *i*.
2. **Regime classifier** (`bot/signals.py`): `TREND_UP`, `TREND_DOWN`, `RANGE`, `CHOP`, `DEAD`
   (too quiet to cover costs) or `SHOCK`. Chop, dead and shock regimes do not trade.
3. **Signal ensemble**: trend pullback and volume-confirmed breakout in hourly trends; mean
   reversion in ranges. 5-minute entries, 1-hour trend filter. Long only (spot account).
4. **ML gate** (`bot/model.py`): logistic regression with Platt calibration predicts the probability
   that a candidate reaches its target before its stop or time limit. Trained daily on the last 120
   days of already-resolved candidates. A trade needs probability ≥ 45% **and** expected value
   ≥ +0.10R after costs.
5. **Exits**: stop 1.5× hourly ATR, target 3× (mean reversion 2.25×), 8-hour time stop, flat at 23:00 IST.

## Costs and fills (`bot/broker.py`)

0.10% taker fee + 0.02% half-spread + 0.03% slippage per side (0.30% round trip) plus a volume-impact
term. Orders fill at the **next** bar's open. Entries are capped at 2% of the fill bar's volume
(partial fill, remainder cancelled) and rejected on bars with no trades or below the $5 minimum.
Stops fill at the worse of the stop and the open; targets need price to trade through; if both are
inside one bar the stop is assumed first. Quantities are floored to each symbol's lot size.

## Risk (`bot/risk.py`)

1% of equity at risk per trade including costs · 33% of equity max per position · 3 concurrent
positions · no averaging down · 3% daily loss halts entries until the next session · 15%
peak-to-trough drawdown flattens everything and stops the bot for good · 2-hour cooldown after 3
straight losses · 30-minute re-entry cooldown per symbol.

## Verdict (pre-registered in `config.yaml`)

PASS needs all five: positive expectancy after costs · profit factor > 1.3 · max drawdown < 15% ·
at least 30 trades · still profitable without the single best trade. FAIL if expectancy ≤ 0 or the
drawdown limit is breached. Otherwise MARGINAL. Bootstrap ranges are shown, and ten days is a small sample.

## Layout

```
bot/config.py     typed config, validation, lock hash      bot/engine.py    bar-by-bar engine (backtest = live)
bot/clock.py      IST session + experiment clock           bot/backtest.py  walk-forward replay
bot/data.py       ccxt candles, retry, cache, gaps         bot/live.py      scheduler: catch-up, staleness, loop
bot/features.py   causal indicators                        bot/report.py    dashboard.json + final report
bot/signals.py    regime classifier + ensemble             bot/metrics.py   stats, bootstrap, verdict
bot/model.py      labels, training, calibration            bot/state.py     atomic persistence
bot/broker.py     paper fills and costs                    bot/notify.py    optional Telegram
bot/risk.py       sizing, limits, kill switch              bot/cli.py       command line
invest/outlook.py      Outlook tab: historical odds with a self-test (separate from the bot)
dashboard/index.html   single-file dashboard (no build step)
research/              download, walk-forward backtest, exploration scripts
tests/                 111 tests
```

## Run it locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/pytest -q
```

```bash
.venv/bin/python -m bot.cli run --once
```

```bash
.venv/bin/python -m bot.cli clock --at 2026-11-12T06:59:00+05:30
```

```bash
.venv/bin/python research/download.py && .venv/bin/python research/run.py --holdout
```

`run --once --now <ISO time>` replays the bot at a past wall-clock time against real candles.

## First-time setup of this repository

1. Push this code to an empty public repository. The push itself starts the first bot run.
2. After that run has created the `state` branch (about two minutes): Settings → Pages → Build and
   deployment → Source "Deploy from a branch" → Branch `state`, folder `/ (root)` → Save.
3. Nothing else. No secrets are required. Scheduled runs then keep it going every 30 minutes.

## Outlook tab (separate from the bot)

The dashboard's **Outlook** tab answers one narrow question for a watchlist of indices, stocks,
commodities and crypto: *how often was this asset higher 1 month, 3 months or 1 year later?*

* It reports measured historical frequencies with an 80% range. It is **not a forecast and not advice**.
* It grades itself. Every estimate is replayed over the asset's own past using only data available
  at the time, then scored (Brier) against a coin flip and against the plain long-run average. The
  page shows that score. On current data the trend-adjusted estimate does **not** beat the plain
  base rate, so the headline numbers are base rates unless a row says "trend-adjusted".
* Overlapping windows are counted honestly: a year-long horizon over 25 years is about 25
  independent observations, which is why the ranges are wide.
* The stock rows are survivors (they are listed because they grew), so their history flatters them.

It is independent of the trading experiment: its own package (`invest/`), its own daily workflow
(`outlook.yml`) and its own branch (`outlook`). It never touches the bot's state or config. Edit
`invest/watchlist.yaml` to change the assets (Yahoo Finance tickers). Data: Yahoo Finance daily
adjusted closes, fetched without a key; if a fetch fails the last good result is shown, marked stale.
It refreshes about once an hour while the bot is running. Refreshing faster would not make it more
accurate: the horizons are months, and the inputs are daily closing prices.

Each row carries a plain-word label, "History leans up / down" or "No clear lean", shown only by
which side of 50% the whole range sits on, and a "Why this number" list of the facts behind it.
The tab deliberately does **not** say BUY or SELL: its own self-test shows no edge worth acting on.

```bash
.venv/bin/python -m invest.outlook --out outlook.json
```

## Optional Telegram alerts

Create a bot with @BotFather, then add two repository secrets: `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID`. You get a daily summary at the close, halt and kill-switch alerts, and the final verdict.

## Known limitations

* GitHub can delay or drop scheduled runs. The bot never trades on stale bars, so a delay means
  missed entries, not bad ones. Exits during a gap are simulated from candles after the fact.
* Fills are simulated from 5-minute bars, not an order book. Real fills can be worse.
* Fees assume a 0.10% taker venue. Many retail tiers are several times higher.
* The dashboard updates when the bot publishes (every 5 minutes) and GitHub Pages rebuilds; allow a
  minute or two. Live ticks for open positions come straight from Coinbase's public websocket.
* The NSE equities module from the original brief is not built.
* The keepalive relies on GitHub's workflow-enable API resetting the inactivity timer. That is the
  common approach but GitHub does not guarantee it.
