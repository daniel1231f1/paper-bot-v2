"""Live paper-trading runner (scheduler).

`LiveBot.step()` is one idempotent pass: fetch closed candles, replay every bar newer than
the last processed one through the Engine, save state atomically. It is safe to call at any
time, any number of times:
  * missed runs  -> the gap is replayed bar by bar (stops, targets, session flatten apply as
                    they would have), but stale bars can never open a trade
  * repeated run -> bars already processed are ignored, so nothing trades twice
  * API outage   -> retry with backoff inside the data layer, then "data stale, not trading"
`LiveBot.run()` loops `step()` on bar closes for a time budget (one GitHub Actions job).
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from . import notify
from .backtest import align, build_candidates
from .clock import DAY_MS, SessionClock
from .config import Config
from .data import COLUMNS, DataUnavailable, MarketData, normalize
from .engine import Bar, Candidate, Engine
from .model import TrainedModel, build_dataset, score, train
from .report import REGIME_TEXT, build_view, final_report
from .state import Store

log = logging.getLogger("bot.live")

GRACE_MS = 60_000  # wait this long for an exchange to publish a bar before treating it as "no trades"
WHY_FIELDS = ("close", "rsi", "bb_z", "vwap_dev", "adx", "adx_h", "trend_h", "atr_rank", "vol_ratio", "stop_pct")


@dataclass(frozen=True)
class StepResult:
    new_bars: int
    status: str
    open_positions: int
    frozen: bool


def engine_code_hash() -> str:
    """Fingerprint of the trading code itself (every .py file in this package).

    Used to flag a mid-experiment change. A commit that only touches the dashboard, docs or the
    separate Outlook tool leaves it unchanged, so those do not raise a false alarm.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _now_ms() -> int:
    return int(time.time() * 1000)


class LiveBot:
    def __init__(self, file_cfg: Config, store: Store, md: MarketData | None = None, cache_dir: Path | None = None,
                 now_fn: Callable[[], int] = _now_ms, sleep_fn: Callable[[float], None] = time.sleep,
                 backtest_summary: dict[str, Any] | None = None) -> None:
        self.store, self.now_fn, self.sleep_fn = store, now_fn, sleep_fn
        saved = store.load()
        if saved is not None:
            # LOCKED: trading parameters come from the snapshot taken at experiment start, not from config.yaml
            self.cfg = Config.from_dict({**saved["config"], "ops": file_cfg.to_dict()["ops"]})
            self.clock = SessionClock.from_cfg(self.cfg.schedule)
            self.engine = Engine.from_dict(self.cfg, self.clock, saved["engine"])
            self.shadow = Engine.from_dict(self.cfg, self.clock, saved["shadow"]) if saved.get("shadow") \
                else Engine(self.cfg, self.clock)
        else:
            self.cfg = file_cfg
            self.clock = SessionClock.from_cfg(self.cfg.schedule)
            self.engine = Engine(self.cfg, self.clock)
            # Shadow book: identical rules and risk limits with the ML gate removed. It is NOT the strategy and
            # never affects it; it measures, live, what the model gate is filtering out.
            self.shadow = Engine(self.cfg, self.clock)
        saved = saved or {}
        self.notified: list[str] = list(saved.get("notified", []))
        self.final: dict[str, Any] | None = saved.get("final_report")
        self.why: dict[str, Any] = dict(saved.get("why", {}))
        self.model_meta: dict[str, Any] = dict(saved.get("model", {"ready": False}))
        self.meta: dict[str, Any] = {
            "config_file_hash": file_cfg.hash, "code_sha": engine_code_hash(),
            "code_sha_start": saved.get("code_sha_start"), "created_ts": saved.get("created_ts"),
            "backtest": backtest_summary if backtest_summary is not None else saved.get("backtest"),
        }
        self.cache_dir = cache_dir
        self.md = md or MarketData(self.cfg.market.exchange, self.cfg.market.timeframe_minutes,
                                   cache_dir / "candles" if cache_dir else None,
                                   max_retries=self.cfg.ops.max_retries, retry_base_seconds=self.cfg.ops.retry_base_seconds)
        self.model: TrainedModel | None = None
        self.data_stale = False
        self.data_error: str | None = None
        self._train_retry_after = 0
        self._model_cutoff_done: int | None = None
        self._upto = 0
        if self.cfg.hash != file_cfg.hash:
            log.warning("config.yaml differs from the locked experiment config (%s vs %s): file changes are IGNORED "
                        "until the experiment ends", file_cfg.hash[:12], self.cfg.hash[:12])

    # ------------------------------------------------------------------ data
    def _fetch(self, now: int, start: int) -> dict[str, pd.DataFrame]:
        """Closed candles for the whole universe on one shared grid ending at self._upto."""
        bar = self.cfg.bar_ms
        frames = {s: self.md.history(s, start, now, persist=False) for s in self.cfg.market.universe}
        last_closed = (now // bar) * bar - bar
        real_last = []
        for s, df in frames.items():
            real = df.loc[~df["synthetic"], "ts"]
            if real.empty:
                raise DataUnavailable(f"no candles returned for {s}")
            real_last.append(int(real.iloc[-1]))
        grace_ts = ((now - GRACE_MS) // bar) * bar - bar  # newest bar that closed at least GRACE ago
        # A symbol with no print for a bar that others DID print gets a synthetic (untradeable) bar after the
        # grace period. If nobody printed, the feed itself is behind: do not invent bars, let staleness trip.
        upto = min(last_closed, max(min(real_last), min(grace_ts, max(real_last))))
        out = {s: normalize(df.loc[~df["synthetic"], COLUMNS], bar, end_ts=upto) for s, df in frames.items()}
        self._upto = upto
        return align(out)

    # ------------------------------------------------------------------ model
    def _model_cutoff(self, decision_ts: int) -> int:
        day = self.clock.local_date(decision_ts)
        opening = self.clock.session_open(day)
        return opening if decision_ts >= opening else self.clock.session_open(day - timedelta(days=1))

    def _ensure_model(self, now: int, decision_ts: int) -> None:
        """Daily retrain: the model for a session is trained on data strictly before that session's open."""
        cfg, cutoff = self.cfg, self._model_cutoff(decision_ts)
        if self._model_cutoff_done == cutoff:
            return  # already trained (or deterministically found untrainable) for this session
        if not self.clock.in_window(decision_ts) or now < self._train_retry_after:
            return
        path = self.cache_dir / "models" / f"model-{cfg.hash[:12]}-{cutoff}.pkl" if self.cache_dir else None
        model: TrainedModel | None = None
        if path is not None and path.exists():
            try:
                model = pickle.loads(path.read_bytes())  # noqa: S301 - our own cache file, written below
            except Exception as exc:  # noqa: BLE001 - any unpickling problem => just retrain
                log.warning("model cache unreadable (%s): retraining", type(exc).__name__)
        if model is None:
            try:
                start = cutoff - (cfg.model.train_days + 1) * DAY_MS - cfg.ops.warmup_bars * cfg.bar_ms
                raw = {s: self.md.history(s, start, now, persist=False) for s in cfg.market.universe}
                self.md.flush()
                bars = align(raw)
                ds = build_dataset(build_candidates(cfg, self.clock, bars), bars, cfg, self.clock, with_labels=True)
                model = train(ds, cutoff, cfg)
                log.info("model trained: cutoff=%d ready=%s candidates=%d", cutoff, model is not None, len(ds))
                if model is not None and path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(pickle.dumps(model))
                    tmp.replace(path)
            except DataUnavailable as exc:
                log.warning("training data unavailable (%s): will retry in 10 minutes", exc)
                self._train_retry_after = now + 600_000
                if self.model is not None and cutoff - self.model.cutoff_ts > 3 * DAY_MS:
                    self.model = None  # too old to trust
                return
        self.model, self._model_cutoff_done = model, cutoff
        self.model_meta = {"ready": model is not None, "cutoff_ts": cutoff, "kind": cfg.model.kind,
                           "n_fit": model.n_fit if model else 0, "n_calib": model.n_calib if model else 0,
                           "base_rate": model.base_rate if model else None,
                           "loss_r": model.loss_r if model else None, "trained_ts": now}

    def _prefetch(self, now: int) -> None:
        """Download the training window ahead of the open so the daily retrain is quick and bar 1 stays fresh."""
        cfg = self.cfg
        start = now - (cfg.model.train_days + 2) * DAY_MS - cfg.ops.warmup_bars * cfg.bar_ms
        try:
            for s in cfg.market.universe:
                self.md.history(s, start, now, persist=False)
            self.md.flush()
        except DataUnavailable as exc:
            log.warning("prefetch failed (%s): the retrain will fetch what it needs", exc)

    # ------------------------------------------------------------------ one pass
    def step(self) -> StepResult:
        now = self.now_fn()
        cfg, eng, bar = self.cfg, self.engine, self.cfg.bar_ms
        new_ts: list[int] = []
        if not eng.frozen:
            try:
                bars = self._fetch(now, now - cfg.ops.warmup_bars * bar)
                self.data_error = None
                self.data_stale = now - (self._upto + bar) > cfg.ops.stale_after_seconds * 1000
                new_ts = self._pending_bars(now)
                if new_ts:
                    self._process(now, bars, new_ts)
            except DataUnavailable as exc:
                self.data_error, self.data_stale = f"market data unavailable: {exc}", True
                log.error("%s -> not trading this pass", self.data_error)
        if eng.frozen and self.final is None:
            self.final = final_report(eng, cfg, self.clock, now)
            log.info("experiment frozen: verdict=%s", self.final["verdict"]["result"])
        self._notify(now)
        view = self._persist(now)
        return StepResult(len(new_ts), view["status"]["code"], len(eng.positions), eng.frozen)

    def _can_start(self, decision_ts: int) -> bool:
        if not self.clock.in_window(decision_ts):
            return False
        left = self.clock.session_close(self.clock.local_date(decision_ts)) - decision_ts
        return left >= self.cfg.experiment.min_first_session_hours * 3_600_000

    def _pending_bars(self, now: int) -> list[int]:
        eng, bar, upto = self.engine, self.cfg.bar_ms, self._upto
        if eng.start_ts is None:
            # the experiment clock starts with the first fresh bar that closes inside the trading window,
            # and only if enough of that session is left for day 1 to be a meaningful day
            if self.data_stale or not self._can_start(upto + bar):
                return []
            return [upto]
        first = eng.last_bar_ts + bar
        return list(range(first, upto + bar, bar)) if upto >= first else []

    def _process(self, now: int, bars: dict[str, pd.DataFrame], new_ts: list[int]) -> None:
        cfg, eng, bar = self.cfg, self.engine, self.cfg.bar_ms
        cands = build_candidates(cfg, self.clock, bars)
        self._ensure_model(now, new_ts[-1] + bar)
        index = {s: pd.Index(df["ts"]) for s, df in bars.items()}
        max_age = cfg.ops.max_signal_age_seconds * 1000
        for ts in new_ts:
            frame: dict[str, Bar] = {}
            proposals: list[Candidate] = []
            for s, df in bars.items():
                try:
                    i = int(index[s].get_loc(ts))
                except KeyError:
                    continue  # older than the fetched window (very long outage): skip this symbol for this bar
                r = df.iloc[i]
                frame[s] = Bar(float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
                               float(r["volume"]), bool(r["synthetic"]))
                c = cands[s].iloc[i]
                if c["signal"]:
                    prob, ev = score(self.model, cands[s].iloc[[i]], cfg)
                    ready = self.model is not None and prob[0] == prob[0]
                    proposals.append(Candidate(s, int(c["side"]), str(c["signal"]), str(c["regime"]),
                                               float(c["stop_dist"]), float(c["tp_dist"]),
                                               float(prob[0]) if ready else None, float(ev[0]) if ready else None,
                                               str(c["skip_reason"])))
            if not frame:
                continue
            # re-read the clock: model training above may have taken minutes, and a late decision is a stale one
            fresh = (self.now_fn() - (ts + bar)) <= max_age and not self.data_stale
            if eng.start_ts is None and self.meta.get("code_sha_start") is None:
                self.meta["code_sha_start"], self.meta["created_ts"] = self.meta["code_sha"], now
            eng.on_bar(ts, frame, proposals, fresh=fresh)
            self.shadow.on_bar(ts, frame, [replace(p, prob=1.0, ev_r=9.0) for p in proposals], fresh=fresh)
            if eng.frozen:
                break
        self._update_why(cands, new_ts[-1])

    def _update_why(self, cands: dict[str, pd.DataFrame], ts: int) -> None:
        """Snapshot of what every signal and the model saw on the latest bar, in plain English."""
        cfg, decision_ts = self.cfg, ts + self.cfg.bar_ms
        latest = {(d["symbol"]): d for d in self.engine.decisions if d["ts"] == decision_ts}
        rows = []
        for s, cf in cands.items():
            hit = cf.index[cf["ts"] == ts]
            if len(hit) == 0:
                continue
            r = cf.loc[hit[0]]
            regime, signal = str(r["regime"]), str(r["signal"])
            d = latest.get(s)
            if d is not None:
                action, reason = d["action"], d["reason"]
            elif s in self.engine.positions:
                action, reason = "hold", "in a position: managing stop and target"
            else:
                action, reason = "none", _no_setup_text(regime, cfg.strategy.allow_short)
            row = {"symbol": s, "regime": regime, "signal": signal, "action": action, "reason": reason,
                   "prob": d["prob"] if d else None, "ev_r": d["ev_r"] if d else None}
            for k in WHY_FIELDS:
                v = r[k]
                row[k] = float(v) if v == v else None
            rows.append(row)
        self.why = {"ts": decision_ts, "symbols": rows, "model_ready": self.model is not None,
                    "min_prob": cfg.model.min_prob, "min_ev_r": cfg.model.min_ev_r}

    # ------------------------------------------------------------------ persistence / notifications
    def _persist(self, now: int) -> dict[str, Any]:
        cfg = self.cfg
        view = build_view(self.engine, cfg, self.clock, now, data_stale=self.data_stale, data_error=self.data_error,
                          why=self.why, model=self.model_meta, meta=self.meta, final_report=self.final,
                          shadow=self.shadow)
        self.store.save({
            "config": cfg.locked_dict(), "config_hash": cfg.hash, "engine": self.engine.to_dict(),
            "shadow": self.shadow.to_dict(),
            "model": self.model_meta, "why": self.why, "notified": self.notified[-200:], "final_report": self.final,
            "heartbeat_ts": now, "code_sha_start": self.meta.get("code_sha_start"),
            "created_ts": self.meta.get("created_ts"), "backtest": self.meta.get("backtest"),
        })
        self.store.save_dashboard(view)
        return view

    def _once(self, key: str, text: str) -> None:
        if key in self.notified:
            return
        self.notified.append(key)  # recorded even when Telegram is off, so enabling it later never replays history
        if notify.enabled():
            notify.send(text)

    def _notify(self, now: int) -> None:
        eng, cap = self.engine, self.cfg.experiment.starting_capital
        if eng.start_ts is None:
            return
        day = self.clock.local_date(now)
        if eng.risk.killed:
            self._once("kill", f"🛑 Paper bot KILL SWITCH: drawdown limit hit. Equity ${eng.equity:.2f}. Trading stopped.")
        if eng.risk.daily_halt and eng.risk.day_id == day.toordinal():
            self._once(f"halt-{day}", f"⏸ Paper bot: daily loss limit hit. Equity ${eng.equity:.2f}. Halted until next session.")
        for d in (day - timedelta(days=1), day):  # also covers a session whose close was only seen in catch-up
            close = self.clock.session_close(d)
            if d >= self.clock.local_date(eng.start_ts) and eng.last_bar_ts + self.cfg.bar_ms >= close:
                n = sum(1 for t in eng.trades if self.clock.local_date(t.exit_ts - 1) == d)
                self._once(f"daily-{d}", f"📊 Paper bot day {self.clock.day_number(eng.start_ts, close - 1)}: equity "
                                         f"${eng.equity:.2f} ({(eng.equity / cap - 1) * 100:+.2f}% total), {n} trades closed.")
        if self.final is not None:
            v = self.final["verdict"]
            self._once("final", f"🏁 Paper bot final verdict: {v['result']}. {v['explanation']}")

    # ------------------------------------------------------------------ session loop
    def run(self, budget_seconds: float, publish: Callable[[], bool] | None = None, pre_open_wait_seconds: int = 2400) -> None:
        """Loop until the time budget is spent, the experiment freezes, or the session is over and we are flat."""
        cfg, bar = self.cfg, self.cfg.bar_ms
        deadline = self.now_fn() + int(budget_seconds * 1000)
        last_publish = 0
        while True:
            res = self.step()
            now = self.now_fn()
            if publish and (res.new_bars or now - last_publish >= cfg.ops.publish_seconds * 1000):
                if not publish():
                    log.error("state publish rejected (another runner owns the state?): stopping this run")
                    return
                last_publish = now
            if res.frozen:
                log.info("experiment is frozen: nothing left to do")
                break
            armed = self.engine.start_ts is None
            caught_up = self._upto <= self.engine.last_bar_ts or armed
            idle = not self.clock.in_window(now) or (armed and not self._can_start(now))  # asleep, or armed too late today
            if idle and caught_up and not self.engine.positions and not self.engine.pending:
                wait = self.clock.next_open(now) - now
                if wait <= pre_open_wait_seconds * 1000 and now + wait < deadline:
                    log.info("session opens in %.0f min: warming the candle cache, then waiting", wait / 60_000)
                    self._prefetch(now)
                    self.sleep_fn(wait / 1000 + 1)
                    continue
                log.info("outside the trading window and flat: sleeping until the next scheduled run")
                break
            if now >= deadline:
                log.info("time budget spent: handing over to the next scheduled run")
                break
            # sleep to just after the next bar close, then poll quickly until the exchange has published it
            to_close = ((now // bar + 1) * bar - now) / 1000 + 3
            if self.data_error:
                pause = 60.0  # exchange is down: the data layer already retried with backoff
            elif res.new_bars or now % bar > 90_000:
                pause = to_close
            else:
                pause = 5.0  # a bar just closed and is not published yet
            self.sleep_fn(max(1.0, min(pause, (deadline - now) / 1000)))
        self.md.flush()
        if publish:
            publish()


def _no_setup_text(regime: str, allow_short: bool) -> str:
    base = REGIME_TEXT.get(regime, regime)
    if regime == "TREND_UP":
        return f"{base}, but no pullback-reclaim or volume breakout on this bar"
    if regime == "TREND_DOWN":
        return f"{base}: " + ("no breakdown or failed-rally setup on this bar" if allow_short
                              else "long-only strategy stays out")
    if regime == "RANGE":
        return f"{base}: waiting for an oversold stretch below the band"
    return f"no setup: {base}"
