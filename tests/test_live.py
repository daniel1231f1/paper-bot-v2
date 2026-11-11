"""Scheduler behaviour with a simulated clock and a fake exchange:
IST gating, catch-up after missed runs, idempotency, stale data, outages, config lock, day-N freeze."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import bot.live as live
from bot.data import MarketData
from bot.live import LiveBot
from bot.state import Store
from conftest import BAR, FakeClient, ist, make_candles, make_config

A, B = "AAA/USD", "BBB/USD"
DATA_START = ist(2026, 1, 1, 0, 0)


class World:
    """Simulated wall clock + exchange. `sleep` advances time, so run() loops can be tested instantly."""

    def __init__(self, tmp_path, monkeypatch, **cfg_sections) -> None:
        self.frames = {A: make_candles(3000, DATA_START, seed=1), B: make_candles(3000, DATA_START, seed=2)}
        self.client = FakeClient(self.frames)
        self.t = 0
        self.tmp = tmp_path
        self.cfg = make_config(**cfg_sections)
        self.forced: dict[str, set[int]] = {A: set(), B: set()}
        real = live.build_candidates

        def forced_candidates(cfg, clock, bars):
            out = real(cfg, clock, bars)
            for sym, cf in out.items():
                m = cf["ts"].isin(self.forced[sym]).to_numpy()
                cf["signal"] = np.where(m, "breakout", "")
                cf["side"] = np.where(m, 1, 0)
                cf["regime"] = np.where(m, "TREND_UP", cf["regime"])
                cf["skip_reason"] = ""
                cf["stop_dist"], cf["tp_dist"] = cf["close"] * 0.01, cf["close"] * 0.02
            return out

        monkeypatch.setattr(live, "build_candidates", forced_candidates)
        monkeypatch.setattr(live, "score", lambda model, rows, cfg: (np.full(len(rows), 0.9), np.full(len(rows), 0.5)))
        monkeypatch.setattr(LiveBot, "_ensure_model", lambda self, now, ts: setattr(self, "model", object()))

    def now(self) -> int:
        self.client.now = self.t
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += int(seconds * 1000)

    def bot(self, cfg=None) -> LiveBot:
        md = MarketData("fake", 5, None, client=self.client, max_retries=2, retry_base_seconds=0.01, sleep=lambda s: None)
        return LiveBot(cfg or self.cfg, Store(self.tmp / "state"), md=md, now_fn=self.now, sleep_fn=self.sleep)

    def view(self) -> dict:
        return json.loads((self.tmp / "state" / "dashboard.json").read_text())

    def set_bar(self, sym: str, ts: int, **cols: float) -> None:
        df = self.frames[sym]
        for k, v in cols.items():
            df.loc[df["ts"] == ts, k] = v


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    return World(tmp_path, monkeypatch)


def test_outside_window_the_experiment_does_not_start(world):
    world.t = ist(2026, 1, 5, 3, 0) + 5_000  # 03:00 IST
    world.forced[A].add(ist(2026, 1, 5, 2, 55))
    bot = world.bot()
    res = bot.step()
    assert res.status == "ARMED" and bot.engine.start_ts is None and not bot.engine.pending
    assert world.view()["status"]["label"] == "Armed, starts 07:00 IST"


def test_first_bar_inside_the_window_starts_the_clock(world):
    world.t = ist(2026, 1, 5, 7, 5) + 5_000  # newest closed bar is 07:00-07:05: decision time 07:05
    bot = world.bot()
    res = bot.step()
    assert res.status == "ACTIVE" and res.new_bars == 1
    assert bot.engine.start_ts == ist(2026, 1, 5, 7, 5)  # first decision time
    assert bot.engine.end_ts == ist(2026, 1, 14, 23, 0)  # close of IST day 10
    v = world.view()
    assert v["experiment"]["day"] == 1 and v["experiment"]["config_hash"] == world.cfg.hash


def test_signal_fills_on_the_next_bar_and_repeat_steps_do_nothing(world):
    sig = ist(2026, 1, 5, 9, 55)
    world.forced[A].add(sig)
    world.t = sig + BAR + 5_000  # 10:00:05, bar 09:55 just closed
    bot = world.bot()
    bot.step()
    assert A in bot.engine.pending and not bot.engine.positions
    world.t += BAR
    bot.step()
    pos = bot.engine.positions[A]
    nxt_open = float(world.frames[A].loc[world.frames[A]["ts"] == sig + BAR, "open"].iloc[0])
    assert pos.entry_ts == sig + BAR and pos.entry_price == pytest.approx(nxt_open * 1.0005)
    before = json.dumps(bot.engine.to_dict(), sort_keys=True)
    for _ in range(3):  # overlapping / repeated runs at the same time: idempotent
        assert bot.step().new_bars == 0
    assert json.dumps(bot.engine.to_dict(), sort_keys=True) == before
    # a second process restoring the saved state must not trade again either
    twin = world.bot()
    assert twin.step().new_bars == 0
    assert json.dumps(twin.engine.to_dict(), sort_keys=True) == before


def test_catch_up_after_missed_runs_applies_exits_but_never_enters_on_stale_bars(world):
    sig = ist(2026, 1, 5, 9, 55)
    world.forced[A].add(sig)
    world.t = sig + BAR + 5_000
    bot = world.bot()
    bot.step()
    world.t += BAR
    bot.step()
    stop = bot.engine.positions[A].stop
    # two hours of missed runs: price collapses through the stop 30 minutes in; B flashes a signal an hour in
    crash = sig + 7 * BAR
    world.set_bar(A, crash, low=stop * 0.99)
    world.forced[B].add(sig + 13 * BAR)
    world.t = sig + 25 * BAR + 5_000
    res = world.bot().step()  # a fresh process, like the next scheduled job
    eng = world.bot().engine
    assert res.new_bars == 23 and eng.last_bar_ts == sig + 24 * BAR  # every missed candle replayed, in order
    assert len(eng.trades) == 1 and eng.trades[0].exit_reason == "stop" and eng.trades[0].exit_ts == crash + BAR
    assert B not in eng.positions and B not in eng.pending  # the stale signal was not acted on
    assert any(d["symbol"] == B and "stale" in d["reason"] for d in eng.decisions)


def test_fresh_signal_right_after_catch_up_is_still_tradeable(world):
    world.t = ist(2026, 1, 5, 9, 0) + 5_000
    world.bot().step()
    latest = ist(2026, 1, 5, 11, 55)
    world.forced[B].add(latest)
    world.t = latest + BAR + 5_000
    bot = world.bot()
    assert bot.step().new_bars > 30
    assert B in bot.engine.pending


def test_position_left_open_is_flattened_at_2300_ist_even_if_the_run_happens_at_night(world):
    world.t = ist(2026, 1, 5, 9, 0) + 5_000  # experiment started in the morning
    world.bot().step()
    sig = ist(2026, 1, 5, 21, 30)
    world.forced[A].add(sig)
    world.t = sig + BAR + 5_000
    bot = world.bot()
    bot.step()
    world.t += BAR
    bot.step()
    pos = bot.engine.positions[A]
    for ts in range(sig + 2 * BAR, ist(2026, 1, 6, 4, 0), BAR):  # keep price between stop and target all night
        world.set_bar(A, ts, open=pos.entry_price, high=pos.entry_price, low=pos.entry_price, close=pos.entry_price)
    world.t = ist(2026, 1, 6, 2, 0) + 5_000  # next run only at 02:00 IST
    night = world.bot()
    res = night.step()
    assert res.status == "SLEEPING" and not night.engine.positions
    t = night.engine.trades[-1]
    assert t.exit_reason == "session close" and t.exit_ts == ist(2026, 1, 5, 23, 0)
    assert "Sleeping until 07:00 IST" == world.view()["status"]["label"]


def test_stale_data_blocks_entries_and_is_reported(world):
    world.t = ist(2026, 1, 5, 9, 0) + 5_000
    bot = world.bot()
    bot.step()
    frozen_at = world.t
    last = ist(2026, 1, 5, 8, 55)
    world.forced[A].add(last)
    # the exchange stops publishing candles; 30 minutes later the bot looks again
    world.now = lambda: (setattr(world.client, "now", frozen_at), world.t)[1]
    world.t = frozen_at + 30 * 60_000
    stale = world.bot()
    res = stale.step()
    assert res.status == "DATA_STALE" and stale.data_stale and not stale.engine.pending
    assert world.view()["status"]["label"] == "Data stale, not trading"


def test_api_outage_is_survived_and_state_is_untouched(world):
    world.t = ist(2026, 1, 5, 9, 0) + 5_000
    bot = world.bot()
    bot.step()
    before = json.dumps(bot.engine.to_dict(), sort_keys=True)
    world.client.fail = 10_000
    world.t += BAR
    res = bot.step()
    assert res.status == "DATA_STALE" and res.new_bars == 0 and "unavailable" in bot.data_error
    assert json.dumps(bot.engine.to_dict(), sort_keys=True) == before
    world.client.fail = 0
    world.t += BAR
    assert bot.step().new_bars == 2 and bot.data_error is None  # recovers and catches up


def test_config_is_locked_for_the_whole_experiment(world):
    world.t = ist(2026, 1, 5, 9, 0) + 5_000
    world.bot().step()
    tweaked = make_config(risk={"risk_per_trade": 0.02, "max_positions": 1}, ops={"poll_seconds": 7})
    assert tweaked.hash != world.cfg.hash
    world.t += BAR
    bot = world.bot(tweaked)
    bot.step()
    assert bot.cfg.hash == world.cfg.hash and bot.cfg.risk.risk_per_trade == 0.01  # edit ignored
    assert bot.cfg.ops.poll_seconds == 7  # operational knobs are allowed to change
    exp = world.view()["experiment"]
    assert exp["config_hash"] == world.cfg.hash and exp["config_dirty"] is True


def test_freeze_and_final_report_on_the_last_day(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch, experiment={"days": 2})
    sig = ist(2026, 1, 5, 9, 55)
    world.forced[A].add(sig)
    world.t = sig + BAR + 5_000
    bot = world.bot()
    bot.step()
    world.t = ist(2026, 1, 6, 22, 0) + 5_000
    mid = world.bot()
    assert mid.step().status == "ACTIVE" and mid.final is None
    assert world.view()["experiment"]["day"] == 2 and world.view()["verdict"]["final"] is False
    world.forced[B].add(ist(2026, 1, 6, 23, 30))
    world.t = ist(2026, 1, 7, 0, 30) + 5_000  # after the close of day 2
    end = world.bot()
    res = end.step()
    assert res.frozen and res.status == "FROZEN" and not end.engine.positions and not end.engine.pending
    assert end.engine.frozen_ts == ist(2026, 1, 6, 23, 0)
    report = end.final
    assert report["verdict"]["final"] is True and report["verdict"]["result"] in {"PASS", "MARGINAL", "FAIL"}
    trades = len(end.engine.trades)
    # days later: still frozen, same report, no trading
    world.forced[A].add(ist(2026, 1, 9, 9, 55))
    world.t = ist(2026, 1, 9, 10, 0) + 5_000
    later = world.bot()
    res = later.step()
    assert res.frozen and len(later.engine.trades) == trades and later.final == report
    v = world.view()
    assert v["status"]["label"] == "Experiment complete" and v["final_report"]["generated_ts"] == report["generated_ts"]
    assert v["experiment"]["day"] == 2 and v["experiment"]["progress"] == 1.0


def test_run_loop_stays_alive_at_night_with_heartbeats_and_never_trades(world):
    start = world.t = ist(2026, 1, 5, 1, 0) + 5_000
    world.forced[A].add(ist(2026, 1, 5, 1, 30))  # a signal in the middle of the night
    published: list[int] = []
    bot = world.bot()
    bot.run(3600, publish=lambda: published.append(world.t) or True)
    assert bot.engine.start_ts is None and not bot.engine.pending  # asleep: nothing starts, nothing trades
    assert start + 3_600_000 <= world.t < start + 3_700_000  # used its whole budget instead of exiting
    assert len(published) >= 3  # heartbeat roughly every 30 minutes, plus the final publish
    assert max(b - a for a, b in zip(published, published[1:])) <= 31 * 60_000


def test_run_loop_waits_for_the_open_then_trades_on_bar_closes(world):
    world.t = ist(2026, 1, 5, 6, 40) + 5_000  # 20 minutes before the session
    first_sig = ist(2026, 1, 5, 7, 10)
    world.forced[A].add(first_sig)
    bot = world.bot()
    published: list[int] = []
    bot.run(3600, publish=lambda: published.append(world.t) or True)
    eng = bot.engine
    assert eng.start_ts == ist(2026, 1, 5, 7, 0)  # nothing before 07:00 IST
    assert A in eng.positions or any(t.symbol == A for t in eng.trades)
    entry_ts = eng.positions[A].entry_ts if A in eng.positions else eng.trades[0].entry_ts
    assert entry_ts == first_sig + BAR
    assert world.t >= ist(2026, 1, 5, 7, 40)  # used its time budget
    assert len(published) >= 5


def test_run_stops_if_another_runner_owns_the_state(world):
    world.t = ist(2026, 1, 5, 9, 0) + 5_000
    bot = world.bot()
    calls: list[int] = []
    bot.run(3600, publish=lambda: calls.append(1) or False)  # publish rejected: lease lost
    assert len(calls) == 1 and world.t < ist(2026, 1, 5, 9, 1)


def test_real_pipeline_runs_end_to_end_without_stubs(tmp_path):
    frames = {A: make_candles(3000, DATA_START, seed=1, sigma=0.004), B: make_candles(3000, DATA_START, seed=2, sigma=0.004)}
    client = FakeClient(frames)
    t = ist(2026, 1, 9, 12, 0) + 5_000
    client.now = t
    md = MarketData("fake", 5, tmp_path / "candles", client=client, sleep=lambda s: None)
    bot = LiveBot(make_config(model={"train_days": 7, "min_samples": 50}), Store(tmp_path / "state"), md=md,
                  cache_dir=tmp_path, now_fn=lambda: t)
    res = bot.step()
    assert res.status == "ACTIVE" and res.new_bars == 1
    view = json.loads((tmp_path / "state" / "dashboard.json").read_text())
    assert {s["symbol"] for s in view["why"]["symbols"]} == {A, B}
    assert all(s["reason"] for s in view["why"]["symbols"])
    assert view["account"]["equity"] == 100.0 and view["model"]["cutoff_ts"] == ist(2026, 1, 9, 7, 0)


def test_shadow_book_trades_what_the_model_gate_rejects_without_touching_the_strategy(world, monkeypatch):
    monkeypatch.setattr(live, "score", lambda model, rows, cfg: (np.full(len(rows), 0.05), np.full(len(rows), -0.5)))
    sig = ist(2026, 1, 5, 9, 55)
    world.forced[A].add(sig)
    world.t = sig + BAR + 5_000
    bot = world.bot()
    bot.step()
    world.t += BAR
    bot.step()
    assert not bot.engine.positions and not bot.engine.pending  # the real strategy declined: p = 5%
    assert "probability" in bot.engine.decisions[-1]["reason"]
    assert A in bot.shadow.positions  # the ungated shadow took it
    assert bot.engine.equity == 100.0 and bot.engine.cash == 100.0
    view = world.view()
    assert view["shadow"]["open_positions"] == 1 and view["account"]["equity"] == 100.0
    twin = world.bot()  # shadow state survives a restart
    assert A in twin.shadow.positions


def test_late_first_run_stays_armed_and_starts_at_the_next_open(world):
    world.t = ist(2026, 1, 5, 16, 0) + 5_000  # only 7h of the session left: not a fair day 1
    world.forced[A].add(ist(2026, 1, 5, 15, 55))
    bot = world.bot()
    res = bot.step()
    assert res.status == "ARMED" and bot.engine.start_ts is None and not bot.engine.pending
    published: list[int] = []
    bot.run(3600, publish=lambda: published.append(world.t) or True)
    assert bot.engine.start_ts is None and published  # stays armed for its whole budget
    assert len(published) <= 4  # naps between heartbeats: no bar-by-bar polling while armed
    world.t = ist(2026, 1, 6, 7, 5) + 5_000
    nxt = world.bot()
    assert nxt.step().status == "ACTIVE"
    assert nxt.engine.start_ts == ist(2026, 1, 6, 7, 5) and nxt.engine.end_ts == ist(2026, 1, 15, 23, 0)


def test_signal_goes_stale_if_the_decision_is_made_late_within_one_pass(world, monkeypatch):
    """Slow work inside a pass (e.g. the daily retrain) must not let an old bar open a trade."""
    sig = ist(2026, 1, 5, 9, 55)
    world.forced[A].add(sig)
    world.t = sig + BAR + 5_000

    def slow_training(self, now, ts):
        self.model = object()
        world.t += 4 * 60_000  # four minutes pass before the decision is taken

    monkeypatch.setattr(LiveBot, "_ensure_model", slow_training)
    bot = world.bot()
    bot.step()
    assert not bot.engine.pending and "stale" in bot.engine.decisions[-1]["reason"]


def test_code_fingerprint_tracks_trading_code_only(tmp_path, monkeypatch):
    first = live.engine_code_hash()
    assert len(first) == 12 and first == live.engine_code_hash()  # stable
    pkg = tmp_path / "bot"
    pkg.mkdir()
    for src in live.Path(live.__file__).parent.glob("*.py"):
        (pkg / src.name).write_bytes(src.read_bytes())
    monkeypatch.setattr(live, "__file__", str(pkg / "live.py"))
    assert live.engine_code_hash() == first  # same files elsewhere -> same fingerprint
    (tmp_path / "dashboard.html").write_text("changed")  # non-trading file: ignored
    assert live.engine_code_hash() == first
    (pkg / "risk.py").write_text((pkg / "risk.py").read_text() + "\n# edit\n")
    assert live.engine_code_hash() != first  # any trading-code edit is detected


def test_one_job_can_sleep_through_the_night_and_start_trading_at_the_open(world):
    """No cron needed: a job that begins at 03:00 IST waits, then trades from 07:00 IST."""
    world.t = ist(2026, 1, 5, 3, 0) + 5_000
    first_sig = ist(2026, 1, 5, 7, 10)
    world.forced[A].add(first_sig)
    bot = world.bot()
    bot.run(5 * 3600, publish=lambda: True)  # 03:00 -> 08:00
    eng = bot.engine
    assert eng.start_ts == ist(2026, 1, 5, 7, 0)
    assert (eng.positions[A].entry_ts if A in eng.positions else eng.trades[0].entry_ts) == first_sig + BAR


def test_cli_marks_the_chain_finished_only_when_the_experiment_is_frozen(tmp_path, monkeypatch):
    import bot.cli as cli

    class Fake:
        def __init__(self, frozen):
            self.engine = type("E", (), {"frozen": frozen, "start_ts": 1})()
            self.cfg = type("C", (), {"hash": "x" * 64})()
            self.md = type("M", (), {"flush": lambda self: None})()
        def run(self, budget, publish):
            return None

    for frozen in (False, True):
        state = tmp_path / str(frozen)
        monkeypatch.setattr(live, "LiveBot", lambda *a, _f=frozen, **k: Fake(_f))
        assert cli.main(["--state-dir", str(state), "run"]) == 0
        assert (state / ".stop_chain").exists() is frozen
