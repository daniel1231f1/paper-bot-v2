"""Command line entry point.

  python -m bot.cli run            # session loop (what GitHub Actions runs)
  python -m bot.cli run --once     # a single idempotent pass
  python -m bot.cli run --once --now 2026-11-12T03:10:00+05:30   # simulate the clock (past times only)
  python -m bot.cli clock --at 2026-11-12T06:59:00+05:30         # show what the schedule would do at a time
  python -m bot.cli status         # print a short summary of the saved state
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .clock import SessionClock
from .config import Config, ConfigError
from .state import StateError, Store

log = logging.getLogger("bot.cli")
LEASE_LOST = 3  # exit code of scripts/publish_state.sh when another writer advanced the state branch


class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to grep in the Actions log."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {"t": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="seconds"),
                   "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(as_json: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if as_json else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger("ccxt").setLevel(logging.WARNING)


def parse_time(text: str) -> int:
    """ISO-8601 with an explicit UTC offset -> epoch ms."""
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise argparse.ArgumentTypeError("time must include a UTC offset, e.g. 2026-11-12T07:00:00+05:30")
    return int(dt.timestamp() * 1000)


def make_publisher(script: str) -> Callable[[], bool]:
    def publish() -> bool:
        try:
            proc = subprocess.run(["bash", script], capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("publish failed (%s): will retry on the next bar", type(exc).__name__)
            return True  # a push hiccup must not stop the bot; state is safe on disk
        if proc.returncode == LEASE_LOST:
            return False
        if proc.returncode != 0:
            log.warning("publish failed (exit %d): %s", proc.returncode, proc.stderr.strip()[-300:])
        return True
    return publish


def cmd_run(args: argparse.Namespace) -> int:
    from .live import LiveBot

    cfg = Config.load(args.config)
    summary = None
    if Path(args.backtest_summary).exists():
        summary = json.loads(Path(args.backtest_summary).read_text())
    now_fn = (lambda: args.now) if args.now is not None else (lambda: int(time.time() * 1000))
    bot = LiveBot(cfg, Store(Path(args.state_dir)), cache_dir=Path(args.cache_dir), now_fn=now_fn,
                  backtest_summary=summary)
    log.info("paper bot ready: locked config %s, experiment day clock %s", bot.cfg.hash[:12],
             "running" if bot.engine.start_ts else "not started")
    if args.once or args.now is not None:
        res = bot.step()
        bot.md.flush()
        log.info("step: status=%s new_bars=%d positions=%d equity=%.4f", res.status, res.new_bars, res.open_positions,
                 bot.engine.equity)
        return 0
    publish = make_publisher(args.publish_script) if os.getenv("PUBLISH") == "1" else None
    bot.run(float(os.getenv("RUN_SECONDS", args.budget)), publish)
    if bot.engine.frozen:  # experiment over: the workflow must not queue another session
        Path(args.state_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.state_dir) / ".stop_chain").write_text("experiment complete\n")
    return 0


def cmd_clock(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    clock = SessionClock.from_cfg(cfg.schedule)
    ts = args.at
    print(json.dumps({
        "utc": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
        "local": clock.local(ts).isoformat(), "in_window": clock.in_window(ts),
        "entries_allowed": clock.entries_allowed(ts),
        "next_open_local": clock.local(clock.next_open(ts)).isoformat(),
    }, indent=1))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.state_dir) / "dashboard.json"
    if not path.exists():
        print("no dashboard.json yet")
        return 1
    v = json.loads(path.read_text())
    print(json.dumps({"status": v["status"]["label"], "day": f"{v['experiment']['day']}/{v['experiment']['days']}",
                      "equity": v["account"]["equity"], "trades": v["performance"]["trades"],
                      "positions": len(v["positions"]), "config_hash": v["experiment"]["config_hash"][:12],
                      "verdict": v["verdict"]["result"], "final": v["verdict"]["final"]}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="paper-bot", description="Paper-trading bot. No real-money code path exists.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--state-dir", default=os.getenv("STATE_DIR", "state"))
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--log-json", action="store_true", default=os.getenv("CI") == "true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the session loop (or one pass with --once)")
    run.add_argument("--once", action="store_true")
    run.add_argument("--now", type=parse_time, default=None, help="simulate the wall clock (implies --once)")
    run.add_argument("--budget", type=float, default=19_800, help="seconds before handing over to the next job")
    run.add_argument("--publish-script", default="scripts/publish_state.sh")
    run.add_argument("--backtest-summary", default="backtest_summary.json")
    run.set_defaults(fn=cmd_run)
    clk = sub.add_parser("clock", help="show schedule gating at a given time")
    clk.add_argument("--at", type=parse_time, required=True)
    clk.set_defaults(fn=cmd_clock)
    st = sub.add_parser("status", help="summarise the saved state")
    st.set_defaults(fn=cmd_status)
    args = ap.parse_args(argv)
    setup_logging(args.log_json)
    try:
        return int(args.fn(args))
    except (ConfigError, StateError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
