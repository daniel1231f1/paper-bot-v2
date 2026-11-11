"""Outlook: how often has this asset been higher 1 month / 3 months / 1 year later?

This is NOT a forecast and NOT advice. It reports measured historical frequencies and then
grades itself: every estimate is replayed over the asset's own past (walk-forward, using only
data available at the time) and scored against a coin flip and against the plain base rate.

Method, per asset and horizon h (in bars):
  base rate      share of all past h-bar windows that ended higher
  conditional    the same share, restricted to windows that started in today's trend state
                 (price above / below its 200-day average), shrunk toward the base rate
  range          80% Wilson interval using the EFFECTIVE sample size n/h, because
                 overlapping windows are not independent observations
  headline       the conditional estimate only where it beat the base rate out of sample;
                 otherwise the base rate

Independent of the trading bot: separate package, separate workflow, separate branch.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from bot.state import atomic_write_json

log = logging.getLogger("invest.outlook")

HORIZONS: tuple[tuple[str, float], ...] = (("1 month", 1 / 12), ("3 months", 3 / 12), ("1 year", 1.0))
PRIOR_WINDOWS = 5.0  # shrinkage strength: the base rate counts as this many independent windows
MIN_TRAIN_YEARS = 3.0  # history required before the first scored estimate
MIN_SCORED = 30  # scored past calls needed before the conditional estimate may be used
MIN_SKILL = 0.02  # ...and it must have beaten the base rate by this much (Brier skill); smaller gaps are noise
Z80 = 1.2816  # two-sided 80% normal quantile
KINDS = ("index", "stock", "commodity", "crypto")
SYMBOL_RE = re.compile(r"^[A-Za-z0-9^=.\-]{1,20}$")
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1=0&period2={now}&interval=1d&events=div%2Csplit"
USER_AGENT = "Mozilla/5.0 (compatible; price-history-client/1.0)"  # generic: identifies nothing about the user

Fetcher = Callable[[str], bytes]


class OutlookError(RuntimeError):
    """An asset could not be fetched or does not have enough usable history."""


# ---------------------------------------------------------------- data
def http_fetch(url: str, retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> bytes:
    """GET with exponential backoff on rate limits / server errors."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https host
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise OutlookError(f"HTTP {exc.code}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        sleep(2.0 * 2 ** attempt)
    raise OutlookError(f"unreachable after {retries} tries: {type(last).__name__}")


def parse_chart(body: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Yahoo chart JSON -> (frame[date, close] of split/dividend-adjusted closes, meta)."""
    try:
        result = json.loads(body)["chart"]["result"][0]
        ts = result["timestamp"]
        ind = result["indicators"]
        closes = (ind.get("adjclose") or [{}])[0].get("adjclose") or ind["quote"][0]["close"]
        meta = result.get("meta", {})
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise OutlookError(f"unexpected response shape: {type(exc).__name__}") from exc
    df = pd.DataFrame({"ts": ts, "close": closes}).dropna()
    df = df[df["close"] > 0].drop_duplicates("ts").sort_values("ts")
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.normalize()
    df = df.drop_duplicates("date", keep="last").reset_index(drop=True)
    return df[["date", "close"]], {"currency": meta.get("currency"), "name": meta.get("longName") or meta.get("shortName")}


def fetch_history(symbol: str, now_s: int, fetch: Fetcher = http_fetch) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not SYMBOL_RE.match(symbol):
        raise OutlookError(f"invalid symbol {symbol!r}")
    return parse_chart(fetch(CHART_URL.format(symbol=urllib.parse.quote(symbol, safe=""), now=now_s)))


# ---------------------------------------------------------------- statistics
def wilson(p: float, n: float, z: float = Z80) -> tuple[float, float]:
    """Wilson score interval for a proportion with (possibly fractional) sample size n."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def shrink(ups: float, n: float, base: float, h: int) -> tuple[float, float]:
    """Conditional frequency shrunk toward the base rate. Returns (estimate, effective windows).

    n overlapping h-bar windows carry roughly n/h independent observations.
    """
    n_eff = n / h
    raw = ups / n if n > 0 else base
    return (raw * n_eff + base * PRIOR_WINDOWS) / (n_eff + PRIOR_WINDOWS), n_eff


@dataclass(frozen=True)
class Windows:
    """All h-bar forward windows of one price series, with causal trend state at each start."""
    h: int
    up: np.ndarray  # bool, window starting at t ended higher (length n - h)
    ret: np.ndarray  # simple return of that window
    state: np.ndarray  # at t: 1 above the long average, 0 below, -1 unknown (length n)
    cum_n: np.ndarray  # prefix counts, shape (3, n - h + 1): rows = all, state 0, state 1
    cum_up: np.ndarray

    @classmethod
    def build(cls, close: np.ndarray, h: int, sma_window: int) -> "Windows":
        close = np.asarray(close, dtype=float)
        logp = np.log(close)
        fwd = logp[h:] - logp[:-h]
        sma = pd.Series(close).rolling(sma_window).mean().to_numpy()  # trailing: uses bars <= t only
        state = np.where(np.isnan(sma), -1, (close > sma).astype(np.int8)).astype(np.int8)
        up = fwd > 0
        masks = [np.ones(len(up), dtype=bool), state[: len(up)] == 0, state[: len(up)] == 1]
        zero = np.zeros((3, 1))
        cum_n = np.concatenate([zero, np.cumsum(np.array(masks, dtype=float), axis=1)], axis=1)
        cum_up = np.concatenate([zero, np.cumsum(np.array([m & up for m in masks], dtype=float), axis=1)], axis=1)
        return cls(h, up, np.expm1(fwd), state, cum_n, cum_up)

    def estimate(self, cut: int, state: int) -> dict[str, float]:
        """Estimates from the first `cut` windows only (those that had resolved by decision time)."""
        n_all, up_all = self.cum_n[0, cut], self.cum_up[0, cut]
        base = up_all / n_all if n_all > 0 else 0.5
        if state < 0:
            return {"base": base, "cond": base, "n_eff": n_all / self.h, "n_eff_base": n_all / self.h}
        cond, n_eff = shrink(self.cum_up[state + 1, cut], self.cum_n[state + 1, cut], base, self.h)
        return {"base": base, "cond": cond, "n_eff": n_eff, "n_eff_base": n_all / self.h}

    def walk_forward(self, min_train: int) -> dict[str, np.ndarray]:
        """Replay: at each past decision bar i (every h bars) estimate from windows resolved by i,
        then look up what actually happened. Outcomes do not overlap, estimates never see the future."""
        idx = np.arange(max(min_train, 2 * self.h), len(self.up), self.h)
        base, cond = np.empty(len(idx)), np.empty(len(idx))
        for k, i in enumerate(idx):
            est = self.estimate(int(i) - self.h + 1, int(self.state[i]))  # window t usable iff t + h <= i
            base[k], cond[k] = est["base"], est["cond"]
        return {"idx": idx, "base": base, "cond": cond, "y": self.up[idx].astype(float)}


def score(wf: Mapping[str, np.ndarray]) -> dict[str, Any]:
    n = len(wf["y"])
    if n == 0:
        return {"n": 0, "brier": None, "brier_base": None, "skill_vs_base": None, "n_60": 0, "hit_60": None}
    y = wf["y"]
    b_model, b_base = float(np.mean((wf["cond"] - y) ** 2)), float(np.mean((wf["base"] - y) ** 2))
    confident = wf["cond"] >= 0.6
    return {"n": n, "brier": b_model, "brier_base": b_base,
            "skill_vs_base": 1 - b_model / b_base if b_base > 0 else None,
            "n_60": int(confident.sum()), "hit_60": float(y[confident].mean()) if confident.any() else None}


def bars_per_year(dates: pd.Series) -> float:
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return (len(dates) - 1) / years if years > 0 else 252.0


def analyse(df: pd.DataFrame) -> dict[str, Any]:
    """All horizons for one asset. O(n) per horizon."""
    if len(df) < 300:
        raise OutlookError("not enough price history")
    close = df["close"].to_numpy(float)
    bpy = bars_per_year(df["date"])
    sma_window = max(20, round(bpy * 200 / 252))
    n = len(close)
    sma_now = float(np.mean(close[-sma_window:])) if n >= sma_window else float("nan")
    horizons, squared = [], []
    for label, years in HORIZONS:
        h = max(1, round(bpy * years))
        min_train = round(bpy * MIN_TRAIN_YEARS)
        if n < min_train + 3 * h:
            horizons.append({"label": label, "bars": h, "prob": None, "note": "not enough history for this horizon"})
            continue
        w = Windows.build(close, h, sma_window)
        state_now = int(w.state[-1])
        est = w.estimate(len(w.up), state_now)
        wf = w.walk_forward(min_train)
        sc = score(wf)
        use_cond = state_now >= 0 and sc["n"] >= MIN_SCORED and (sc["skill_vs_base"] or 0) >= MIN_SKILL
        prob = est["cond"] if use_cond else est["base"]
        n_eff = (est["n_eff"] + PRIOR_WINDOWS) if use_cond else est["n_eff_base"]
        lo, hi = wilson(prob, n_eff)
        rets = w.ret[w.state[: len(w.up)] == state_now] if use_cond else w.ret
        p10, p50, p90 = (float(x) for x in np.percentile(rets, [10, 50, 90]))
        horizons.append({
            "label": label, "bars": h, "prob": prob, "lo": lo, "hi": hi, "base": est["base"], "cond": est["cond"],
            "used": "conditional" if use_cond else "base", "n_eff": n_eff, "windows": int(len(rets)),
            "dist": {"p10": p10, "p50": p50, "p90": p90}, "score": sc,
        })
        squared.append((label, wf))
    return {
        "as_of": df["date"].iloc[-1].strftime("%Y-%m-%d"), "last": float(close[-1]),
        "history_years": round((df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25, 1),
        "trend": {"state": "unknown" if math.isnan(sma_now) else "above" if close[-1] > sma_now else "below",
                  "average": None if math.isnan(sma_now) else sma_now,
                  "gap": None if math.isnan(sma_now) else float(close[-1] / sma_now - 1)},
        "horizons": horizons, "_wf": squared,
    }


# ---------------------------------------------------------------- orchestration
def load_watchlist(path: Path) -> list[dict[str, str]]:
    raw = yaml.safe_load(path.read_text()) or {}
    items = raw.get("assets") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise OutlookError("watchlist must contain a non-empty 'assets' list")
    out = []
    for item in items:
        symbol, name, kind = str(item.get("symbol", "")), str(item.get("name", "")), str(item.get("kind", ""))
        if not SYMBOL_RE.match(symbol) or not name or kind not in KINDS:
            raise OutlookError(f"bad watchlist entry: {item!r} (need symbol, name, kind in {KINDS})")
        out.append({"symbol": symbol, "name": name, "kind": kind})
    if len({a["symbol"] for a in out}) != len(out):
        raise OutlookError("watchlist has duplicate symbols")
    return out


def build(watchlist: Sequence[Mapping[str, str]], now_s: int, fetch: Fetcher = http_fetch,
          previous: Mapping[str, Any] | None = None, pause: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Outlook for every asset. A failed asset keeps its previous entry (flagged stale) if there is one."""
    old = {a["symbol"]: a for a in (previous or {}).get("assets", []) if a.get("horizons")}
    assets, pooled = [], {label: [] for label, _ in HORIZONS}
    for k, item in enumerate(watchlist):
        if k:
            pause(1.0)  # be polite to a free endpoint
        entry: dict[str, Any] = {**item, "currency": None, "stale": False, "error": None}
        try:
            df, meta = fetch_history(item["symbol"], now_s, fetch)
            result = analyse(df)
            for label, wf in result.pop("_wf"):
                pooled[label].append(wf)
            entry.update(result, currency=meta.get("currency"))
        except OutlookError as exc:
            log.warning("%s failed: %s", item["symbol"], exc)
            if item["symbol"] in old:
                entry = {**old[item["symbol"]], **item, "stale": True, "error": str(exc)}
            else:
                entry.update(horizons=[], error=str(exc))
        assets.append(entry)
    summary = []
    for label, runs in pooled.items():
        if not runs:
            continue
        y = np.concatenate([r["y"] for r in runs])
        cond, base = np.concatenate([r["cond"] for r in runs]), np.concatenate([r["base"] for r in runs])
        summary.append({"label": label, "calls": int(len(y)), "brier": float(np.mean((cond - y) ** 2)),
                        "brier_base": float(np.mean((base - y) ** 2)), "brier_coin": 0.25,
                        "up_rate": float(y.mean())})
    return {"generated_ts": now_s * 1000, "assets": assets, "summary": summary,
            "ok": sum(1 for a in assets if a.get("horizons") and not a["stale"]),
            "method": {"prior_windows": PRIOR_WINDOWS, "interval": "80% Wilson on effective sample size",
                       "trend": "price vs 200-day average", "source": "Yahoo Finance daily adjusted closes"}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Historical odds for a watchlist. Not a forecast, not advice.")
    ap.add_argument("--watchlist", default="invest/watchlist.yaml")
    ap.add_argument("--out", default="outlook.json")
    ap.add_argument("--previous", default="", help="last published outlook.json, reused for assets that fail today")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        watchlist = load_watchlist(Path(args.watchlist))
        previous = json.loads(Path(args.previous).read_text()) if args.previous and Path(args.previous).exists() else None
    except (OutlookError, OSError, ValueError) as exc:
        log.error("cannot start: %s", exc)
        return 2
    view = build(watchlist, int(time.time()), previous=previous)
    if view["ok"] == 0 and not any(a.get("horizons") for a in view["assets"]):
        log.error("no asset could be analysed: not publishing an empty outlook")
        return 1
    atomic_write_json(Path(args.out), view)
    log.info("outlook written: %d/%d assets fresh", view["ok"], len(view["assets"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
