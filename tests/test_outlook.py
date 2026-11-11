"""Outlook estimator: honest statistics, no look-ahead, resilient to data failures."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from invest import outlook as ol
from invest.outlook import OutlookError, Windows, analyse, build, load_watchlist, parse_chart, shrink, wilson


def series(n: int = 3000, drift: float = 0.0003, sigma: float = 0.012, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, sigma, n)))
    return pd.DataFrame({"date": pd.bdate_range("2010-01-01", periods=n, tz="UTC"), "close": close})


def chart_json(df: pd.DataFrame, nulls: tuple[int, ...] = ()) -> bytes:
    closes = [None if i in nulls else float(c) for i, c in enumerate(df["close"])]
    ts = [int(d.timestamp()) for d in df["date"]]
    return json.dumps({"chart": {"result": [{"meta": {"currency": "USD", "longName": "Test Co"}, "timestamp": ts,
                                             "indicators": {"quote": [{"close": closes}], "adjclose": [{"adjclose": closes}]}}]}}).encode()


def test_wilson_interval_is_sane():
    lo, hi = wilson(0.6, 50)
    assert 0 <= lo < 0.6 < hi <= 1
    lo2, hi2 = wilson(0.6, 500)
    assert hi2 - lo2 < hi - lo  # more independent evidence -> tighter range
    assert wilson(0.5, 0) == (0.0, 1.0)  # no evidence -> no claim
    assert wilson(1.0, 10)[0] < 1.0  # never certain


def test_shrinkage_pulls_thin_evidence_toward_the_base_rate():
    est, n_eff = shrink(ups=0, n=0, base=0.6, h=21)
    assert est == pytest.approx(0.6) and n_eff == 0
    thin, _ = shrink(ups=21, n=21, base=0.6, h=21)  # one independent window, 100% up
    thick, _ = shrink(ups=21 * 200, n=21 * 200, base=0.6, h=21)
    assert 0.6 < thin < 0.7 < thick < 1.0


def test_overlapping_windows_count_as_few_independent_ones():
    _, n_eff = shrink(ups=500, n=1000, base=0.5, h=250)
    assert n_eff == pytest.approx(4.0)  # 1000 daily windows of ~1 year are only ~4 independent years


def test_always_rising_and_always_falling_series():
    up = pd.DataFrame({"date": pd.bdate_range("2010-01-01", periods=2500, tz="UTC"), "close": np.linspace(50, 500, 2500)})
    down = up.assign(close=np.linspace(500, 50, 2500))
    for h in analyse(up)["horizons"]:
        assert h["prob"] > 0.95 and h["hi"] <= 1.0 and h["lo"] < 1.0
    for h in analyse(down)["horizons"]:
        assert h["prob"] < 0.05


def test_probabilities_and_ranges_are_well_formed():
    res = analyse(series())
    assert res["trend"]["state"] in {"above", "below"}
    for h in res["horizons"]:
        assert 0 <= h["lo"] <= h["prob"] <= h["hi"] <= 1
        assert h["dist"]["p10"] <= h["dist"]["p50"] <= h["dist"]["p90"]
        assert h["used"] in {"base", "conditional"} and h["score"]["n"] > 0
    assert [h["bars"] for h in res["horizons"]] == [21, 63, 252] or res["horizons"][0]["bars"] in (20, 21, 22)


def test_crypto_calendar_uses_365_bars_per_year():
    df = series(2500).assign(date=pd.date_range("2015-01-01", periods=2500, tz="UTC"))  # trades every day
    bars = [h["bars"] for h in analyse(df)["horizons"]]
    assert bars[2] in (364, 365, 366) and bars[0] in (30, 31)


def test_walk_forward_never_sees_the_future():
    df = series(3000, seed=3)
    close = df["close"].to_numpy()
    w = Windows.build(close, 21, 200)
    wf = w.walk_forward(756)
    pivot = 2000
    tampered = close.copy()
    tampered[pivot + 1:] *= np.linspace(1, 5, len(close) - pivot - 1)  # rewrite everything after the pivot
    wf2 = Windows.build(tampered, 21, 200).walk_forward(756)
    early = wf["idx"] <= pivot  # decisions taken at or before the pivot
    assert early.sum() > 20
    np.testing.assert_allclose(wf["cond"][early], wf2["cond"][early])
    np.testing.assert_allclose(wf["base"][early], wf2["base"][early])


def test_walk_forward_outcomes_do_not_overlap():
    w = Windows.build(series(3000)["close"].to_numpy(), 63, 200)
    idx = w.walk_forward(756)["idx"]
    assert (np.diff(idx) == 63).all() and idx[0] >= 756


def test_conditional_estimate_is_used_only_when_it_earned_it():
    res = analyse(series(4000, seed=11))  # pure random walk: trend has no information
    assert all(h["used"] == "base" for h in res["horizons"])


def test_too_little_history_is_refused_or_marked():
    with pytest.raises(OutlookError):
        analyse(series(200))
    res = analyse(series(900))
    assert res["horizons"][-1]["prob"] is None and "not enough history" in res["horizons"][-1]["note"]


def test_parse_chart_drops_missing_prices_and_rejects_garbage():
    df = series(400)
    parsed, meta = parse_chart(chart_json(df, nulls=(5, 6, 399)))
    assert len(parsed) == 397 and meta == {"currency": "USD", "name": "Test Co"}
    with pytest.raises(OutlookError):
        parse_chart(b'{"chart": {"result": null, "error": {"code": "Not Found"}}}')
    with pytest.raises(OutlookError):
        parse_chart(b"<html>blocked</html>")


def test_symbols_are_validated_before_they_reach_a_url():
    for bad in ("../etc/passwd", "AAPL?x=1", "A B", "", "X" * 30, "AAPL/../x"):
        with pytest.raises(OutlookError):
            ol.fetch_history(bad, 0, fetch=lambda url: b"")
    seen = []
    ol.fetch_history("^GSPC", 123, fetch=lambda url: seen.append(url) or chart_json(series(400)))
    assert "%5EGSPC" in seen[0] and "period2=123" in seen[0]


def test_build_reuses_the_previous_entry_when_an_asset_fails(tmp_path):
    good = chart_json(series(3000))
    wl = [{"symbol": "AAA", "name": "A", "kind": "stock"}, {"symbol": "BBB", "name": "B", "kind": "index"}]
    first = build(wl, 1_700_000_000, fetch=lambda url: good, pause=lambda s: None)
    assert first["ok"] == 2 and len(first["summary"]) == 3

    def flaky(url: str) -> bytes:
        if "BBB" in url:
            raise OutlookError("HTTP 429")
        return good

    second = build(wl, 1_700_086_400, fetch=flaky, previous=first, pause=lambda s: None)
    b = next(a for a in second["assets"] if a["symbol"] == "BBB")
    assert second["ok"] == 1 and b["stale"] is True and b["error"] == "HTTP 429" and b["horizons"]
    fresh = build(wl, 1_700_086_400, fetch=flaky, pause=lambda s: None)  # no previous data to fall back on
    assert next(a for a in fresh["assets"] if a["symbol"] == "BBB")["horizons"] == []
    json.dumps(ol.atomic_write_json and second, allow_nan=False)  # strict JSON, no NaN leaks


def test_main_refuses_to_publish_an_empty_outlook(tmp_path, monkeypatch):
    wl = tmp_path / "w.yaml"
    wl.write_text("assets:\n  - {symbol: AAA, name: A, kind: stock}\n")
    monkeypatch.setattr(ol, "http_fetch", lambda url, **k: (_ for _ in ()).throw(OutlookError("down")))
    monkeypatch.setattr(ol.build, "__defaults__", (ol.http_fetch, None, lambda s: None))
    out = tmp_path / "outlook.json"
    assert ol.main(["--watchlist", str(wl), "--out", str(out)]) == 1 and not out.exists()


def test_watchlist_validation(tmp_path):
    assert len(load_watchlist(__import__("pathlib").Path("invest/watchlist.yaml"))) >= 10
    bad = tmp_path / "w.yaml"
    bad.write_text("assets:\n  - {symbol: 'A B', name: A, kind: stock}\n")
    with pytest.raises(OutlookError):
        load_watchlist(bad)
    bad.write_text("assets:\n  - {symbol: AAA, name: A, kind: bond}\n")
    with pytest.raises(OutlookError):
        load_watchlist(bad)
    bad.write_text("assets:\n  - {symbol: AAA, name: A, kind: stock}\n  - {symbol: AAA, name: A2, kind: stock}\n")
    with pytest.raises(OutlookError):
        load_watchlist(bad)
