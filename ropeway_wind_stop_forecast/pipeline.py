"""End-to-end: evaluate on held-out winters (parallel over station x threshold), write tables, figures and an example feed."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CALIB_WINTERS, STATIONS, TEST_WINTERS, TRAIN_WINTERS, coverage_table, load_fixture, station_series
from .evaluate import evaluate_station, lead_breakdown, reliability_table
from .features import HORIZONS
from .model import make_model
from .ops import feed_payload

THRESHOLDS = (60, 70, 80)
FAST = dict(
    stations=("SAE", "CMA"), thresholds=(70,), kinds=("climatology", "persistence", "logreg"),
    train=tuple(range(2015, 2019)), calib=CALIB_WINTERS, test=(2024, 2025),
)  # fmt: skip
PRIMARY_THR = 70


def _job(args):
    from threadpoolctl import threadpool_limits

    df, station, thr, kinds, train, calib, test, want_example = args
    with threadpool_limits(1):
        s = station_series(df, station)
        rows, ctx = evaluate_station(s, station, thr, kinds, train, calib, test)
        lead = lead_breakdown(ctx["longs"], station, thr)
        rel = []
        for kind, L in ctx["longs"].items():
            t = reliability_table(L["p"].to_numpy(), L["y"].to_numpy())
            t.insert(0, "model", kind)
            t.insert(0, "thr", thr)
            t.insert(0, "station", station)
            rel.append(t)
        example = pick_example(s, ctx, rows, thr, station) if want_example else None
    return rows, lead, pd.concat(rel, ignore_index=True), example


def pick_example(s: pd.DataFrame, ctx: dict, rows: list[dict], thr: float, station: str) -> dict | None:
    """A held-out storm the best learned model warned >= 3 h ahead: gust / +3 h probability timeline and a feed record."""
    learned = [r for r in rows if r["model"] in ("gboost", "logreg")]
    if not learned or not ctx["episodes"]:
        return None
    best = max(learned, key=lambda r: np.nan_to_num(r["bss_clim"], nan=-9))["model"]
    P, p_star = ctx["preds"][best]
    cands = []
    for e in ctx["episodes"]:
        t0 = e - pd.Timedelta(hours=3)
        quiet = s["gust"].loc[e - pd.Timedelta(hours=24) : e - pd.Timedelta(hours=1)]
        if t0 in P.index and P.at[t0, 3] >= p_star and quiet.notna().sum() >= 20 and quiet.max() < thr:
            win = s["gust"].loc[e : e + pd.Timedelta(hours=12)]
            cands.append((float(win.max()), e))
    if not cands:
        return None
    _, e = max(cands)
    idx = pd.date_range(e - pd.Timedelta(hours=36), e + pd.Timedelta(hours=24), freq="h")
    tl = pd.DataFrame({"gust": s["gust"].reindex(idx)})
    tl["p_lead3"] = [P.at[t - pd.Timedelta(hours=3), 3] if (t - pd.Timedelta(hours=3)) in P.index else np.nan for t in idx]
    issue = e - pd.Timedelta(hours=3)
    probs = {int(k): float(P.at[issue, k]) for k in HORIZONS}
    feed = feed_payload(STATIONS[station]["name"], thr, issue.to_pydatetime(), probs, p_star, best)
    return dict(model=best, station=station, thr=thr, episode_start=e, p_star=p_star, timeline=tl, feed=feed)


def run_evaluation(df: pd.DataFrame, cfg: dict, jobs: int | None = None) -> dict:
    todo = [
        (df[df["station"] == st], st, thr, cfg["kinds"], cfg["train"], cfg["calib"], cfg["test"],
         thr == PRIMARY_THR and st == cfg["stations"][0])
        for st in cfg["stations"] for thr in cfg["thresholds"]
    ]  # fmt: skip
    jobs = jobs or min(len(todo), os.cpu_count() or 1)
    if jobs <= 1:
        res = [_job(a) for a in todo]
    else:
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(_job, todo))
    return dict(
        metrics=pd.DataFrame([r for x in res for r in x[0]]),
        lead=pd.DataFrame([r for x in res for r in x[1]]),
        reliability=pd.concat([x[2] for x in res], ignore_index=True),
        example=next((x[3] for x in res if x[3]), None),
        cfg=cfg,
    )


def pooled_reliability(rel: pd.DataFrame, model: str, thr: float) -> pd.DataFrame:
    r = rel[(rel["model"] == model) & (rel["thr"] == thr)].copy()
    r["pm"], r["fy"] = r["p_mean"] * r["n"], r["freq"] * r["n"]
    g = r.groupby("bin").agg(pm=("pm", "sum"), fy=("fy", "sum"), n=("n", "sum"))
    return pd.DataFrame({"p_mean": g["pm"] / g["n"], "freq": g["fy"] / g["n"], "n": g["n"]}).reset_index()


def _r(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def _span(w) -> str:
    return f"{min(w)}/{str(min(w) + 1)[-2:]} to {max(w)}/{str(max(w) + 1)[-2:]}"


def summarise(res: dict, cov: pd.DataFrame) -> dict:
    m, cfg = res["metrics"], res["cfg"]
    out = {
        "train_winters": _span(cfg["train"]),
        "test_winters": _span(cfg["test"]),
        "stations": list(cfg["stations"]),
        "fixture_valid_gust_hours": int(cov["valid_gust_hours"].sum()),
        "by_threshold": {},
    }
    for thr, g in m.groupby("thr"):
        d = {}
        for kind, k in g.groupby("model"):
            d[kind] = dict(
                stations_scored=int(k["bss_clim"].notna().sum()),
                mean_bss_clim=_r(k["bss_clim"].mean()),
                mean_bss_pers=_r(k["bss_pers"].mean()),
                mean_ece=_r(k["ece"].mean()),
                pod=_r((k["pod"].fillna(0) * k["n_events"]).sum() / max(k["n_events"].sum(), 1)),
                episode_hit_rate=_r(k["episodes_warned"].sum() / max(k["episodes"].sum(), 1)),
                mean_lead_h=_r(k["mean_lead_h"].mean()),
                test_events=int(k["n_events"].sum()),
            )
        out["by_threshold"][int(thr)] = d
    return out


def run_demo(out: Path, fast: bool = False, jobs: int | None = None) -> dict:
    from . import plots

    out = Path(out)
    df = load_fixture()
    cfg = FAST if fast else dict(
        stations=tuple(STATIONS), thresholds=THRESHOLDS, kinds=("climatology", "persistence", "logreg", "gboost"),
        train=TRAIN_WINTERS, calib=CALIB_WINTERS, test=TEST_WINTERS,
    )  # fmt: skip
    res = run_evaluation(df, cfg, jobs)
    cov = coverage_table(df)
    summ = summarise(res, cov)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    res["metrics"].round(4).to_csv(out / "results" / "metrics.csv", index=False)
    res["lead"].round(4).to_csv(out / "results" / "lead_breakdown.csv", index=False)
    res["reliability"].round(4).to_csv(out / "results" / "reliability.csv", index=False)
    cov.to_csv(out / "results" / "coverage.csv", index=False)
    (out / "results" / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    if res["example"]:
        (out / "docs" / "example_feed.json").write_text(
            json.dumps(res["example"]["feed"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    plots.main_figure(res, out / "docs" / "figure.png")
    plots.post_figure(res, out / "docs" / "post-figure.png")
    return summ


def live_forecast(station: str, thr: float, kind: str = "logreg", at: str | None = None) -> dict:
    """Feed record for one station. With `at` (UTC hour end, inside the fixture) replay a past hour using only
    pre-test training data; otherwise fetch the latest MeteoSwiss hours and train on all winters except calibration."""
    from .fetch import fetch_latest

    station = station.upper()
    if station not in STATIONS:
        raise KeyError(f"unknown station {station}; choose from {', '.join(STATIONS)}")
    df = load_fixture()
    if at:
        issue = pd.Timestamp(at)
        s = station_series(df, station)
        if issue not in s.index or pd.isna(s.at[issue, "gust"]):
            raise ValueError(f"no observation for {station} at {issue}")
        m = make_model(kind, thr).fit(s, TRAIN_WINTERS, CALIB_WINTERS)
    else:
        latest = fetch_latest(station)
        merged = pd.concat([df[df["station"] == station], latest]).drop_duplicates(["station", "time_utc"], keep="last")
        s = station_series(merged, station)
        issue = s["gust"].last_valid_index()
        m = make_model(kind, thr).fit(s, TRAIN_WINTERS + TEST_WINTERS, CALIB_WINTERS)
    P = m.predict_wide(s)
    if issue not in P.index:
        raise ValueError("not enough recent observations to build features")
    probs = {int(k): float(P.at[issue, k]) for k in HORIZONS}
    return feed_payload(STATIONS[station]["name"], thr, issue.to_pydatetime(), probs, m.p_star, kind)
