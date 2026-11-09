"""Verification on held-out winters: Brier score / skill, reliability, hit and false-alarm rates, warning lead time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CALIB_WINTERS, TEST_WINTERS, TRAIN_WINTERS, winter_of
from .features import HORIZONS
from .model import KINDS, make_model

MIN_TEST_EVENTS = 30  # below this, skill numbers for a station/threshold are reported as n/a


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def brier_skill(b: float, b_ref: float) -> float:
    """1 - B/B_ref (1 = perfect, 0 = as good as the reference, < 0 = worse)."""
    if not np.isfinite(b) or not np.isfinite(b_ref) or b_ref <= 0:
        return float("nan")
    return 1.0 - b / b_ref


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins: mean forecast, observed frequency, count."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for i in range(bins):
        sel = b == i
        if sel.any():
            rows.append(dict(bin=i, p_mean=float(p[sel].mean()), freq=float(y[sel].mean()), n=int(sel.sum())))
    return pd.DataFrame(rows, columns=["bin", "p_mean", "freq", "n"])


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    t = reliability_table(p, y, bins)
    if t.empty:
        return float("nan")
    return float((t["n"] * (t["p_mean"] - t["freq"]).abs()).sum() / t["n"].sum())


def contingency(p: np.ndarray, y: np.ndarray, p_star: float) -> dict:
    """Hit rate (POD), false-alarm ratio (FAR = FP/(TP+FP)), false-alarm rate (POFD = FP/(FP+TN)) and CSI at p_star."""
    p, y = np.asarray(p, float), np.asarray(y).astype(bool)
    w = p >= p_star
    tp, fp = int((w & y).sum()), int((w & ~y).sum())
    fn, tn = int((~w & y).sum()), int((~w & ~y).sum())
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        pod=tp / (tp + fn) if tp + fn else float("nan"),
        far=fp / (tp + fp) if tp + fp else float("nan"),
        pofd=fp / (fp + tn) if fp + tn else float("nan"),
        csi=tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
    )  # fmt: skip


def find_episodes(gust: pd.Series, thr: float, winters, quiet_hours: int = 6) -> list[pd.Timestamp]:
    """Start hour of each exceedance episode: gust > thr after `quiet_hours` known hours without exceedance."""
    g = gust
    over = (g > thr).to_numpy()
    known = g.notna().to_numpy()
    in_w = np.isin(winter_of(g.index), list(winters))
    starts = []
    for i in np.flatnonzero(over & in_w):
        if i < quiet_hours:
            continue
        prev = slice(i - quiet_hours, i)
        if known[prev].all() and not over[prev].any():
            starts.append(g.index[i])
    return starts


def lead_time_stats(P: pd.DataFrame, starts: list[pd.Timestamp], p_star: float) -> dict:
    """For each episode start e: warned at issue t if P[t, k=e-t] >= p_star. Lead time = e - earliest such t (hours)."""
    leads = []
    for e in starts:
        best = 0
        for k in HORIZONS:
            t = e - pd.Timedelta(hours=k)
            if t in P.index and P.at[t, k] >= p_star:
                best = k
        leads.append(best)
    n = len(leads)
    det = [x for x in leads if x > 0]
    return dict(
        episodes=n,
        episodes_warned=len(det),
        episode_hit_rate=len(det) / n if n else float("nan"),
        mean_lead_h=float(np.mean(det)) if det else float("nan"),
    )


def long_predictions(P: pd.DataFrame, s: pd.DataFrame, thr: float, winters) -> pd.DataFrame:
    """Stack wide probabilities with their outcomes for issue hours in `winters` (rows with unknown target dropped)."""
    keep = np.isin(winter_of(P.index), list(winters))
    parts = []
    for k in HORIZONS:
        fut = s["gust"].shift(-k).reindex(P.index)
        ok = keep & fut.notna().to_numpy()
        parts.append(
            pd.DataFrame(
                {"t": P.index[ok], "k": k, "p": P.loc[ok, k].to_numpy(), "y": (fut[ok] > thr).astype(int).to_numpy()}
            )
        )
    return pd.concat(parts, ignore_index=True)


def evaluate_station(
    s: pd.DataFrame,
    station: str,
    thr: float,
    kinds=KINDS,
    train=TRAIN_WINTERS,
    calib=CALIB_WINTERS,
    test=TEST_WINTERS,
) -> tuple[list[dict], dict]:
    """Fit all `kinds` on `train` (+ calibrate/tune on `calib`), verify on `test`.

    Returns (metric rows, {kind: (wide probabilities, p_star, calibrated reliability table)}).
    """
    preds, longs = {}, {}
    for kind in kinds:
        m = make_model(kind, thr).fit(s, train, calib)
        P = m.predict_wide(s)
        preds[kind] = (P, m.p_star)
        longs[kind] = long_predictions(P, s, thr, test)
    starts = find_episodes(s["gust"], thr, test)
    ref_clim = brier(longs["climatology"]["p"], longs["climatology"]["y"]) if "climatology" in longs else float("nan")
    ref_pers = brier(longs["persistence"]["p"], longs["persistence"]["y"]) if "persistence" in longs else float("nan")
    rows = []
    for kind in kinds:
        L = longs[kind]
        P, p_star = preds[kind]
        n_ev = int(L["y"].sum())
        b = brier(L["p"], L["y"])
        enough = n_ev >= MIN_TEST_EVENTS
        c = contingency(L["p"], L["y"], p_star)
        lt = lead_time_stats(P, starts, p_star)
        rows.append(
            dict(
                station=station, thr=thr, model=kind, n_test=len(L), n_events=n_ev, base_rate=float(L["y"].mean()) if len(L) else float("nan"),
                brier=b,
                bss_clim=brier_skill(b, ref_clim) if enough else float("nan"),
                bss_pers=brier_skill(b, ref_pers) if enough else float("nan"),
                ece=expected_calibration_error(L["p"], L["y"]),
                p_star=p_star, pod=c["pod"], far=c["far"], pofd=c["pofd"], csi=c["csi"],
                episodes=lt["episodes"], episodes_warned=lt["episodes_warned"], episode_hit_rate=lt["episode_hit_rate"],
                mean_lead_h=lt["mean_lead_h"],
            )
        )  # fmt: skip
    return rows, {"preds": preds, "longs": longs, "episodes": starts}


def lead_breakdown(longs: dict, station: str, thr: float) -> list[dict]:
    """Brier skill vs climatology by lead hour for each model."""
    out = []
    if "climatology" not in longs:
        return out
    for kind, L in longs.items():
        for k in HORIZONS:
            a = L[L["k"] == k]
            r = longs["climatology"]
            r = r[r["k"] == k]
            out.append(
                dict(
                    station=station, thr=thr, model=kind, k=k, n_events=int(a["y"].sum()),
                    brier=brier(a["p"], a["y"]),
                    bss_clim=brier_skill(brier(a["p"], a["y"]), brier(r["p"], r["y"])) if a["y"].sum() >= MIN_TEST_EVENTS else float("nan"),
                )
            )  # fmt: skip
    return out
