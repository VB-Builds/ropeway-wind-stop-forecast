"""Features and targets on the hourly grid.

Issue time t = end of the last observed hour. Target for lead k (hours): hourly max gust of hour t+k exceeds the threshold.
Only data up to and including t is used for features (checked by tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3, 4, 5, 6)
FEATURES = [
    "g0", "gmax3", "gmax6", "gmax12", "gmax24", "w0", "wmean3", "wmean6", "gust_factor",
    "dg3", "dg6", "dw3", "dir_sin", "dir_cos", "dp3", "dp6", "dp12", "dtemp3", "nexc6", "nexc24",
    "g_over_thr", "hod_sin", "hod_cos", "doy_sin", "doy_cos",
]  # fmt: skip


def _roll(s: pd.Series, w: int, how: str) -> pd.Series:
    r = s.rolling(w, min_periods=int(np.ceil(0.75 * w)))
    return getattr(r, how)()


def make_features(s: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Feature frame (one row per issue hour of the complete hourly grid `s`) for gust threshold `thr` km/h.

    `s` needs columns gust, wind, dir, pres, temp (NaN allowed). Columns of the result are FEATURES; rows with an
    unknown current gust are NaN in g0 and should be dropped by the caller.
    """
    g, w = s["gust"], s["wind"]
    f = pd.DataFrame(index=s.index)
    f["g0"] = g
    f["gmax3"] = _roll(g, 3, "max")
    f["gmax6"] = _roll(g, 6, "max")
    f["gmax12"] = _roll(g, 12, "max")
    f["gmax24"] = _roll(g, 24, "max")
    f["w0"] = w
    f["wmean3"] = _roll(w, 3, "mean")
    f["wmean6"] = _roll(w, 6, "mean")
    f["gust_factor"] = (g / w.clip(lower=5.0)).clip(upper=6.0)
    f["dg3"] = g - g.shift(3)
    f["dg6"] = g - g.shift(6)
    f["dw3"] = w - w.shift(3)
    rad = np.deg2rad(s["dir"])
    f["dir_sin"] = np.sin(rad)
    f["dir_cos"] = np.cos(rad)
    f["dp3"] = s["pres"] - s["pres"].shift(3)
    f["dp6"] = s["pres"] - s["pres"].shift(6)
    f["dp12"] = s["pres"] - s["pres"].shift(12)
    f["dtemp3"] = s["temp"] - s["temp"].shift(3)
    over = (g > thr).astype(float).where(g.notna())
    f["nexc6"] = _roll(over, 6, "sum")
    f["nexc24"] = _roll(over, 24, "sum")
    f["g_over_thr"] = g / thr
    return f[[c for c in FEATURES if c in f.columns and not c.startswith(("hod", "doy"))]]


def add_time_features(f: pd.DataFrame, k: int) -> pd.DataFrame:
    """Add hour-of-day / day-of-year of the TARGET hour (issue + k)."""
    tgt = f.index + pd.Timedelta(hours=k)
    out = f.copy()
    hod = (tgt.hour.to_numpy() % 24) / 24.0
    doy = tgt.day_of_year.to_numpy() / 366.0
    out["hod_sin"] = np.sin(2 * np.pi * hod)
    out["hod_cos"] = np.cos(2 * np.pi * hod)
    out["doy_sin"] = np.sin(2 * np.pi * doy)
    out["doy_cos"] = np.cos(2 * np.pi * doy)
    return out[FEATURES]


def build_xy(s: pd.DataFrame, thr: float, k: int, need_target: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Rows (issue hours) with known current gust and, if `need_target`, a known target for lead k.

    y = 1 if gust at t+k > thr else 0. Index = issue time.
    """
    f = add_time_features(make_features(s, thr), k)
    fut = s["gust"].shift(-k)
    ok = f["g0"].notna()
    if need_target:
        ok &= fut.notna()
    y = (fut > thr).astype(int)
    return f[ok], y[ok]


def exceed_series(s: pd.DataFrame, thr: float) -> pd.Series:
    """Boolean per hour: gust > thr (NaN-aware: unknown hours are False but flagged by `s['gust'].isna()`)."""
    return (s["gust"] > thr).fillna(False)
