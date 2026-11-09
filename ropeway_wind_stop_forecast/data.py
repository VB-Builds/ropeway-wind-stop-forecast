"""MeteoSwiss SwissMetNet 10-minute wind data (open data) -> compact hourly winter fixture.

Source: MeteoSwiss, https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn (open data, source must be cited).
Raw 10-min timestamps are UTC and mark the END of the interval. Parameters used:
  fu3010z1  gust peak within the 10 min (km/h)      fu3010z0  wind speed scalar, 10-min mean (km/h)
  dkl010z0  wind direction, 10-min mean (deg)       prestas0  station pressure (hPa)
  tre200s0  air temperature 2 m (deg C)
Hourly layout (index label = END of the hour, UTC): gust = max of the 10-min gust peaks, wind = mean of 10-min means.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
DATA_DIR = Path(__file__).parent / "data"
FIXTURE = DATA_DIR / "winters_hourly.csv.gz"

# MeteoSwiss station metadata (ogd-smn_meta_stations.csv). These are weather stations at / near ski areas,
# NOT measurements at lift towers.
STATIONS: dict[str, dict] = {
    "SAE": dict(name="Säntis", elev=2501, lat=47.249447, lon=9.343469, canton="AI"),
    "PIL": dict(name="Pilatus", elev=2105, lat=46.978869, lon=8.252256, canton="OW"),
    "GUE": dict(name="Gütsch (Andermatt)", elev=2286, lat=46.652475, lon=8.615531, canton="UR"),
    "WFJ": dict(name="Weissfluhjoch", elev=2691, lat=46.833342, lon=9.806397, canton="GR"),
    "TIT": dict(name="Titlis", elev=3045, lat=46.770608, lon=8.425981, canton="OW"),
    "CMA": dict(name="Crap Masegn", elev=2468, lat=46.842275, lon=9.180042, canton="GR"),
    "ATT": dict(name="Les Attelas", elev=2734, lat=46.0991, lon=7.26865, canton="VS"),
    "NAS": dict(name="Naluns / Schlivera", elev=2380, lat=46.817164, lon=10.2614, canton="GR"),
}

PARAMS = {"fu3010z1": "gust", "fu3010z0": "wind", "dkl010z0": "dir", "prestas0": "pres", "tre200s0": "temp"}
COLUMNS = ["station", "time_utc", "gust", "wind", "dir", "pres", "temp"]
WINTER_MONTHS = (11, 12, 1, 2, 3, 4)
MIN_VALID_PER_HOUR = 4  # of 6 ten-minute values

# winter label = calendar year in which the season starts (2022 -> winter 2022/23)
TRAIN_WINTERS = tuple(range(2010, 2019))  # 2010/11 .. 2018/19
CALIB_WINTERS = (2019, 2020, 2021)  # 2019/20 .. 2021/22 (probability calibration + operating threshold)
TEST_WINTERS = (2022, 2023, 2024, 2025)  # 2022/23 .. 2025/26 (held out, never used for fitting or tuning)


def winter_of(ts) -> np.ndarray:
    """Winter label (start year) for each timestamp; months outside Nov-Apr give -1."""
    idx = pd.DatetimeIndex(ts)
    m = idx.month.to_numpy()
    y = idx.year.to_numpy()
    lab = np.where(m >= 11, y, y - 1)
    return np.where(np.isin(m, WINTER_MONTHS), lab, -1)


def parse_10min_csv(text: str) -> pd.DataFrame:
    """Parse one MeteoSwiss 10-min CSV into columns station, time_utc (end of interval) and PARAMS names."""
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    need = {"station_abbr", "reference_timestamp", "fu3010z1", "fu3010z0"}
    if not need <= set(df.columns):
        raise ValueError(f"unexpected MeteoSwiss file, missing columns {sorted(need - set(df.columns))}")
    out = pd.DataFrame(
        {
            "station": df["station_abbr"].str.upper(),
            "time_utc": pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M", errors="coerce"),
        }
    )
    for raw, name in PARAMS.items():
        out[name] = pd.to_numeric(df[raw], errors="coerce") if raw in df.columns else np.nan
    return out.dropna(subset=["time_utc"]).reset_index(drop=True)


def to_hourly(df10: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 10-min rows of ONE station to hours (label = end of hour, UTC).

    gust = max of 10-min peaks, wind/pres/temp = mean, dir = speed-weighted vector mean. Hours with fewer than
    MIN_VALID_PER_HOUR valid gust values are dropped.
    """
    if df10.empty:
        return pd.DataFrame(columns=COLUMNS)
    d = df10.copy()
    d["hour"] = (d["time_utc"] - pd.Timedelta(minutes=1)).dt.floor("h") + pd.Timedelta(hours=1)
    rad = np.deg2rad(d["dir"])
    wgt = d["wind"].fillna(1.0).clip(lower=0.1)
    d["_u"] = np.sin(rad) * wgt
    d["_v"] = np.cos(rad) * wgt
    g = d.groupby("hour")
    out = pd.DataFrame(
        {
            "gust": g["gust"].max(),
            "wind": g["wind"].mean(),
            "pres": g["pres"].mean(),
            "temp": g["temp"].mean(),
            "n": g["gust"].count(),
            "_u": g["_u"].mean(),
            "_v": g["_v"].mean(),
        }
    )
    out["dir"] = (np.rad2deg(np.arctan2(out["_u"], out["_v"])) + 360.0) % 360.0
    out = out[out["n"] >= MIN_VALID_PER_HOUR].drop(columns=["_u", "_v", "n"])
    out.index.name = "time_utc"
    out.insert(0, "station", d["station"].iloc[0])
    return out.reset_index()[COLUMNS]


def load_fixture(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or FIXTURE, parse_dates=["time_utc"])
    return df.sort_values(["station", "time_utc"]).reset_index(drop=True)


def station_series(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """One station on a complete hourly grid (NaN where missing), indexed by hour-end UTC."""
    s = df[df["station"] == station].drop(columns="station").set_index("time_utc").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        raise KeyError(f"no data for station {station}")
    return s.reindex(pd.date_range(s.index[0], s.index[-1], freq="h"))


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per station: first/last hour, number of winters present, valid gust hours."""
    rows = []
    for st, g in df.groupby("station"):
        w = winter_of(g["time_utc"])
        rows.append(
            dict(
                station=st,
                first=g["time_utc"].min().strftime("%Y-%m-%d"),
                last=g["time_utc"].max().strftime("%Y-%m-%d"),
                winters=len({int(x) for x in w if x >= 0}),
                valid_gust_hours=int(g["gust"].notna().sum()),
            )
        )
    return pd.DataFrame(rows)
