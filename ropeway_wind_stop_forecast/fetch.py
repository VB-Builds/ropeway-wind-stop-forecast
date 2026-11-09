"""Network tools: rebuild the committed fixture, fetch the latest MeteoSwiss data, SLF IMIS wind, Open-Meteo gust forecast.

    python -m ropeway_wind_stop_forecast fetch-fixture    # MeteoSwiss 10-min CSVs -> data/winters_hourly.csv.gz (large download)

Sources (cite when you use the output):
  MeteoSwiss SwissMetNet open data: https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn
  SLF IMIS public API (only the most recent days are served): https://measurement-api.slf.ch/public/api/imis
  Open-Meteo forecast API (NWP model gusts, CC BY 4.0): https://open-meteo.com
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BASE_URL, FIXTURE, STATIONS, parse_10min_csv, to_hourly, winter_of

PERIODS = ("historical_2010-2019", "historical_2020-2029", "recent")
IMIS_URL = "https://measurement-api.slf.ch/public/api/imis/station/{code}/measurements"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MS_TO_KMH = 3.6


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ropeway-wind-stop-forecast/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.write_bytes(_get(url))


def fetch_fixture(cache: Path, out: Path = FIXTURE) -> pd.DataFrame:
    """Download the 10-min files (cached in `cache`), aggregate to hours, keep winters >= 2010/11."""
    frames = []
    for abbr in STATIONS:
        low = abbr.lower()
        parts = []
        for period in PERIODS:
            dest = cache / f"{low}_{period}.csv"
            _download(f"{BASE_URL}/{low}/ogd-smn_{low}_t_{period}.csv", dest)
            parts.append(parse_10min_csv(dest.read_text(encoding="utf-8")))
        d10 = pd.concat(parts, ignore_index=True).drop_duplicates("time_utc").sort_values("time_utc")
        h = to_hourly(d10)
        h = h[winter_of(h["time_utc"]) >= 2010]
        frames.append(h)
    full = pd.concat(frames, ignore_index=True).sort_values(["station", "time_utc"])
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False, float_format="%.4g", compression="gzip")
    return full


def fetch_latest(station: str, cache: Path | None = None) -> pd.DataFrame:
    """Latest hours for one station: 'recent' (Jan 1 .. yesterday) + 'now' (today) 10-min files -> hourly."""
    low = station.lower()
    parts = []
    for period in ("recent", "now"):
        url = f"{BASE_URL}/{low}/ogd-smn_{low}_t_{period}.csv"
        raw = _get(url).decode("utf-8")
        if cache is not None:
            (cache / f"{low}_{period}.csv").write_text(raw, encoding="utf-8")
        parts.append(parse_10min_csv(raw))
    d10 = pd.concat(parts, ignore_index=True).drop_duplicates("time_utc").sort_values("time_utc")
    return to_hourly(d10)


def parse_imis(payload: list[dict]) -> pd.DataFrame:
    """SLF IMIS 30-min records -> frame with gust/wind in km/h (IMIS reports m/s), time_utc = measure_date.

    VW_30MIN_MAX is the maximum wind speed in the 30-min interval (a gust-like value, not the 3 s peak of the
    MeteoSwiss stations), so it is NOT mixed into the training data; it is for live context only.
    """
    rows = []
    for rec in payload:
        t = rec.get("measure_date")
        if not t:
            continue
        mean, mx, d = rec.get("VW_30MIN_MEAN"), rec.get("VW_30MIN_MAX"), rec.get("DW_30MIN_MEAN")
        rows.append(
            dict(
                time_utc=pd.to_datetime(t, utc=True).tz_convert(None),
                wind=np.nan if mean is None else float(mean) * MS_TO_KMH,
                gust=np.nan if mx is None else float(mx) * MS_TO_KMH,
                dir=np.nan if d is None else float(d),
            )
        )
    return pd.DataFrame(rows, columns=["time_utc", "wind", "gust", "dir"]).sort_values("time_utc").reset_index(drop=True)


def fetch_imis(code: str) -> pd.DataFrame:
    return parse_imis(json.loads(_get(IMIS_URL.format(code=code), timeout=60)))


def parse_open_meteo(payload: dict) -> pd.DataFrame:
    """Open-Meteo hourly response (wind_gusts_10m, wind_speed_10m in km/h, UTC) -> frame time_utc, nwp_gust, nwp_wind."""
    h = payload.get("hourly")
    if not h or "time" not in h or "wind_gusts_10m" not in h:
        raise ValueError("unexpected Open-Meteo response (no hourly wind_gusts_10m)")
    out = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(h["time"]),
            "nwp_gust": pd.to_numeric(pd.Series(h["wind_gusts_10m"]), errors="coerce"),
            "nwp_wind": pd.to_numeric(pd.Series(h.get("wind_speed_10m", [None] * len(h["time"]))), errors="coerce"),
        }
    )
    return out


def fetch_open_meteo(lat: float, lon: float, days: int = 2) -> pd.DataFrame:
    q = urllib.parse.urlencode(
        dict(
            latitude=lat,
            longitude=lon,
            hourly="wind_gusts_10m,wind_speed_10m",
            wind_speed_unit="kmh",
            forecast_days=days,
            timezone="UTC",
        )
    )
    return parse_open_meteo(json.loads(_get(f"{OPEN_METEO_URL}?{q}", timeout=60)))
