import numpy as np
import pandas as pd
import pytest


def synth_series(winters=(2010, 2011, 2012, 2013), seed=0) -> pd.DataFrame:
    """Hourly Nov-Apr series with persistent wind regimes (AR(1) in log space), hour-end UTC index on a full grid."""
    rng = np.random.default_rng(seed)
    parts = []
    for w in winters:
        idx = pd.date_range(f"{w}-11-01 01:00", f"{w + 1}-04-30 23:00", freq="h")
        z = np.zeros(len(idx))
        for i in range(1, len(idx)):
            z[i] = 0.96 * z[i - 1] + rng.normal(0, 0.35)
        wind = 25 * np.exp(0.55 * z)
        gust = wind * rng.uniform(1.2, 1.8, len(idx))
        parts.append(
            pd.DataFrame(
                dict(gust=gust, wind=wind, dir=rng.uniform(0, 360, len(idx)), pres=750 + 5 * z, temp=-5 + rng.normal(0, 2, len(idx))),
                index=idx,
            )
        )
    s = pd.concat(parts)
    return s.reindex(pd.date_range(s.index[0], s.index[-1], freq="h"))


@pytest.fixture(scope="session")
def series():
    return synth_series()


@pytest.fixture(scope="session")
def long_df():
    s = synth_series().dropna(subset=["gust"]).reset_index(names="time_utc")
    s.insert(0, "station", "SAE")
    return s
