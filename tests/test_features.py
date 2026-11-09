import numpy as np
import pandas as pd

from ropeway_wind_stop_forecast import features as F


def test_no_future_leakage(series):
    """Features at issue time t must not change when data after t is altered."""
    t = series.index[3000]
    a = F.make_features(series, 70).loc[t]
    s2 = series.copy()
    s2.loc[s2.index > t, ["gust", "wind", "dir", "pres", "temp"]] = 999.0
    b = F.make_features(s2, 70).loc[t]
    pd.testing.assert_series_equal(a, b)


def test_target_is_future_exceedance(series):
    X, y = F.build_xy(series, 70, 3)
    t = X.index[100]
    assert y.loc[t] == int(series["gust"].loc[t + pd.Timedelta(hours=3)] > 70)


def test_targets_drop_unknown_future(series):
    X, y = F.build_xy(series, 70, 6)
    fut = series["gust"].shift(-6).reindex(X.index)
    assert fut.notna().all() and len(X) == len(y)


def test_unknown_current_gust_rows_dropped(series):
    X, _ = F.build_xy(series, 70, 1)
    assert X["g0"].notna().all()


def test_feature_columns_and_order(series):
    X, _ = F.build_xy(series, 70, 2)
    assert list(X.columns) == F.FEATURES


def test_time_features_use_target_hour():
    idx = pd.date_range("2024-01-01 00:00", periods=3, freq="h")
    f = pd.DataFrame({c: 1.0 for c in F.FEATURES if not c.startswith(("hod", "doy"))}, index=idx)
    a, b = F.add_time_features(f, 6), F.add_time_features(f, 12)
    assert not np.allclose(a["hod_sin"], b["hod_sin"])
    assert np.isclose(a["hod_sin"].iloc[0], 1.0)  # 00:00 + 6 h = 06:00, sin(pi/2)


def test_gust_factor_capped_and_guarded():
    idx = pd.date_range("2024-01-01", periods=30, freq="h")
    s = pd.DataFrame(dict(gust=100.0, wind=0.0, dir=0.0, pres=750.0, temp=0.0), index=idx)
    assert F.make_features(s, 70)["gust_factor"].max() <= 6.0


def test_nexc_counts_exceedances():
    idx = pd.date_range("2024-01-01", periods=30, freq="h")
    g = np.full(30, 10.0)
    g[20:23] = 100.0
    s = pd.DataFrame(dict(gust=g, wind=5.0, dir=0.0, pres=750.0, temp=0.0), index=idx)
    f = F.make_features(s, 70)
    assert f["nexc6"].iloc[22] == 3 and f["nexc6"].iloc[25] == 3 and f["nexc6"].iloc[26] == 2 and f["nexc6"].iloc[29] == 0
