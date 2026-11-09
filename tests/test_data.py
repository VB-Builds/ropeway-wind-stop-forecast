import numpy as np
import pandas as pd
import pytest

from ropeway_wind_stop_forecast import data


def _csv(rows):
    head = "station_abbr;reference_timestamp;fu3010z1;fu3010z0;dkl010z0;prestas0;tre200s0"
    return "\n".join([head, *rows])


def test_winter_of_labels():
    idx = pd.to_datetime(["2022-11-01", "2023-01-15", "2023-04-30", "2023-05-01", "2023-07-01", "2022-12-31"])
    assert list(data.winter_of(idx)) == [2022, 2022, 2022, -1, -1, 2022]


def test_split_is_disjoint_and_ordered():
    tr, ca, te = set(data.TRAIN_WINTERS), set(data.CALIB_WINTERS), set(data.TEST_WINTERS)
    assert not (tr & ca) and not (tr & te) and not (ca & te)
    assert max(tr) < min(ca) and max(ca) < min(te)


def test_parse_10min_basic():
    df = data.parse_10min_csv(_csv(["SAE;01.01.2024 00:10;85.2;50.1;270;740.0;-8.1"]))
    assert df.loc[0, "station"] == "SAE"
    assert df.loc[0, "time_utc"] == pd.Timestamp("2024-01-01 00:10")
    assert df.loc[0, "gust"] == pytest.approx(85.2)


def test_parse_10min_missing_values_become_nan():
    df = data.parse_10min_csv(_csv(["SAE;01.01.2024 00:10;;50;;;"]))
    assert np.isnan(df.loc[0, "gust"]) and np.isnan(df.loc[0, "pres"])


def test_parse_10min_rejects_wrong_file():
    with pytest.raises(ValueError, match="missing columns"):
        data.parse_10min_csv("a;b\n1;2")


def test_parse_10min_drops_bad_timestamps():
    df = data.parse_10min_csv(_csv(["SAE;not a date;10;10;1;1;1", "SAE;01.01.2024 00:10;10;10;1;1;1"]))
    assert len(df) == 1


def _six(gusts, end="2024-01-01 01:00"):
    t = pd.date_range(end=end, periods=len(gusts), freq="10min")
    n = len(gusts)
    return pd.DataFrame(
        dict(station="SAE", time_utc=t, gust=gusts, wind=[20.0] * n, dir=[90.0] * n, pres=[740.0] * n, temp=[-5.0] * n)
    )


def test_to_hourly_takes_max_gust_and_labels_hour_end():
    h = data.to_hourly(_six([30, 40, 90, 50, 20, 10]))
    assert len(h) == 1
    assert h.loc[0, "gust"] == 90
    assert h.loc[0, "time_utc"] == pd.Timestamp("2024-01-01 01:00")  # 00:10..01:00 belongs to the hour ending 01:00


def test_to_hourly_drops_sparse_hours():
    assert len(data.to_hourly(_six([30, np.nan, np.nan, np.nan, 20, 10]))) == 0


def test_to_hourly_empty_input():
    assert data.to_hourly(pd.DataFrame(columns=data.COLUMNS)).empty


def test_to_hourly_vector_mean_direction_wraps():
    d = _six([30] * 6)
    d["dir"] = [350, 355, 0, 5, 10, 350]
    out = data.to_hourly(d).loc[0, "dir"]
    assert min(out, 360 - out) < 10  # an arithmetic mean would give about 178


def test_station_series_full_grid(long_df):
    s = data.station_series(long_df, "SAE")
    assert s.index.is_monotonic_increasing
    assert (s.index[1:] - s.index[:-1] == pd.Timedelta(hours=1)).all()


def test_station_series_unknown_station(long_df):
    with pytest.raises(KeyError):
        data.station_series(long_df, "XXX")


def test_fixture_loads_and_covers_all_stations():
    df = data.load_fixture()
    assert set(df["station"]) == set(data.STATIONS)
    assert df["gust"].dropna().between(0, 400).all()
    assert not df.duplicated(["station", "time_utc"]).any()


def test_coverage_table_counts_winters(long_df):
    assert data.coverage_table(long_df).loc[0, "winters"] == 4
