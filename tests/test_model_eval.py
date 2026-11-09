import json

import numpy as np
import pandas as pd
import pytest

from ropeway_wind_stop_forecast import cli
from ropeway_wind_stop_forecast import evaluate as E
from ropeway_wind_stop_forecast import model as M
from ropeway_wind_stop_forecast.features import HORIZONS

TRAIN, CALIB, TEST = (2010, 2011), (2012,), (2013,)


def test_brier_known_values():
    assert E.brier([1, 0], [1, 0]) == 0.0
    assert E.brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert np.isnan(E.brier([], []))


def test_brier_skill_known_values():
    assert E.brier_skill(0.1, 0.2) == pytest.approx(0.5)
    assert E.brier_skill(0.3, 0.2) == pytest.approx(-0.5)
    assert np.isnan(E.brier_skill(0.1, 0.0))


def test_reliability_table_bins():
    t = E.reliability_table(np.array([0.05, 0.05, 0.95]), np.array([0, 1, 1]))
    assert list(t["n"]) == [2, 1] and t.loc[0, "freq"] == 0.5


def test_ece_perfect_and_biased():
    p = np.r_[np.full(100, 0.2), np.full(100, 0.8)]
    y = np.r_[np.r_[np.ones(20), np.zeros(80)], np.r_[np.ones(80), np.zeros(20)]]
    assert E.expected_calibration_error(p, y) == pytest.approx(0.0)
    assert E.expected_calibration_error(np.full(10, 0.9), np.zeros(10)) == pytest.approx(0.9)


def test_contingency_counts_and_rates():
    c = E.contingency(np.array([0.9, 0.9, 0.1, 0.1]), np.array([1, 0, 1, 0]), 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
    assert c["pod"] == 0.5 and c["far"] == 0.5 and c["pofd"] == 0.5 and c["csi"] == pytest.approx(1 / 3)


def test_contingency_no_events_gives_nan_not_error():
    c = E.contingency(np.array([0.1, 0.2]), np.array([0, 0]), 0.5)
    assert np.isnan(c["pod"]) and np.isnan(c["far"])


def test_best_csi_threshold_separable_and_no_events():
    y = np.array([1, 1, 0, 0])
    assert 0.2 < M.best_csi_threshold(y, np.array([0.9, 0.8, 0.1, 0.2])) <= 0.8
    assert M.best_csi_threshold(np.zeros(4), np.full(4, 0.5)) == 0.5


def test_calibrator_kinds():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 5000)
    y = (rng.uniform(size=5000) < p**2).astype(int)  # over-forecasting model
    c = M.Calibrator().fit(p, y)
    assert c.kind == "isotonic"
    assert abs(c.transform(np.array([0.5]))[0] - 0.25) < 0.08
    few = M.Calibrator().fit(p[:200], np.r_[np.ones(5), np.zeros(195)].astype(int))
    assert few.kind == "platt"
    assert M.Calibrator().fit(p, np.zeros(5000, int)).kind == "identity"


def test_find_episodes_needs_quiet_period():
    idx = pd.date_range("2022-12-01", periods=40, freq="h")
    g = pd.Series(10.0, index=idx)
    g.iloc[10:12] = 100.0  # episode
    g.iloc[14] = 100.0  # same storm, not a new episode (only 2 quiet hours)
    g.iloc[30] = 100.0  # new episode
    starts = E.find_episodes(g, 70, (2022,))
    assert list(starts) == [idx[10], idx[30]]


def test_lead_time_stats():
    idx = pd.date_range("2022-12-01", periods=30, freq="h")
    P = pd.DataFrame(0.0, index=idx, columns=list(HORIZONS))
    e = idx[20]
    P.at[e - pd.Timedelta(hours=4), 4] = 0.9
    P.at[e - pd.Timedelta(hours=2), 2] = 0.9
    st = E.lead_time_stats(P, [e, idx[10]], 0.5)
    assert st["episodes"] == 2 and st["episodes_warned"] == 1 and st["mean_lead_h"] == 4.0


def test_climatology_uses_month_hour_rate(series):
    m = M.make_model("climatology", 60).fit(series, TRAIN)
    P = m.predict_wide(series)
    assert P.shape[1] == 6 and P.min().min() >= 0 and P.max().max() <= 1


def test_persistence_probabilities_ordered(series):
    m = M.make_model("persistence", 60).fit(series, TRAIN)
    assert m.rate[(1, True)] > m.rate[(1, False)]


def test_unknown_model_kind_rejected():
    with pytest.raises(ValueError):
        M.Learned(70, "magic")


def test_learned_model_beats_climatology_on_persistent_series(series):
    rows, ctx = E.evaluate_station(series, "SYN", 55, ("climatology", "persistence", "logreg"), TRAIN, CALIB, TEST)
    by = {r["model"]: r for r in rows}
    assert by["climatology"]["bss_clim"] == pytest.approx(0.0, abs=1e-9)
    assert by["logreg"]["bss_clim"] > 0.2
    assert 0 < by["logreg"]["p_star"] <= 1
    P, _ = ctx["preds"]["logreg"]
    assert P.min().min() >= 0 and P.max().max() <= 1


def test_too_few_test_events_reports_nan_skill(series):
    rows, _ = E.evaluate_station(series, "SYN", 400, ("climatology", "persistence"), TRAIN, CALIB, TEST)
    assert all(np.isnan(r["bss_clim"]) for r in rows)


def test_predictions_do_not_use_future(series):
    m = M.make_model("logreg", 55).fit(series, TRAIN, CALIB)
    t = series.index[4000]
    a = m.predict_wide(series).loc[t]
    s2 = series.copy()
    s2.loc[s2.index > t, ["gust", "wind", "pres", "temp", "dir"]] = 300.0
    b = m.predict_wide(s2).loc[t]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_lead_breakdown_has_all_leads(series):
    _, ctx = E.evaluate_station(series, "SYN", 55, ("climatology", "persistence"), TRAIN, CALIB, TEST)
    lb = E.lead_breakdown(ctx["longs"], "SYN", 55)
    assert sorted({r["k"] for r in lb}) == list(HORIZONS)


def test_cli_stations_lists_all(capsys):
    assert cli.main(["stations"]) == 0
    assert "SAE" in capsys.readouterr().out


def test_cli_forecast_unknown_station(capsys):
    assert cli.main(["forecast", "--station", "ZZZ"]) == 2


def test_cli_forecast_replay_from_fixture(capsys):
    assert cli.main(["forecast", "--station", "SAE", "--threshold", "70", "--at", "2023-01-10T12:00"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["level"] in ("ok", "watch", "hold") and set(out["p_exceed"]) == {str(k) for k in HORIZONS}


def test_cli_forecast_replay_missing_hour(capsys):
    assert cli.main(["forecast", "--station", "SAE", "--at", "2023-07-10T12:00"]) == 1


def test_pooled_reliability_weights_by_count():
    from ropeway_wind_stop_forecast.pipeline import pooled_reliability

    rel = pd.DataFrame(
        dict(station=["A", "B"], thr=[70, 70], model=["logreg"] * 2, bin=[3, 3], p_mean=[0.3, 0.4], freq=[0.2, 0.5], n=[100, 300])
    )
    r = pooled_reliability(rel, "logreg", 70)
    assert r.loc[0, "n"] == 400 and r.loc[0, "freq"] == pytest.approx(0.425) and r.loc[0, "p_mean"] == pytest.approx(0.375)


def test_run_evaluation_on_synthetic_station(long_df):
    from ropeway_wind_stop_forecast.pipeline import run_evaluation, summarise

    cfg = dict(stations=("SAE",), thresholds=(70,), kinds=("climatology", "persistence", "logreg"), train=TRAIN, calib=CALIB, test=TEST)
    res = run_evaluation(long_df, cfg, jobs=1)
    assert set(res["metrics"]["model"]) == {"climatology", "persistence", "logreg"}
    cov = pd.DataFrame(dict(valid_gust_hours=[len(long_df)]))
    s = summarise(res, cov)
    assert s["by_threshold"][70]["climatology"]["mean_bss_clim"] == 0.0
