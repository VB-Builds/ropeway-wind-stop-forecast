"""Baselines (climatology, persistence) and learned models (logistic regression, gradient boosting) with calibration.

All models share one interface:
    m = make_model(kind, thr); m.fit(series, train_winters, calib_winters)
    m.predict_wide(series) -> DataFrame (index = issue hour, columns = lead hours 1..6) of P(gust at t+k > thr)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import winter_of
from .features import HORIZONS, add_time_features, build_xy, make_features

KINDS = ("climatology", "persistence", "logreg", "gboost")
MIN_CALIB_EVENTS_ISOTONIC = 100
EPS = 1e-4


def best_csi_threshold(y: np.ndarray, p: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Probability threshold maximising the critical success index TP/(TP+FP+FN); 0.5 if there are no events."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return 0.5
    best, best_t = -1.0, 0.5
    for t in grid if grid is not None else np.arange(0.05, 0.96, 0.05):
        w = p >= t
        tp = int((w & y).sum())
        fp = int((w & ~y).sum())
        fn = int((~w & y).sum())
        csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        if csi > best:
            best, best_t = csi, float(t)
    return round(best_t, 2)


class Calibrator:
    """Isotonic regression when there are enough calibration events, Platt scaling otherwise, identity if no events."""

    def __init__(self) -> None:
        self.kind = "identity"
        self._m = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        y = np.asarray(y).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            self.kind = "identity"
        elif y.sum() >= MIN_CALIB_EVENTS_ISOTONIC:
            self.kind = "isotonic"
            self._m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        else:
            self.kind = "platt"
            self._m = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        if self.kind == "isotonic":
            return self._m.predict(p)
        if self.kind == "platt":
            return self._m.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
        return p


def _winter_mask(index: pd.DatetimeIndex, winters) -> np.ndarray:
    return np.isin(winter_of(index), list(winters))


class Climatology:
    """P(exceed) by calendar month x hour of day of the target hour, shrunk toward the overall rate."""

    kind = "climatology"
    PSEUDO = 20.0

    def __init__(self, thr: float) -> None:
        self.thr = thr
        self.table: dict[tuple[int, int], float] = {}
        self.overall = 0.0
        self.p_star = 0.5

    def fit(self, s: pd.DataFrame, train_winters, calib_winters=()) -> Climatology:
        g = s["gust"]
        ok = g.notna() & _winter_mask(s.index, train_winters)
        y = (g[ok] > self.thr).astype(float)
        self.overall = float(y.mean()) if len(y) else 0.0
        df = pd.DataFrame({"y": y.to_numpy(), "m": y.index.month, "h": y.index.hour})
        agg = df.groupby(["m", "h"])["y"].agg(["sum", "count"])
        for (m, h), r in agg.iterrows():
            self.table[(int(m), int(h))] = (r["sum"] + self.PSEUDO * self.overall) / (r["count"] + self.PSEUDO)
        if len(list(calib_winters)):
            self.p_star = _tune_p_star(self, s, calib_winters)
        return self

    def predict_wide(self, s: pd.DataFrame) -> pd.DataFrame:
        idx = s.index[s["gust"].notna()]
        out = {}
        for k in HORIZONS:
            tgt = idx + pd.Timedelta(hours=k)
            out[k] = [self.table.get((int(m), int(h)), self.overall) for m, h in zip(tgt.month, tgt.hour, strict=True)]
        return pd.DataFrame(out, index=idx)


class Persistence:
    """Calibrated persistence: P(exceed at t+k | current hour exceeds / does not), estimated per lead on the training winters."""

    kind = "persistence"

    def __init__(self, thr: float) -> None:
        self.thr = thr
        self.rate: dict[tuple[int, bool], float] = {}
        self.p_star = 0.5

    def fit(self, s: pd.DataFrame, train_winters, calib_winters=()) -> Persistence:
        now = s["gust"] > self.thr
        for k in HORIZONS:
            fut = s["gust"].shift(-k)
            ok = s["gust"].notna() & fut.notna() & _winter_mask(s.index, train_winters)
            y = (fut > self.thr)[ok]
            n = now[ok]
            for flag in (False, True):
                sel = n == flag
                self.rate[(k, flag)] = float(y[sel].mean()) if sel.sum() >= 5 else float(y.mean() if len(y) else 0.0)
        if len(list(calib_winters)):
            self.p_star = _tune_p_star(self, s, calib_winters)
        return self

    def predict_wide(self, s: pd.DataFrame) -> pd.DataFrame:
        idx = s.index[s["gust"].notna()]
        now = (s.loc[idx, "gust"] > self.thr).to_numpy()
        return pd.DataFrame({k: np.where(now, self.rate[(k, True)], self.rate[(k, False)]) for k in HORIZONS}, index=idx)


class Learned:
    """One classifier per lead hour on `features.FEATURES`, pooled calibrator over all leads, CSI-tuned operating threshold."""

    def __init__(self, thr: float, kind: str) -> None:
        if kind not in ("logreg", "gboost"):
            raise ValueError(f"unknown learned model kind {kind!r}")
        self.thr, self.kind = thr, kind
        self.models: dict[int, object] = {}
        self.cal = Calibrator()
        self.p_star = 0.5

    def _new(self):
        if self.kind == "logreg":
            return make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=500)
            )
        return HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.1, max_depth=4, min_samples_leaf=60, l2_regularization=1.0, random_state=0
        )

    def fit(self, s: pd.DataFrame, train_winters, calib_winters=()) -> Learned:
        for k in HORIZONS:
            X, y = build_xy(s, self.thr, k)
            m = _winter_mask(X.index, train_winters)
            if y[m].nunique() < 2:
                self.models[k] = None
                continue
            self.models[k] = self._new().fit(X[m], y[m])
        calib_winters = list(calib_winters)
        if calib_winters:
            ps, ys = [], []
            for k in HORIZONS:
                X, y = build_xy(s, self.thr, k)
                m = _winter_mask(X.index, calib_winters)
                if m.sum():
                    ps.append(self._raw(k, X[m]))
                    ys.append(y[m].to_numpy())
            if ps:
                p_all, y_all = np.concatenate(ps), np.concatenate(ys)
                self.cal.fit(p_all, y_all)
                self.p_star = best_csi_threshold(y_all, self.cal.transform(p_all))
        return self

    def _raw(self, k: int, X: pd.DataFrame) -> np.ndarray:
        m = self.models.get(k)
        if m is None:
            return np.zeros(len(X))
        return m.predict_proba(X)[:, 1]

    def predict_wide(self, s: pd.DataFrame) -> pd.DataFrame:
        base = make_features(s, self.thr)
        base = base[base["g0"].notna()]
        out = {}
        for k in HORIZONS:
            X = add_time_features(base, k)
            out[k] = self.cal.transform(self._raw(k, X))
        return pd.DataFrame(out, index=base.index)


def _tune_p_star(model, s: pd.DataFrame, calib_winters) -> float:
    P = model.predict_wide(s)
    ps, ys = [], []
    for k in HORIZONS:
        fut = s["gust"].shift(-k).reindex(P.index)
        ok = fut.notna() & _winter_mask(P.index, calib_winters)
        ps.append(P.loc[ok, k].to_numpy())
        ys.append((fut[ok] > model.thr).to_numpy())
    return best_csi_threshold(np.concatenate(ys), np.concatenate(ps))


def make_model(kind: str, thr: float):
    if kind == "climatology":
        return Climatology(thr)
    if kind == "persistence":
        return Persistence(thr)
    return Learned(thr, kind)
