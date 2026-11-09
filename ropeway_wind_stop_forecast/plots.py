"""Figures: README figure (1600x900) and phone-sized post figure (960x700), dark palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"
SOURCE = "Source: MeteoSwiss (SwissMetNet, open data)"
MODEL_COLOR = {"climatology": GREY, "persistence": AMBER, "logreg": VIOLET, "gboost": ACCENT}
MODEL_LABEL = {
    "climatology": "climatology", "persistence": "persistence", "logreg": "logistic regression", "gboost": "gradient boosting",
}  # fmt: skip
PRIMARY_THR = 70

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "text.color": TEXT, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.facecolor": BG,
    "figure.facecolor": BG, "savefig.facecolor": BG,
})  # fmt: skip


def _style(ax, fs: int) -> None:
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=fs)
    for s in ax.spines.values():
        s.set_color(GRID)


def _learned(metrics) -> list[str]:
    return [k for k in ("persistence", "logreg", "gboost") if k in set(metrics["model"])]


def _thr(res: dict) -> int:
    t = sorted(res["metrics"]["thr"].unique())
    return PRIMARY_THR if PRIMARY_THR in t else int(t[0])


def panel_bss_station(ax, res: dict, fs: int) -> None:
    m = res["metrics"]
    thr = _thr(res)
    m = m[m["thr"] == thr]
    kinds = [k for k in ("persistence", "logreg", "gboost") if k in set(m["model"])]
    stations = list(res["cfg"]["stations"])
    w = 0.8 / max(len(kinds), 1)
    for i, k in enumerate(kinds):
        v = [m[(m["station"] == s) & (m["model"] == k)]["bss_clim"].iloc[0] for s in stations]
        ax.bar(np.arange(len(stations)) + (i - (len(kinds) - 1) / 2) * w, np.nan_to_num(v, nan=0.0), w * 0.92,
               color=MODEL_COLOR[k], label=MODEL_LABEL[k])  # fmt: skip
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(stations)), stations)
    ax.set_ylabel("Brier skill vs climatology", fontsize=fs)
    ax.set_title(f"Skill by station, gust > {thr} km/h, +1..6 h", fontsize=fs + 1, loc="left", color=TEXT)
    ax.set_ylim(min(0.0, ax.get_ylim()[0]), ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=fs - 1, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper right", ncol=len(kinds))
    _style(ax, fs)


def panel_bss_lead(ax, res: dict, fs: int, lw: float = 2.4) -> None:
    ld = res["lead"]
    thr = _thr(res)
    ld = ld[ld["thr"] == thr]
    for k in _learned(res["metrics"]):
        g = ld[ld["model"] == k].groupby("k")["bss_clim"].mean()
        ax.plot(g.index, g.values, color=MODEL_COLOR[k], lw=lw, marker="o", ms=5, label=MODEL_LABEL[k])
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("lead time (h)", fontsize=fs)
    ax.set_ylabel("Brier skill vs climatology", fontsize=fs)
    ax.set_title(f"Skill by lead time, gust > {thr} km/h (mean of stations)", fontsize=fs + 1, loc="left", color=TEXT)
    ax.set_xticks(range(1, 7))
    ax.legend(fontsize=fs - 1, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="best")
    _style(ax, fs)


def panel_reliability(ax, res: dict, fs: int) -> None:
    from .pipeline import pooled_reliability

    thr = _thr(res)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for k in _learned(res["metrics"]):
        r = pooled_reliability(res["reliability"], k, thr)
        r = r[r["n"] >= 30]
        ax.plot(r["p_mean"], r["freq"], color=MODEL_COLOR[k], lw=2.2, marker="o", ms=5, label=MODEL_LABEL[k])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("forecast probability", fontsize=fs)
    ax.set_ylabel("observed frequency", fontsize=fs)
    ax.set_title("Reliability on held-out winters (bins with n >= 30)", fontsize=fs + 1, loc="left", color=TEXT)
    ax.legend(fontsize=fs - 1, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left")
    _style(ax, fs)


def panel_storm(ax, res: dict, fs: int, compact: bool = False) -> None:
    ex = res.get("example")
    _style(ax, fs)
    if not ex:
        ax.text(0.5, 0.5, "no example storm in this run", ha="center", color=MUTED, fontsize=fs, transform=ax.transAxes)
        return
    tl, thr = ex["timeline"], ex["thr"]
    t0 = ex["episode_start"]
    hrs = (tl.index - t0) / np.timedelta64(1, "h")
    ax.fill_between(hrs, tl["gust"], color=GREY, alpha=0.55, lw=0)
    ax.plot(hrs, tl["gust"], color=TEXT, lw=1.4)
    ax.axhline(ex["thr"], color=RED, lw=1.6, ls="--")
    ax.text(hrs.max() - 1, thr + 3, f"{thr} km/h", color=RED, fontsize=fs - 1, ha="right")
    ax.set_ylabel("observed gust (km/h)", fontsize=fs)
    ax.set_xlabel("hours relative to first gust above threshold", fontsize=fs)
    a2 = ax.twinx()
    a2.plot(hrs, tl["p_lead3"], color=ACCENT, lw=2.6)
    a2.axhline(ex["p_star"], color=ACCENT, lw=1, ls=":")
    a2.set_ylim(0, 1)
    a2.set_ylabel("P(exceed) forecast 3 h earlier", fontsize=fs, color=ACCENT)
    a2.tick_params(labelsize=fs, colors=ACCENT)
    a2.grid(False)
    for s in a2.spines.values():
        s.set_color(GRID)
    name = ex["feed"]["station"]
    ax.set_title(f"Held-out storm at {name}: the +3 h forecast rises before the gusts" if not compact else f"Held-out storm, {name}",
                 fontsize=fs + 1, loc="left", color=TEXT)  # fmt: skip


def _title(fig, title: str, sub: str, fs_title: int, fs_sub: int, y: float = 0.965) -> None:
    fig.text(0.02, y, title, fontsize=fs_title, fontweight="bold", color=TEXT, va="top")
    fig.text(0.02, y - 0.055 * 19 / fs_title * 1.0, sub, fontsize=fs_sub, color=MUTED, va="top")


def main_figure(res: dict, path: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=100, facecolor=BG)
    fs = 12
    panel_bss_station(axs[0, 0], res, fs)
    panel_reliability(axs[0, 1], res, fs)
    panel_bss_lead(axs[1, 0], res, fs)
    panel_storm(axs[1, 1], res, fs)
    c = res["cfg"]
    sub = f"Hourly gust exceedance on held-out winters {min(c['test'])}/{str(min(c['test']) + 1)[-2:]} to " \
          f"{max(c['test'])}/{str(max(c['test']) + 1)[-2:]}, {len(c['stations'])} mountain stations. {SOURCE}. Station gusts, not lift-tower measurements."  # fmt: skip
    fig.text(0.02, 0.975, "Ropeway wind-hold forecast: calibrated 1-6 h gust exceedance probability", fontsize=20,
             fontweight="bold", color=TEXT, va="top")  # fmt: skip
    fig.text(0.02, 0.935, sub, fontsize=11, color=MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.5, w_pad=2.5)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def post_figure(res: dict, path: Path) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    fs = 12
    panel_storm(axs[0], res, fs, compact=True)
    panel_bss_lead(axs[1], res, fs)
    axs[1].set_title("Brier skill vs climatology by lead time", fontsize=fs + 1, loc="left", color=TEXT)
    fig.text(0.02, 0.975, "Warning a wind hold hours ahead", fontsize=19, fontweight="bold", color=TEXT, va="top")
    fig.text(0.02, 0.925, f"{SOURCE}, held-out winters. Station gusts, not lift towers.", fontsize=11, color=MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=2.0)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
