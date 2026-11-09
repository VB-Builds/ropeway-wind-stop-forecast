"""From lead-hour probabilities to an operator message, guest text (de/fr/it/en) and a JSON / MQTT feed.

Levels per lead hour, with p* the CSI-tuned operating probability of the model:
    hold   p >= p*          wind hold likely
    watch  p >= WATCH_FRAC * p*     keep an eye on it
    ok     otherwise
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

LEVELS = ("ok", "watch", "hold")
WATCH_FRAC = 0.5
LANGS = ("en", "de", "fr", "it")

OPERATOR = {
    "ok": "{station}: no wind hold expected in the next 6 h (peak P(gust > {thr:.0f} km/h) = {pmax:.0%}).",
    "watch": "{station}: watch the wind. P(gust > {thr:.0f} km/h) reaches {pmax:.0%} at +{lead_peak} h; hold threshold is {pstar:.0%}.",
    "hold": "{station}: wind hold likely from +{lead_hold} h ({t_hold} UTC). P(gust > {thr:.0f} km/h) = {pmax:.0%} at +{lead_peak} h.",
}
GUEST = {
    "en": {
        "ok": "The lift is expected to run normally.",
        "watch": "Wind is picking up. The lift may stop at short notice.",
        "hold": "Strong wind is expected from about {t_hold}. The lift will probably stop. Please plan your return trip.",
    },
    "de": {
        "ok": "Die Bahn sollte normal fahren.",
        "watch": "Der Wind nimmt zu. Die Bahn kann kurzfristig den Betrieb einstellen.",
        "hold": "Ab etwa {t_hold} wird starker Wind erwartet. Die Bahn wird voraussichtlich stehen bleiben. Bitte planen Sie die Rückfahrt.",
    },
    "fr": {
        "ok": "L'installation devrait fonctionner normalement.",
        "watch": "Le vent se renforce. L'installation peut s'arrêter à court terme.",
        "hold": "Un vent fort est attendu vers {t_hold}. L'installation sera probablement arrêtée. Veuillez prévoir votre retour.",
    },
    "it": {
        "ok": "L'impianto dovrebbe funzionare normalmente.",
        "watch": "Il vento sta aumentando. L'impianto potrebbe fermarsi con breve preavviso.",
        "hold": "Verso le {t_hold} è previsto vento forte. L'impianto probabilmente si fermerà. Pianificate il rientro.",
    },
}


def level_for(p: float, p_star: float) -> str:
    if p >= p_star:
        return "hold"
    if p >= WATCH_FRAC * p_star:
        return "watch"
    return "ok"


def assess(probs: dict[int, float], p_star: float) -> dict:
    """Overall level, first lead hour at hold level, and the peak probability with its lead."""
    if not probs:
        raise ValueError("no lead-hour probabilities")
    if not 0.0 < p_star <= 1.0:
        raise ValueError(f"p_star must be in (0, 1], got {p_star}")
    per = {k: level_for(float(p), p_star) for k, p in sorted(probs.items())}
    worst = max(per.values(), key=LEVELS.index)
    hold = [k for k, lv in per.items() if lv == "hold"]
    lead_peak = max(probs, key=lambda k: (probs[k], -k))
    return dict(
        level=worst,
        per_lead=per,
        lead_hold=hold[0] if hold else None,
        lead_peak=lead_peak,
        p_peak=float(probs[lead_peak]),
    )


def _t_hold(issue_utc: datetime, lead: int | None) -> str:
    return "" if lead is None else (issue_utc + timedelta(hours=lead)).strftime("%H:%M")


def operator_message(station: str, thr: float, issue_utc: datetime, probs: dict[int, float], p_star: float) -> str:
    a = assess(probs, p_star)
    return OPERATOR[a["level"]].format(
        station=station, thr=thr, pmax=a["p_peak"], lead_peak=a["lead_peak"], lead_hold=a["lead_hold"],
        t_hold=_t_hold(issue_utc, a["lead_hold"]), pstar=p_star,
    )  # fmt: skip


def guest_message(level: str, lang: str = "en", t_hold: str = "") -> str:
    if lang not in GUEST:
        raise ValueError(f"unsupported language {lang!r}; choose from {', '.join(LANGS)}")
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    return GUEST[lang][level].format(t_hold=t_hold)


def feed_payload(station: str, thr: float, issue_utc: datetime, probs: dict[int, float], p_star: float, model: str) -> dict:
    """JSON-serialisable feed record. `issue_utc` is the end of the last observed hour (UTC)."""
    a = assess(probs, p_star)
    th = _t_hold(issue_utc, a["lead_hold"])
    return {
        "station": station,
        "issue_utc": issue_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "threshold_kmh": thr,
        "model": model,
        "p_star": round(p_star, 3),
        "level": a["level"],
        "lead_hold_h": a["lead_hold"],
        "p_exceed": {str(k): round(float(p), 4) for k, p in sorted(probs.items())},
        "operator": operator_message(station, thr, issue_utc, probs, p_star),
        "guest": {lg: guest_message(a["level"], lg, th) for lg in LANGS},
    }


def to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def publish_mqtt(payload: dict, host: str, topic: str = "ropeway/wind", port: int = 1883) -> None:
    """Publish the feed record (retained, QoS 1). Needs the optional `mqtt` extra (paho-mqtt)."""
    try:
        import paho.mqtt.publish as publish
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("MQTT output needs paho-mqtt: pip install 'ropeway-wind-stop-forecast[mqtt]'") from e
    publish.single(f"{topic}/{payload['station']}", json.dumps(payload, ensure_ascii=False), qos=1, retain=True, hostname=host, port=port)
