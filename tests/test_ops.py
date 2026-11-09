import json
import sys
import types
from datetime import datetime

import pytest

from ropeway_wind_stop_forecast import ops

T0 = datetime(2024, 1, 15, 6, 0)
CALM = {k: 0.02 for k in range(1, 7)}
STORM = {1: 0.1, 2: 0.2, 3: 0.65, 4: 0.8, 5: 0.9, 6: 0.9}


def test_level_thresholds():
    assert ops.level_for(0.5, 0.5) == "hold"
    assert ops.level_for(0.25, 0.5) == "watch"
    assert ops.level_for(0.24, 0.5) == "ok"


def test_assess_calm():
    a = ops.assess(CALM, 0.4)
    assert a["level"] == "ok" and a["lead_hold"] is None


def test_assess_storm_first_hold_lead_and_peak():
    a = ops.assess(STORM, 0.5)
    assert a["level"] == "hold" and a["lead_hold"] == 3
    assert a["lead_peak"] == 5 and a["p_peak"] == 0.9  # ties go to the earlier lead


def test_assess_watch_only():
    assert ops.assess({1: 0.1, 2: 0.3}, 0.5)["level"] == "watch"


def test_assess_validates_input():
    with pytest.raises(ValueError):
        ops.assess({}, 0.5)
    with pytest.raises(ValueError):
        ops.assess(CALM, 0.0)


def test_operator_message_contains_time_and_numbers():
    m = ops.operator_message("Saentis", 70, T0, STORM, 0.5)
    assert "+3 h" in m and "09:00" in m and "90%" in m and "70 km/h" in m


def test_guest_messages_all_languages_nonempty():
    for lg in ops.LANGS:
        for lv in ops.LEVELS:
            assert ops.guest_message(lv, lg, "09:00")


def test_guest_hold_mentions_time():
    for lg in ops.LANGS:
        assert "09:00" in ops.guest_message("hold", lg, "09:00")


def test_guest_rejects_unknown_language_and_level():
    with pytest.raises(ValueError):
        ops.guest_message("hold", "xx")
    with pytest.raises(ValueError):
        ops.guest_message("panic", "en")


def test_feed_payload_roundtrips_json():
    q = json.loads(ops.to_json(ops.feed_payload("Saentis", 70, T0, STORM, 0.5, "logreg")))
    assert q["level"] == "hold" and q["lead_hold_h"] == 3 and q["issue_utc"] == "2024-01-15T06:00:00Z"
    assert set(q["guest"]) == set(ops.LANGS) and q["p_exceed"]["3"] == 0.65


def test_mqtt_publish_uses_topic_and_retain(monkeypatch):
    sent = {}
    pub = types.ModuleType("paho.mqtt.publish")
    pub.single = lambda topic, payload, **kw: sent.update(topic=topic, payload=payload, **kw)
    mqtt = types.ModuleType("paho.mqtt")
    mqtt.publish = pub
    monkeypatch.setitem(sys.modules, "paho", types.ModuleType("paho"))
    monkeypatch.setitem(sys.modules, "paho.mqtt", mqtt)
    monkeypatch.setitem(sys.modules, "paho.mqtt.publish", pub)
    ops.publish_mqtt(ops.feed_payload("SAE", 70, T0, STORM, 0.5, "logreg"), "broker", "t")
    assert sent["topic"] == "t/SAE" and sent["retain"] is True and sent["hostname"] == "broker"
