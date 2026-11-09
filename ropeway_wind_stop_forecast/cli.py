"""Command line: demo, forecast, fetch-met, stations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ropeway-wind", description="Wind-hold forecast for cable cars, gondolas and chairlifts.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="evaluate on held-out winters, write results/ and docs/ figures")
    d.add_argument("--out", type=Path, default=Path("."), help="directory that receives results/ and docs/")
    d.add_argument("--fast", action="store_true", help="2 stations, 1 threshold, no gradient boosting (CI smoke test)")
    d.add_argument("--jobs", type=int, default=None, help="worker processes (default: CPU count)")

    f = sub.add_parser("forecast", help="feed record (JSON) for one station")
    f.add_argument("--station", required=True, help="SAE PIL GUE WFJ TIT CMA ATT NAS")
    f.add_argument("--threshold", type=float, default=70.0, help="gust threshold in km/h (see data/lift_types.json)")
    f.add_argument("--model", choices=("logreg", "gboost"), default="logreg")
    f.add_argument("--at", default=None, help="replay a past hour from the fixture, UTC hour end, e.g. 2024-01-15T06:00")
    f.add_argument("--mqtt-host", default=None, help="also publish to this MQTT broker (needs the mqtt extra)")
    f.add_argument("--mqtt-topic", default="ropeway/wind")

    m = sub.add_parser("fetch-met", help="rebuild the MeteoSwiss winter fixture from the open-data portal (large download)")
    m.add_argument("--cache", type=Path, default=Path("cache"))

    sub.add_parser("stations", help="list the bundled stations")

    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "demo":
        from .pipeline import run_demo

        print(json.dumps(run_demo(a.out, fast=a.fast, jobs=a.jobs), indent=2))
    elif a.cmd == "forecast":
        from .data import STATIONS
        from .ops import publish_mqtt, to_json
        from .pipeline import live_forecast

        if a.station.upper() not in STATIONS:
            print(f"unknown station {a.station}; choose from {', '.join(STATIONS)}", file=sys.stderr)
            return 2
        try:
            payload = live_forecast(a.station, a.threshold, a.model, a.at)
        except (ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(to_json(payload))
        if a.mqtt_host:
            publish_mqtt(payload, a.mqtt_host, a.mqtt_topic)
    elif a.cmd == "fetch-met":
        from .fetch import fetch_fixture

        print(f"fixture rows: {len(fetch_fixture(a.cache))}")
    elif a.cmd == "stations":
        from .data import STATIONS

        for k, v in STATIONS.items():
            print(f"{k}  {v['name']:<22} {v['elev']} m  {v['canton']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
