#!/usr/bin/env python3
"""Run the monitor as a local HTTP service (stdlib only; no AWS by default).

POST one telemetry reading per request; the service holds per-asset state in
memory (swap InMemoryStateStore for DynamoStateStore to persist).

    PYTHONPATH=src python3 scripts/run_server.py --port 8080

    curl -s localhost:8080 -H 'content-type: application/json' -d '{
      "reading": {"asset_id":"PUMP-1","tick":0,"vibration_rms":2.0,
                  "temperature_c":55,"current_a":42,"rpm":1770,"asset_type":"pump"}}'
"""

from __future__ import annotations

import argparse

from fleet_health.serving import MonitorService, Service, make_http_server
from fleet_health.state_store import InMemoryStateStore


def main() -> int:
    ap = argparse.ArgumentParser(description="Local predictive-maintenance HTTP service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    service = Service(MonitorService(InMemoryStateStore()), emit_metrics=False)
    server = make_http_server(service, host=args.host, port=args.port)
    print(f"listening on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
