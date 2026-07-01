"""Serving layer — the request handler + thin runtime adapters (SPEC.md §6, §11).

`MonitorService.process(reading)` is the stateful hot path: load the asset's
rolling state → append the reading → features (slow + fast) → health → RUL →
detector → persist state → optionally draft a work order on an alert → return a
structured result that doubles as the §11 audit record. `Service.handle(request)`
wraps it with validation + metrics over a JSON-shaped dict, so the same handler
runs unchanged on Lambda, Fargate, or EC2 — only the adapter differs (the HTTP
one is stdlib-only).

Error model maps to HTTP-ish status codes:
  400 bad request · 404 unknown asset type · 502 store/inference failure · 200 ok.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .asset_types import get_asset_type
from .detector import DetectorConfig, step as detector_step
from .features import FEAT_V, extract
from .health import HEALTH_V, compose
from .metrics import emit as emit_emf, reading_emf
from .model import Reading
from .rul import estimate
from .state_store import AssetState, bounded, touch

_REQUIRED_FIELDS = ("asset_id", "tick", "vibration_rms", "temperature_c", "current_a", "rpm")


def _reading_from_dict(d: Any) -> Reading:
    if not isinstance(d, dict):
        raise ValueError("'reading' must be a JSON object")
    missing = [f for f in _REQUIRED_FIELDS if f not in d]
    if missing:
        raise ValueError(f"missing reading fields: {missing}")
    try:
        return Reading(
            asset_id=str(d["asset_id"]),
            tick=int(d["tick"]),
            vibration_rms=float(d["vibration_rms"]),
            temperature_c=float(d["temperature_c"]),
            current_a=float(d["current_a"]),
            rpm=float(d["rpm"]),
            asset_type=str(d.get("asset_type", "pump")),
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid reading field: {e}")


def _reading_to_window(r: Reading) -> dict:
    """Only the channels that feed features are persisted (no ground truth)."""
    return {
        "tick": r.tick,
        "vibration_rms": r.vibration_rms,
        "temperature_c": r.temperature_c,
        "current_a": r.current_a,
        "rpm": r.rpm,
    }


def _window_to_readings(window: list[dict], asset_id: str, asset_type: str) -> list[Reading]:
    return [
        Reading(
            asset_id=asset_id,
            tick=int(w["tick"]),
            vibration_rms=float(w["vibration_rms"]),
            temperature_c=float(w["temperature_c"]),
            current_a=float(w["current_a"]),
            rpm=float(w["rpm"]),
            asset_type=asset_type,
        )
        for w in window
    ]


@dataclass
class ServeResult:
    """The §11 audit record for one processed reading."""

    asset_id: str
    asset_type: str
    tick: int
    health_index: float
    anomaly_score: float
    rul_ticks: float
    alert: Optional[dict]        # serialized Alert, or None
    work_order: Optional[dict]   # LLM-drafted, or None
    latency_ms: float
    feat_v: str = FEAT_V
    health_v: str = HEALTH_V

    def to_response(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "tick": self.tick,
            "health_index": round(self.health_index, 4),
            "anomaly_score": round(self.anomaly_score, 4),
            "rul_ticks": None if self.rul_ticks > 1e8 else round(self.rul_ticks, 1),
            "alert": self.alert,
            "work_order": self.work_order,
            "latency_ms": round(self.latency_ms, 3),
            "feat_v": self.feat_v,
            "health_v": self.health_v,
        }


class MonitorService:
    """Stateful per-asset monitor. Store-backed, so it is stateless between calls."""

    def __init__(
        self,
        store: Any,
        *,
        detector_config: Optional[DetectorConfig] = None,
        window: int = 24,
        fast_window: int = 6,
        rul_window: int = 48,
        min_points: int = 6,
        ttl_days: int = 90,
        now_fn=time.time,
        work_order_gen: Any = None,   # optional WorkOrderGenerator (§9)
        budget: Any = None,           # optional BudgetGuard for work orders
        historian: Any = None,        # optional Timestream historian (§6)
    ) -> None:
        self.store = store
        self.cfg = detector_config or DetectorConfig()
        self.window = window
        self.fast_window = fast_window
        self.rul_window = rul_window
        self.min_points = min_points
        self.ttl_seconds = ttl_days * 86400
        self.now_fn = now_fn
        self.work_order_gen = work_order_gen
        self.budget = budget
        self.historian = historian
        self.alerts_fired = 0
        self.history_writes = 0

    def process(self, reading: Reading) -> ServeResult:
        t0 = time.perf_counter()
        asset = get_asset_type(reading.asset_type)  # KeyError -> unknown asset type

        # Durable history is written best-effort (§6): the DynamoDB hot path owns
        # correctness; a historian hiccup must never drop a reading or an alert.
        self._maybe_historian(reading)

        st = self.store.get(reading.asset_id)
        if st is None:
            st = AssetState(asset_id=reading.asset_id, asset_type=reading.asset_type)

        st.window.append(_reading_to_window(reading))
        bounded(st)
        readings = _window_to_readings(st.window, reading.asset_id, reading.asset_type)

        feat_slow = extract(readings, asset, window=self.window)
        feat_fast = extract(readings, asset, window=self.fast_window)
        health = compose(feat_slow, feat_fast, asset)

        st.health_history.append(health.health_index)
        bounded(st)
        rul = estimate(st.health_history, window=self.rul_window, min_points=self.min_points)

        alert = detector_step(self.cfg, st, health, rul, asset_id=reading.asset_id, tick=reading.tick)
        st.last_tick = reading.tick

        now = int(self.now_fn())
        touch(st, now, self.ttl_seconds)
        self.store.put(st)

        alert_dict: Optional[dict] = None
        work_order: Optional[dict] = None
        if alert is not None:
            self.alerts_fired += 1
            alert_dict = {
                "reason": alert.reason,
                "severity": alert.severity,
                "health_index": alert.health_index,
                "anomaly_score": alert.anomaly_score,
                "rul_ticks": None if alert.rul_ticks > 1e8 else round(alert.rul_ticks, 1),
                "predicted_failure_tick": alert.predicted_failure_tick,
                "dominant": alert.dominant,
            }
            work_order = self._maybe_work_order(alert, reading.asset_type)

        return ServeResult(
            asset_id=reading.asset_id,
            asset_type=reading.asset_type,
            tick=reading.tick,
            health_index=health.health_index,
            anomaly_score=health.anomaly_score,
            rul_ticks=rul.ticks_remaining,
            alert=alert_dict,
            work_order=work_order,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _maybe_historian(self, reading: Reading) -> None:
        if self.historian is None:
            return
        try:
            self.historian.write(reading, now_ms=int(self.now_fn() * 1000))
            self.history_writes += 1
        except Exception:
            # Best-effort: never let the historian break the live hot path.
            pass

    def _maybe_work_order(self, alert, asset_type: str) -> Optional[dict]:
        if self.work_order_gen is None:
            return None
        try:
            if self.budget is not None:
                self.budget.check()
            wo = self.work_order_gen.draft(alert, asset_type)
            if self.budget is not None:
                self.budget.record(wo.cost_usd)
            return wo.order
        except Exception as e:  # never let work-order drafting break the hot path
            return {"_error": str(e)}


class Service:
    """Wraps a MonitorService into a request/response handler with metrics."""

    def __init__(
        self,
        monitor: MonitorService,
        *,
        emit_metrics: bool = True,
        emit_fn: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.monitor = monitor
        self.emit_metrics = emit_metrics
        self.emit_fn = emit_fn or emit_emf

    def handle(self, request: Any, *, timestamp_ms: Optional[int] = None) -> tuple[int, dict]:
        ts = int(timestamp_ms if timestamp_ms is not None else time.time() * 1000)
        try:
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            reading = _reading_from_dict(request.get("reading"))
        except ValueError as e:
            return 400, {"error": "bad_request", "message": str(e)}

        try:
            result = self.monitor.process(reading)
        except KeyError as e:
            return 404, {"error": "unknown_asset_type", "message": str(e).strip('"')}
        except Exception as e:  # store / inference failure
            return 502, {"error": "store_or_inference_error", "message": str(e)}

        if self.emit_metrics:
            self.emit_fn(reading_emf(result, ts))
        return 200, result.to_response()


# --- Adapters (thin; pick one per runtime) ---------------------------------
def make_lambda_handler(service: Service) -> Callable[[Any, Any], dict]:
    """AWS Lambda adapter (Function URL / API Gateway proxy / direct invoke)."""

    def lambda_handler(event: Any, context: Any = None) -> dict:
        body = event
        if isinstance(event, dict) and "body" in event:
            body = event["body"]
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                return {"statusCode": 400, "headers": {"content-type": "application/json"},
                        "body": json.dumps({"error": "bad_request", "message": "invalid JSON"})}
        status, resp = service.handle(body)
        return {"statusCode": status, "headers": {"content-type": "application/json"},
                "body": json.dumps(resp)}

    return lambda_handler


def make_http_server(service: Service, host: str = "0.0.0.0", port: int = 8080):
    """Stdlib HTTP adapter for Fargate/EC2/local. POST a reading request to /."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                n = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                status, resp = 400, {"error": "bad_request", "message": "invalid JSON body"}
            else:
                status, resp = service.handle(body)
            payload = json.dumps(resp).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            return

    return ThreadingHTTPServer((host, port), _Handler)
