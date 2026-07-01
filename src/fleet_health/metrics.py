"""CloudWatch metrics emission (SPEC.md §6, §11) — Embedded Metric Format.

Emit per-reading health telemetry and per-alert events as EMF JSON. Printed to
stdout (or a Lambda/container log), CloudWatch Logs auto-extracts these into
metrics — no PutMetricData calls, no extra IAM, works from any runtime. AssetType
is the dimension so a per-class breakdown (§8) is free.

Timestamp is passed in (not read from a clock) so emission is pure and testable.
"""

from __future__ import annotations

import json
from typing import Any, Optional

NAMESPACE = "FleetHealthPdM"


def build_emf(
    namespace: str,
    dimensions: dict[str, str],
    metrics: dict[str, tuple[float, str]],
    timestamp_ms: int,
) -> dict[str, Any]:
    """Construct one EMF document. ``metrics`` maps name -> (value, CW unit)."""
    metric_defs = [{"Name": name, "Unit": unit} for name, (_, unit) in metrics.items()]
    body: dict[str, Any] = dict(dimensions)
    body.update({name: value for name, (value, _) in metrics.items()})
    body["_aws"] = {
        "Timestamp": int(timestamp_ms),
        "CloudWatchMetrics": [
            {
                "Namespace": namespace,
                "Dimensions": [list(dimensions.keys())],
                "Metrics": metric_defs,
            }
        ],
    }
    return body


def reading_emf(
    result: Any,
    timestamp_ms: int,
    *,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """EMF for one processed reading (health, anomaly, RUL, whether it alerted)."""
    metrics: dict[str, tuple[float, str]] = {
        "HealthIndex": (round(result.health_index, 4), "None"),
        "AnomalyScore": (round(result.anomaly_score, 4), "None"),
        "Alerted": (1 if result.alert else 0, "Count"),
    }
    rul = result.rul_ticks
    if rul == rul and rul < 1e8:  # finite (not inf, not nan)
        metrics["RulTicks"] = (round(rul, 1), "None")
    if result.alert:
        sev = result.alert.get("severity")
        metrics["CriticalAlert"] = (1 if sev == "critical" else 0, "Count")
    return build_emf(namespace, {"AssetType": result.asset_type}, metrics, timestamp_ms)


def fleet_emf(
    report: Any,
    timestamp_ms: int,
    *,
    cost_saved_usd: Optional[float] = None,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """EMF for an aggregate replay/monitoring report (§8: recall, precision, lead)."""
    metrics: dict[str, tuple[float, str]] = {
        "Failures": (report.failures, "Count"),
        "Detected": (report.detected, "Count"),
        "Missed": (report.missed, "Count"),
        "FalseAlarms": (report.false_alarms, "Count"),
        "Recall": (round(report.recall * 100, 3), "Percent"),
        "Precision": (round(report.precision * 100, 3), "Percent"),
        "FalseAlarmRate": (round(report.false_alarm_rate * 100, 4), "Percent"),
        "MeanLeadTime": (round(report.mean_lead_time, 1), "None"),
    }
    if cost_saved_usd is not None:
        metrics["DowntimeCostAvoidedUsd"] = (round(cost_saved_usd, 2), "None")
    return build_emf(namespace, {"Fleet": "all"}, metrics, timestamp_ms)


def emit(emf_doc: dict[str, Any]) -> None:
    """Write one EMF document to stdout for CloudWatch Logs ingestion."""
    print(json.dumps(emf_doc, separators=(",", ":")))
