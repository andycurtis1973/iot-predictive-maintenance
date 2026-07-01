"""AWS Lambda entry point — the same serving layer, wired to DynamoDB.

Cold-start once: build the DynamoDB-backed MonitorService and the serving
adapter; then every invocation is a pure call through it. Work-order generation
(Bedrock) is left off in the demo Lambda to keep it fast and cheap — wire a
WorkOrderGenerator here to enable it.
"""

from __future__ import annotations

import os

import boto3
from botocore.config import Config

from fleet_health.historian import InfluxHistorian, TimestreamHistorian
from fleet_health.serving import MonitorService, Service, make_lambda_handler
from fleet_health.state_store import DynamoStateStore

_REGION = os.environ.get("AWS_REGION", "us-east-1")
_TABLE = os.environ["STATE_TABLE"]                       # required
_TTL_DAYS = int(os.environ.get("STATE_TTL_DAYS", "90"))  # optional

_cfg = Config(retries={"mode": "adaptive", "total_max_attempts": 6})
_store = DynamoStateStore(
    boto3.resource("dynamodb", region_name=_REGION, config=_cfg).Table(_TABLE)
)


def _build_historian():
    """Optional telemetry historian (§6). Prefer Timestream for InfluxDB (open to
    new accounts) if its env vars are set; else Timestream LiveAnalytics; else
    none (the DynamoDB hot path is complete on its own)."""
    endpoint = os.environ.get("INFLUX_ENDPOINT")
    if endpoint and (os.environ.get("INFLUX_TOKEN") or os.environ.get("INFLUX_USERNAME")):
        return InfluxHistorian(
            endpoint,
            token=os.environ.get("INFLUX_TOKEN"),
            organization=os.environ.get("INFLUX_ORG", "fleet-health"),
            bucket=os.environ.get("INFLUX_BUCKET", "telemetry"),
            username=os.environ.get("INFLUX_USERNAME"),
            password=os.environ.get("INFLUX_PASSWORD"),
        )
    ts_db, ts_table = os.environ.get("TIMESTREAM_DB"), os.environ.get("TIMESTREAM_TABLE")
    if ts_db and ts_table:
        return TimestreamHistorian(
            boto3.client("timestream-write", region_name=_REGION, config=_cfg),
            boto3.client("timestream-query", region_name=_REGION, config=_cfg),
            ts_db, ts_table,
        )
    return None


_service = Service(MonitorService(_store, ttl_days=_TTL_DAYS, historian=_build_historian()))
_handler = make_lambda_handler(_service)


def lambda_handler(event, context):
    return _handler(event, context)
