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

from fleet_health.historian import TimestreamHistorian
from fleet_health.serving import MonitorService, Service, make_lambda_handler
from fleet_health.state_store import DynamoStateStore

_REGION = os.environ.get("AWS_REGION", "us-east-1")
_TABLE = os.environ["STATE_TABLE"]                       # required
_TTL_DAYS = int(os.environ.get("STATE_TTL_DAYS", "90"))  # optional

# Optional Timestream historian (§6): raw telemetry lands here for dashboards /
# backtests. Absent env vars -> no historian (DynamoDB hot path still complete).
_TS_DB = os.environ.get("TIMESTREAM_DB")
_TS_TABLE = os.environ.get("TIMESTREAM_TABLE")

_cfg = Config(retries={"mode": "adaptive", "total_max_attempts": 6})
_store = DynamoStateStore(
    boto3.resource("dynamodb", region_name=_REGION, config=_cfg).Table(_TABLE)
)

_historian = None
if _TS_DB and _TS_TABLE:
    _historian = TimestreamHistorian(
        boto3.client("timestream-write", region_name=_REGION, config=_cfg),
        boto3.client("timestream-query", region_name=_REGION, config=_cfg),
        _TS_DB, _TS_TABLE,
    )

_service = Service(MonitorService(_store, ttl_days=_TTL_DAYS, historian=_historian))
_handler = make_lambda_handler(_service)


def lambda_handler(event, context):
    return _handler(event, context)
