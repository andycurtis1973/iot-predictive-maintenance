"""Fleet Health — IoT predictive maintenance.

A streaming pipeline that turns rotating-equipment telemetry into early-warning
alerts: windowed features → health index + anomaly score → remaining-useful-life
→ edge-triggered detector. The whole library is pure Python and has no AWS
dependency; boto3 is only needed for the live scripts and the deploy. See SPEC.md.
"""

from .model import Reading
from .asset_types import ASSET_TYPES, AssetType, get_asset_type
from .features import FEAT_V, Features, extract
from .health import HEALTH_V, Health, compose, score
from .rul import RUL, RUL_INF, estimate
from .detector import Alert, Detector, DetectorConfig, AssetAlertState, step
from .generator import AssetRun, SimConfig, generate
from .replay import ClassStats, ReplayReport, process_run, replay
from .state_store import (
    AssetState,
    DynamoStateStore,
    InMemoryStateStore,
    StateStore,
    create_table,
)
from .cost_model import CostInputs, CostModel, cost_model
from .historian import (
    Historian,
    InfluxHistorian,
    InMemoryHistorian,
    TimestreamHistorian,
    create_database_and_table,
    create_influxdb_instance,
    influx_auth_from_secret,
    wait_influxdb_available,
)
from .serving import MonitorService, ServeResult, Service, make_http_server, make_lambda_handler

__all__ = [
    "Reading",
    "ASSET_TYPES",
    "AssetType",
    "get_asset_type",
    "FEAT_V",
    "Features",
    "extract",
    "HEALTH_V",
    "Health",
    "compose",
    "score",
    "RUL",
    "RUL_INF",
    "estimate",
    "Alert",
    "Detector",
    "DetectorConfig",
    "AssetAlertState",
    "step",
    "AssetRun",
    "SimConfig",
    "generate",
    "ClassStats",
    "ReplayReport",
    "process_run",
    "replay",
    "AssetState",
    "DynamoStateStore",
    "InMemoryStateStore",
    "StateStore",
    "create_table",
    "CostInputs",
    "CostModel",
    "cost_model",
    "Historian",
    "InMemoryHistorian",
    "TimestreamHistorian",
    "InfluxHistorian",
    "create_database_and_table",
    "create_influxdb_instance",
    "influx_auth_from_secret",
    "wait_influxdb_available",
    "MonitorService",
    "ServeResult",
    "Service",
    "make_http_server",
    "make_lambda_handler",
]
