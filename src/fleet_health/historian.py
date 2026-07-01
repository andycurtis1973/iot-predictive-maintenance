"""Telemetry historian on Amazon Timestream (SPEC.md §6).

The hot path (state_store.py) keeps only a small bounded rolling context per asset
in DynamoDB — that's all the streaming detector needs, at KV point-read latency.
But the raw telemetry itself is worth keeping: dashboards, ad-hoc RUL backtests,
tuning the detector against real history, and long-range trend queries all want
the full time series. That durable history lands here, in Timestream — AWS's
purpose-built time-series store, with an in-memory tier for recent data and a
magnetic tier for cheap long retention.

Two stores, two jobs: DynamoDB answers "what is this asset's state right now?" in
single-digit ms on every reading; Timestream answers "show me this asset's last
month of vibration" for analytics. Writing to the historian is best-effort on the
hot path (serving.py) — a historian hiccup must never drop a live reading or
suppress an alert.

The store is an interface (Protocol) so the serving path is identical whether
backed by Timestream (production) or an in-memory list (tests/local).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from .model import Reading

_CHANNELS = ("vibration_rms", "temperature_c", "current_a", "rpm")


class Historian(Protocol):
    """Write-mostly telemetry history. Timestream and in-memory both satisfy it."""

    def write(self, reading: Reading, *, now_ms: int) -> None: ...
    def recent(self, asset_id: str, limit: int = 50) -> list[dict]: ...


# --- In-memory historian (tests / local) -----------------------------------
class InMemoryHistorian:
    """Append-only per-asset history; recent() returns newest-first."""

    def __init__(self) -> None:
        self._d: dict[str, list[dict]] = {}

    def write(self, reading: Reading, *, now_ms: int) -> None:
        row = {
            "asset_id": reading.asset_id,
            "asset_type": reading.asset_type,
            "tick": reading.tick,
            "time_ms": now_ms,
            "vibration_rms": reading.vibration_rms,
            "temperature_c": reading.temperature_c,
            "current_a": reading.current_a,
            "rpm": reading.rpm,
        }
        self._d.setdefault(reading.asset_id, []).append(row)

    def recent(self, asset_id: str, limit: int = 50) -> list[dict]:
        rows = self._d.get(asset_id, [])
        return list(reversed(rows[-limit:]))

    def count(self, asset_id: str) -> int:
        return len(self._d.get(asset_id, []))


# --- Timestream historian --------------------------------------------------
class TimestreamHistorian:
    """Backed by boto3 ``timestream-write`` + ``timestream-query`` clients.

    Each reading is one multi-measure record (all channels under a single
    ``telemetry`` measure), dimensioned by asset_id + asset_type. The logical
    tick rides along as a measure so backtests can reconstruct the sequence.
    """

    def __init__(self, write_client: Any, query_client: Any, database: str, table: str) -> None:
        self._w = write_client
        self._q = query_client
        self.database = database
        self.table = table

    def write(self, reading: Reading, *, now_ms: int) -> None:
        record = {
            "Dimensions": [
                {"Name": "asset_id", "Value": reading.asset_id},
                {"Name": "asset_type", "Value": reading.asset_type},
            ],
            "MeasureName": "telemetry",
            "MeasureValueType": "MULTI",
            "MeasureValues": [
                {"Name": "vibration_rms", "Value": str(reading.vibration_rms), "Type": "DOUBLE"},
                {"Name": "temperature_c", "Value": str(reading.temperature_c), "Type": "DOUBLE"},
                {"Name": "current_a", "Value": str(reading.current_a), "Type": "DOUBLE"},
                {"Name": "rpm", "Value": str(reading.rpm), "Type": "DOUBLE"},
                {"Name": "tick", "Value": str(reading.tick), "Type": "BIGINT"},
            ],
            "Time": str(int(now_ms)),
            "TimeUnit": "MILLISECONDS",
        }
        self._w.write_records(DatabaseName=self.database, TableName=self.table, Records=[record])

    def recent(self, asset_id: str, limit: int = 50) -> list[dict]:
        # asset_id comes from our own generated ids; still, keep it single-quoted
        # safe by rejecting quotes rather than interpolating them.
        if "'" in asset_id:
            raise ValueError("asset_id must not contain quotes")
        cols = ", ".join(_CHANNELS)
        q = (
            f'SELECT tick, {cols}, time FROM "{self.database}"."{self.table}" '
            f"WHERE asset_id = '{asset_id}' ORDER BY time DESC LIMIT {int(limit)}"
        )
        resp = self._q.query(QueryString=q)
        return _parse_rows(resp)


def _parse_rows(resp: dict) -> list[dict]:
    """Flatten a Timestream query response into a list of {column: value} dicts."""
    cols = [c["Name"] for c in resp.get("ColumnInfo", [])]
    out: list[dict] = []
    for row in resp.get("Rows", []):
        rec: dict[str, Any] = {}
        for name, cell in zip(cols, row["Data"]):
            val = cell.get("ScalarValue")
            if val is None:
                rec[name] = None
            elif name == "tick":
                rec[name] = int(val)
            elif name == "time":
                rec[name] = val
            else:
                try:
                    rec[name] = float(val)
                except ValueError:
                    rec[name] = val
        out.append(rec)
    return out


def create_database_and_table(
    write_client: Any,
    database: str,
    table: str,
    *,
    memory_hours: int = 24,
    magnetic_days: int = 180,
) -> None:
    """Create the Timestream database + table (idempotent).

    Memory-store retention keeps recent data hot for fast queries; the magnetic
    tier keeps a long, cheap history for backtests. Both are configurable per the
    fleet's dashboard/analytics needs.
    """
    from botocore.exceptions import ClientError

    def _ignore_conflict(fn):
        try:
            fn()
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise

    _ignore_conflict(lambda: write_client.create_database(DatabaseName=database))
    _ignore_conflict(lambda: write_client.create_table(
        DatabaseName=database,
        TableName=table,
        RetentionProperties={
            "MemoryStoreRetentionPeriodInHours": memory_hours,
            "MagneticStoreRetentionPeriodInDays": magnetic_days,
        },
    ))
