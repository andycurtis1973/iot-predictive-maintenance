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


# ===========================================================================
# Amazon Timestream for InfluxDB (a managed InfluxDB *database* instance).
#
# Timestream for LiveAnalytics (above) is serverless but closed to new
# customers; Timestream for InfluxDB is a provisioned managed InfluxDB instance
# that new accounts CAN use. It speaks the InfluxDB 2.x HTTP API (line-protocol
# writes, Flux queries) with token auth, so this historian talks to it directly
# over HTTPS — no extra SDK, just urllib. Same Historian Protocol as the others.
# ===========================================================================
def _line_protocol(reading: Reading, now_ms: int) -> str:
    """InfluxDB line protocol for one reading (ms precision)."""
    return (
        "telemetry,"
        f"asset_id={reading.asset_id},asset_type={reading.asset_type} "
        f"vibration_rms={reading.vibration_rms},"
        f"temperature_c={reading.temperature_c},"
        f"current_a={reading.current_a},"
        f"rpm={reading.rpm},"
        f"tick={reading.tick}i "
        f"{int(now_ms)}"
    )


def _parse_flux_csv(text: str) -> list[dict]:
    """Flatten InfluxDB annotated-CSV query output into a list of row dicts.

    Skips annotation (#) lines and the result/table bookkeeping columns; numeric
    strings are coerced to float, ``tick`` to int.
    """
    import csv
    import io

    rows: list[dict] = []
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if not lines:
        return rows
    reader = csv.reader(io.StringIO("\n".join(lines)))
    header = next(reader, None)
    if not header:
        return rows
    skip = {"", "result", "table", "_start", "_stop", "_measurement"}
    for raw in reader:
        rec: dict[str, Any] = {}
        for name, val in zip(header, raw):
            if name in skip:
                continue
            if name == "tick":
                rec[name] = int(float(val)) if val else None
            elif name in _CHANNELS:
                rec[name] = float(val) if val else None
            else:
                rec[name] = val
        if rec:
            rows.append(rec)
    return rows


class InfluxHistorian:
    """Telemetry historian on a Timestream for InfluxDB instance (InfluxDB 2.x API).

    Authenticates either with an API ``token`` (Token header) or, when only
    admin credentials are available, with ``username``/``password`` via
    ``/api/v2/signin`` (session cookie, re-issued on expiry). Timestream for
    InfluxDB stores admin username/password in Secrets Manager but not a token,
    so the credential path is the one used live.
    """

    def __init__(
        self,
        endpoint: str,
        token: Optional[str] = None,
        organization: str = "",
        bucket: str = "",
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        port: int = 8086,
        verify_tls: bool = True,
        timeout: float = 15.0,
    ) -> None:
        self.base = f"https://{endpoint}:{port}"
        self.token = token
        self.username = username
        self.password = password
        self.org = organization
        self.bucket = bucket
        self.timeout = timeout
        self._cookie: Optional[str] = None
        if not token and not (username and password):
            raise ValueError("InfluxHistorian needs a token or username+password")
        import ssl

        self._ctx = ssl.create_default_context()
        if not verify_tls:
            self._ctx.check_hostname = False
            self._ctx.verify_mode = ssl.CERT_NONE
        if not token:
            self._signin()

    def _signin(self) -> None:
        import base64
        import urllib.request

        cred = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        req = urllib.request.Request(
            self.base + "/api/v2/signin", data=b"", method="POST",
            headers={"Authorization": "Basic " + cred},
        )
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
            cookies = r.headers.get_all("Set-Cookie") or []
        # keep just the name=value of the session cookie
        self._cookie = "; ".join(c.split(";", 1)[0] for c in cookies) or None

    def _auth_headers(self) -> dict:
        if self.token:
            return {"Authorization": f"Token {self.token}"}
        return {"Cookie": self._cookie or ""}

    def _request(self, path: str, data: bytes, extra_headers: dict) -> str:
        import urllib.error
        import urllib.request

        def _do() -> str:
            headers = dict(self._auth_headers())
            headers.update(extra_headers)
            req = urllib.request.Request(self.base + path, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
                return r.read().decode("utf-8", "replace")

        try:
            return _do()
        except urllib.error.HTTPError as e:
            # session expired -> sign in again once (credential mode only)
            if e.code == 401 and not self.token and self.username:
                self._signin()
                return _do()
            raise

    def write(self, reading: Reading, *, now_ms: int) -> None:
        from urllib.parse import quote

        path = f"/api/v2/write?org={quote(self.org)}&bucket={quote(self.bucket)}&precision=ms"
        self._request(path, _line_protocol(reading, now_ms).encode(),
                      {"Content-Type": "text/plain; charset=utf-8"})

    def recent(self, asset_id: str, limit: int = 50) -> list[dict]:
        if '"' in asset_id:
            raise ValueError("asset_id must not contain quotes")
        from urllib.parse import quote

        flux = (
            f'from(bucket:"{self.bucket}") |> range(start:-30d) '
            f'|> filter(fn:(r) => r._measurement == "telemetry" and r.asset_id == "{asset_id}") '
            f'|> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value") '
            f'|> sort(columns:["_time"], desc:true) |> limit(n:{int(limit)})'
        )
        csv_text = self._request(
            f"/api/v2/query?org={quote(self.org)}", flux.encode(),
            {"Content-Type": "application/vnd.flux", "Accept": "application/csv"},
        )
        return _parse_flux_csv(csv_text)


# --- Provisioning (boto3 ``timestream-influxdb``) --------------------------
def create_influxdb_instance(
    client: Any,
    name: str,
    *,
    username: str,
    password: str,
    organization: str,
    bucket: str,
    subnet_ids: list,
    security_group_ids: list,
    instance_type: str = "db.influx.medium",
    allocated_storage: int = 20,
    storage_type: str = "InfluxIOIncludedT1",
    publicly_accessible: bool = True,
) -> str:
    """Create a Timestream for InfluxDB instance; returns its id. ~15-20 min to
    become AVAILABLE (poll with wait_influxdb_available)."""
    resp = client.create_db_instance(
        name=name,
        username=username,
        password=password,
        organization=organization,
        bucket=bucket,
        dbInstanceType=instance_type,
        allocatedStorage=allocated_storage,
        dbStorageType=storage_type,
        vpcSubnetIds=subnet_ids,
        vpcSecurityGroupIds=security_group_ids,
        publiclyAccessible=publicly_accessible,
    )
    return resp["id"]


def wait_influxdb_available(client, instance_id: str, *, timeout_s: int = 1800, poll_s: int = 30):
    """Poll get_db_instance until AVAILABLE (or DELETED/FAILED). Returns the
    instance description (includes 'endpoint' and 'influxAuthParametersSecretArn')."""
    import time

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        d = client.get_db_instance(identifier=instance_id)
        status = d.get("status")
        if status == "AVAILABLE":
            return d
        if status in ("FAILED", "DELETING", "DELETED"):
            raise RuntimeError(f"InfluxDB instance {instance_id} entered status {status}")
        time.sleep(poll_s)
    raise TimeoutError(f"InfluxDB instance {instance_id} not AVAILABLE within {timeout_s}s")


def influx_auth_from_secret(secrets_client, secret_arn: str) -> dict:
    """Read the InfluxDB admin auth (token/org/bucket/username/password) that
    Timestream for InfluxDB stores in Secrets Manager on instance creation."""
    import json

    raw = secrets_client.get_secret_value(SecretId=secret_arn)["SecretString"]
    return json.loads(raw)
