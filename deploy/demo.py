#!/usr/bin/env python3
"""Provision → run → tear down a complete predictive-maintenance stack on AWS.

    PYTHONPATH=src deploy/demo.py up            # DynamoDB + IAM role + Lambda + Function URL
    PYTHONPATH=src deploy/demo.py call          # a few SigV4-signed sample readings
    PYTHONPATH=src deploy/demo.py drive 8 220   # stream a fleet through the live cache (concurrent)
    PYTHONPATH=src deploy/demo.py capture 12 220 # live run -> animated deploy/web/demo.html
    PYTHONPATH=src deploy/demo.py capture-local 12 220  # same visual, no AWS (in-process)
    PYTHONPATH=src deploy/demo.py down           # delete everything

The Function URL uses AWS_IAM auth (anonymous URLs are commonly org-blocked); the
client SigV4-signs. Zero cost when off: Lambda scales to zero, on-demand DynamoDB
is idle-free.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fleet_health.asset_types import ASSET_TYPES
from fleet_health.bedrock_models import HAIKU_4_5
from fleet_health.generator import SimConfig, generate

ROOT = Path(__file__).resolve().parents[1]
REGION = "us-east-1"
NAME = "fleet-health-demo"
ROLE = "fleet-health-demo-role"
TABLE = NAME
TS_DB = "fleet-health-demo"      # Timestream database (telemetry historian)
TS_TABLE = "telemetry"          # Timestream table


# ----------------------------------------------------------------------------
# Provisioning
# ----------------------------------------------------------------------------
def _build_zip() -> bytes:
    buf = io.BytesIO()
    src = ROOT / "src" / "fleet_health"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob("*.py"):
            z.write(p, f"fleet_health/{p.relative_to(src)}")
        z.write(ROOT / "deploy" / "lambda_app.py", "lambda_app.py")
    return buf.getvalue()


def up() -> int:
    import boto3
    from botocore.exceptions import ClientError
    from _awsclock import ensure_clock_synced
    from fleet_health.historian import create_database_and_table
    from fleet_health.state_store import create_table

    ensure_clock_synced(REGION)
    sess = boto3.Session(region_name=REGION)
    acct = sess.client("sts").get_caller_identity()["Account"]
    ddb, iam, lam = sess.resource("dynamodb"), sess.client("iam"), sess.client("lambda")

    print(f"[1/5] DynamoDB table {TABLE} (hot per-asset state) ...")
    try:
        create_table(ddb, TABLE)
        print("      created (KMS-encrypted, on-demand, TTL).")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print("      already exists.")
        else:
            raise

    print(f"[2/5] Timestream {TS_DB}.{TS_TABLE} (telemetry historian) ...")
    ts_enabled = True
    try:
        create_database_and_table(sess.client("timestream-write"), TS_DB, TS_TABLE)
        print("      created (memory 24h + magnetic 180d retention).")
    except Exception as e:  # region without Timestream, etc. — degrade gracefully
        ts_enabled = False
        print(f"      skipped ({type(e).__name__}); running DynamoDB-only.")

    print(f"[3/5] IAM role {ROLE} ...")
    trust = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
         "Action": "sts:AssumeRole"}]}
    try:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust))
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
    statements = [
        {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": "arn:aws:logs:*:*:*"},
        {"Effect": "Allow",
         "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem"],
         "Resource": f"arn:aws:dynamodb:{REGION}:{acct}:table/{TABLE}"},
        {"Effect": "Allow", "Action": ["bedrock:InvokeModel"], "Resource": "*"}]
    if ts_enabled:
        statements += [
            {"Effect": "Allow", "Action": ["timestream:WriteRecords"],
             "Resource": f"arn:aws:timestream:{REGION}:{acct}:database/{TS_DB}/table/{TS_TABLE}"},
            # Timestream requires DescribeEndpoints on * for any write/query call.
            {"Effect": "Allow", "Action": ["timestream:DescribeEndpoints"], "Resource": "*"}]
    policy = {"Version": "2012-10-17", "Statement": statements}
    iam.put_role_policy(RoleName=ROLE, PolicyName="access", PolicyDocument=json.dumps(policy))
    role_arn = f"arn:aws:iam::{acct}:role/{ROLE}"
    time.sleep(10)  # let the new role propagate before Lambda assumes it

    print(f"[4/5] Lambda {NAME} ...")
    code = _build_zip()
    env = {"STATE_TABLE": TABLE}
    if ts_enabled:
        env.update({"TIMESTREAM_DB": TS_DB, "TIMESTREAM_TABLE": TS_TABLE})
    common = dict(FunctionName=NAME, Runtime="python3.12", Role=role_arn,
                  Handler="lambda_app.lambda_handler", Timeout=30, MemorySize=256,
                  Environment={"Variables": env})
    try:
        lam.create_function(Code={"ZipFile": code}, **common)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            lam.update_function_code(FunctionName=NAME, ZipFile=code)
            lam.get_waiter("function_updated_v2").wait(FunctionName=NAME)
            lam.update_function_configuration(**{k: v for k, v in common.items() if k != "FunctionName"})
        else:
            raise
    lam.get_waiter("function_active_v2").wait(FunctionName=NAME)

    print(f"[5/5] Function URL (AWS_IAM) ...")
    try:
        url = lam.create_function_url_config(FunctionName=NAME, AuthType="AWS_IAM")["FunctionUrl"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            lam.update_function_url_config(FunctionName=NAME, AuthType="AWS_IAM")
            url = lam.get_function_url_config(FunctionName=NAME)["FunctionUrl"]
        else:
            raise
    try:
        lam.add_permission(FunctionName=NAME, StatementId="iam-invoke",
                           Action="lambda:InvokeFunctionUrl", Principal=acct,
                           FunctionUrlAuthType="AWS_IAM")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceConflictException":
            raise

    print("\nup. Function URL:")
    print(f"  {url}")
    print("next:  deploy/demo.py call   |   drive 8 220   |   capture 12 220   |   down")
    return 0


# ----------------------------------------------------------------------------
# SigV4-signed client
# ----------------------------------------------------------------------------
def _signer():
    import boto3
    from _awsclock import ensure_clock_synced

    ensure_clock_synced(REGION)
    sess = boto3.Session(region_name=REGION)
    creds = sess.get_credentials().get_frozen_credentials()
    url = sess.client("lambda").get_function_url_config(FunctionName=NAME)["FunctionUrl"]
    return creds, url


def _post(creds, url, payload: dict) -> dict:
    import urllib.request
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    body = json.dumps(payload).encode()
    aws_req = AWSRequest(method="POST", url=url, data=body,
                         headers={"content-type": "application/json"})
    SigV4Auth(creds, "lambda", REGION).add_auth(aws_req)
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers=dict(aws_req.prepare().headers))
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _payload(r) -> dict:
    return {"reading": {
        "asset_id": r.asset_id, "tick": r.tick, "vibration_rms": r.vibration_rms,
        "temperature_c": r.temperature_c, "current_a": r.current_a, "rpm": r.rpm,
        "asset_type": r.asset_type}}


def call() -> int:
    creds, url = _signer()
    runs = generate(SimConfig(n_assets=1, horizon=400, seed=7))
    run = runs[0]
    print(f"asset {run.asset_id} ({run.asset_type}), class={run.gen_class}")
    for tick in (0, len(run.readings) // 2, len(run.readings) - 3):
        resp = _post(creds, url, _payload(run.readings[tick]))
        print(f"  tick {tick:>4}: health={resp.get('health_index')}  "
              f"rul={resp.get('rul_ticks')}  alert={bool(resp.get('alert'))}")
    return 0


# ----------------------------------------------------------------------------
# Shared streaming — build the record list the web demo replays
# ----------------------------------------------------------------------------
def _asset_meta(runs) -> dict:
    meta = {}
    for run in runs:
        at = ASSET_TYPES[run.asset_type]
        meta[run.asset_id] = {
            "type": run.asset_type,
            "gen_class": run.gen_class,
            "failure_tick": run.failure_tick,
            "cu": at.unplanned_cost_usd,
            "cp": at.planned_cost_usd,
        }
    return meta


def _record(i, tick, asset_id, atype, resp) -> dict:
    return {
        "i": i, "tick": tick, "asset_id": asset_id, "type": atype,
        "health": resp.get("health_index"), "anomaly": resp.get("anomaly_score"),
        "rul": resp.get("rul_ticks"), "alert": resp.get("alert"),
    }


def _stream(process, runs) -> list:
    """Tick-major stream so the animation sweeps the whole fleet each tick.
    ``process`` maps a reading -> response dict (live SigV4 or in-process)."""
    horizon = max(len(r.readings) for r in runs)
    records, i = [], 0
    for tick in range(horizon):
        for run in runs:
            if tick >= len(run.readings):
                continue
            resp = process(run.readings[tick])
            records.append(_record(i, tick, run.asset_id, run.asset_type, resp))
            i += 1
    return records


def _inject(summary: dict, meta: dict, records: list) -> Path:
    data = {"summary": summary, "asset_meta": meta, "records": records}
    tpl = (ROOT / "deploy" / "web" / "template.html").read_text()
    html = re.sub(r'(<script id="rundata"[^>]*>).*?(</script>)',
                  lambda m: m.group(1) + json.dumps(data, separators=(",", ":")) + m.group(2),
                  tpl, flags=re.S)
    out = ROOT / "deploy" / "web" / "demo.html"
    out.write_text(html)
    return out


def _args(default_assets=16, default_horizon=300):
    a = int(sys.argv[2]) if len(sys.argv) > 2 else default_assets
    h = int(sys.argv[3]) if len(sys.argv) > 3 else default_horizon
    return a, h


def capture() -> int:
    assets, horizon = _args()
    creds, url = _signer()
    runs = generate(SimConfig(n_assets=assets, horizon=horizon, seed=11))
    print(f"streaming {assets} assets x {horizon} ticks through the live Lambda ...")
    records = _stream(lambda r: _post(creds, url, _payload(r)), runs)
    summary = {"assets": assets, "horizon": horizon, "region": REGION,
               "model": HAIKU_4_5, "ticks": len(records)}
    out = _inject(summary, _asset_meta(runs), records)
    print(f"wrote {out}  ({len(records)} records). open it in a browser.")
    return 0


def capture_local() -> int:
    assets, horizon = _args()
    from fleet_health.serving import MonitorService, Service
    from fleet_health.state_store import InMemoryStateStore

    svc = Service(MonitorService(InMemoryStateStore()), emit_metrics=False)

    def process(r):
        return svc.handle(_payload(r))[1]

    runs = generate(SimConfig(n_assets=assets, horizon=horizon, seed=11))
    print(f"streaming {assets} assets x {horizon} ticks in-process (no AWS) ...")
    records = _stream(process, runs)
    summary = {"assets": assets, "horizon": horizon, "region": "local",
               "model": HAIKU_4_5, "ticks": len(records)}
    out = _inject(summary, _asset_meta(runs), records)
    print(f"wrote {out}  ({len(records)} records). open it in a browser.")
    return 0


def drive() -> int:
    assets, horizon = _args(8, 220)
    creds, url = _signer()
    runs = generate(SimConfig(n_assets=assets, horizon=horizon, seed=11))
    readings = [rr for run in runs for rr in run.readings]
    print(f"driving {len(readings):,} readings ({assets} assets) concurrently ...")
    t0 = time.perf_counter()
    alerts = 0
    lat = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(_post, creds, url, _payload(r)) for r in readings]
        for f in as_completed(futs):
            resp = f.result()
            if resp.get("alert"):
                alerts += 1
            if resp.get("latency_ms") is not None:
                lat.append(resp["latency_ms"])
    dt = time.perf_counter() - t0
    lat.sort()
    p50 = lat[len(lat) // 2] if lat else 0
    print(f"  {len(readings):,} readings in {dt:.1f}s, {alerts} alerts, server p50 {p50:.2f} ms")
    return 0


# ----------------------------------------------------------------------------
# Teardown
# ----------------------------------------------------------------------------
def down() -> int:
    import boto3
    from botocore.exceptions import ClientError
    from _awsclock import ensure_clock_synced

    ensure_clock_synced(REGION)
    sess = boto3.Session(region_name=REGION)
    lam, iam, ddb = sess.client("lambda"), sess.client("iam"), sess.client("dynamodb")
    ts = sess.client("timestream-write")

    def _try(fn, what):
        try:
            fn()
            print(f"  deleted {what}")
        except ClientError as e:
            print(f"  {what}: {e.response['Error']['Code']}")
        except Exception as e:
            print(f"  {what}: {type(e).__name__}")

    _try(lambda: lam.delete_function_url_config(FunctionName=NAME), "function url")
    _try(lambda: lam.delete_function(FunctionName=NAME), "lambda")
    _try(lambda: iam.delete_role_policy(RoleName=ROLE, PolicyName="access"), "role policy")
    _try(lambda: iam.delete_role(RoleName=ROLE), "role")
    _try(lambda: ddb.delete_table(TableName=TABLE), "table")
    _try(lambda: ts.delete_table(DatabaseName=TS_DB, TableName=TS_TABLE), "timestream table")
    _try(lambda: ts.delete_database(DatabaseName=TS_DB), "timestream database")
    print("down. zero cost when off.")
    return 0


_CMDS = {"up": up, "call": call, "drive": drive, "capture": capture,
         "capture-local": capture_local, "down": down}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        print(__doc__)
        return 2
    return _CMDS[sys.argv[1]]()


if __name__ == "__main__":
    raise SystemExit(main())
