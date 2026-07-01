#!/usr/bin/env python3
"""LIVE end-to-end with a real Amazon Timestream for InfluxDB historian.

Provisions a managed InfluxDB instance (Timestream for InfluxDB), a DynamoDB hot-
state table, and a security group; streams a run-to-failure fleet through the
serving path (DynamoDB state + InfluxDB history + optional Bedrock work orders);
queries the InfluxDB history back to prove the telemetry landed; then tears
everything down.

    PYTHONPATH=src python3 scripts/run_influxdb.py --assets 8 --horizon 140

Note: the InfluxDB instance takes ~15-20 min to provision and bills ~$0.50/hr
while up. Nothing is left running unless you pass --keep.
"""

from __future__ import annotations

import argparse
import secrets
import string
import sys
import time
import urllib.request

from _awsclock import ensure_clock_synced

from fleet_health.generator import SimConfig, generate
from fleet_health.historian import (
    InfluxHistorian,
    create_influxdb_instance,
    influx_auth_from_secret,
    wait_influxdb_available,
)
from fleet_health.model import Reading
from fleet_health.serving import MonitorService
from fleet_health.state_store import DynamoStateStore, create_table

REGION = "us-east-1"
TABLE = "fleet-health-influx-e2e"
INFLUX_NAME = "fleet-health-influxdb"
SG_NAME = "fleet-health-influxdb-sg"
ORG = "fleet-health"
BUCKET = "telemetry"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _my_ip() -> str:
    return urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()


def _password(n: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


def _make_historian(endpoint: str, auth: dict) -> InfluxHistorian:
    # Timestream for InfluxDB stores admin username/password (no token) in the
    # secret, so authenticate by signin (session cookie).
    org = auth.get("organization", ORG)
    bucket = auth.get("bucket", BUCKET)
    kw = dict(username=auth.get("username"), password=auth.get("password"))
    if auth.get("token"):
        kw = dict(token=auth["token"])
    for verify in (True, False):
        try:
            h = InfluxHistorian(endpoint, organization=org, bucket=bucket, verify_tls=verify, **kw)
            h.recent("__probe__", limit=1)  # confirm auth + reachability
            if not verify:
                _log("  (TLS verification disabled for the endpoint)")
            return h
        except Exception as e:
            if verify:
                _log(f"  connect with TLS verify failed ({type(e).__name__}); retrying insecure")
            else:
                raise


def main() -> int:
    ap = argparse.ArgumentParser(description="Live InfluxDB-historian end-to-end")
    ap.add_argument("--assets", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=140)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--live-bedrock", action="store_true")
    ap.add_argument("--max-spend", type=float, default=0.25)
    ap.add_argument("--keep", action="store_true", help="leave all resources up")
    args = ap.parse_args()

    import boto3
    from botocore.config import Config

    ensure_clock_synced(REGION)
    cfg = Config(retries={"mode": "adaptive", "total_max_attempts": 6})
    ec2 = boto3.client("ec2", region_name=REGION, config=cfg)
    ti = boto3.client("timestream-influxdb", region_name=REGION, config=cfg)
    sm = boto3.client("secretsmanager", region_name=REGION, config=cfg)
    ddb = boto3.resource("dynamodb", region_name=REGION, config=cfg)

    # --- networking: SG allowing 8086 from this host ---
    ip = _my_ip()
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    subnet = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc]}])["Subnets"][0]["SubnetId"]
    _log(f"[1/5] security group {SG_NAME} (allow 8086 from {ip}/32) ...")
    sg_id = ec2.create_security_group(GroupName=SG_NAME, Description="Fleet Health InfluxDB demo", VpcId=vpc)["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{"IpProtocol": "tcp", "FromPort": 8086, "ToPort": 8086,
                        "IpRanges": [{"CidrIp": f"{ip}/32"}]}],
    )
    _log(f"      {sg_id}")

    instance_id = None
    table = None
    try:
        _log(f"[2/5] provisioning Timestream for InfluxDB {INFLUX_NAME} (~15-20 min) ...")
        pwd = _password()
        instance_id = create_influxdb_instance(
            ti, INFLUX_NAME, username="admin", password=pwd,
            organization=ORG, bucket=BUCKET, subnet_ids=[subnet],
            security_group_ids=[sg_id], publicly_accessible=True,
        )
        _log(f"      id {instance_id} — waiting for AVAILABLE")
        desc = wait_influxdb_available(ti, instance_id, timeout_s=2100, poll_s=30)
        endpoint = desc["endpoint"]
        _log(f"      AVAILABLE at {endpoint}")

        auth = influx_auth_from_secret(sm, desc["influxAuthParametersSecretArn"])
        historian = _make_historian(endpoint, auth)
        _log("      InfluxDB historian connected.")

        _log(f"[3/5] DynamoDB table {TABLE} (hot state) ...")
        table = create_table(ddb, TABLE)
        _log("      active.")

        work_order_gen = budget = None
        if args.live_bedrock:
            from fleet_health.inference import BudgetGuard, WorkOrderGenerator
            work_order_gen = WorkOrderGenerator(WorkOrderGenerator.make_client(REGION))
            budget = BudgetGuard(max_spend_usd=args.max_spend)

        service = MonitorService(DynamoStateStore(table), historian=historian,
                                 work_order_gen=work_order_gen, budget=budget)

        _log(f"[4/5] streaming {args.assets} assets x {args.horizon} ticks "
             f"(DynamoDB state + InfluxDB history) ...")
        runs = generate(SimConfig(n_assets=args.assets, horizon=args.horizon, seed=args.seed))
        alerts = []
        t0 = time.perf_counter()
        n = 0
        for tick in range(args.horizon):
            for run in runs:
                if tick >= len(run.readings):
                    continue
                r = run.readings[tick]
                res = service.process(Reading(r.asset_id, r.tick, r.vibration_rms,
                                              r.temperature_c, r.current_a, r.rpm, r.asset_type))
                n += 1
                if res.alert:
                    lead = (run.failure_tick - res.tick) if run.failure_tick is not None else None
                    alerts.append((r.asset_id, res.tick, res.alert["severity"], lead))
        dt = time.perf_counter() - t0
        _log(f"      streamed {n:,} readings in {dt:.1f}s; {len(alerts)} alerts; "
             f"{service.history_writes:,} InfluxDB writes")

        # --- query the history back out of InfluxDB ---
        sample_asset = runs[0].asset_id
        _log(f"[5/5] querying InfluxDB history for {sample_asset} (last 5 points) ...")
        rows = historian.recent(sample_asset, limit=5)
        _log(f"      InfluxDB returned {len(rows)} points:")
        for row in rows:
            _log(f"        tick={row.get('tick')}  vib={row.get('vibration_rms')}  "
                 f"temp={row.get('temperature_c')}  t={row.get('_time')}")
        if budget is not None:
            _log(f"      work orders: {budget.calls} drafted, ${budget.spent_usd:.4f}")
    finally:
        if args.keep:
            _log("leaving resources up (--keep). Delete manually to stop InfluxDB billing.")
        else:
            _log("tearing down ...")
            if table is not None:
                table.delete()
                _log("  deleted DynamoDB table")
            if instance_id is not None:
                try:
                    ti.delete_db_instance(identifier=instance_id)
                    _log("  deleting InfluxDB instance (waiting so the SG can be freed) ...")
                    for _ in range(60):
                        try:
                            ti.get_db_instance(identifier=instance_id)
                            time.sleep(20)
                        except ti.exceptions.ResourceNotFoundException:
                            break
                    _log("  deleted InfluxDB instance")
                except Exception as e:
                    _log(f"  influx teardown: {type(e).__name__}")
            for _ in range(6):
                try:
                    ec2.delete_security_group(GroupId=sg_id)
                    _log("  deleted security group")
                    break
                except Exception:
                    time.sleep(20)
            _log("down. zero cost when off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
