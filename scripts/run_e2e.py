#!/usr/bin/env python3
"""LIVE end-to-end on real AWS — DynamoDB per-asset state (+ optional Bedrock).

Self-provisions a KMS-encrypted DynamoDB state table, streams a run-to-failure
fleet through the real serving path (DynamoStateStore; write-back on every
reading), reports the alerts that fire with their lead time, and then tears the
table down. Zero cost when off (on-demand DynamoDB is idle-free). Spend is a few
cents of DynamoDB I/O; add --live-bedrock to also draft real work orders.

    PYTHONPATH=src python3 scripts/run_e2e.py --assets 40 --horizon 400

Requires AWS credentials in us-east-1 with DynamoDB permissions (and Bedrock
access to Claude Haiku 4.5 if --live-bedrock).
"""

from __future__ import annotations

import argparse
import time

from _awsclock import ensure_clock_synced

from fleet_health.generator import SimConfig, generate
from fleet_health.model import Reading
from fleet_health.serving import MonitorService
from fleet_health.state_store import DynamoStateStore, create_table

REGION = "us-east-1"
TABLE = "fleet-health-e2e"
TS_DB = "fleet-health-e2e"
TS_TABLE = "telemetry"


def main() -> int:
    ap = argparse.ArgumentParser(description="Live end-to-end predictive-maintenance run")
    ap.add_argument("--assets", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--live-bedrock", action="store_true", help="draft real work orders on alerts")
    ap.add_argument("--timestream", action="store_true", help="also land raw telemetry in Timestream")
    ap.add_argument("--max-spend", type=float, default=0.25)
    ap.add_argument("--keep", action="store_true", help="do not tear the tables down")
    args = ap.parse_args()

    import boto3
    from botocore.config import Config

    ensure_clock_synced(REGION)
    cfg = Config(retries={"mode": "adaptive", "total_max_attempts": 6})
    ddb = boto3.resource("dynamodb", region_name=REGION, config=cfg)

    print(f"provisioning DynamoDB table {TABLE!r} (KMS-encrypted, on-demand, TTL)...")
    table = create_table(ddb, TABLE)
    print("  active.")

    work_order_gen = budget = None
    if args.live_bedrock:
        from fleet_health.inference import BudgetGuard, WorkOrderGenerator
        work_order_gen = WorkOrderGenerator(WorkOrderGenerator.make_client(REGION))
        budget = BudgetGuard(max_spend_usd=args.max_spend)

    historian = None
    if args.timestream:
        from fleet_health.historian import TimestreamHistorian, create_database_and_table
        tsw = boto3.client("timestream-write", region_name=REGION, config=cfg)
        print(f"provisioning Timestream {TS_DB}.{TS_TABLE} (telemetry historian)...")
        create_database_and_table(tsw, TS_DB, TS_TABLE)
        historian = TimestreamHistorian(
            tsw, boto3.client("timestream-query", region_name=REGION, config=cfg), TS_DB, TS_TABLE
        )
        print("  active.")

    service = MonitorService(
        DynamoStateStore(table), work_order_gen=work_order_gen, budget=budget, historian=historian
    )

    try:
        runs = generate(SimConfig(n_assets=args.assets, horizon=args.horizon, seed=args.seed))
        failure_tick = {r.asset_id: r.failure_tick for r in runs}
        alerts = []
        t0 = time.perf_counter()
        n = 0
        for tick in range(args.horizon):
            for run in runs:
                if tick >= len(run.readings):
                    continue
                r = run.readings[tick]
                clean = Reading(r.asset_id, r.tick, r.vibration_rms, r.temperature_c,
                                r.current_a, r.rpm, r.asset_type)
                res = service.process(clean)
                n += 1
                if res.alert:
                    ft = failure_tick.get(r.asset_id)
                    lead = (ft - res.tick) if ft is not None else None
                    alerts.append((r.asset_id, res.tick, res.alert["severity"], lead))
        dt = time.perf_counter() - t0

        print("-" * 60)
        print(f"streamed {n:,} readings through DynamoDB in {dt:.1f}s")
        print(f"alerts fired: {len(alerts)}")
        for aid, tk, sev, lead in alerts[:15]:
            lead_s = f"{lead} ticks lead" if lead and lead > 0 else ("late" if lead is not None else "n/a")
            print(f"  {aid:<14} tick {tk:>4}  {sev:<8} {lead_s}")
        if len(alerts) > 15:
            print(f"  ... and {len(alerts) - 15} more")
        if budget is not None:
            print(f"work orders: {budget.calls} drafted, ${budget.spent_usd:.4f} spent")
        if historian is not None and alerts:
            aid = alerts[0][0]
            print(f"Timestream history for {aid} (last 3 readings):")
            for row in historian.recent(aid, limit=3):
                print(f"  tick {row.get('tick')}  vib {row.get('vibration_rms')}  temp {row.get('temperature_c')}")
    finally:
        if args.keep:
            print(f"leaving tables in place (--keep). Delete later with the console.")
        else:
            print(f"tearing down table {TABLE!r}...")
            table.delete()
            if args.timestream:
                tsw = boto3.client("timestream-write", region_name=REGION, config=cfg)
                try:
                    tsw.delete_table(DatabaseName=TS_DB, TableName=TS_TABLE)
                    tsw.delete_database(DatabaseName=TS_DB)
                except Exception as e:
                    print(f"  timestream teardown: {type(e).__name__}")
            print("  deleted. zero cost when off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
