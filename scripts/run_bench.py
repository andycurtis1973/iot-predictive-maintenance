#!/usr/bin/env python3
"""Hot-path latency bench — per-reading processing cost through the serving path.

Dry by default (no AWS): streams a fleet through the in-memory serving path and
reports per-reading latency percentiles + alerts fired. With --live, also drafts
a Bedrock work order for each alert (budget-capped) to measure that cost.

    PYTHONPATH=src python3 scripts/run_bench.py --assets 120 --horizon 480
    PYTHONPATH=src python3 scripts/run_bench.py --assets 60 --horizon 480 --live --max-spend 0.25
"""

from __future__ import annotations

import argparse

from fleet_health.bench import run_bench
from fleet_health.generator import SimConfig, generate
from fleet_health.serving import MonitorService
from fleet_health.state_store import InMemoryStateStore


def main() -> int:
    ap = argparse.ArgumentParser(description="Predictive-maintenance hot-path bench")
    ap.add_argument("--assets", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=480)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--live", action="store_true", help="draft Bedrock work orders on alerts")
    ap.add_argument("--max-spend", type=float, default=0.25)
    args = ap.parse_args()

    runs = generate(SimConfig(n_assets=args.assets, horizon=args.horizon, seed=args.seed))

    work_order_gen = budget = None
    if args.live:
        from fleet_health.inference import BudgetGuard, WorkOrderGenerator
        from _awsclock import ensure_clock_synced

        ensure_clock_synced()
        work_order_gen = WorkOrderGenerator(WorkOrderGenerator.make_client())
        budget = BudgetGuard(max_spend_usd=args.max_spend)

    service = MonitorService(InMemoryStateStore(), work_order_gen=work_order_gen, budget=budget)
    rep = run_bench(runs, service=service)

    print(f"processed {rep.readings:,} readings across {rep.assets} assets")
    print(f"alerts fired : {rep.alerts}")
    print(f"latency (ms) : mean {rep.mean_ms:.3f}   p50 {rep.p50_ms:.3f}   p99 {rep.p99_ms:.3f}")
    if args.live and budget is not None:
        print(f"work orders  : {budget.calls} drafted, ${budget.spent_usd:.4f} spent "
              f"(cap ${args.max_spend:.2f})")
    else:
        print("dry run — no Bedrock calls (add --live to draft work orders on alerts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
