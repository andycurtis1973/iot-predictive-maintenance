#!/usr/bin/env python3
"""§10 cost projection — downtime avoided per year from the MEASURED recall.

No AWS spend. Runs the offline replay, scales its failure count + false alarms to
an annual rate, and applies the per-asset-type downtime economics to project the
baseline (run-to-failure) vs predictive-maintenance annual cost.

    PYTHONPATH=src python3 scripts/run_cost.py --assets 240 --horizon 720
"""

from __future__ import annotations

import argparse

from fleet_health.asset_types import ASSET_TYPES
from fleet_health.cost_model import cost_model
from fleet_health.generator import SimConfig, generate
from fleet_health.replay import replay

HOURS_PER_YEAR = 8760.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Downtime-avoided cost projection")
    ap.add_argument("--assets", type=int, default=240)
    ap.add_argument("--horizon", type=int, default=720, help="ticks (hours) per asset")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fa-cost", type=float, default=800.0, help="wasted inspection $/false alarm")
    args = ap.parse_args()

    runs = generate(SimConfig(n_assets=args.assets, horizon=args.horizon, seed=args.seed))
    rep = replay(runs)

    # Scale the simulated window (horizon hours) up to a full year.
    scale = HOURS_PER_YEAR / args.horizon
    failures_per_year = rep.failures * scale
    false_alarms_per_year = rep.false_alarms * scale

    # Fleet-average downtime economics (the generator cycles asset types evenly).
    cu = sum(a.unplanned_cost_usd for a in ASSET_TYPES.values()) / len(ASSET_TYPES)
    cp = sum(a.planned_cost_usd for a in ASSET_TYPES.values()) / len(ASSET_TYPES)

    m = cost_model(
        failures_per_year=failures_per_year,
        recall=rep.recall,
        unplanned_cost=cu,
        planned_cost=cp,
        false_alarms_per_year=false_alarms_per_year,
        false_alarm_cost=args.fa_cost,
    )

    print(f"measured early-warning recall : {rep.recall*100:.2f}%   (false-alarm rate {rep.false_alarm_rate*100:.3f}%)")
    print(f"projected failures/year       : {failures_per_year:,.0f}   across {args.assets} assets")
    print(f"avg unplanned failure cost Cu : ${cu:,.0f}")
    print(f"avg planned intervention  Cp  : ${cp:,.0f}")
    print("-" * 56)
    print(f"baseline (run-to-failure)/yr  : ${m.baseline_per_year:,.0f}")
    print(f"with predictive maintenance/yr: ${m.pdm_per_year:,.0f}")
    print(f"savings/yr                    : ${m.savings_per_year:,.0f}   ({m.savings_ratio*100:.1f}%)")
    print("-" * 56)
    print("Savings scale with the unplanned/planned cost gap and the measured recall;")
    print("the abrupt-fault floor caps recall, so bigger Cu (heavier assets) pays most.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
