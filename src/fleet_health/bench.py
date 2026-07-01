"""Hot-path bench (SPEC.md §8a "end-to-end latency").

Drive a whole fleet's telemetry through the stateful serving path with an
in-memory store (a DynamoDB stand-in) and measure per-reading processing latency
— load state → features → health → RUL → detector → write state — split out from
any network/Bedrock time. The narrative mirrors the reference bench: the per-
reading cost is tiny and flat, so the pipeline scales to a large fleet cheaply;
the only expensive calls are the handful of Bedrock work-order drafts on alerts.

Interleaves assets by tick so it looks like a live fleet stream rather than one
asset processed to completion at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .generator import AssetRun
from .model import Reading
from .serving import MonitorService
from .state_store import InMemoryStateStore


@dataclass
class BenchReport:
    assets: int
    readings: int
    alerts: int
    mean_ms: float
    p50_ms: float
    p99_ms: float

    def summary(self) -> dict:
        return {
            "assets": self.assets,
            "readings": self.readings,
            "alerts": self.alerts,
            "mean_ms": round(self.mean_ms, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
        }


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


def run_bench(runs: list[AssetRun], service: Optional[MonitorService] = None) -> BenchReport:
    """Process every reading through the serving path and report latency."""
    service = service or MonitorService(InMemoryStateStore())
    horizon = max((len(r.readings) for r in runs), default=0)
    latencies: list[float] = []
    alerts = 0
    n = 0

    for tick in range(horizon):
        for run in runs:
            if tick >= len(run.readings):
                continue
            r = run.readings[tick]
            # Strip ground-truth labels — production readings don't carry them.
            clean = Reading(
                asset_id=r.asset_id, tick=r.tick, vibration_rms=r.vibration_rms,
                temperature_c=r.temperature_c, current_a=r.current_a, rpm=r.rpm,
                asset_type=r.asset_type,
            )
            res = service.process(clean)
            latencies.append(res.latency_ms)
            n += 1
            if res.alert:
                alerts += 1

    mean = sum(latencies) / len(latencies) if latencies else 0.0
    return BenchReport(
        assets=len(runs),
        readings=n,
        alerts=alerts,
        mean_ms=mean,
        p50_ms=_pct(latencies, 0.50),
        p99_ms=_pct(latencies, 0.99),
    )
