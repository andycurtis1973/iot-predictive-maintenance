"""The telemetry reading record (SPEC.md §8a).

We model only the sensor channels that can influence a health/RUL decision for
rotating equipment (pumps, motors, fans, compressors). Plant mechanics we do not
need — flow rates, control setpoints, SCADA tags — are deliberately omitted; they
don't change how the health model behaves and would be simulation theater.

Volatile, per-sample noise lives in the numeric channels and is smoothed by the
windowed feature extractor (features.py). The ground-truth labels are populated
only by the simulation generator (§8a) — real telemetry never carries them; they
are what let the offline replay *measure* lead time and false-alarm rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Reading:
    """A single timestamped telemetry sample from one asset.

    ``tick`` is a monotonic per-asset sample index (e.g. one reading per hour).
    Using an integer tick instead of a wall-clock timestamp keeps the whole
    pipeline deterministic and clock-free for testing; a real deployment maps
    tick → epoch on ingest.
    """

    asset_id: str
    tick: int
    vibration_rms: float   # ISO 10816 broadband velocity, mm/s (the primary wear signal)
    temperature_c: float   # bearing / winding temperature
    current_a: float       # motor current draw, amps
    rpm: float             # shaft speed
    asset_type: str = "pump"

    # Ground-truth labels — populated by the simulation generator (§8a) ONLY.
    # Real telemetry lacks these; they exist so the replay can score itself.
    true_health: Optional[float] = None    # 0.0 (failed) .. 1.0 (healthy)
    gen_class: Optional[str] = None        # "stable" | "wearout" | "sudden" | "transient"
    fault_mode: Optional[str] = None       # "bearing_wear" | "imbalance" | "overheat" | None
    failure_tick: Optional[int] = None     # tick at which this asset fails; None if it never does
