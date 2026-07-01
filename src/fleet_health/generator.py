"""Synthetic run-to-failure fleet generator (SPEC.md §8a).

Lead time and false-alarm rate are properties of the *failure distribution*, not
the detector, so — exactly as the hit ratio is a property of the transaction
distribution in the reference design — this generator is the most important
component of the simulation. It emits per-tick telemetry for a fleet of assets
drawn from four classes:

  * stable    — health stays nominal for the whole horizon; never fails. The
                true negatives. Any alert here is a false alarm. Anchors
                specificity.
  * wearout   — a monotonic degradation (bearing wear, fouling) from an onset
                tick to a failure tick. Vibration climbs steadily, so the health
                trend is predictable and the RUL path should give real lead time.
                The bread-and-butter of predictive maintenance.
  * sudden    — healthy until an abrupt fault in the last few ticks. Little or no
                warning; bounds how much lead time is achievable and is caught,
                if at all, by the persistence backstop, not RUL. The irreducible
                floor (the analog of "genuinely novel" misses).
  * transient — a stable asset with a short vibration spike that recovers. An
                adversarial case: a naive per-sample threshold would fire, but
                windowing + persistence + hysteresis must NOT. Never fails.

Emission is the inverse of the health model: a target true_health h∈[0,1] maps to
vibration nominal + (1−h)·(failure−nominal) plus noise, so a degrading asset
really does produce the rising signal the model keys on. Ground-truth labels
(true_health, gen_class, failure_tick) ride on every reading — the advantage the
simulator has over real traffic, and what lets the replay score itself (§8a).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .asset_types import ASSET_TYPES, AssetType, get_asset_type
from .model import Reading


@dataclass(frozen=True)
class AssetRun:
    """One asset's full telemetry history plus its ground truth."""

    asset_id: str
    asset_type: str
    gen_class: str                    # stable | wearout | sudden | transient
    failure_tick: Optional[int]       # None if the asset never fails in-horizon
    fault_mode: Optional[str]
    readings: list[Reading] = field(default_factory=list)

    @property
    def fails(self) -> bool:
        return self.failure_tick is not None


@dataclass
class SimConfig:
    """Generation parameters. Tune the class mix, then MEASURE lead time and the
    false-alarm rate (§8a) — don't assert them."""

    n_assets: int = 240
    horizon: int = 720          # ticks per asset (e.g. 30 days at hourly sampling)
    seed: int = 7
    # Class mix (need not sum to 1; normalized internally).
    w_stable: float = 0.55
    w_wearout: float = 0.25
    w_sudden: float = 0.10
    w_transient: float = 0.10
    # Sensor noise as a fraction of each channel's nominal→failure band. Small
    # enough that a stable asset stays well under the alarm, large enough to be
    # realistic and to make windowing earn its keep.
    noise_frac: float = 0.03


_ASSET_TYPE_NAMES = list(ASSET_TYPES.keys())
_FAULTS = {"wearout": "bearing_wear", "sudden": "imbalance"}


def _emit(
    asset: AssetType,
    rng: random.Random,
    asset_id: str,
    tick: int,
    health: float,
    gen_class: str,
    failure_tick: Optional[int],
    noise_frac: float,
) -> Reading:
    """Render one reading from a target true_health via the inverse health map."""
    deficit = 1.0 - health  # 0 healthy .. 1 failed
    vib_band = asset.vib_failure - asset.vib_nominal
    temp_band = asset.temp_alarm - asset.temp_nominal

    vib = asset.vib_nominal + deficit * vib_band
    # Temperature lags mechanical wear: only ~70% coupled, so it's the weaker
    # signal the health model already down-weights.
    temp = asset.temp_nominal + 0.7 * deficit * temp_band
    # Current creeps up modestly as the machine works against a developing fault.
    current = asset.current_nominal * (1.0 + 0.15 * deficit)
    rpm = asset.rpm_nominal

    vib += rng.gauss(0.0, noise_frac * vib_band)
    temp += rng.gauss(0.0, noise_frac * temp_band)
    current += rng.gauss(0.0, 0.02 * asset.current_nominal)
    rpm += rng.gauss(0.0, 0.004 * asset.rpm_nominal)

    return Reading(
        asset_id=asset_id,
        tick=tick,
        vibration_rms=round(max(0.0, vib), 4),
        temperature_c=round(temp, 4),
        current_a=round(max(0.0, current), 4),
        rpm=round(rpm, 2),
        asset_type=asset.name,
        true_health=round(health, 4),
        gen_class=gen_class,
        fault_mode=_FAULTS.get(gen_class) if health < 1.0 else None,
        failure_tick=failure_tick,
    )


def _health_curve_stable(horizon: int, rng: random.Random) -> list[float]:
    return [1.0 for _ in range(horizon)]


def _health_curve_transient(horizon: int, rng: random.Random) -> list[float]:
    """Nominal health throughout, with a brief dip that fully recovers — the raw
    vibration spike lives in emission noise below; the underlying health does not
    actually degrade (the asset is fine)."""
    hs = [1.0] * horizon
    spike_at = rng.randint(int(horizon * 0.25), int(horizon * 0.8))
    # Span ≤ 2 by design: a genuine transient is shorter than the detector's
    # persistence requirement, so it can never latch an alert — the adversarial
    # guarantee the replay checks.
    span = rng.randint(1, 2)
    for t in range(spike_at, min(horizon, spike_at + span)):
        hs[t] = rng.uniform(0.45, 0.6)  # transient excursion, then back to 1.0
    return hs


def _health_curve_wearout(horizon: int, rng: random.Random) -> tuple[list[float], int]:
    """Monotonic decline from an onset tick to failure (health 0)."""
    onset = rng.randint(int(horizon * 0.10), int(horizon * 0.45))
    # Degradation ramp long enough for a trend to be estimable — capped so short
    # (test-sized) horizons don't invert the range, but ~120 ticks on a real run.
    min_ramp = max(20, min(120, horizon // 4))
    failure = rng.randint(onset + min_ramp, horizon - 1)
    p = rng.uniform(1.0, 1.6)  # ≥1 => gently accelerating decline
    hs = []
    for t in range(horizon):
        if t < onset:
            hs.append(1.0)
        elif t >= failure:
            hs.append(0.0)
        else:
            frac = (t - onset) / (failure - onset)
            hs.append(max(0.0, 1.0 - frac ** p))
    return hs, failure


def _health_curve_sudden(horizon: int, rng: random.Random) -> tuple[list[float], int]:
    """Healthy until an abrupt drop over the last few ticks — minimal warning."""
    failure = rng.randint(int(horizon * 0.30), horizon - 1)
    drop = rng.randint(2, 6)
    start = failure - drop
    hs = []
    for t in range(horizon):
        if t < start:
            hs.append(1.0)
        elif t >= failure:
            hs.append(0.0)
        else:
            frac = (t - start) / (failure - start)
            hs.append(max(0.0, 1.0 - frac))
    return hs, failure


def generate(cfg: Optional[SimConfig] = None) -> list[AssetRun]:
    """Generate a fleet of run-to-failure telemetry histories with ground truth."""
    cfg = cfg or SimConfig()
    rng = random.Random(cfg.seed)

    classes = ["stable", "wearout", "sudden", "transient"]
    weights = [cfg.w_stable, cfg.w_wearout, cfg.w_sudden, cfg.w_transient]

    runs: list[AssetRun] = []
    for i in range(cfg.n_assets):
        atype_name = _ASSET_TYPE_NAMES[i % len(_ASSET_TYPE_NAMES)]
        asset = get_asset_type(atype_name)
        asset_id = f"{atype_name.upper()}-{i:04d}"
        cls = rng.choices(classes, weights=weights, k=1)[0]

        failure_tick: Optional[int] = None
        if cls == "wearout":
            healths, failure_tick = _health_curve_wearout(cfg.horizon, rng)
        elif cls == "sudden":
            healths, failure_tick = _health_curve_sudden(cfg.horizon, rng)
        elif cls == "transient":
            healths = _health_curve_transient(cfg.horizon, rng)
        else:
            healths = _health_curve_stable(cfg.horizon, rng)

        readings = [
            _emit(asset, rng, asset_id, t, healths[t], cls, failure_tick, cfg.noise_frac)
            for t in range(cfg.horizon)
        ]
        runs.append(
            AssetRun(
                asset_id=asset_id,
                asset_type=atype_name,
                gen_class=cls,
                failure_tick=failure_tick,
                fault_mode=_FAULTS.get(cls),
                readings=readings,
            )
        )
    return runs
