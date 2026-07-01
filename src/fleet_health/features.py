"""Windowed feature extraction (SPEC.md §5) — versioned FEAT_V.

The engineering risk in predictive maintenance lives here and in the health
model (health.py), the way key design is where the risk lives for a result
cache. Raw telemetry is noisy per-sample; a decision made off a single reading
either chatters (false alarms) or misses a real trend. So we reduce a rolling
window of readings to a small, stable feature vector: smoothed levels plus the
*trend* (least-squares slope), which is the leading indicator — vibration that
is rising is more informative than vibration that is merely high.

FEAT_V versions the feature definition. Any change to what a feature means (new
channel, different window, changed slope units) changes model behavior and must
bump FEAT_V so downstream state/thresholds re-segment instead of silently
comparing old features to new ones (§7, mirrors norm_v in the reference design).
"""

from __future__ import annotations

from dataclasses import dataclass

from .asset_types import AssetType
from .model import Reading

FEAT_V = "feat_v1"


@dataclass(frozen=True)
class Features:
    """A stable, low-dimensional summary of an asset's recent behavior."""

    asset_id: str
    asset_type: str
    tick: int
    window: int  # number of readings actually used (<= configured window)

    vib_mean: float
    vib_slope: float   # mm/s per tick — the leading degradation indicator
    temp_mean: float
    temp_slope: float  # °C per tick
    current_mean: float

    # Deviations normalized to the asset's own envelope (dimensionless), so the
    # health model is asset-type-agnostic. 0.0 == nominal, 1.0 == failure level.
    vib_frac: float
    temp_frac: float


def _slope(values: list[float]) -> float:
    """Least-squares slope of ``values`` against tick index 0..n-1.

    Pure and deterministic; returns 0.0 for fewer than two points or a
    degenerate (zero-variance) x-axis. Units are value-per-tick.
    """
    n = len(values)
    if n < 2:
        return 0.0
    xs = range(n)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (v - mean_y) for x, v in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den else 0.0


def _frac(value: float, nominal: float, top: float) -> float:
    """Fraction of the way from ``nominal`` to ``top`` (failure/alarm level).

    Clamped at the bottom to 0 (better-than-nominal isn't negative health) but
    NOT at the top — a value past the failure level should read > 1.0 so the
    health index can saturate to 0 and the anomaly score keeps rising.
    """
    span = top - nominal
    if span <= 0:
        return 0.0
    return max(0.0, (value - nominal) / span)


def extract(readings: list[Reading], asset: AssetType, *, window: int = 24) -> Features:
    """Reduce the most recent ``window`` readings to a Features vector.

    ``readings`` must be in ascending tick order and non-empty. The latest
    reading supplies asset_id / tick; the window supplies smoothed levels and
    slopes. Callers hold the rolling window (state_store.py) so this stays a
    pure function of its inputs.
    """
    if not readings:
        raise ValueError("extract() requires at least one reading")
    w = readings[-window:]
    vib = [r.vibration_rms for r in w]
    temp = [r.temperature_c for r in w]
    cur = [r.current_a for r in w]
    latest = w[-1]

    vib_mean = sum(vib) / len(vib)
    temp_mean = sum(temp) / len(temp)

    return Features(
        asset_id=latest.asset_id,
        asset_type=asset.name,
        tick=latest.tick,
        window=len(w),
        vib_mean=vib_mean,
        vib_slope=_slope(vib),
        temp_mean=temp_mean,
        temp_slope=_slope(temp),
        current_mean=sum(cur) / len(cur),
        vib_frac=_frac(vib_mean, asset.vib_nominal, asset.vib_failure),
        temp_frac=_frac(temp_mean, asset.temp_nominal, asset.temp_alarm),
    )
