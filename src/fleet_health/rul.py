"""Remaining-useful-life estimation (SPEC.md §5, §8) — the early-warning engine.

Degradation shows up as a downward trend in the health index (health.py). Fit a
line to the recent health trajectory and extrapolate to the failure level; the
horizontal distance to that crossing is the remaining useful life. This is the
whole premise of predictive maintenance made measurable: a monotonic health
decline gives days of warning before the level itself reaches failure.

We report the fit quality (r²) alongside the estimate so the detector can refuse
to act on a noisy, low-confidence trend — an unstable extrapolation is worse than
none. A flat or improving trajectory yields an infinite RUL (nothing to predict).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

RUL_INF = float("inf")


@dataclass(frozen=True)
class RUL:
    """A remaining-useful-life estimate for one asset."""

    ticks_remaining: float   # RUL_INF when not declining / unpredictable
    slope_per_tick: float    # health-index change per tick (negative == degrading)
    r2: float                # fit quality in [0, 1]; confidence in the estimate

    @property
    def declining(self) -> bool:
        return self.slope_per_tick < 0 and math.isfinite(self.ticks_remaining)


def estimate(
    health_history: list[float],
    *,
    failure_level: float = 0.0,
    window: int = 24,
    min_points: int = 6,
) -> RUL:
    """Estimate RUL from a chronological health-index history.

    Fits ``health ≈ a + b·t`` by least squares over the last ``window`` points.
    If the trend is downward, RUL = (fitted_now − failure_level) / −b ticks;
    otherwise RUL is infinite. Fewer than ``min_points`` samples → infinite
    (not enough evidence to extrapolate).
    """
    ys = list(health_history[-window:])
    n = len(ys)
    if n < min_points:
        return RUL(RUL_INF, 0.0, 0.0)

    mean_x = (n - 1) / 2.0
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in range(n))
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(ys))
    if sxx == 0:
        return RUL(RUL_INF, 0.0, 0.0)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    # Coefficient of determination (confidence in the linear trend).
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot == 0:
        r2 = 1.0 if slope == 0 else 0.0
    else:
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in enumerate(ys))
        r2 = max(0.0, 1.0 - ss_res / ss_tot)

    if slope >= 0:
        return RUL(RUL_INF, round(slope, 8), round(r2, 6))

    fitted_now = intercept + slope * (n - 1)
    ticks = (fitted_now - failure_level) / (-slope)
    ticks = max(0.0, ticks)
    return RUL(round(ticks, 3), round(slope, 8), round(r2, 6))
