"""Edge-triggered alerting (SPEC.md §5, §7, §11).

Combines the two signals into one alert decision per asset, and fires each alert
*once* (on the transition into the alerted state) so a degrading asset produces a
single work order, not one per tick. Two independent trips:

  * RUL trip — a confident downward health trend predicts failure within the
    lead-time horizon. This is the early warning; it fires before the level is
    itself alarming.
  * Persistence trip — the anomaly score stays above threshold for N consecutive
    windows. This is the "clearly bad now" backstop for faults too abrupt to
    trend (the §8a sudden class). Hysteresis (a separate, lower clear threshold)
    stops a single transient spike from latching an alert — the excursion must
    persist, not just occur.

Firing once, and only on a confident/persistent signal, is the direct analog of
the reference design's edge-triggered drift alert: the point is a trustworthy
signal an operator will act on, not a stream of noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .health import Health
from .rul import RUL


@dataclass
class DetectorConfig:
    """Alerting policy. Defaults bias toward catching real failures (a missed
    failure is the costly direction) while hysteresis + confidence gates hold the
    false-alarm rate down."""

    anomaly_threshold: float = 0.35   # enter: anomaly_score at/above this counts
    clear_threshold: float = 0.20     # exit: below this resets the streak (hysteresis)
    persist: int = 3                  # consecutive over-threshold windows to trip
    lead_time_ticks: int = 72         # RUL horizon that counts as "act now"
    min_r2: float = 0.5               # minimum trend confidence for an RUL trip
    critical_health: float = 0.35     # health at/below this escalates to critical


@dataclass
class AssetAlertState:
    consecutive: int = 0
    alerted: bool = False


@dataclass(frozen=True)
class Alert:
    """A fired alert — the unit an operator (or the work-order generator) acts on."""

    asset_id: str
    tick: int
    reason: str            # "rul" | "anomaly_persist"
    severity: str          # "warning" | "critical"
    health_index: float
    anomaly_score: float
    rul_ticks: float
    predicted_failure_tick: Optional[int]
    dominant: str          # which channel drives it (from Health)


def step(
    cfg: DetectorConfig,
    state: Any,
    health: Health,
    rul: RUL,
    *,
    asset_id: str,
    tick: int,
) -> Optional[Alert]:
    """Pure alert transition over a mutable ``state`` (anything with integer
    ``consecutive`` and bool ``alerted`` attributes).

    Factored out so the in-memory Detector (replay) and the DynamoDB-backed
    serving path run byte-identical decision logic — only where the state lives
    differs. Mutates ``state`` in place; returns an Alert only on the edge into
    the alerted state.
    """
    # Persistence with hysteresis: count consecutive clearly-abnormal windows;
    # only a drop below the (lower) clear threshold resets the streak, so the
    # band between the two thresholds holds state and a single spike can't latch.
    if health.anomaly_score >= cfg.anomaly_threshold:
        state.consecutive += 1
    elif health.anomaly_score <= cfg.clear_threshold:
        state.consecutive = 0
        if not (rul.declining and rul.r2 >= cfg.min_r2):
            # Fully recovered and no downward trend — re-arm so a future genuine
            # degradation can alert again.
            state.alerted = False

    rul_trip = (
        rul.declining
        and rul.r2 >= cfg.min_r2
        and rul.ticks_remaining <= cfg.lead_time_ticks
    )
    persist_trip = state.consecutive >= cfg.persist

    if state.alerted or not (rul_trip or persist_trip):
        return None

    state.alerted = True
    reason = "rul" if rul_trip else "anomaly_persist"
    near = rul_trip and rul.ticks_remaining <= cfg.lead_time_ticks / 2
    severity = "critical" if (health.health_index <= cfg.critical_health or near) else "warning"
    predicted = (
        tick + int(rul.ticks_remaining)
        if rul.declining and math.isfinite(rul.ticks_remaining)
        else None
    )
    return Alert(
        asset_id=asset_id,
        tick=tick,
        reason=reason,
        severity=severity,
        health_index=health.health_index,
        anomaly_score=health.anomaly_score,
        rul_ticks=rul.ticks_remaining,
        predicted_failure_tick=predicted,
        dominant=health.dominant,
    )


class Detector:
    """Per-asset alert state machine. One instance monitors a whole fleet by
    holding each asset's AssetAlertState in memory (used by the offline replay)."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.cfg = config or DetectorConfig()
        self.states: dict[str, AssetAlertState] = {}

    def update(
        self, health: Health, rul: RUL, *, asset_id: str, tick: int
    ) -> Optional[Alert]:
        st = self.states.setdefault(asset_id, AssetAlertState())
        return step(self.cfg, st, health, rul, asset_id=asset_id, tick=tick)
