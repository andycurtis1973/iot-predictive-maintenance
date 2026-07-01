"""Offline replay / early-warning measurement (SPEC.md §8, §8a).

Stream each asset's telemetry through the full pipeline — window → features →
health → RUL → detector — with NO AWS, and measure against the ground truth the
simulator knows and real traffic doesn't:

  * lead time    = failure_tick − first_alert_tick for detected failures. The
                   headline metric: how many ticks of warning the model buys.
  * recall       = detected failures / total failures. Did we catch them?
  * precision    = detected / (detected + false positives). Were our alerts real?
  * false-alarm  = alerts on assets that never fail / non-failing assets. The
    rate           operational nuisance rate; the adversarial transient class
                   must not inflate it.

A missed catastrophic failure is the dangerous direction and a false alarm is the
merely-annoying one, so the detector is tuned to favor recall — but both are
reported here so the trade-off is measured, not assumed. This is the predictive-
maintenance analog of proving hit ratio AND collision-freedom together.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .asset_types import get_asset_type
from .detector import Detector, DetectorConfig
from .features import extract
from .generator import AssetRun
from .health import compose
from .rul import estimate

# Fast anomaly window — short enough to react to an abrupt fault the slow health
# window would smooth away, long enough that a 1–2 tick transient cannot persist.
FAST_WINDOW = 6


def process_run(
    run: AssetRun,
    detector: Detector,
    *,
    window: int = 24,
    fast_window: int = FAST_WINDOW,
    rul_window: int = 48,
    min_points: int = 6,
) -> Optional[int]:
    """Replay one asset tick-by-tick; return the tick of its FIRST alert (or None).

    Mirrors exactly what the stateful serving path does per reading, but holds
    the rolling window in memory instead of DynamoDB. The detector carries the
    per-asset alert state (keyed by asset_id), so one Detector serves the fleet.
    Both the slow (health/RUL) and fast (anomaly) channels are cut from the same
    rolling window, so the second view costs no extra storage.
    """
    asset = get_asset_type(run.asset_type)
    win: list = []
    health_hist: list[float] = []
    first_alert: Optional[int] = None

    for r in run.readings:
        win.append(r)
        if len(win) > window:
            win = win[-window:]
        feat_slow = extract(win, asset, window=window)
        feat_fast = extract(win, asset, window=fast_window)
        h = compose(feat_slow, feat_fast, asset)
        health_hist.append(h.health_index)
        rul = estimate(health_hist, window=rul_window, min_points=min_points)
        alert = detector.update(h, rul, asset_id=r.asset_id, tick=r.tick)
        if alert is not None and first_alert is None:
            first_alert = alert.tick
    return first_alert


@dataclass
class ClassStats:
    assets: int = 0
    failures: int = 0
    detected: int = 0          # alerted at or before failure (in time to act)
    late: int = 0              # alerted, but only after failure (real, not useful)
    false_alarms: int = 0      # alerted on an asset that never fails
    lead_times: list[int] = field(default_factory=list)


@dataclass
class ReplayReport:
    """Measured early-warning outcomes.

    The taxonomy separates the two ways an alert can relate to reality: an alert
    on a genuinely failing asset is a TRUE alert (early if before failure, late
    if after); an alert on an asset that never fails is a FALSE ALARM. Late
    detections are real (the asset was failing) so they don't hurt precision, but
    they buy no lead time, so they don't count toward recall — this is exactly
    how the abrupt-fault floor shows up honestly in the numbers.
    """

    total_assets: int
    horizon: int
    failures: int
    detected: int                 # early detections (lead time > 0)
    late: int                     # detected only after failure
    non_failing: int
    false_alarms: int             # alerts on never-failing assets
    lead_times: list[int] = field(default_factory=list)
    by_class: dict[str, ClassStats] = field(default_factory=dict)

    @property
    def missed(self) -> int:
        return self.failures - self.detected - self.late

    @property
    def recall(self) -> float:
        """Early-warning recall: failures caught IN TIME to act on."""
        return self.detected / self.failures if self.failures else 0.0

    @property
    def precision(self) -> float:
        """Of all alerts raised, the fraction about a genuinely failing asset."""
        alerts = self.detected + self.late + self.false_alarms
        return (self.detected + self.late) / alerts if alerts else 0.0

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarms / self.non_failing if self.non_failing else 0.0

    def _pct(self, p: float) -> float:
        if not self.lead_times:
            return 0.0
        xs = sorted(self.lead_times)
        idx = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
        return float(xs[idx])

    @property
    def mean_lead_time(self) -> float:
        return sum(self.lead_times) / len(self.lead_times) if self.lead_times else 0.0

    @property
    def median_lead_time(self) -> float:
        return self._pct(0.5)

    @property
    def p10_lead_time(self) -> float:
        return self._pct(0.10)


def replay(
    runs: list[AssetRun],
    *,
    config: Optional[DetectorConfig] = None,
    window: int = 24,
    rul_window: int = 48,
    min_points: int = 6,
) -> ReplayReport:
    """Replay a whole fleet and produce a measured early-warning report."""
    detector = Detector(config)
    by_class: dict[str, ClassStats] = defaultdict(ClassStats)

    detected = late = false_alarms = 0
    failures = non_failing = 0
    lead_times: list[int] = []
    horizon = max((len(r.readings) for r in runs), default=0)

    for run in runs:
        cs = by_class[run.gen_class]
        cs.assets += 1
        alert_tick = process_run(
            run, detector, window=window, rul_window=rul_window, min_points=min_points
        )

        if run.fails:
            failures += 1
            cs.failures += 1
            if alert_tick is not None and alert_tick <= run.failure_tick:
                detected += 1
                cs.detected += 1
                lead = run.failure_tick - alert_tick
                lead_times.append(lead)
                cs.lead_times.append(lead)
            elif alert_tick is not None:
                # Fired only after the asset had already failed: a real alert
                # (the asset was genuinely failing) but too late to prevent it.
                late += 1
                cs.late += 1
        else:
            non_failing += 1
            if alert_tick is not None:
                false_alarms += 1
                cs.false_alarms += 1

    return ReplayReport(
        total_assets=len(runs),
        horizon=horizon,
        failures=failures,
        detected=detected,
        late=late,
        non_failing=non_failing,
        false_alarms=false_alarms,
        lead_times=lead_times,
        by_class=dict(by_class),
    )
