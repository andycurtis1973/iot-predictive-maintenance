"""Detector: edge-triggering, persistence, hysteresis, RUL trip, severity."""

from fleet_health.detector import Detector, DetectorConfig
from fleet_health.health import Health
from fleet_health.rul import RUL, RUL_INF

NO_TREND = RUL(RUL_INF, 0.0, 0.0)


def _h(anomaly, health=0.8):
    return Health(health_index=health, anomaly_score=anomaly, dominant="vibration")


def _feed(det, healths, ruls=None):
    ruls = ruls or [NO_TREND] * len(healths)
    out = []
    for t, (h, r) in enumerate(zip(healths, ruls)):
        out.append(det.update(h, r, asset_id="A", tick=t))
    return out


def test_persistence_fires_once_on_the_edge():
    det = Detector(DetectorConfig(persist=3))
    alerts = _feed(det, [_h(0.5)] * 5)
    fired = [a for a in alerts if a is not None]
    assert len(fired) == 1                 # edge-triggered: fires exactly once
    assert alerts[2] is not None           # on the 3rd consecutive over-threshold
    assert alerts[0] is None and alerts[1] is None
    assert fired[0].reason == "anomaly_persist"


def test_single_transient_spike_does_not_latch():
    det = Detector(DetectorConfig(persist=3, clear_threshold=0.2))
    # spike, spike, then recover below the clear threshold -> streak resets.
    alerts = _feed(det, [_h(0.5), _h(0.5), _h(0.05), _h(0.5), _h(0.05)])
    assert all(a is None for a in alerts)


def test_hysteresis_band_holds_streak():
    det = Detector(DetectorConfig(persist=3, anomaly_threshold=0.35, clear_threshold=0.2))
    # over, then in-band (0.3, between clear and enter) does NOT reset, then over.
    alerts = _feed(det, [_h(0.5), _h(0.3), _h(0.5)])
    # counts: 1 (0.5), still 1 (0.3 in-band, no increment/reset), 2 (0.5) -> not yet 3.
    assert all(a is None for a in alerts)
    a = det.update(_h(0.5), NO_TREND, asset_id="A", tick=99)
    assert a is not None


def test_rul_trip_fires_within_lead_horizon():
    det = Detector(DetectorConfig(lead_time_ticks=72, min_r2=0.5))
    rul = RUL(ticks_remaining=50.0, slope_per_tick=-0.01, r2=0.9)
    a = det.update(_h(0.1, health=0.7), rul, asset_id="A", tick=10)
    assert a is not None
    assert a.reason == "rul"
    assert a.predicted_failure_tick == 60


def test_rul_low_confidence_does_not_trip():
    det = Detector(DetectorConfig(min_r2=0.5))
    rul = RUL(ticks_remaining=10.0, slope_per_tick=-0.01, r2=0.2)  # r2 below gate
    a = det.update(_h(0.1, health=0.7), rul, asset_id="A", tick=0)
    assert a is None


def test_rul_beyond_horizon_does_not_trip():
    det = Detector(DetectorConfig(lead_time_ticks=72))
    rul = RUL(ticks_remaining=500.0, slope_per_tick=-0.001, r2=0.9)
    a = det.update(_h(0.1, health=0.7), rul, asset_id="A", tick=0)
    assert a is None


def test_critical_severity_when_health_low():
    det = Detector(DetectorConfig(persist=1, critical_health=0.35))
    a = det.update(_h(0.5, health=0.2), NO_TREND, asset_id="A", tick=0)
    assert a is not None and a.severity == "critical"


def test_recovered_asset_can_alert_again():
    det = Detector(DetectorConfig(persist=1, clear_threshold=0.2))
    a1 = det.update(_h(0.5, health=0.8), NO_TREND, asset_id="A", tick=0)
    assert a1 is not None
    # fully recover (below clear, no trend) -> re-arm
    det.update(_h(0.05, health=1.0), NO_TREND, asset_id="A", tick=1)
    a2 = det.update(_h(0.5, health=0.8), NO_TREND, asset_id="A", tick=2)
    assert a2 is not None


def test_independent_assets_have_independent_state():
    det = Detector(DetectorConfig(persist=2))
    assert det.update(_h(0.5), NO_TREND, asset_id="A", tick=0) is None
    assert det.update(_h(0.5), NO_TREND, asset_id="B", tick=0) is None
    # A reaches 2 consecutive; B is still at 1.
    assert det.update(_h(0.5), NO_TREND, asset_id="A", tick=1) is not None
    assert det.update(_h(0.5), NO_TREND, asset_id="B", tick=1) is not None
