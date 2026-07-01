"""Health index + anomaly score: bounds, monotonicity, two-channel composition."""

from fleet_health.asset_types import get_asset_type
from fleet_health.features import extract
from fleet_health.health import HEALTH_V, compose, score
from fleet_health.model import Reading

PUMP = get_asset_type("pump")


def _win(vib, temp=55.0, n=24):
    return [Reading("A", t, vib, temp, 42.0, 1770.0, "pump") for t in range(n)]


def _score(vib, temp=55.0):
    return score(extract(_win(vib, temp), PUMP), PUMP)


def test_health_v_is_versioned():
    assert isinstance(HEALTH_V, str) and HEALTH_V


def test_nominal_is_full_health():
    h = _score(PUMP.vib_nominal)
    assert h.health_index == 1.0
    assert h.dominant == "nominal"


def test_failure_level_is_zero_health():
    h = _score(PUMP.vib_failure)
    assert h.health_index == 0.0
    assert h.dominant == "vibration"


def test_health_is_monotonic_in_vibration():
    vibs = [PUMP.vib_nominal + f * (PUMP.vib_failure - PUMP.vib_nominal) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    healths = [_score(v).health_index for v in vibs]
    assert healths == sorted(healths, reverse=True)


def test_health_index_never_out_of_bounds():
    for v in (0.0, PUMP.vib_nominal, PUMP.vib_alarm, PUMP.vib_failure, PUMP.vib_failure * 3):
        h = _score(v)
        assert 0.0 <= h.health_index <= 1.0


def test_temperature_can_dominate():
    # Vibration nominal but temperature at alarm -> temperature drives severity.
    h = _score(PUMP.vib_nominal, temp=PUMP.temp_alarm)
    assert h.dominant == "temperature"
    assert h.health_index < 1.0


def test_compose_takes_level_from_slow_anomaly_from_fast():
    # Nominal for the first 22 samples, elevated for the last 2 (a short
    # excursion). Slow (24) health barely moves; fast (2) anomaly sees it.
    win = [Reading("A", t, PUMP.vib_nominal, 55.0, 42.0, 1770.0, "pump") for t in range(22)]
    win += [Reading("A", 22 + i, PUMP.vib_failure, 55.0, 42.0, 1770.0, "pump") for i in range(2)]
    feat_slow = extract(win, PUMP, window=24)
    feat_fast = extract(win, PUMP, window=2)
    h = compose(feat_slow, feat_fast, PUMP)
    assert h.health_index > 0.85
    assert h.anomaly_score > score(feat_slow, PUMP).anomaly_score
