"""Feature extraction: slope correctness, normalization, window bounding."""

from fleet_health.asset_types import get_asset_type
from fleet_health.features import FEAT_V, _slope, extract
from fleet_health.model import Reading

PUMP = get_asset_type("pump")


def _r(tick, vib, temp=55.0, cur=42.0, rpm=1770.0):
    return Reading("A", tick, vib, temp, cur, rpm, "pump")


def test_feat_v_is_versioned():
    assert isinstance(FEAT_V, str) and FEAT_V


def test_slope_of_a_line_is_exact():
    # y = 2t + 1 over t=0..4 => slope 2.
    assert abs(_slope([1, 3, 5, 7, 9]) - 2.0) < 1e-9


def test_slope_of_flat_is_zero():
    assert _slope([5, 5, 5, 5]) == 0.0
    assert _slope([]) == 0.0
    assert _slope([7]) == 0.0


def test_nominal_readings_have_zero_fractions():
    win = [_r(t, PUMP.vib_nominal, PUMP.temp_nominal) for t in range(10)]
    f = extract(win, PUMP)
    assert f.vib_frac < 1e-9
    assert f.temp_frac < 1e-9
    assert abs(f.vib_mean - PUMP.vib_nominal) < 1e-9


def test_failure_level_gives_vib_frac_one():
    win = [_r(t, PUMP.vib_failure) for t in range(10)]
    f = extract(win, PUMP)
    assert abs(f.vib_frac - 1.0) < 1e-9


def test_better_than_nominal_clamps_to_zero_not_negative():
    win = [_r(t, PUMP.vib_nominal - 1.0) for t in range(5)]
    f = extract(win, PUMP)
    assert f.vib_frac == 0.0


def test_window_is_bounded_to_last_n():
    win = [_r(t, 2.0 + t * 0.1) for t in range(100)]
    f = extract(win, PUMP, window=24)
    assert f.window == 24
    assert f.tick == 99  # latest reading supplies the tick


def test_rising_vibration_has_positive_slope():
    win = [_r(t, 2.0 + 0.5 * t) for t in range(10)]
    f = extract(win, PUMP)
    assert f.vib_slope > 0
