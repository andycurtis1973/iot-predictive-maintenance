"""Remaining-useful-life: extrapolation math, confidence, flat/improving cases."""

from fleet_health.rul import RUL_INF, estimate


def test_flat_history_is_infinite_rul():
    r = estimate([0.9] * 20)
    assert r.ticks_remaining == RUL_INF
    assert not r.declining


def test_improving_history_is_infinite_rul():
    r = estimate([0.5 + 0.02 * i for i in range(20)])
    assert r.ticks_remaining == RUL_INF
    assert not r.declining


def test_too_few_points_is_infinite():
    r = estimate([0.9, 0.8, 0.7], min_points=6)
    assert r.ticks_remaining == RUL_INF


def test_linear_decline_extrapolates_to_failure():
    # health drops 0.01/tick from 1.0; at tick 19 fitted health ~0.81.
    # RUL to 0 = 0.81 / 0.01 ≈ 81 ticks. Slope -0.01, r2 ~ 1.
    hist = [1.0 - 0.01 * i for i in range(20)]
    r = estimate(hist)
    assert r.declining
    assert abs(r.slope_per_tick + 0.01) < 1e-6
    assert r.r2 > 0.999
    assert abs(r.ticks_remaining - 81.0) < 1.0


def test_custom_failure_level_shrinks_rul():
    hist = [1.0 - 0.01 * i for i in range(20)]
    to_zero = estimate(hist, failure_level=0.0).ticks_remaining
    to_point3 = estimate(hist, failure_level=0.3).ticks_remaining
    assert to_point3 < to_zero


def test_window_limits_history_used():
    # A long flat prefix then a decline: with a short window only the decline is fit.
    hist = [0.9] * 40 + [0.9 - 0.01 * i for i in range(20)]
    r = estimate(hist, window=20)
    assert r.declining
