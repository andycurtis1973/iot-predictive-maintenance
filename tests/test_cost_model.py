"""Cost model: downtime-avoided arithmetic (SPEC.md §10)."""

from fleet_health.cost_model import cost_model


def test_worked_example():
    # 100 failures/yr, recall 0.75, unplanned $48k, planned $3.5k, no false alarms.
    m = cost_model(failures_per_year=100, recall=0.75, unplanned_cost=48_000, planned_cost=3_500)
    assert m.baseline_per_year == 100 * 48_000
    # caught 75 -> planned; not caught 25 -> unplanned.
    assert m.pdm_per_year == 75 * 3_500 + 25 * 48_000
    # savings = 75 * (48000 - 3500)
    assert m.savings_per_year == 75 * (48_000 - 3_500)


def test_false_alarms_reduce_savings():
    base = cost_model(100, 0.75, 48_000, 3_500)
    with_fa = cost_model(100, 0.75, 48_000, 3_500,
                         false_alarms_per_year=50, false_alarm_cost=800)
    assert with_fa.savings_per_year == base.savings_per_year - 50 * 800


def test_zero_recall_saves_nothing():
    m = cost_model(100, 0.0, 48_000, 3_500)
    assert m.savings_per_year == 0.0
    assert m.savings_ratio == 0.0


def test_savings_ratio_bounded():
    m = cost_model(100, 1.0, 48_000, 3_500)
    # perfect recall, no false alarms: savings = F*(Cu-Cp), ratio = (Cu-Cp)/Cu.
    assert abs(m.savings_ratio - (48_000 - 3_500) / 48_000) < 1e-9


def test_summary_keys_present():
    s = cost_model(100, 0.75, 48_000, 3_500).summary()
    for k in ("baseline_per_year_usd", "pdm_per_year_usd", "savings_per_year_usd", "savings_pct"):
        assert k in s
