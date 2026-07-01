"""Generator: class mix, ground truth, degradation shapes, determinism."""

from fleet_health.generator import SimConfig, generate


def test_all_classes_present():
    runs = generate(SimConfig(n_assets=200, horizon=300, seed=7))
    classes = {r.gen_class for r in runs}
    assert classes == {"stable", "wearout", "sudden", "transient"}


def test_readings_carry_ground_truth():
    runs = generate(SimConfig(n_assets=20, horizon=100, seed=1))
    for run in runs:
        for r in run.readings:
            assert r.gen_class == run.gen_class
            assert r.failure_tick == run.failure_tick
            assert r.true_health is not None
            assert r.asset_type == run.asset_type


def test_stable_and_transient_never_fail():
    runs = generate(SimConfig(n_assets=200, horizon=300, seed=7))
    for run in runs:
        if run.gen_class in ("stable", "transient"):
            assert run.failure_tick is None
            assert not run.fails


def test_wearout_and_sudden_fail_within_horizon():
    runs = generate(SimConfig(n_assets=200, horizon=400, seed=7))
    for run in runs:
        if run.gen_class in ("wearout", "sudden"):
            assert run.failure_tick is not None
            assert 0 <= run.failure_tick < 400


def test_wearout_health_declines_monotonically():
    runs = generate(SimConfig(n_assets=300, horizon=400, seed=3))
    wear = next(r for r in runs if r.gen_class == "wearout")
    healths = [rd.true_health for rd in wear.readings]
    # true_health (before noise) is non-increasing for a wear-out asset.
    assert all(healths[i] >= healths[i + 1] - 1e-9 for i in range(len(healths) - 1))
    assert healths[0] == 1.0
    assert healths[wear.failure_tick] == 0.0


def test_transient_recovers_to_full_health():
    runs = generate(SimConfig(n_assets=300, horizon=400, seed=5))
    tr = next(r for r in runs if r.gen_class == "transient")
    healths = [rd.true_health for rd in tr.readings]
    assert min(healths) < 1.0          # there is a dip
    assert healths[-1] == 1.0          # and it recovers


def test_generation_is_deterministic():
    a = generate(SimConfig(n_assets=50, horizon=120, seed=9))
    b = generate(SimConfig(n_assets=50, horizon=120, seed=9))
    assert [r.asset_id for r in a] == [r.asset_id for r in b]
    assert a[0].readings[10].vibration_rms == b[0].readings[10].vibration_rms
