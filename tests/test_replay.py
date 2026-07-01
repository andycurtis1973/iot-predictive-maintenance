"""Replay: the measured early-warning guarantees on the synthetic fleet.

These are the tests that earn trust, the analog of proving hit-ratio AND
collision-freedom together: strong lead time on predictable failures, and zero
false alarms on healthy + adversarial-transient assets.
"""

from fleet_health.generator import SimConfig, generate
from fleet_health.replay import replay

# One fixed fleet reused across the assertions (kept modest for test speed).
RUNS = generate(SimConfig(n_assets=140, horizon=520, seed=7))
REP = replay(RUNS)


def test_no_false_alarms_on_healthy_or_transient_assets():
    # The dangerous-nuisance direction: alerting on an asset that never fails.
    assert REP.false_alarms == 0
    assert REP.false_alarm_rate == 0.0
    assert REP.by_class["stable"].false_alarms == 0
    assert REP.by_class["transient"].false_alarms == 0


def test_precision_is_perfect_when_there_are_no_false_alarms():
    # Every alert raised is about a genuinely failing asset.
    assert REP.precision == 1.0


def test_every_failure_is_noticed_eventually():
    # missed == 0: no failure slips through entirely (early or at least late).
    assert REP.missed == 0


def test_all_wearout_failures_are_caught_early():
    wear = REP.by_class["wearout"]
    assert wear.failures > 0
    assert wear.detected == wear.failures     # 100% early-warning recall on wear-out


def test_wearout_lead_time_is_substantial():
    wear = REP.by_class["wearout"]
    leads = sorted(wear.lead_times)
    median = leads[len(leads) // 2]
    assert median > 40          # many ticks of warning before failure
    assert min(leads) > 0       # every wear-out catch is strictly before failure


def test_overall_recall_in_expected_band():
    # Overall recall is dragged down only by the abrupt-fault floor (by design),
    # not by missed wear-out; it should still clear a healthy bar.
    assert REP.recall > 0.6


def test_abrupt_faults_are_the_floor_not_early_catches():
    # Sudden faults give little warning: they show up as late detections, not
    # early ones. This documents the honest limitation.
    sudden = REP.by_class["sudden"]
    assert sudden.failures > 0
    assert sudden.detected < sudden.failures
