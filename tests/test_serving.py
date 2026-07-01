"""Serving: stateful hot path, edge-triggered alerts, JSON handler, adapters."""

import json
from types import SimpleNamespace

from fleet_health.generator import SimConfig, generate
from fleet_health.inference import BudgetGuard
from fleet_health.model import Reading
from fleet_health.serving import (
    MonitorService,
    Service,
    make_lambda_handler,
)
from fleet_health.state_store import InMemoryStateStore

READING = {
    "asset_id": "PUMP-1", "tick": 0, "vibration_rms": 1.8,   # pump nominal
    "temperature_c": 55.0, "current_a": 42.0, "rpm": 1770.0, "asset_type": "pump",
}


def _clean(r: Reading) -> Reading:
    return Reading(r.asset_id, r.tick, r.vibration_rms, r.temperature_c,
                   r.current_a, r.rpm, r.asset_type)


def _wearout_run():
    runs = generate(SimConfig(n_assets=140, horizon=520, seed=7))
    return next(r for r in runs if r.gen_class == "wearout")


def test_process_returns_health_and_persists_state():
    store = InMemoryStateStore()
    svc = MonitorService(store)
    res = svc.process(Reading(**READING))
    assert res.health_index == 1.0               # nominal reading -> full health
    assert store.get("PUMP-1") is not None       # state written back


def test_wearout_stream_fires_exactly_one_alert():
    svc = MonitorService(InMemoryStateStore())
    run = _wearout_run()
    fired = [res for r in run.readings
             if (res := svc.process(_clean(r))).alert]
    assert len(fired) == 1                          # edge-triggered
    assert fired[0].tick <= run.failure_tick        # in time
    assert fired[0].alert["dominant"] in ("vibration", "temperature")


def test_handler_status_codes():
    h = Service(MonitorService(InMemoryStateStore()), emit_metrics=False)
    assert h.handle({"reading": READING})[0] == 200
    assert h.handle({"reading": {"asset_id": "X", "tick": 0}})[0] == 400   # missing fields
    bad_type = dict(READING, asset_type="spaceship")
    assert h.handle({"reading": bad_type})[0] == 404                        # unknown type
    assert h.handle("not a dict")[0] == 400


def test_work_order_drafted_on_alert_and_budget_recorded():
    gen = SimpleNamespace(
        draft=lambda alert, asset_type: SimpleNamespace(
            order={"failure_mode": "bearing_wear", "priority": alert.severity},
            cost_usd=0.0002,
        )
    )
    budget = BudgetGuard(max_spend_usd=1.0)
    svc = MonitorService(InMemoryStateStore(), work_order_gen=gen, budget=budget)
    run = _wearout_run()
    got = None
    for r in run.readings:
        res = svc.process(_clean(r))
        if res.alert:
            got = res
            break
    assert got is not None
    assert got.work_order["failure_mode"] == "bearing_wear"
    assert budget.calls == 1 and budget.spent_usd > 0


def test_work_order_failure_never_breaks_hot_path():
    # A budget already exhausted -> check() raises -> the hot path still serves.
    gen = SimpleNamespace(draft=lambda a, t: SimpleNamespace(order={}, cost_usd=0.0))
    budget = BudgetGuard(max_spend_usd=0.0)
    svc = MonitorService(InMemoryStateStore(), work_order_gen=gen, budget=budget)
    run = _wearout_run()
    for r in run.readings:
        res = svc.process(_clean(r))
        if res.alert:
            assert "_error" in res.work_order    # captured, not raised
            break


def test_lambda_adapter_shapes():
    lh = make_lambda_handler(Service(MonitorService(InMemoryStateStore()), emit_metrics=False))
    out = lh({"reading": READING})
    assert out["statusCode"] == 200
    body = json.loads(out["body"])
    assert "health_index" in body and "rul_ticks" in body

    out2 = lh({"body": json.dumps({"reading": READING})})
    assert out2["statusCode"] == 200

    out3 = lh({"body": "{not json"})
    assert out3["statusCode"] == 400
