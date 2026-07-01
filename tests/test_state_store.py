"""State store: round-trip serialization, TTL, size bounds, delete."""

from fleet_health.state_store import (
    HEALTH_CAP,
    WINDOW_CAP,
    AssetState,
    InMemoryStateStore,
    bounded,
    touch,
)


def _state(asset_id="A"):
    return AssetState(
        asset_id=asset_id,
        asset_type="pump",
        window=[{"tick": 0, "vibration_rms": 2.0, "temperature_c": 55.0, "current_a": 42.0, "rpm": 1770.0}],
        health_history=[1.0, 0.99, 0.98],
        consecutive=2,
        alerted=True,
        last_tick=0,
    )


def test_put_get_round_trip_preserves_fields():
    store = InMemoryStateStore()
    st = _state()
    store.put(st)
    got = store.get("A")
    assert got is not None
    assert got.window == st.window
    assert got.health_history == st.health_history
    assert got.consecutive == 2
    assert got.alerted is True


def test_dynamo_serialization_is_reversible():
    st = _state()
    round_tripped = AssetState.from_dynamo(st.to_dynamo())
    assert round_tripped.to_dynamo() == st.to_dynamo()


def test_missing_asset_returns_none():
    assert InMemoryStateStore().get("nope") is None


def test_ttl_expiry_is_enforced_on_read():
    clock = {"t": 1000}
    store = InMemoryStateStore(now_fn=lambda: clock["t"])
    st = _state()
    touch(st, now=1000, ttl_seconds=60)
    store.put(st)
    assert store.get("A") is not None
    clock["t"] = 1000 + 61
    assert store.get("A") is None


def test_delete_removes_state():
    store = InMemoryStateStore()
    store.put(_state())
    store.delete("A")
    assert store.get("A") is None


def test_bounded_caps_window_and_history():
    st = AssetState(asset_id="A", asset_type="pump")
    st.window = [{"tick": i} for i in range(WINDOW_CAP + 30)]
    st.health_history = [1.0] * (HEALTH_CAP + 30)
    bounded(st)
    assert len(st.window) == WINDOW_CAP
    assert len(st.health_history) == HEALTH_CAP
    # keeps the most recent entries
    assert st.window[-1]["tick"] == WINDOW_CAP + 29
