"""Telemetry historian: in-memory semantics, serving integration, best-effort."""

from types import SimpleNamespace

from fleet_health.historian import (
    InMemoryHistorian,
    _line_protocol,
    _parse_flux_csv,
    _parse_rows,
)
from fleet_health.model import Reading
from fleet_health.serving import MonitorService
from fleet_health.state_store import InMemoryStateStore

R = Reading("PUMP-1", 0, 1.8, 55.0, 42.0, 1770.0, "pump")   # pump nominal vibration


def test_write_and_recent_newest_first():
    h = InMemoryHistorian()
    for t in range(5):
        h.write(Reading("PUMP-1", t, 2.0 + t, 55.0, 42.0, 1770.0, "pump"), now_ms=1000 + t)
    assert h.count("PUMP-1") == 5
    recent = h.recent("PUMP-1", limit=3)
    assert [r["tick"] for r in recent] == [4, 3, 2]     # newest first
    assert recent[0]["vibration_rms"] == 6.0


def test_recent_empty_for_unknown_asset():
    assert InMemoryHistorian().recent("nope") == []


def test_serving_writes_every_reading_to_historian():
    h = InMemoryHistorian()
    svc = MonitorService(InMemoryStateStore(), historian=h)
    for t in range(10):
        svc.process(Reading("PUMP-1", t, 2.0, 55.0, 42.0, 1770.0, "pump"))
    assert h.count("PUMP-1") == 10
    assert svc.history_writes == 10


def test_historian_failure_never_breaks_hot_path():
    def _boom(reading, *, now_ms):
        raise RuntimeError("timestream throttled")

    bad = SimpleNamespace(write=_boom)
    svc = MonitorService(InMemoryStateStore(), historian=bad)
    res = svc.process(R)                       # must not raise
    assert res.health_index == 1.0
    assert svc.history_writes == 0             # write failed, but serving is fine


def test_parse_rows_flattens_timestream_response():
    resp = {
        "ColumnInfo": [{"Name": "tick"}, {"Name": "vibration_rms"}, {"Name": "time"}],
        "Rows": [
            {"Data": [{"ScalarValue": "12"}, {"ScalarValue": "3.5"}, {"ScalarValue": "2026-07-01 00:00:00"}]},
            {"Data": [{"ScalarValue": "11"}, {"ScalarValue": "3.2"}, {"ScalarValue": "2026-06-30 23:59:00"}]},
        ],
    }
    rows = _parse_rows(resp)
    assert rows[0] == {"tick": 12, "vibration_rms": 3.5, "time": "2026-07-01 00:00:00"}
    assert rows[1]["tick"] == 11 and rows[1]["vibration_rms"] == 3.2


# --- Timestream for InfluxDB (managed InfluxDB instance) --------------------
def test_influx_line_protocol():
    lp = _line_protocol(Reading("MOTOR-1", 5, 2.1, 60.0, 68.0, 3550.0, "motor"), now_ms=1719800000000)
    assert lp == (
        "telemetry,asset_id=MOTOR-1,asset_type=motor "
        "vibration_rms=2.1,temperature_c=60.0,current_a=68.0,rpm=3550.0,tick=5i "
        "1719800000000"
    )


def test_parse_flux_csv_pivoted():
    csv_text = (
        "#datatype,string,long,dateTime:RFC3339,string,string,double,double,double,double,long\r\n"
        ",result,table,_time,asset_id,asset_type,current_a,rpm,temperature_c,vibration_rms,tick\r\n"
        ",_result,0,2026-07-01T00:00:00Z,MOTOR-1,motor,68.0,3550.0,60.0,2.1,5\r\n"
        ",_result,0,2026-06-30T23:59:00Z,MOTOR-1,motor,68.0,3550.0,60.0,2.0,4\r\n"
    )
    rows = _parse_flux_csv(csv_text)
    assert len(rows) == 2
    assert rows[0]["asset_id"] == "MOTOR-1"
    assert rows[0]["vibration_rms"] == 2.1
    assert rows[0]["tick"] == 5
    assert "table" not in rows[0] and "result" not in rows[0]


def test_parse_flux_csv_empty():
    assert _parse_flux_csv("") == []
    assert _parse_flux_csv("#datatype,string\n") == []
