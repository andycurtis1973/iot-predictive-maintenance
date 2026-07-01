"""Asset-type baselines, alarm thresholds, and downtime economics.

Single source of truth for what "healthy" looks like per class of equipment and
what a failure costs — shared by the health model (health.py), the RUL estimator
(rul.py), the generator (generator.py), and the cost model (cost_model.py). This
is the predictive-maintenance analog of a pricing table: the numbers that turn
raw signals into decisions live in exactly one reviewed place.

Vibration levels follow the shape of ISO 10816/20816 velocity zones for medium
machines (Zone A/B nominal, Zone C alarm, Zone D shutdown), rounded to round
numbers for a synthetic fleet — CONFIRM against the actual machine class and
mounting before trusting absolute values. Temperatures are illustrative bearing
limits. Costs are order-of-magnitude placeholders for the §10 model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetType:
    """Nominal operating envelope + failure economics for a class of machine."""

    name: str
    # Vibration velocity (mm/s RMS): healthy setpoint, alarm (investigate),
    # failure (imminent trip). Monotonic: nominal < alarm < failure.
    vib_nominal: float
    vib_alarm: float
    vib_failure: float
    # Bearing/winding temperature (°C): nominal and alarm.
    temp_nominal: float
    temp_alarm: float
    # Steady-state electrical + mechanical setpoints (used by the generator to
    # emit realistic secondary channels; minor contributors to the health index).
    current_nominal: float
    rpm_nominal: float
    # §10 economics (USD): an UNPLANNED failure (downtime + secondary damage +
    # emergency labor) vs a PLANNED intervention caught early. Cp ≪ Cu is the
    # whole reason predictive maintenance pays for itself.
    unplanned_cost_usd: float
    planned_cost_usd: float


ASSET_TYPES: dict[str, AssetType] = {
    "pump": AssetType(
        name="pump",
        vib_nominal=1.8, vib_alarm=7.1, vib_failure=11.0,
        temp_nominal=55.0, temp_alarm=85.0,
        current_nominal=42.0, rpm_nominal=1770.0,
        unplanned_cost_usd=48_000.0, planned_cost_usd=3_500.0,
    ),
    "motor": AssetType(
        name="motor",
        vib_nominal=1.4, vib_alarm=4.5, vib_failure=7.1,
        temp_nominal=60.0, temp_alarm=95.0,
        current_nominal=68.0, rpm_nominal=3550.0,
        unplanned_cost_usd=72_000.0, planned_cost_usd=5_000.0,
    ),
    "fan": AssetType(
        name="fan",
        vib_nominal=2.2, vib_alarm=9.0, vib_failure=14.0,
        temp_nominal=45.0, temp_alarm=75.0,
        current_nominal=30.0, rpm_nominal=1180.0,
        unplanned_cost_usd=26_000.0, planned_cost_usd=2_200.0,
    ),
    "compressor": AssetType(
        name="compressor",
        vib_nominal=2.0, vib_alarm=6.5, vib_failure=10.5,
        temp_nominal=70.0, temp_alarm=105.0,
        current_nominal=120.0, rpm_nominal=2950.0,
        unplanned_cost_usd=140_000.0, planned_cost_usd=9_000.0,
    ),
}


def get_asset_type(name: str) -> AssetType:
    """Look up an asset type, failing closed on an unknown class."""
    try:
        return ASSET_TYPES[name]
    except KeyError:
        raise KeyError(
            f"unknown asset_type {name!r}; register it in asset_types.ASSET_TYPES"
        ) from None
