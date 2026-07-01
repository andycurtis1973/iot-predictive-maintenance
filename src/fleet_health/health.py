"""Health index + anomaly score (SPEC.md §5) — versioned HEALTH_V.

Turns a Features vector into two deterministic numbers:

  * health_index ∈ [0, 1] — current condition. 1.0 is nominal; 0.0 is at or
    past the failure level. Level-based: it answers "how worn is this asset
    right now?"
  * anomaly_score ≥ 0 — how far outside normal the asset is *now*, with a modest
    contribution from the rising trend so a sharp excursion registers
    immediately. It answers "how abnormal is this sample?" and drives the
    persistence backstop in the detector.

The split is deliberate: the health index is the level, the anomaly score is the
"clearly bad now" signal, and the *early warning* comes from projecting the
health trajectory forward (rul.py). Keeping those three roles separate is what
lets each be reasoned about — the predictive-maintenance analog of pinning field
sensitivity per task in the reference design.

HEALTH_V versions the scoring formula; changing weights or the failure mapping
must bump it so stored health history doesn't mix scales (§7).
"""

from __future__ import annotations

from dataclasses import dataclass

from .asset_types import AssetType
from .features import Features

HEALTH_V = "health_v1"

# Temperature contributes to severity but is a weaker, laggier signal than
# vibration for mechanical wear, so it's down-weighted.
TEMP_WEIGHT = 0.6
# How strongly a rising vibration trend bumps the anomaly score. A rise fast
# enough to traverse the whole nominal→failure band in ~1/SLOPE_GAIN ticks adds
# ~1.0 to the score. Kept modest so steady wear-out flags via RUL, not chatter.
SLOPE_GAIN = 3.0


@dataclass(frozen=True)
class Health:
    """Deterministic condition summary for one asset at one tick."""

    health_index: float   # 0.0 (failed) .. 1.0 (nominal)
    anomaly_score: float   # 0.0 nominal, grows without a hard cap
    dominant: str          # "vibration" | "temperature" | "nominal"


def _severity(feat: Features) -> tuple[float, str]:
    """Level severity (0 nominal .. 1 failure) and the channel that drives it."""
    vib_sev = feat.vib_frac
    temp_sev = TEMP_WEIGHT * feat.temp_frac
    severity = max(vib_sev, temp_sev)
    if severity <= 1e-9:
        dominant = "nominal"
    elif vib_sev >= temp_sev:
        dominant = "vibration"
    else:
        dominant = "temperature"
    return severity, dominant


def _slope_frac(vib_slope: float, asset: AssetType) -> float:
    band = asset.vib_failure - asset.vib_nominal
    return max(0.0, vib_slope) / band if band > 0 else 0.0


def _anomaly(level: Features, trend: Features, asset: AssetType) -> float:
    """How abnormal this window is now: level severity (from ``level``) plus a
    rising-trend bump (from ``trend``). Sourcing the trend from a longer window
    is deliberate — a 1–2 tick transient contributes almost no slope there, so it
    can't spike the anomaly, while a genuine multi-tick fault still does."""
    severity, _ = _severity(level)
    return severity + SLOPE_GAIN * _slope_frac(trend.vib_slope, asset)


def score(feat: Features, asset: AssetType) -> Health:
    """Single-window scoring: health index and anomaly from the same features.

    Pure function of the feature vector and the asset envelope — no clock, no
    state — so it is trivially testable and reproducible. The pipeline uses
    ``compose`` (below) to run the anomaly on a shorter, faster window; this is
    the convenience form for tests and simple one-window callers.
    """
    severity, dominant = _severity(feat)
    return Health(
        health_index=round(min(1.0, max(0.0, 1.0 - severity)), 6),
        anomaly_score=round(_anomaly(feat, feat, asset), 6),
        dominant=dominant,
    )


def compose(feat_slow: Features, feat_fast: Features, asset: AssetType) -> Health:
    """Two-channel scoring: a SLOW channel for the level/health (stable, drives
    RUL) and a FAST short-window channel for the anomaly (responsive to abrupt
    faults). Separating them lets the health trend stay smooth for RUL while the
    anomaly path still reacts to a sudden fault that a long average would hide —
    without letting a single transient spike (too short to persist) latch."""
    severity, dominant = _severity(feat_slow)
    return Health(
        health_index=round(min(1.0, max(0.0, 1.0 - severity)), 6),
        anomaly_score=round(_anomaly(feat_fast, feat_slow, asset), 6),
        dominant=dominant,
    )
