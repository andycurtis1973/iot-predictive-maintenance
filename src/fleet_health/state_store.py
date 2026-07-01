"""Per-asset state store on DynamoDB (SPEC.md §6, §7, §11).

The transaction cache in the reference design is stateless per request; a
predictive-maintenance monitor is not — each reading updates a rolling per-asset
context (recent window, health trajectory, alert latch). That state is what lives
in DynamoDB here: keyed by asset_id, it is loaded, updated, and written back on
every reading so the serving path is horizontally scalable and stateless between
invocations (any Lambda can pick up any asset).

Correctness controls, mirroring the reference:
  * The store is an interface (Protocol) so the serving hot path is identical on
    DynamoDB (production) or an in-memory dict (tests/local).
  * Versioned (§7): FEAT_V/HEALTH_V are recorded on each item; a bump lets an
    upgrade discard state computed under the old feature/health definition
    instead of mixing scales.
  * TTL (§7): a decommissioned asset's state ages out instead of lingering.
  * Bounded state: the window and health history are capped so an item can't grow
    without bound over a long-lived asset.
  * Encryption at rest (KMS) + tight IAM on the table (§11) — telemetry is
    operational data and treated as sensitive.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from .features import FEAT_V
from .health import HEALTH_V

# Bounds on the persisted per-asset context. WINDOW_CAP must be >= the feature
# window; HEALTH_CAP >= the RUL window. Kept small so each item stays tiny.
WINDOW_CAP = 24
HEALTH_CAP = 48


@dataclass
class AssetState:
    """The rolling context the monitor keeps for one asset."""

    asset_id: str
    asset_type: str
    window: list[dict] = field(default_factory=list)      # recent readings (bounded)
    health_history: list[float] = field(default_factory=list)  # recent health (bounded)
    consecutive: int = 0        # detector persistence counter
    alerted: bool = False       # detector alert latch (edge-trigger)
    last_tick: int = -1
    feat_v: str = FEAT_V
    health_v: str = HEALTH_V
    updated_at: str = ""        # ISO 8601
    ttl_epoch: int = 0          # DynamoDB TTL attribute (epoch seconds)

    def to_dynamo(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "window": json.dumps(self.window, separators=(",", ":")),
            "health_history": json.dumps(self.health_history, separators=(",", ":")),
            "consecutive": int(self.consecutive),
            "alerted": bool(self.alerted),
            "last_tick": int(self.last_tick),
            "feat_v": self.feat_v,
            "health_v": self.health_v,
            "updated_at": self.updated_at,
            "ttl_epoch": int(self.ttl_epoch),
        }

    @staticmethod
    def from_dynamo(item: dict[str, Any]) -> "AssetState":
        return AssetState(
            asset_id=item["asset_id"],
            asset_type=item["asset_type"],
            window=json.loads(item.get("window", "[]")),
            health_history=json.loads(item.get("health_history", "[]")),
            consecutive=int(item.get("consecutive", 0)),
            alerted=bool(item.get("alerted", False)),
            last_tick=int(item.get("last_tick", -1)),
            feat_v=item.get("feat_v", FEAT_V),
            health_v=item.get("health_v", HEALTH_V),
            updated_at=item.get("updated_at", ""),
            ttl_epoch=int(item.get("ttl_epoch", 0)),
        )


class StateStore(Protocol):
    """Hot-path per-asset state interface. DynamoDB and in-memory both satisfy it."""

    def get(self, asset_id: str) -> Optional[AssetState]: ...
    def put(self, state: AssetState) -> None: ...
    def delete(self, asset_id: str) -> None: ...


# --- In-memory store (tests / local; same semantics as DynamoDB) -----------
class InMemoryStateStore:
    def __init__(self, now_fn=time.time) -> None:
        self._d: dict[str, AssetState] = {}
        self._now = now_fn

    def get(self, asset_id: str) -> Optional[AssetState]:
        st = self._d.get(asset_id)
        if st is None:
            return None
        if st.ttl_epoch and st.ttl_epoch <= int(self._now()):
            del self._d[asset_id]
            return None
        # Round-trip through the serialized form so tests exercise the same
        # (de)serialization the DynamoDB path uses.
        return AssetState.from_dynamo(st.to_dynamo())

    def put(self, state: AssetState) -> None:
        self._d[state.asset_id] = AssetState.from_dynamo(state.to_dynamo())

    def delete(self, asset_id: str) -> None:
        self._d.pop(asset_id, None)


# --- DynamoDB store --------------------------------------------------------
class DynamoStateStore:
    """Backed by a boto3 DynamoDB Table resource (Key schema: asset_id HASH)."""

    def __init__(self, table: Any, now_fn=time.time) -> None:
        self._t = table
        self._now = now_fn

    def get(self, asset_id: str) -> Optional[AssetState]:
        resp = self._t.get_item(Key={"asset_id": asset_id})
        item = resp.get("Item")
        if not item:
            return None
        st = AssetState.from_dynamo(item)
        if st.ttl_epoch and st.ttl_epoch <= int(self._now()):
            return None  # TTL deletion lags; enforce expiry on read too
        return st

    def put(self, state: AssetState) -> None:
        self._t.put_item(Item=state.to_dynamo())

    def delete(self, asset_id: str) -> None:
        self._t.delete_item(Key={"asset_id": asset_id})


def bounded(state: AssetState) -> AssetState:
    """Trim the window/history to their caps and stamp the update time + TTL is
    the caller's job; this only enforces the size bounds."""
    if len(state.window) > WINDOW_CAP:
        state.window = state.window[-WINDOW_CAP:]
    if len(state.health_history) > HEALTH_CAP:
        state.health_history = state.health_history[-HEALTH_CAP:]
    return state


def touch(state: AssetState, now: int, ttl_seconds: int) -> AssetState:
    """Stamp updated_at + ttl_epoch from a unix ``now``."""
    state.updated_at = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
    state.ttl_epoch = now + ttl_seconds
    return state


# --- Provisioning helper (§6, §11) -----------------------------------------
def create_table(dynamodb_resource: Any, table_name: str, *, enable_ttl: bool = True) -> Any:
    """Create the state table: on-demand billing, KMS encryption at rest (§11),
    TTL on ttl_epoch (§7). Raises if the table already exists."""
    table = dynamodb_resource.create_table(
        TableName=table_name,
        KeySchema=[{"AttributeName": "asset_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "asset_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
        SSESpecification={"Enabled": True},
    )
    table.wait_until_exists()
    if enable_ttl:
        table.meta.client.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl_epoch"},
        )
    return table
