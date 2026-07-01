"""Optional work-order generation via Amazon Bedrock (SPEC.md §9).

The predictive pipeline (features → health → RUL → detector) is pure numerics and
needs no LLM. But when an alert fires, an operator wants more than a number: a
concise work order — likely failure mode, recommended action, priority, parts to
stage. That last-mile translation from a signal to an instruction is what Bedrock
does here, on the alert path only (a few calls per fleet-day), not the hot path.

Determinism is pinned (temperature 0) and a BudgetGuard caps spend so a batch of
alerts can never run away. The boto3 client is injected so tests supply a fake and
never spend. Claude Haiku 4.5 is reached via its regional inference profile — the
bare id 400s on Bedrock (see bedrock_models.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

from .bedrock_models import HAIKU_4_5, price_for
from .detector import Alert

DEFAULT_MODEL_ID = HAIKU_4_5

_SYSTEM_PROMPT = (
    "You are a reliability engineer writing a maintenance work order for a piece "
    "of rotating equipment that a predictive-maintenance model has flagged. You "
    "are given the asset type, the dominant degrading signal, a health index "
    "(1.0 healthy, 0.0 failed), and an estimated remaining useful life in hours. "
    "Respond with ONLY a JSON object, no prose, of the form "
    '{"failure_mode": "<short likely failure mode>", '
    '"recommended_action": "<one concrete maintenance action>", '
    '"priority": "<low|medium|high|critical>", '
    '"parts": ["<part>", ...], '
    '"summary": "<one-sentence operator summary>"}. '
    "Be specific to the asset type and dominant signal, and deterministic: the "
    "same input must always yield the same output."
)


@dataclass
class WorkOrder:
    order: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    cost_usd: float
    raw_text: str = ""


class BudgetExceeded(RuntimeError):
    """Raised when a call would push spend past the guard's cap."""


@dataclass
class BudgetGuard:
    """Hard cap on cumulative Bedrock spend for a batch of alerts (§8a)."""

    max_spend_usd: float
    spent_usd: float = 0.0
    calls: int = 0

    def remaining(self) -> float:
        return max(0.0, self.max_spend_usd - self.spent_usd)

    def check(self) -> None:
        if self.spent_usd >= self.max_spend_usd:
            raise BudgetExceeded(
                f"budget cap ${self.max_spend_usd:.4f} reached "
                f"(spent ${self.spent_usd:.4f} over {self.calls} calls)"
            )

    def record(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        self.calls += 1


def alert_user_text(alert: Alert, asset_type: str, *, tick_hours: float = 1.0) -> str:
    """The per-alert user message describing the flagged condition."""
    rul_hours = alert.rul_ticks * tick_hours
    rul_str = "n/a (abrupt fault, no trend)" if rul_hours != rul_hours or rul_hours > 1e8 else f"{rul_hours:.0f} hours"
    return (
        f"asset_type: {asset_type}\n"
        f"dominant_signal: {alert.dominant}\n"
        f"health_index: {alert.health_index:.2f}\n"
        f"remaining_useful_life: {rul_str}\n"
        f"alert_reason: {alert.reason}\n"
        f"severity: {alert.severity}"
    )


class WorkOrderGenerator:
    """Wraps Bedrock Converse to draft a work order from a fired alert."""

    def __init__(
        self,
        client: Any,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        cross_region_factor: float = 1.0,
        max_tokens: int = 300,
        tick_hours: float = 1.0,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.cross_region_factor = cross_region_factor
        self.max_tokens = max_tokens
        self.tick_hours = tick_hours
        self._in_price, self._out_price = price_for(model_id)

    @staticmethod
    def make_client(region_name: str = "us-east-1", config: Any = None) -> Any:
        import boto3
        from botocore.config import Config

        cfg = config or Config(retries={"mode": "standard", "total_max_attempts": 8})
        return boto3.client("bedrock-runtime", region_name=region_name, config=cfg)

    def _cost(self, in_tok: int, out_tok: int) -> float:
        base = (in_tok / 1e6) * self._in_price + (out_tok / 1e6) * self._out_price
        return base * self.cross_region_factor

    def draft(self, alert: Alert, asset_type: str) -> WorkOrder:
        """Make one real Converse call and return the parsed work order + telemetry."""
        t0 = time.perf_counter()
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [
                {"text": alert_user_text(alert, asset_type, tick_hours=self.tick_hours)}
            ]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = resp["output"]["message"]["content"][0]["text"]
        usage = resp.get("usage", {})
        in_tok = int(usage.get("inputTokens", 0))
        out_tok = int(usage.get("outputTokens", 0))
        return WorkOrder(
            order=_parse_json(text),
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            model_id=self.model_id,
            cost_usd=self._cost(in_tok, out_tok),
            raw_text=text,
        )


def _parse_json(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model response, tolerating stray prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"_unparsed": text}
