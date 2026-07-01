"""Bedrock model identifiers and on-demand pricing (SPEC.md §9).

Single source of truth for the model used by the optional work-order generator
(inference.py). The predictive core is pure numerics and needs no LLM; Bedrock is
used only to turn a fired alert into a human-readable maintenance work order, so
this table is small.

Pricing is per MILLION tokens (input, output), USD, us-east-1 on-demand. CONFIRM
against current Bedrock pricing before trusting cost figures — rates change.
"""

from __future__ import annotations

# Claude Haiku 4.5 must be called via a regional inference profile on Bedrock —
# the bare "anthropic.claude-haiku-4-5" id 400s (on-demand not offered on it).
HAIKU_4_5 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Amazon Nova — a cheaper alternative for the short work-order generation task.
NOVA_LITE = "amazon.nova-lite-v1:0"
NOVA_MICRO = "amazon.nova-micro-v1:0"


# Keyed by a substring of the model id so inference-profile ids resolve too.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "haiku-4-5": (1.00, 5.00),
    "sonnet-4-6": (3.00, 15.00),
    "nova-lite": (0.06, 0.24),
    "nova-micro": (0.035, 0.14),
}


def price_for(model_id: str) -> tuple[float, float]:
    """(input, output) $/MTok for a model id, matching by substring."""
    for needle, price in PRICING_PER_MTOK.items():
        if needle in model_id:
            return price
    raise KeyError(
        f"no pricing for model {model_id!r}; add it to PRICING_PER_MTOK "
        f"(rates change — confirm against current Bedrock pricing)"
    )
