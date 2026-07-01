"""Cost model (SPEC.md §10) — downtime avoided, from MEASURED inputs.

The reference design saves money by skipping inference calls; predictive
maintenance saves money by converting UNPLANNED failures (downtime + secondary
damage + emergency labor, cost Cu) into PLANNED interventions caught early (cost
Cp ≪ Cu). The lever is the early-warning recall measured by the replay (§8), not
an assumed number.

    Baseline/yr (run-to-failure) = F · Cu
    With PdM/yr                  ≈ F·r·Cp + F·(1−r)·Cu + A·Cf
    Savings/yr                   ≈ F·r·(Cu − Cp) − A·Cf

where F = failures/yr, r = early-warning recall, A = false alarms/yr, Cf =
wasted-inspection cost of a false alarm. As in §10, recompute this with the
*measured* r and A rather than a hoped-for detection rate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostInputs:
    failures_per_year: float     # F — expected failures across the monitored fleet
    recall: float                # r — early-warning recall (measured, §8)
    unplanned_cost: float        # Cu — cost of an unplanned failure
    planned_cost: float          # Cp — cost of a planned intervention, Cp ≪ Cu
    false_alarms_per_year: float = 0.0   # A — measured false alarms/yr
    false_alarm_cost: float = 0.0        # Cf — wasted inspection per false alarm


@dataclass
class CostModel:
    inputs: CostInputs

    @property
    def baseline_per_year(self) -> float:
        i = self.inputs
        return i.failures_per_year * i.unplanned_cost

    @property
    def caught(self) -> float:
        i = self.inputs
        return i.failures_per_year * i.recall

    @property
    def not_caught(self) -> float:
        i = self.inputs
        return i.failures_per_year * (1.0 - i.recall)

    @property
    def false_alarm_cost_per_year(self) -> float:
        i = self.inputs
        return i.false_alarms_per_year * i.false_alarm_cost

    @property
    def pdm_per_year(self) -> float:
        i = self.inputs
        return (
            self.caught * i.planned_cost
            + self.not_caught * i.unplanned_cost
            + self.false_alarm_cost_per_year
        )

    @property
    def savings_per_year(self) -> float:
        return self.baseline_per_year - self.pdm_per_year

    @property
    def savings_ratio(self) -> float:
        b = self.baseline_per_year
        return self.savings_per_year / b if b else 0.0

    def summary(self) -> dict:
        return {
            "failures_per_year": round(self.inputs.failures_per_year, 2),
            "recall": round(self.inputs.recall, 4),
            "unplanned_cost_usd": self.inputs.unplanned_cost,
            "planned_cost_usd": self.inputs.planned_cost,
            "baseline_per_year_usd": round(self.baseline_per_year, 2),
            "pdm_per_year_usd": round(self.pdm_per_year, 2),
            "  caught_planned": round(self.caught, 2),
            "  not_caught_unplanned": round(self.not_caught, 2),
            "  false_alarm_cost_usd": round(self.false_alarm_cost_per_year, 2),
            "savings_per_year_usd": round(self.savings_per_year, 2),
            "savings_pct": round(self.savings_ratio * 100, 2),
        }


def cost_model(
    failures_per_year: float,
    recall: float,
    unplanned_cost: float,
    planned_cost: float,
    false_alarms_per_year: float = 0.0,
    false_alarm_cost: float = 0.0,
) -> CostModel:
    return CostModel(
        CostInputs(
            failures_per_year=failures_per_year,
            recall=recall,
            unplanned_cost=unplanned_cost,
            planned_cost=planned_cost,
            false_alarms_per_year=false_alarms_per_year,
            false_alarm_cost=false_alarm_cost,
        )
    )
