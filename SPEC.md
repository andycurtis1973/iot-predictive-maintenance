# Fleet Health — IoT Predictive Maintenance — Specification

**Status:** Draft v0.1
**Target runtime:** AWS (Lambda + DynamoDB + Timestream + Bedrock)
**Audience:** implementation agent

---

## 1. Summary

A streaming pipeline that sits in front of a fleet of rotating equipment (pumps,
motors, fans, compressors) and turns raw telemetry into **early-warning alerts**:
each reading updates a per-asset **health index**, a **remaining-useful-life
(RUL)** estimate is fit to the health trajectory, and an **edge-triggered
detector** fires a single alert when a confident downward trend crosses a
lead-time horizon — days before the machine actually fails.

The premise: **most mechanical failures announce themselves.** Wear is gradual —
bearings roughen, vibration climbs, temperature creeps — for days or weeks before
a machine quits. If you score health on every reading and extrapolate the trend,
you convert an *unplanned* failure (line down, secondary damage, emergency crew)
into a *planned* intervention scheduled in advance. The lever is **lead time**,
and it is measurable.

**Goal:** cut unplanned downtime on a fleet without a flood of false alarms —
catch the predictable (gradual) failures early, and be honest about the
unpredictable (abrupt) ones.

> **Framing for the implementer:** this is a genuine anomaly-detection / RUL
> pipeline, not an LLM cache. Bedrock appears only on the *alert* path, to draft a
> human-readable work order from a fired alert — a few calls per fleet-day, never
> the hot path.

---

## 2. Problem & motivation

Run-to-failure and fixed-interval maintenance both waste money. Run-to-failure
pays the full cost of an unplanned failure `Cu` — downtime, secondary damage,
emergency labor — often $25k–$140k per event for industrial rotating equipment.
Fixed-interval servicing avoids some failures but services healthy machines on a
calendar, paying for maintenance that wasn't needed and still missing off-cycle
faults.

Condition-based / predictive maintenance is the third path: watch the machine's
actual condition and act just in time. The engineering question is whether you
can (a) catch real failures early enough to act, and (b) not cry wolf. Both must
be **measured**, not assumed — which is what the simulation harness (§8) is for.

---

## 3. Core idea

```
reading ─▶ window ─▶ features ─▶ health index ─▶ RUL (trend extrapolation) ─┐
                                        │                                     │
                                        └────────────▶ anomaly score ────────┤
                                                                             ▼
                                                              edge-triggered detector
                                                                             │
                                                              alert (once) ──┴─▶ Bedrock work order
```

Every decision is a pure function of a **windowed** view of an asset's telemetry.
Smooth the per-sample noise into stable features; map features to a health index
(current condition) and an anomaly score (how abnormal now); fit a line to the
recent health trajectory to get RUL; fire one alert when either a confident RUL
crosses the lead-time horizon or the anomaly persists.

The engineering risk lives in **feature / health / threshold design** (§5). Get it
wrong and you either miss real failures (the dangerous direction) or false-alarm
constantly (the costly-nuisance direction). We bias toward catching failures but
control false alarms with persistence + hysteresis + a trend-confidence gate.

---

## 4. Scope

### In scope
- Deterministic condition monitoring of rotating equipment from vibration,
  temperature, and current telemetry.
- Health-index scoring, RUL estimation, edge-triggered alerting, per-asset state.
- Durable telemetry history for dashboards and backtesting.
- Optional LLM work-order generation on alerts (Bedrock).

### Out of scope / must stay elsewhere
- **Real-time safety trips.** Protective shutdowns (overspeed, vibration trip)
  belong on the PLC / protection relay, not a predictive model. Predictive
  maintenance *advises*; it does not *control*.
- **Predicting truly abrupt/random faults.** A fault with no precursor cannot be
  forewarned; we detect it at onset (late) and report it honestly as the floor,
  rather than trading away specificity to chase it (§8).

---

## 5. Feature / health / threshold design (the critical part)

### 5.1 Windowed features (`FEAT_V`)
A single reading is noisy; a decision off one sample chatters or misses. Reduce a
rolling window to a small, stable feature vector: smoothed levels **and the trend
(least-squares slope)**, which is the leading indicator — vibration that is
*rising* is more informative than vibration that is merely high. Deviations are
normalized to each asset type's own envelope (0 = nominal, 1 = failure level) so
the health model is asset-type-agnostic. `FEAT_V` versions the definition; any
change re-segments stored state (§7).

### 5.2 Health index + anomaly score (`HEALTH_V`)
Two deterministic numbers per reading:
- **health_index ∈ [0,1]** — current condition (level-based). 1 nominal, 0 at
  failure level.
- **anomaly_score ≥ 0** — how abnormal *now*: level severity plus a rising-trend
  bump.

The roles are deliberately separated: the health index is the level, the anomaly
is the "clearly bad now" signal, and the **early warning** comes from projecting
the health trajectory forward (RUL). Keeping the three roles distinct is what
lets each be reasoned about.

**Two channels.** The anomaly takes its *level* from a short, fast window and its
*trend* from the long, slow window. A short window reacts to an abrupt fault the
long average would hide; sourcing the trend from the long window means a 1–2 tick
transient contributes almost no slope, so it cannot spike the anomaly. A genuine
multi-tick fault does.

### 5.3 RUL
Fit `health ≈ a + b·t` over the recent trajectory; if `b < 0`, RUL =
`(health_now − failure_level) / −b`. Report r² (confidence) so the detector can
refuse to act on a noisy extrapolation. Flat/improving ⇒ infinite RUL.

### 5.4 Detector (edge-triggered)
Fire **once** per asset (on the transition into alerted), on either:
- **RUL trip** — confident (`r² ≥ min_r2`) downward trend predicts failure within
  `lead_time_ticks`. The early warning.
- **Persistence trip** — anomaly ≥ threshold for `persist` consecutive windows,
  with **hysteresis** (a separate lower clear threshold), so a single spike can't
  latch. The "clearly bad now" backstop for faults too abrupt to trend.

Bias: a missed catastrophic failure is the dangerous direction, so defaults favor
recall; hysteresis + the confidence gate hold the false-alarm rate down.

---

## 6. Architecture on AWS

```
             ┌───────────────────────────── Monitor Service (Lambda) ─────────────────────────────┐
telemetry ─▶ │ load asset state (DynamoDB) → features → health → RUL → detector → write state       │
             │        │best-effort                                        │on alert                  │
             │        ▼                                                    ▼                          │
             │   Timestream (raw history)                          Bedrock (work order)              │
             └────────────────────────────────────────────────────────────────────────────────────┘
```

**Two stores, two jobs:**
- **DynamoDB (hot path).** Per-asset rolling state — the last window of readings,
  the health history, the detector's persistence counter + alert latch — keyed by
  `asset_id`. Single-digit-ms point read/update on every reading; the pipeline is
  stateless between invocations (any worker serves any asset). KMS-encrypted, TTL,
  on-demand (idle-free).
- **Timestream (historian).** Every raw reading is written best-effort as a
  multi-measure record for dashboards, ad-hoc RUL backtests, and long-range trend
  queries. Memory tier for recent data, magnetic tier for cheap long retention.
  A historian hiccup must **never** drop a live reading or suppress an alert — the
  DynamoDB hot path owns correctness.

**Compute:** stateless Lambda (or a container on Fargate/EC2 for steady high
volume). **Inference:** Bedrock `Converse` on the alert path only. **Metrics:**
CloudWatch EMF (structured logs → metrics; no PutMetricData, works from any
runtime).

Why not Timestream for the hot path? The per-reading access pattern is
"read-last-N + update latch" — a KV point operation. Timestream is analytical
(time-range queries), not a low-latency point store; using it for the hot path
would be slower and more complex. DynamoDB for state *now*, Timestream for history.

---

## 7. State versioning & invalidation

- **Version-keyed state.** `FEAT_V` / `HEALTH_V` ride on each stored item; a bump
  lets an upgrade discard state computed under the old definition rather than
  mixing scales. This is the primary invalidation mechanism.
- **TTL.** A decommissioned asset's state ages out.
- **Bounded state.** Window and health history are capped so an item can't grow
  without bound over a long-lived asset.
- **Threshold changes** are config, not code — re-tunable per asset class from the
  measured false-alarm/lead-time trade-off (§8).

---

## 8. Validation — does the early-warning premise hold? (simulation harness)

Don't take lead time or the false-alarm rate as given — **measure** them on
synthetic-but-realistic data with ground truth the simulator knows and real
traffic doesn't.

### 8.1 Run-to-failure generator (the part that determines the results)
Lead time and false-alarm rate are properties of the *failure distribution*, so
the generator is the most important component. It emits per-tick telemetry for a
fleet drawn from four classes:
- **stable** — nominal for the whole horizon; never fails. True negatives; any
  alert here is a false alarm. Anchors specificity.
- **wearout** — monotonic degradation from an onset tick to failure. Vibration
  climbs steadily; the trend is predictable and RUL should give real lead time.
  The bread-and-butter of predictive maintenance.
- **sudden** — healthy until an abrupt fault in the last few ticks. Little/no
  warning; the irreducible floor. Caught, if at all, by the persistence backstop.
- **transient** — a stable asset with a short spike that recovers. Adversarial: a
  naive per-sample threshold fires; windowing + persistence + hysteresis must not.

Emission is the inverse of the health model, so a degrading asset really produces
the rising signal the model keys on. Ground-truth labels ride on every reading.

### 8.2 The three goals proven
- **Lead time** — `failure_tick − first_alert_tick` for detected failures.
- **Correctness / no false-alarm storm** — alerts on never-failing (stable +
  transient) assets. The dangerous-nuisance direction; must stay ~0.
- **Downtime avoided** — §10, computed from the *measured* recall.

The taxonomy separates **early** detection (before failure, buys lead time),
**late** detection (real but too late — the abrupt floor), and **false alarm** (on
a never-failing asset). Late detections don't hurt precision (the asset really was
failing) but don't count toward recall.

### 8.3 Measured result (reference fleet, 240 assets × 720 ticks)
- **Wear-out: 100% caught early**, median lead ~150–180 ticks (~6–7 days at hourly
  sampling), zero false alarms.
- **Precision ~100%** — every alert is about a genuinely failing asset.
- **False-alarm rate 0%** across ~160 healthy + transient assets.
- **Overall early-warning recall ~0.70** — dragged down only by the abrupt class,
  by design, not by missed wear-out. `missed = 0`: no failure goes unnoticed
  (early or at least late).

---

## 9. Relationship to LLM work-order generation (Bedrock)

The predictive core is pure numerics. On an alert, Bedrock (`Converse`, Claude
Haiku 4.5 via its regional inference profile, temperature 0, budget-guarded)
drafts a concise work order — likely failure mode, recommended action, priority,
parts. This is the last-mile translation from a signal to an instruction, on the
alert path only (a handful of calls per fleet-day). It is optional; the pipeline
is complete without it.

---

## 10. Cost model

```
Baseline/yr (run-to-failure) = F · Cu
With PdM/yr                  ≈ F·r·Cp + F·(1−r)·Cu + A·Cf
Savings/yr                   ≈ F·r·(Cu − Cp) − A·Cf
```
`F` failures/yr · `r` measured early-warning recall · `Cu` unplanned cost · `Cp`
planned cost (Cp ≪ Cu) · `A` false alarms/yr · `Cf` wasted-inspection cost.
Recompute with the *measured* `r` and `A`. Savings scale with the `Cu − Cp` gap,
so heavy assets (large `Cu`) pay most, and the abrupt-fault floor caps `r`.

---

## 11. Non-functional requirements

- **Latency:** per-reading hot-path processing p99 < 10 ms (excluding network);
  bench measures it. Alert-path Bedrock latency is bounded by the model.
- **Correctness:** the detector fires once per asset; the same reading history
  always produces the same decision (deterministic).
- **Auditability:** every processed reading returns health/anomaly/RUL and, on an
  alert, the reason + severity + predicted failure tick — a reconstructable record.
- **Isolation / safety:** predictive alerts advise; protective trips stay on the
  PLC (§4).
- **Security:** telemetry is operational data; encrypt at rest (KMS on DynamoDB),
  scope IAM tightly (per-table DynamoDB, per-table Timestream write +
  DescribeEndpoints), least-privilege Lambda role.
- **Resilience:** historian writes and work-order drafting are best-effort and can
  never break the hot path.

---

## 12. Implementation milestones

1. **Features + health + RUL** with `FEAT_V`/`HEALTH_V` versioning and unit tests
   covering slope, bounds, extrapolation. *(done)*
2. **Detector** — edge-triggered, persistence + hysteresis + RUL trip. *(done)*
3. **Simulation harness** (§8) — run-to-failure generator + offline replay proving
   lead time, precision/recall, zero false alarms against ground truth. *(done)*
4. **Serving path** — DynamoDB per-asset state, stateless Lambda/HTTP adapters,
   metrics. *(done)*
5. **Timestream historian** — best-effort raw-telemetry write + query helper. *(done)*
6. **Deploy** — provision/run/teardown stack + animated HTML demo. *(done)*
7. **Bedrock work orders** on alerts, budget-guarded. *(done)*

---

## 13. Open questions

- Per-asset-class thresholds tuned from that class's own history vs global defaults.
- Multi-modal faults (imbalance vs bearing vs looseness) — classify the fault mode,
  not just detect degradation.
- Batched Timestream writes (vs one record per reading) at high fleet volume.
- Gazetteer of known maintenance events to suppress alerts right after a service.
