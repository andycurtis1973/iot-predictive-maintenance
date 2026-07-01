# Fleet Health — IoT Predictive Maintenance

Turn rotating-equipment telemetry into **early-warning alerts**. Every reading
updates a per-asset **health index**, a **remaining-useful-life** estimate is fit
to the health trend, and an **edge-triggered detector** fires one alert when a
confident decline crosses a lead-time horizon — *days before the machine fails*.
See `SPEC.md` for the full design.

> The premise: **most mechanical failures announce themselves.** Wear is gradual
> — vibration climbs and temperature creeps for days before a machine quits. Watch
> the trend and you convert an unplanned failure into a planned repair. The lever
> is **lead time**, and it is measured, not assumed.

This is a genuine anomaly-detection / RUL pipeline, **not** an LLM cache. Bedrock
appears only on the *alert* path, to draft a work order from a fired alert.

## Quick start (no AWS needed)

The library and tests are pure Python — the offline parts never call AWS.

```bash
git clone https://github.com/andycurtis1973/iot-predictive-maintenance
cd iot-predictive-maintenance
python3 -m pip install pytest          # the only test dependency

pytest -q                              # full suite, AWS mocked — never spends
PYTHONPATH=src python3 scripts/run_replay.py --assets 240 --horizon 720   # measure lead time
PYTHONPATH=src python3 scripts/run_cost.py                                # §10 downtime-avoided
```

**To run it live on AWS**, see [the demo](#demo-spin-up--run--turn-off-real-aws).
You'll need AWS credentials in `us-east-1` with DynamoDB + Timestream permissions
(the demo provisions and tears down its own tables; spend is a few cents), and
Bedrock access to Claude Haiku 4.5 if you enable work-order generation.

## The result (measured, reference fleet 240 × 720)

| metric | value |
|---|---|
| **wear-out failures caught early** | **100%**, median lead ~150–180 ticks (~6–7 days hourly) |
| **precision** | **~100%** — every alert is about a genuinely failing asset |
| **false-alarm rate** | **0%** across ~160 healthy + transient assets |
| overall early-warning recall | ~0.70 (capped only by the abrupt-fault floor, by design) |
| failures fully missed | **0** — every failure is at least detected (early or late) |
| hot-path latency | p50 ~1.4 ms, p99 ~7 ms per reading |

**The honest story:** gradual wear-out — the majority of mechanical failures — is
caught early with a week of lead time at zero false alarms. Abrupt/random faults
are the irreducible floor: detected at onset (late), not forewarned. We report
them as such rather than trade away specificity to chase a signal that isn't
there yet.

## How it works

```
reading ─▶ window ─▶ features ─▶ health index ─▶ RUL (trend) ─┐
                            └────────▶ anomaly score ─────────┤
                                                              ▼
                                              edge-triggered detector ─▶ alert ─▶ Bedrock work order
```

- **Two channels.** The anomaly takes its *level* from a short, fast window (reacts
  to abrupt faults) and its *trend* from the long, slow window (so a 1–2 tick
  transient can't spike it). The health index + RUL use the slow window.
- **Fire once.** The detector is edge-triggered — one alert per asset, on a
  confident RUL trip or a persistent anomaly (with hysteresis so a single spike
  can't latch).
- **Two stores.** DynamoDB holds the hot per-asset rolling state (point read/write
  every reading); Timestream is the durable telemetry historian (dashboards,
  backtests). A historian hiccup never breaks the hot path.

## Architecture on AWS

```
telemetry ─▶ Lambda (stateless) ──┬─▶ DynamoDB    (hot per-asset state)
                                  ├─▶ Timestream  (raw telemetry history, best-effort)
                                  └─▶ Bedrock     (work order, on alert only)
```

DynamoDB answers *"what is this asset's state right now?"* at KV latency on every
reading; Timestream answers *"show me this asset's last month of vibration"* for
analytics. Using Timestream for the hot path would be slower (it's analytical, not
a low-latency point store), so the design splits the two jobs — see `SPEC.md` §6.

## Layout

```
src/fleet_health/
  model.py        Reading — the telemetry record (fields that affect a decision)
  asset_types.py  per-class baselines, alarm/failure thresholds, downtime economics
  features.py     windowed feature extraction (levels + slope) — versioned FEAT_V
  health.py       health index + anomaly score (two-channel) — versioned HEALTH_V
  rul.py          remaining-useful-life: trend fit → time-to-failure + confidence
  detector.py     edge-triggered alerting: RUL trip + persistence/hysteresis
  generator.py    synthetic run-to-failure fleet, 4 classes + ground truth
  replay.py       offline eval: lead time, precision/recall, false-alarm rate
  state_store.py  DynamoDB per-asset state: store interface + service + TTL
  historian.py    Timestream telemetry historian: write + query + provisioning
  cost_model.py   §10 downtime-avoided projection from measured inputs
  metrics.py      CloudWatch EMF (health, anomaly, recall, cost avoided)
  inference.py    optional Bedrock work-order generation (budget-guarded)
  serving.py      request handler + Lambda/HTTP adapters (runtime-agnostic)
scripts/
  run_replay.py   generate a fleet + print the measured early-warning report
  run_cost.py     §10 downtime-avoided projection (no spend)
  run_bench.py    hot-path latency bench; dry by default, --live for work orders
  run_e2e.py      LIVE on real DynamoDB (+ --timestream, --live-bedrock); self-tears-down
  run_server.py   run the monitor as a local HTTP service
tests/            AWS-free deterministic suite (66 tests)
deploy/           demo.py (up/call/drive/capture/down) + Lambda handler + web demo
video/            explainer pipeline: script → slides → narration → MP4
```

## Demo: spin up → run → turn off (real AWS)

`deploy/demo.py` provisions a complete stack — DynamoDB + Timestream + IAM role +
Lambda (python3.12) + Function URL — runs it, and deletes it. **Zero cost when
off** (Lambda scales to zero, on-demand DynamoDB + Timestream are idle-free).

```bash
PYTHONPATH=src deploy/demo.py up             # provision; prints the Function URL
PYTHONPATH=src deploy/demo.py call           # SigV4-signed sample readings
PYTHONPATH=src deploy/demo.py drive 8 220    # stream a fleet through the live pipeline
PYTHONPATH=src deploy/demo.py down           # delete everything
```

The Function URL uses **AWS_IAM auth** (anonymous URLs are commonly org-blocked);
the client SigV4-signs.

### Customer-facing visual

`deploy/demo.py capture` streams a fleet through the live pipeline, records each
result, and injects it into `deploy/web/template.html` to produce a self-contained
animated **`deploy/web/demo.html`** — a replay of a real AWS run (no creds/spend to
view; opens anywhere). It animates a fleet of asset tiles, health bars falling as
assets degrade, alerts firing with lead time and drafted work orders, and metric
counters (assets monitored, alerts, mean lead time, downtime cost avoided).

```bash
PYTHONPATH=src deploy/demo.py up
PYTHONPATH=src deploy/demo.py capture 16 300     # writes deploy/web/demo.html
PYTHONPATH=src deploy/demo.py down
open deploy/web/demo.html

# no AWS? generate the same visual in-process:
PYTHONPATH=src deploy/demo.py capture-local 16 300
```

## Video

`video/` renders a ~90-second explainer: `build_assets.py` draws the slides and
writes `assets/script.json`; `make_audio_local.py` narrates it with macOS `say`
(the offline stand-in for the production F5-TTS GPU worker in `launch.py`);
`assemble_local.py` muxes slides + narration into `video/out/fleet_health_demo.mp4`
(timing derived from each segment's audio length).

```bash
python3 video/build_assets.py
python3 video/make_audio_local.py
python3 video/assemble_local.py
```

## Key design decisions

- **Versioned features/health (`FEAT_V`, `HEALTH_V`).** Any change to what a
  feature means re-segments stored state instead of silently comparing old to new.
- **Bias to catching failures.** A missed catastrophic failure is the dangerous
  direction; a false alarm is the merely-annoying one. Defaults favor recall, and
  hysteresis + a trend-confidence gate hold false alarms down.
- **Don't chase the unpredictable.** Abrupt faults are reported as the floor, not
  hidden by loosening thresholds until specificity collapses.
- **Advise, don't control.** Protective trips stay on the PLC; this system schedules
  maintenance, it doesn't shut machines down.

## License

MIT — see [LICENSE](LICENSE).
