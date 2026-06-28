# agent-rx

> **Close the loop on agent-triage. Don't just rank multi-agent failures, fix them and prove the fix worked.**

[![CI](https://github.com/thebharathkumar/agent-rx/actions/workflows/ci.yml/badge.svg)](https://github.com/thebharathkumar/agent-rx/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[agent-triage](https://github.com/thebharathkumar/agent-triage) answers *"what should I look at this morning?"* It ranks multi-agent trace failures by severity. The obvious next question is *"so what do I do about it, and did it actually help?"* `agent-rx` is that next step: a closed remediation loop that consumes a triage-style ranking, proposes a fix from a constrained action space, A/B-tests the fix against the unpatched system, and accepts it only on a statistically significant improvement. A learned prioritizer decides which incidents are worth a fix attempt in the first place, and it trains on the loop's own outcomes.

The headline run, fully reproducible offline with `agent-rx demo`:

```
failure rate 49.9% -> 15.0% (-70%), p < 0.0001 (significant)
4 fixes accepted, 100% targeted the correct ground-truth lever
prioritizer: learned model AUC 0.99 vs heuristic prior 0.94 on a held-out split
```

---

## Why this exists

A severity ranking is a recommendation, not a resolution. The hard parts of acting on it are the parts a dashboard skips:

1. **Which incident is even fixable?** The worst incident by severity may have no lever that moves it. Spending your next fix attempt there is wasted budget.
2. **What is the right fix?** Mapping a failure signature to a concrete, safe change.
3. **Did it work, or did you just get lucky on a noisy batch?** Without a controlled comparison and a significance test, "it looks better" is not evidence.

`agent-rx` treats remediation as a measurable control loop and puts a learned model on the one decision that is genuinely a prediction problem: expected fix value per incident.

---

## The loop

```
   ┌─────────────────────────────────────────────────────────────┐
   │  current config (starts buggy)                               │
   │        │                                                     │
   │        ▼                                                     │
   │   simulate ──► score incidents ──► PRIORITIZER (learned)     │
   │                                        │ rank by P(fix helps)│
   │                                        ▼                     │
   │                                    PROPOSER                  │
   │                              (heuristic or Claude)           │
   │                                        │ pick 1 patch        │
   │                                        ▼                     │
   │             A/B on held-out seeds: control vs patched        │
   │                                        │                     │
   │                                        ▼                     │
   │                 two-proportion z-test on failure rate        │
   │                          significant? ── no ──► reject       │
   │                                 │ yes                        │
   │                                 ▼                            │
   │                    accept patch, update config               │
   │                                 │                            │
   │             log (features -> outcome), retrain prioritizer   │
   └─────────────────────────────────────────────────────────────┘
```

The evaluation seeds are disjoint from the seeds used to surface incidents, so a fix has to generalize rather than just fit the batch it was found in. Control and treatment share identical seeds, so the before/after delta is a matched A/B, not noise.

---

## Quick start

No API key required. The whole loop runs offline against the bundled environment.

```bash
pip install -e ".[dev]"   # or: pip install -e .

agent-rx demo             # run the end-to-end loop, print a remediation report
agent-rx train            # pretrain the prioritizer, print held-out AUC / Brier
agent-rx run --max-iters 8 --output report.md
```

Use Claude to propose fixes instead of the heuristic proposer (optional):

```bash
pip install -e ".[ai]"
export ANTHROPIC_API_KEY=sk-ant-...
agent-rx run --llm
```

### Run it on real traces

`agent-rx analyze` runs the diagnose + prioritize + recommend half of the loop
over real trace files (anything in the agent-triage NDJSON schema, including
real [dungeon-traces](https://github.com/thebharathkumar/dungeon-traces)
output). It produces a ranked remediation plan:

```bash
agent-rx analyze runs/*.ndjson --top 5            # heuristic ranking
agent-rx analyze runs/ --pretrain                  # learned-prioritizer ranking
```

This is **read-only by design**: it recommends fixes ranked by expected value,
but does not A/B-test them. Verification (the significance gate) needs a
re-runnable system, which static files can't provide, so the closed loop with
acceptance runs against the bundled environment (`agent-rx demo`). A sample plan
is at [`examples/remediation-plan.md`](examples/remediation-plan.md).

### Interop with agent-triage

The environment emits NDJSON in the exact agent-triage schema, so the two tools share data:

```bash
agent-rx trace --output runs/baseline.ndjson
triage report runs/baseline.ndjson      # agent-triage ranks the same failures
```

A full generated remediation report is checked in at [`examples/remediation-report.md`](examples/remediation-report.md).

---

## The learned prioritizer (the ML)

agent-triage ranks by a hand-tuned severity formula. That is the right tool for "what is worst," but the loop needs "what is worth my next fix attempt," which is a different and genuinely predictive question. An incident can be high severity and unfixable.

So `agent-rx` learns **P(an attempted fix yields a statistically significant improvement)** per incident.

**Model.** L2-regularized logistic regression, implemented from scratch in numpy (gradient descent on standardized features). No scikit-learn, deliberately: the modeling, calibration, and evaluation are all explicit and auditable in `prioritizer.py`.

**Features** (per incident): frequency score, severity score, recovery rate, final score, confidence, run coverage, median recovery latency, and a one-hot of the failure classification.

**Labels are self-generated.** Every loop iteration tries a fix and records whether it produced a significant improvement. The system manufactures its own training set as it runs (closed-loop / online learning). For a model worth deploying on day one, `agent-rx train` pretrains offline across many randomized broken configs.

**It is evaluated honestly**, not asserted. On a held-out 30% split of self-generated labels:

| metric | learned model | heuristic prior |
|--------|--------------|-----------------|
| ROC AUC (ranking quality) | **0.99** | 0.94 |
| Brier score (calibration) | **0.038** | 0.238 (base rate) |

Until enough labels exist, the loop falls back to a transparent heuristic prior and **reports which regime it is in**, so a thin-data run is never dressed up as a learned one.

---

## What is real and what is not

This is the part most "self-healing agent" demos leave out, so it goes up front.

The multi-agent system under test is **synthetic and bundled** (`environment.py`): three agents navigating a grid, with bugs injected through known config knobs and known ground-truth fixes. That is a deliberate choice, not a shortcut. A remediation loop is only trustworthy if you can verify that the fix it applied actually addressed the bug it diagnosed, and against a black-box production system you cannot. The synthetic environment gives ground truth, which is what lets the loop report a real **targeting precision** (did it pick the correct lever) alongside the failure-rate delta.

What that means for reading the results:

- The **machinery is real and general**: scoring, the constrained action space, the A/B + significance gate, the from-scratch learned prioritizer with a held-out evaluation, the LLM proposer interface. None of it is hard-coded to the demo.
- The **numbers are from the bundled environment**, not production traffic. They demonstrate that the loop closes and that learning beats the prior on this task. They are not a claim about any real system.
- Pointing `agent-rx` at a real system means two things: write an adapter that loads your traces (the schema is agent-triage's), and replace the in-process `simulate` re-run with however you re-run your system under a candidate config. The decision logic does not change.

---

## Architecture

| Module | Responsibility |
|--------|----------------|
| `schema.py` | `TraceEvent` + `RunConfig`. agent-triage-compatible NDJSON I/O. Dataclasses, no pydantic, so the core is stdlib-only. |
| `environment.py` | Self-contained 3-agent task with injectable bugs and known ground-truth fixes. Deterministic and seedable. |
| `severity.py` | Group failures into incidents and score them. A compact reimplementation of the agent-triage scoring contract. |
| `actions.py` | The constrained action space. A fix is a typed diff against `RunConfig`, never arbitrary code. |
| `proposer.py` | Diagnose an incident, propose a patch. Heuristic (offline) and Claude-backed (live) behind one interface. |
| `prioritizer.py` | From-scratch logistic regression, feature extraction, AUC/Brier, heuristic fallback. The ML core. |
| `training.py` | Offline dataset generation + train/test evaluation of the prioritizer. |
| `stats.py` | Two-proportion z-test (normal CDF via `math.erfc`), with a tentative flag for small samples. |
| `loop.py` | Orchestrates diagnose, propose, A/B, accept/reject, log, retrain. |
| `reporter.py` | Render the markdown remediation report. |
| `analyze.py` | Read-only mode over real trace files: load, score, prioritize, recommend (no A/B). |
| `cli.py` | `demo`, `run`, `train`, `analyze`, `trace`. |

---

## Design decisions

### Why a constrained action space instead of free-form code edits?
Safety, reproducibility, and attribution. A patch is data (one entry from a registry applied as a typed diff), so it cannot execute arbitrary code, it serializes cleanly, and any measured improvement is attributable to exactly one lever. Free-form edits would make the A/B uninterpretable.

### Why gate on significance instead of "did the number go down"?
Because on a noisy batch the number goes down half the time by chance. The loop runs a matched control/treatment on held-out seeds and accepts only when a two-proportion z-test clears alpha. Small-sample wins are tagged tentative rather than trusted.

### Why a learned prioritizer when a heuristic already ranks by severity?
Severity answers "what is worst." The loop needs "what is worth a fix attempt," which depends on whether any lever can move the incident at all. That is a prediction, and the held-out AUC shows the learned model captures it better than the severity-based prior, mostly by learning to deprioritize unfixable noise like environment friction.

### Why self-generated labels?
Because the loop already performs the exact experiment that produces a label (try a fix, measure significance). Logging those outcomes turns operation into training data for free, which is the natural shape of an online control system.

### Why logistic regression from scratch instead of importing a library?
The dataset is small and the value is in showing the modeling explicitly: standardization, L2, gradient descent, plus honest evaluation with AUC and Brier against a baseline. A one-line `LogisticRegression()` import would hide exactly the part worth demonstrating. Swapping in a stronger model later is trivial; the interface is two methods.

---

## Development

```bash
pip install -e ".[dev]"
pytest --cov=agent_rx     # 30 tests
ruff check src tests
mypy src/agent_rx
```

The loop, the environment ground truth, the stats gate, and the learned-vs-heuristic comparison are all covered by tests, so CI fails if the loop stops closing or the model stops beating the prior.

---

## Relationship to agent-triage

`agent-triage` is the detector: it reads traces and ranks failures. `agent-rx` is the actuator: it reads that ranking and resolves it under a measurement gate. They share a trace schema and a scoring contract on purpose, so output from one flows into the other. Triage tells you what is on fire; rx puts it out and shows you the before-and-after.

---

## License

[MIT](LICENSE) (c) 2026 Bharath kumar R
