# agent-rx remediation plan

Analyzed **720** events, found **4** incident patterns at a **46.7%** failure rate. Ranked by the **heuristic** prioritizer (expected fix value).

> Read-only mode: these are recommended fixes ranked by priority. Verification (the A/B significance gate) requires re-running the system under each patch, which static trace files can't provide.

| # | priority | incident | recovery | confidence | recommended fix |
|---|----------|----------|----------|------------|-----------------|
| 1 | 0.66 | [navigator] move / agent_error / no-divergence | 14% | high | `add_navigator_retry` |
| 2 | 0.56 | [planner] plan / coordination_failure / target | 84% | high | `raise_planner_consistency` |
| 3 | 0.34 | [scout] observe / information_lag / belief_age | 84% | high | `raise_timeout` |
| 4 | 0.32 | [navigator] move / environment_constraint / no-divergence | 40% | high | `no action (self-resolving)` |

**#1 [navigator] move / agent_error / no-divergence** — agent_error on [navigator] move with 14% recovery over 102 events; Add one navigator retry so transient tool faults recover in-window.

**#2 [planner] plan / coordination_failure / target** — coordination_failure on [planner] plan with 84% recovery over 100 events; Raise planner_consistency by +0.35 (cap 0.95) to reduce target desync.

**#3 [scout] observe / information_lag / belief_age** — information_lag on [scout] observe with 84% recovery over 81 events; Double the timeout budget (cap 3000ms) so belief state refreshes in time.
