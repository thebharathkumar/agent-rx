# agent-rx remediation report

**Outcome:** failure rate 49.9% -> 15.0% (-70%), p=0.0000 (**significant**).

**Fixes accepted:** 4 (raise_planner_consistency, add_navigator_retry, raise_timeout, raise_planner_consistency).  **Targeting precision:** 100% of accepted fixes hit the correct ground-truth lever.

## Config diff (baseline -> remediated)

| knob | baseline | remediated |
|------|----------|------------|
| `planner_consistency` | 0.55 | 0.95  ←changed |
| `navigator_retries` | 0 | 1  ←changed |
| `timeout_ms` | 800 | 1600  ←changed |
| `friction_rate` | 0.2 | 0.2 |

## Iterations

| # | incident | action | regime | result | accepted | right lever |
|---|----------|--------|--------|--------|----------|-------------|
| 1 | [planner] plan / coordination_failure / target | `raise_planner_consistency` | learned | 49.9% -> 38.9% (-22%, improved, p=0.000, significant) | yes | yes |
| 2 | [navigator] move / agent_error / no-divergence | `add_navigator_retry` | learned | 38.9% -> 27.4% (-30%, improved, p=0.000, significant) | yes | yes |
| 3 | [scout] observe / information_lag / belief_age | `raise_timeout` | learned | 27.4% -> 17.3% (-37%, improved, p=0.000, significant) | yes | yes |
| 4 | [planner] plan / coordination_failure / target | `raise_planner_consistency` | learned | 17.3% -> 15.0% (-13%, improved, p=0.028, significant) | yes | yes |
| 5 | [scout] observe / information_lag / belief_age | `raise_timeout` | learned | 15.0% -> 15.0% (-0%, improved, p=0.968, not significant) | no | yes |
| 6 | [navigator] move / agent_error / no-divergence | `add_navigator_retry` | learned | 15.0% -> 15.0% (+0%, regressed, p=1.000, not significant) | no | yes |

## Prioritizer

Final regime: **learned**, trained on 246 self-generated outcome labels.

Offline ranking quality (AUC): learned **0.99** vs heuristic prior 0.96, Brier 0.029.
