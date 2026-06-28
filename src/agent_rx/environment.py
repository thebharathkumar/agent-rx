"""environment.py - a self-contained multi-agent system with injectable bugs.

Why this exists: a remediation loop is only trustworthy if you can verify
that the fix it applied actually addressed the bug it diagnosed. Against a
black-box production system you cannot. So agent-rx ships its own small
multi-agent task with *known* failure modes and *known* ground-truth fixes.
Every knob the remediation agent can turn (RunConfig) maps to a failure
classification, so we can check both halves of the loop: did it pick the
right lever, and did the trace measurably improve.

The task: three agents cooperate to navigate a grid.
  - planner   emits a target cell           -> coordination_failure when inconsistent
  - navigator moves toward the target       -> agent_error on un-retried tool faults
                                            -> environment_constraint on wall bumps
  - scout     refreshes shared belief state -> information_lag when its budget is tight

Ground-truth fixes (what the remediation agent should discover):
  coordination_failure  <- raise planner_consistency
  agent_error           <- raise navigator_retries (0 -> >=1)
  information_lag        <- raise timeout_ms
  environment_constraint<- (none) self-resolving noise; must be left alone
"""

from __future__ import annotations

import random

from agent_rx.schema import RunConfig, TraceEvent

# Recovery window shared with the scorer: a failure "recovered" if the same
# agent succeeds within this many turns.
RECOVERY_WINDOW = 3

# Which config lever fixes which classification. Returned as (field, "up").
# Used by evaluation to score whether the agent diagnosed correctly. None
# means "no fix should be attempted" (the failure is irreducible noise).
GROUND_TRUTH_FIX: dict[str, str | None] = {
    "coordination_failure": "planner_consistency",
    "agent_error": "navigator_retries",
    "information_lag": "timeout_ms",
    "environment_constraint": None,
}


def _coord_recovery_p(consistency: float) -> float:
    """Per-turn chance a stuck planner re-syncs. Rises with consistency."""
    return 0.15 + 0.60 * consistency


def _lag_rate(timeout_ms: int) -> float:
    """Chance the scout acts on stale belief. Falls as the budget grows."""
    if timeout_ms >= 1500:
        return 0.04
    if timeout_ms >= 1000:
        return 0.10
    return 0.30


def _simulate_planner(
    rng: random.Random, run_id: str, num_turns: int, cfg: RunConfig
) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    stuck = False
    for turn in range(num_turns):
        if not stuck:
            if rng.random() > cfg.planner_consistency:
                stuck = True
                events.append(_ev(run_id, turn, "planner", "plan", False,
                                  "coordination_failure", ("target",)))
            else:
                events.append(_ev(run_id, turn, "planner", "plan", True))
        else:
            if rng.random() < _coord_recovery_p(cfg.planner_consistency):
                stuck = False
                events.append(_ev(run_id, turn, "planner", "plan", True))
            else:
                events.append(_ev(run_id, turn, "planner", "plan", False,
                                  "coordination_failure", ("target",)))
    return events


def _simulate_navigator(
    rng: random.Random, run_id: str, num_turns: int, cfg: RunConfig
) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    retry_budget = 0
    in_fault = False
    for turn in range(num_turns):
        # Environment friction is independent background noise: a wall bump
        # that self-resolves on the next move. Low severity, high recovery.
        if rng.random() < cfg.friction_rate:
            events.append(_ev(run_id, turn, "navigator", "move", False,
                              "environment_constraint", ()))
            continue

        if not in_fault:
            if rng.random() < 0.18:  # transient tool fault
                in_fault = True
                retry_budget = cfg.navigator_retries
                events.append(_ev(run_id, turn, "navigator", "move", False,
                                  "agent_error", ()))
            else:
                events.append(_ev(run_id, turn, "navigator", "move", True))
        else:
            if retry_budget > 0:
                retry_budget -= 1
                in_fault = False
                events.append(_ev(run_id, turn, "navigator", "move", True))
            elif rng.random() < 0.10:  # rare unaided recovery
                in_fault = False
                events.append(_ev(run_id, turn, "navigator", "move", True))
            else:
                events.append(_ev(run_id, turn, "navigator", "move", False,
                                  "agent_error", ()))
    return events


def _simulate_scout(
    rng: random.Random, run_id: str, num_turns: int, cfg: RunConfig
) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    lag_rate = _lag_rate(cfg.timeout_ms)
    stale = False
    for turn in range(num_turns):
        if not stale:
            if rng.random() < lag_rate:
                stale = True
                events.append(_ev(run_id, turn, "scout", "observe", False,
                                  "information_lag", ("belief_age",)))
            else:
                events.append(_ev(run_id, turn, "scout", "observe", True))
        else:
            # Belief refreshes faster with a larger timeout budget.
            refresh_p = 0.35 + min(0.45, cfg.timeout_ms / 4000)
            if rng.random() < refresh_p:
                stale = False
                events.append(_ev(run_id, turn, "scout", "observe", True))
            else:
                events.append(_ev(run_id, turn, "scout", "observe", False,
                                  "information_lag", ("belief_age",)))
    return events


def _ev(
    run_id: str,
    turn: int,
    agent: str,
    tool: str,
    ok: bool,
    classification: str | None = None,
    divergence: tuple[str, ...] = (),
) -> TraceEvent:
    return TraceEvent(
        event_id=f"{run_id}-{turn}-{agent}",
        run_id=run_id,
        turn=turn,
        agent_id=agent,
        tool_name=tool,
        action_succeeded=ok,
        failure_classification=classification,
        divergence_fields=divergence,
        latency_ms=40 if ok else 120,
    )


def simulate(
    config: RunConfig,
    num_runs: int = 12,
    num_turns: int = 20,
    seed: int = 42,
) -> list[TraceEvent]:
    """Run the multi-agent task and return a flat list of trace events.

    Runs are seeded deterministically as ``seed + run_index`` so a control
    config and a treatment config can be compared on identical randomness,
    which is what makes the before/after delta a clean A/B rather than noise.
    """
    events: list[TraceEvent] = []
    for r in range(num_runs):
        run_id = f"run{r:03d}"
        rng = random.Random(seed + r)
        events.extend(_simulate_planner(rng, run_id, num_turns, config))
        events.extend(_simulate_navigator(rng, run_id, num_turns, config))
        events.extend(_simulate_scout(rng, run_id, num_turns, config))
    return events
