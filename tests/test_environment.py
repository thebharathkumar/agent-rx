from agent_rx.environment import simulate
from agent_rx.schema import RunConfig
from agent_rx.severity import failure_rate, score_events


def _count(events, classification):
    return sum(1 for e in events if e.failure_classification == classification)


def test_simulation_is_deterministic():
    a = simulate(RunConfig(), seed=42)
    b = simulate(RunConfig(), seed=42)
    assert a == b


def test_raising_consistency_reduces_coordination_failures():
    buggy = simulate(RunConfig(planner_consistency=0.5), seed=1)
    fixed = simulate(RunConfig(planner_consistency=0.95), seed=1)
    assert _count(fixed, "coordination_failure") < _count(buggy, "coordination_failure")


def test_navigator_retries_improve_agent_error_recovery():
    no_retry = score_events(simulate(RunConfig(navigator_retries=0), seed=3))
    retry = score_events(simulate(RunConfig(navigator_retries=2), seed=3))
    rec_no = [s for s in no_retry if s.classification == "agent_error"][0].recovery_rate
    rec_yes = [s for s in retry if s.classification == "agent_error"][0].recovery_rate
    assert rec_yes > rec_no


def test_overall_failure_rate_drops_when_all_knobs_fixed():
    buggy = failure_rate(simulate(RunConfig(), seed=7))
    good = failure_rate(simulate(
        RunConfig(planner_consistency=0.95, navigator_retries=2, timeout_ms=2000), seed=7))
    assert good < buggy
