from agent_rx.schema import TraceEvent
from agent_rx.severity import score_events


def _ev(run, turn, agent, tool, ok, cls=None):
    return TraceEvent(event_id=f"{run}-{turn}-{agent}", run_id=run, turn=turn,
                      agent_id=agent, tool_name=tool, action_succeeded=ok,
                      failure_classification=cls)


def test_grouping_and_ordering_by_final_score():
    events = [
        _ev("r", 0, "planner", "plan", False, "coordination_failure"),
        _ev("r", 1, "planner", "plan", False, "coordination_failure"),
        _ev("r", 0, "nav", "move", False, "environment_constraint"),
        _ev("r", 1, "nav", "move", True),
    ]
    scored = score_events(events)
    assert scored[0].classification == "coordination_failure"
    # coordination_failure (unrecovered, high weight) outranks environment_constraint
    assert scored[0].final_score > scored[-1].final_score


def test_recovery_multiplier_applies_when_no_recovery():
    # never recovers -> severity multiplied
    no_rec = score_events([
        _ev("r", 0, "a", "t", False, "agent_error"),
        _ev("r", 1, "a", "t", False, "agent_error"),
    ])[0]
    rec = score_events([
        _ev("s", 0, "a", "t", False, "agent_error"),
        _ev("s", 1, "a", "t", True),
    ])[0]
    assert no_rec.recovery_rate == 0.0
    assert rec.recovery_rate > 0.0
    assert no_rec.severity_score > rec.severity_score
