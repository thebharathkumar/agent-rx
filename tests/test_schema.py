import json

from agent_rx.schema import RunConfig, TraceEvent


def test_trace_event_ndjson_roundtrip():
    ev = TraceEvent(
        event_id="r-1-a", run_id="r", turn=1, agent_id="planner",
        tool_name="plan", action_succeeded=False,
        failure_classification="coordination_failure", divergence_fields=("target",),
        latency_ms=120,
    )
    obj = json.loads(ev.to_json_line())
    back = TraceEvent.from_obj(obj)
    assert back == ev


def test_from_obj_coerces_null_classification():
    obj = {"event_id": "x", "run_id": "r", "turn": 0, "agent_id": "a",
           "action_taken": {"tool_name": "t"}, "action_succeeded": True,
           "failure_classification": "null"}
    assert TraceEvent.from_obj(obj).failure_classification is None


def test_runconfig_with_changes_is_immutable_copy():
    base = RunConfig()
    changed = base.with_changes(navigator_retries=3)
    assert base.navigator_retries == 0
    assert changed.navigator_retries == 3
