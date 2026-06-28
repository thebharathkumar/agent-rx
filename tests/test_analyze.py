from pathlib import Path

from agent_rx.analyze import analyze_events, load_paths
from agent_rx.environment import simulate
from agent_rx.schema import RunConfig, TraceEvent, write_ndjson


def test_load_paths_reads_file_and_directory(tmp_path: Path):
    events = simulate(RunConfig(), num_runs=3, num_turns=5, seed=1)
    f = tmp_path / "a.ndjson"
    write_ndjson(events, f)

    loaded_file, errs1 = load_paths([f])
    loaded_dir, errs2 = load_paths([tmp_path])
    assert not errs1 and not errs2
    assert len(loaded_file) == len(events)
    assert len(loaded_dir) == len(events)


def test_load_paths_reports_parse_errors(tmp_path: Path):
    bad = tmp_path / "bad.ndjson"
    bad.write_text('{"not":"valid"}\n')  # missing required fields
    events, errors = load_paths([bad])
    assert events == []
    assert errors


def test_analyze_ranks_and_recommends():
    events = simulate(RunConfig(), num_runs=12, num_turns=20, seed=42)
    report = analyze_events(events, top=3)
    assert report.n_events == len(events)
    assert len(report.recommendations) == 3
    assert report.regime == "heuristic"
    # top recommendation should carry an actionable fix
    assert report.recommendations[0].proposal is not None


def test_environment_constraint_gets_no_action():
    # an incident that only self-resolving friction would produce
    events = [
        TraceEvent(event_id=f"r-{t}-nav", run_id="r", turn=t, agent_id="nav",
                   tool_name="move", action_succeeded=False,
                   failure_classification="environment_constraint")
        for t in range(3)
    ]
    report = analyze_events(events, top=1)
    assert report.recommendations[0].proposal is None
