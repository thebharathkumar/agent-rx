from agent_rx.actions import action_by_id, all_actions, candidate_patches_for
from agent_rx.schema import RunConfig


def test_each_classification_has_a_candidate():
    for cls in ("coordination_failure", "agent_error", "information_lag"):
        assert candidate_patches_for(cls), cls


def test_add_navigator_retry_changes_config():
    patch = action_by_id("add_navigator_retry")
    cfg = RunConfig(navigator_retries=0)
    assert patch.changed(cfg)
    assert patch.apply(cfg).navigator_retries == 1


def test_acknowledge_environment_is_inert():
    patch = action_by_id("acknowledge_environment")
    cfg = RunConfig()
    assert not patch.changed(cfg)


def test_timeout_and_consistency_are_capped():
    cfg = RunConfig(timeout_ms=2000, planner_consistency=0.9)
    assert action_by_id("raise_timeout").apply(cfg).timeout_ms <= 3000
    assert action_by_id("raise_planner_consistency").apply(cfg).planner_consistency <= 0.95


def test_registry_round_trip():
    for patch in all_actions():
        assert action_by_id(patch.action_id) is not None
