from agent_rx.loop import run_loop
from agent_rx.training import generate_dataset, train_and_evaluate


def test_loop_significantly_reduces_failure_rate():
    r = run_loop()
    assert r.overall.significant
    assert r.overall.treatment_rate < r.overall.control_rate
    assert r.accepted  # at least one fix accepted


def test_accepted_fixes_target_correct_ground_truth_lever():
    r = run_loop()
    assert r.precision == 1.0


def test_loop_converges_without_thrashing():
    # Once fixed, the loop should stop accepting no-op patches.
    r = run_loop(max_iters=12)
    # final failure rate is low and stable
    assert r.overall.treatment_rate < 0.25


def test_pretrained_prioritizer_runs_in_learned_regime():
    prio, report = train_and_evaluate(generate_dataset(30, seed=7), seed=0)
    assert report.auc_learned >= report.auc_heuristic - 0.05
    r = run_loop(prioritizer=prio)
    assert r.final_regime == "learned"
    assert r.overall.significant
