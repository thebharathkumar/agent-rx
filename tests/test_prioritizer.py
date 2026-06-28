from agent_rx.prioritizer import (
    FEATURE_NAMES,
    Prioritizer,
    _LogReg,
    auc_score,
    extract_features,
    heuristic_score,
)
from agent_rx.schema import TraceEvent
from agent_rx.severity import score_events


def _incident(cls="coordination_failure"):
    events = [
        TraceEvent(event_id=f"r-{t}-a", run_id="r", turn=t, agent_id="a",
                   tool_name="t", action_succeeded=False, failure_classification=cls)
        for t in range(4)
    ]
    return score_events(events)[0]


def test_feature_vector_length_matches_names():
    inc = _incident()
    assert len(extract_features(inc)) == len(FEATURE_NAMES)


def test_heuristic_score_bounded():
    s = heuristic_score(_incident())
    assert 0.0 <= s <= 1.0


def test_auc_perfect_separation():
    assert auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc_score([0, 1], [0.5, 0.5]) == 0.5


def test_logreg_learns_separable_data():
    X = [[0.0], [0.1], [0.2], [0.9], [1.0], [0.8]]
    y = [0, 0, 0, 1, 1, 1]
    m = _LogReg(epochs=400)
    m.fit(X, y)
    probs = m.predict_proba(X)
    assert auc_score(y, probs) == 1.0


def test_prioritizer_falls_back_to_heuristic_until_trained():
    p = Prioritizer()
    assert p.regime == "heuristic"
    inc = _incident()
    assert p.score(inc) == heuristic_score(inc)
