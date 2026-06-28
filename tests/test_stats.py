from agent_rx.stats import two_proportion_test


def test_detects_significant_improvement():
    res = two_proportion_test(200, 400, 80, 400)
    assert res.abs_delta < 0
    assert res.significant
    assert not res.tentative


def test_no_difference_is_not_significant():
    res = two_proportion_test(100, 400, 100, 400)
    assert not res.significant
    assert abs(res.abs_delta) < 1e-9


def test_small_samples_flagged_tentative():
    res = two_proportion_test(2, 20, 1, 20)
    assert res.tentative
