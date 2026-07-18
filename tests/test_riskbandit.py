from riskbandit import evaluate_policies, fit_models, make_log


def test_log_is_deterministic() -> None:
    assert make_log(seed=1, n=5) == make_log(seed=1, n=5)


def test_models_fit_three_actions() -> None:
    models = fit_models(make_log(n=300))
    assert len(models["reward"]) == 3
    assert len(models["cost"]) == 3


def test_conformal_reduces_violations() -> None:
    rows = evaluate_policies(make_log())
    assert rows["conformal"][1] < rows["reward_greedy"][1]
    assert rows["conformal"][0] > rows["safe_action"][0]
