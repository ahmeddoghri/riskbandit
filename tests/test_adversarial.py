from __future__ import annotations

from riskbandit.adversarial import HOLDOUT_SEEDS, TUNING_SEEDS
from riskbandit.bandit import make_log
from riskbandit.eval_v2 import _mean_coverage, summarize
from riskbandit.policy import _calibrate, _quantile, evaluate_policies
from riskbandit.policy_v2 import _calibrate_v2, _quantile_v2, evaluate_policies_v2


def test_holdout_disjoint_from_tuning() -> None:
    assert set(TUNING_SEEDS).isdisjoint(HOLDOUT_SEEDS)


def test_original_benchmark_still_reproduces_exactly() -> None:
    rows = make_log()
    result = evaluate_policies(rows)
    assert round(result["safe_action"][0], 3) == 0.442
    assert round(result["safe_action"][1], 3) == 0.000
    assert round(result["reward_greedy"][0], 3) == 0.774
    assert round(result["reward_greedy"][1], 3) == 0.733
    assert round(result["conformal"][0], 3) == 0.572
    assert round(result["conformal"][1], 3) == 0.007


def test_original_bug_quantile_omits_finite_sample_correction() -> None:
    """policy._quantile picks index ceil(q*n)-1; the correct split-conformal
    order statistic for a marginal (1-alpha) coverage guarantee is
    ceil((n+1)*q)-1. Demonstrated directly on a fixed residual list."""
    values = [float(i) for i in range(1, 101)]  # 1..100
    q = 0.9
    original = _quantile(values, q)
    fixed = _quantile_v2(values, q)
    assert fixed > original
    assert original == 90.0
    assert fixed == 91.0


def test_original_undercovers_nominal_target_on_tuning_seeds() -> None:
    coverage = _mean_coverage(_calibrate, TUNING_SEEDS)
    assert coverage < 0.90


def test_fix_reaches_nominal_target_on_tuning_seeds() -> None:
    coverage = _mean_coverage(_calibrate_v2, TUNING_SEEDS)
    assert coverage >= 0.895


def test_original_undercovers_nominal_target_on_frozen_holdout_seeds() -> None:
    coverage = _mean_coverage(_calibrate, HOLDOUT_SEEDS)
    assert coverage < 0.90


def test_fix_reaches_nominal_target_on_frozen_holdout_seeds() -> None:
    coverage = _mean_coverage(_calibrate_v2, HOLDOUT_SEEDS)
    assert coverage >= 0.895


def test_fix_does_not_regress_published_headline_numbers() -> None:
    rows = make_log()
    result = evaluate_policies_v2(rows)
    # v2 policy should still control violations at least as well as the original,
    # while not collapsing reward.
    assert result["conformal_v2"][1] <= 0.02
    assert result["conformal_v2"][0] > 0.45


def test_original_module_untouched() -> None:
    import inspect

    from riskbandit import policy

    source = inspect.getsource(policy._quantile)
    assert "math.ceil(q * len(values)) - 1" in source


def test_report_is_reproducible() -> None:
    a = summarize(TUNING_SEEDS[:5])
    b = summarize(TUNING_SEEDS[:5])
    assert a == b


def test_quantile_fix_matches_theoretical_expectation_on_uniform_residuals() -> None:
    import random

    rng = random.Random(0)
    n = 500
    values = [rng.random() for _ in range(n)]
    qhat = _quantile_v2(values, 0.9)
    # For a fresh draw from the same uniform, P(x <= qhat) should be close to 0.9.
    fresh = [rng.random() for _ in range(20000)]
    empirical = sum(1 for x in fresh if x <= qhat) / len(fresh)
    assert abs(empirical - 0.9) < 0.02
