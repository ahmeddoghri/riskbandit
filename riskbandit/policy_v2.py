from __future__ import annotations

import math

from .bandit import Event
from .linear import features, pred
from .policy import _choose_conformal, _evaluate, fit_models


def _quantile_v2(values: list[float], q: float) -> float:
    """Standard split-conformal quantile with the finite-sample (n+1) correction.

    policy._quantile picks index ceil(q*n)-1, which omits the "+1" that split
    conformal prediction requires for its marginal coverage guarantee
    (Vovk et al.; Lei et al. 2018): the correct order statistic is the
    ceil((n+1)*q)-th smallest, not the ceil(n*q)-th. Without it, the upper
    cost bound undercovers the nominal target by roughly 1/(2n).
    """
    values = sorted(values)
    n = len(values)
    k = math.ceil((n + 1) * q)
    idx = min(n - 1, max(0, k - 1))
    return values[idx]


def _calibrate_v2(models: dict[str, list[list[float]]], rows: list[Event], alpha: float = 0.1) -> list[float]:
    residuals = [[] for _ in range(3)]
    for row in rows:
        predicted = pred(models["cost"][row.action], features(row.x1, row.x2))
        residuals[row.action].append(row.cost - predicted)
    return [_quantile_v2(items or [0.0], 1.0 - alpha) for items in residuals]


def evaluate_policies_v2(rows: list[Event], budget: float = 0.30) -> dict[str, tuple[float, float]]:
    train = rows[:900]
    calibration = rows[900:1200]
    test = rows[1200:]
    models = fit_models(train)
    qhat = _calibrate_v2(models, calibration)
    return {
        "safe_action": _evaluate(lambda x1, x2: 0, test, budget),
        "reward_greedy": _evaluate(
            lambda x1, x2: max(range(3), key=lambda a: pred(models["reward"][a], features(x1, x2))), test, budget
        ),
        "conformal_v2": _evaluate(lambda x1, x2: _choose_conformal(models, qhat, x1, x2, budget), test, budget),
    }
