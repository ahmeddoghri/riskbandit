from __future__ import annotations

import math

from .bandit import Event, outcome
from .linear import features, fit, pred


def fit_models(rows: list[Event]) -> dict[str, list[list[float]]]:
    reward_models = []
    cost_models = []
    for action in range(3):
        subset = [row for row in rows if row.action == action]
        x = [features(row.x1, row.x2) for row in subset]
        reward_models.append(fit(x, [row.reward for row in subset]))
        cost_models.append(fit(x, [row.cost for row in subset]))
    return {"reward": reward_models, "cost": cost_models}


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[idx]


def _calibrate(models: dict[str, list[list[float]]], rows: list[Event], alpha: float = 0.1) -> list[float]:
    residuals = [[] for _ in range(3)]
    for row in rows:
        predicted = pred(models["cost"][row.action], features(row.x1, row.x2))
        residuals[row.action].append(row.cost - predicted)
    return [_quantile(items or [0.0], 1.0 - alpha) for items in residuals]


def _choose_greedy(models: dict[str, list[list[float]]], x1: float, x2: float) -> int:
    phi = features(x1, x2)
    return max(range(3), key=lambda action: pred(models["reward"][action], phi))


def _choose_conformal(
    models: dict[str, list[list[float]]],
    qhat: list[float],
    x1: float,
    x2: float,
    budget: float,
) -> int:
    phi = features(x1, x2)
    safe = []
    for action in range(3):
        upper_cost = pred(models["cost"][action], phi) + qhat[action]
        if upper_cost <= budget:
            safe.append(action)
    if not safe:
        return 0
    return max(safe, key=lambda action: pred(models["reward"][action], phi))


def _evaluate(policy, rows: list[Event], budget: float) -> tuple[float, float]:
    rewards = []
    violations = 0
    for row in rows:
        action = policy(row.x1, row.x2)
        reward, cost = outcome(row.x1, row.x2, action)
        rewards.append(reward)
        violations += int(cost > budget)
    return sum(rewards) / len(rewards), violations / len(rows)


def evaluate_policies(rows: list[Event], budget: float = 0.30) -> dict[str, tuple[float, float]]:
    train = rows[:900]
    calibration = rows[900:1200]
    test = rows[1200:]
    models = fit_models(train)
    qhat = _calibrate(models, calibration)
    return {
        "safe_action": _evaluate(lambda x1, x2: 0, test, budget),
        "reward_greedy": _evaluate(lambda x1, x2: _choose_greedy(models, x1, x2), test, budget),
        "conformal": _evaluate(lambda x1, x2: _choose_conformal(models, qhat, x1, x2, budget), test, budget),
    }
