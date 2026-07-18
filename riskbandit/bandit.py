from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    x1: float
    x2: float
    action: int
    reward: float
    cost: float


def outcome(x1: float, x2: float, action: int) -> tuple[float, float]:
    if action == 0:
        return 0.45 + 0.12 * x1, 0.08 + 0.02 * max(0.0, x2)
    if action == 1:
        return 0.78 + 0.32 * x1 + 0.12 * x2, 0.34 + 0.22 * max(0.0, x1 + x2)
    return 0.62 + 0.2 * x1 - 0.05 * x2, 0.18 + 0.08 * max(0.0, x1)


def make_log(seed: int = 27, n: int = 1500) -> list[Event]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x1 = rng.gauss(0.0, 1.0)
        x2 = rng.gauss(0.0, 1.0)
        action = rng.choices([0, 1, 2], weights=[0.45, 0.30, 0.25])[0]
        reward, cost = outcome(x1, x2, action)
        rows.append(Event(x1, x2, action, reward + rng.gauss(0, 0.05), cost + rng.gauss(0, 0.03)))
    return rows
