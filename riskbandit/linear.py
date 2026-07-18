from __future__ import annotations


def features(x1: float, x2: float) -> list[float]:
    return [1.0, x1, x2, x1 * x2]


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col] or 1e-12
        aug[col] = [value / div for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * ref for value, ref in zip(aug[row], aug[col])]
    return [row[-1] for row in aug]


def fit(x: list[list[float]], y: list[float], lam: float = 1e-3) -> list[float]:
    p = len(x[0])
    a = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for row, target in zip(x, y):
        for i in range(p):
            b[i] += row[i] * target
            for j in range(p):
                a[i][j] += row[i] * row[j]
    for i in range(p):
        a[i][i] += lam
    return solve(a, b)


def pred(weights: list[float], row: list[float]) -> float:
    return sum(a * b for a, b in zip(weights, row))
