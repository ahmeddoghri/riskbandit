from __future__ import annotations

from .adversarial import HOLDOUT_SEEDS, TUNING_SEEDS
from .bandit import make_log
from .linear import features, pred
from .policy import _calibrate, fit_models
from .policy_v2 import _calibrate_v2

FRESH_N = 4000


def _mean_coverage(calibrate_fn, seeds: list[int], alpha: float = 0.1) -> float:
    covs = []
    for seed in seeds:
        rows = make_log(seed=seed, n=900 + 300 + FRESH_N)
        train, calibration, fresh = rows[:900], rows[900:1200], rows[1200:]
        models = fit_models(train)
        qhat = calibrate_fn(models, calibration, alpha)
        hits = 0
        for row in fresh:
            upper = pred(models["cost"][row.action], features(row.x1, row.x2)) + qhat[row.action]
            hits += int(row.cost <= upper)
        covs.append(hits / len(fresh))
    return sum(covs) / len(covs)


def summarize(seeds: list[int]) -> dict[str, float]:
    return {
        "n": len(seeds),
        "original_coverage": _mean_coverage(_calibrate, seeds),
        "fixed_coverage": _mean_coverage(_calibrate_v2, seeds),
    }


def main() -> None:
    print("riskbandit eval_v2: conformal quantile finite-sample correction")
    print("target marginal coverage: 0.900 (alpha=0.1)")
    print()
    print("seeds        n    original coverage   fixed coverage")
    for label, seeds in (("tuning", TUNING_SEEDS), ("holdout", HOLDOUT_SEEDS)):
        result = summarize(seeds)
        print(f"{label:9s}{result['n']:5d}    {result['original_coverage']:17.4f}   {result['fixed_coverage']:14.4f}")


if __name__ == "__main__":
    main()
