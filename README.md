# riskbandit

A contextual bandit benchmark where the highest reward action can break a risk
budget. The conformal policy trains reward and cost models, calibrates cost
residuals on held-out logged data, then chooses the best action whose upper cost
bound fits the budget.

![CI](https://github.com/ahmeddoghri/riskbandit/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

## Run it

```bash
git clone https://github.com/ahmeddoghri/riskbandit
cd riskbandit
pip install -e ".[dev]"
python -m riskbandit.benchmark
```

## Verified benchmark

These numbers were generated locally with `python -m riskbandit.benchmark`:

```text
policy         avg_reward  violation_rate
safe_action        0.442           0.000
reward_greedy      0.774           0.733
conformal          0.572           0.007
violation_drop      0.727
```

## Research trail

- Conformal risk control, 2024: https://arxiv.org/abs/2208.02814
- Uncertainty-aware distributional offline RL, 2024: https://arxiv.org/html/2403.17646v1
- Automatically adaptive conformal risk control, 2024: https://arxiv.org/abs/2406.17819
- PAC off-policy prediction of contextual bandits, 2025: https://arxiv.org/abs/2507.16236

## Tests

```bash
pytest -q
ruff check .
```

MIT © Ahmed Doghri
