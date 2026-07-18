# riskbandit

The reward-greedy policy does exactly what you told it to do and exactly what you were afraid it would do. riskbandit builds the guardrail that stops it.

![CI](https://github.com/ahmeddoghri/riskbandit/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

Optimize a contextual bandit for reward alone and it will happily find the
action that blows through your risk budget, because nobody told it a budget
existed. riskbandit trains separate reward and cost models on synthetic
logged data, calibrates the cost residuals with conformal prediction on a
held-out split, and at decision time only considers actions whose upper cost
bound actually fits inside the budget before picking the best reward among
the survivors. Safety becomes a selection rule baked into the policy instead
of a paragraph in the README that nobody enforces.

## Run it

```bash
git clone https://github.com/ahmeddoghri/riskbandit
cd riskbandit
pip install -e ".[dev]"
python -m riskbandit.benchmark
```

## Verified benchmark

Generated locally with `python -m riskbandit.benchmark`:

```text
policy         avg_reward  violation_rate
safe_action        0.442           0.000
reward_greedy      0.774           0.733
conformal          0.572           0.007
violation_drop      0.727
```

Reward-greedy racks up 0.774 average reward and blows the budget on 73.3% of
decisions, which is what "optimize for reward, don't ask questions" looks
like in practice. The conformal policy gives up some reward, landing at
0.572, but drops the violation rate to 0.7%, a 0.727 point collapse. That
trade is the whole product.

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
