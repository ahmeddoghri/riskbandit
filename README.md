# riskbandit

The reward-greedy policy does exactly what you told it to do and exactly what you were afraid it would do. riskbandit builds the guardrail that stops it.

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

**Update:** the conformal quantile itself didn't hit its stated 90% coverage
target. `_quantile` picked the `ceil(q*n)`-th order statistic instead of the
`ceil((n+1)*q)`-th that split conformal prediction requires for the marginal
guarantee, undercovering by roughly 1-1.5 points across 20 tuning and 15
disjoint holdout seeds (large fresh test sets, evaluated once). The safety
mechanism the whole repo is built around wasn't delivering the risk bound it
claimed. `policy_v2._quantile_v2` adds the missing `+1` correction and lands
right at the 90% target on both. `python -m riskbandit.eval_v2` runs the
comparison. Headline numbers above are untouched. Details below.

## The 90% coverage target wasn't actually 90%

`policy._quantile(values, q)` computes `values[ceil(q * n) - 1]`. The
correct split-conformal order statistic for a marginal `(1-alpha)` coverage
guarantee (Vovk et al.; Lei et al. 2018) is `values[ceil((n+1) * q) - 1]` --
one index further out. Missing that `+1` undercovers by roughly `1/(2n)`,
worse for the smaller per-action calibration slices this repo actually uses.

```bash
python -m riskbandit.eval_v2
```
```
seeds        n    original coverage   fixed coverage
tuning      20               0.8924           0.9015
holdout     15               0.8882           0.9014
```

Both sweeps draw a fresh 900/300/4000 train/calibration/test split per seed
and measure whether the realized cost actually falls under the calibrated
upper bound, at the nominal `alpha=0.1` (90% target). The original formula
lands at 89.2% (tuning) / 88.8% (holdout, evaluated once) -- consistently
short of the guarantee it's supposed to deliver. `policy_v2._calibrate_v2`
fixes the index and lands at 90.1% on both. `bandit.py`/`policy.py` are
untouched, and the published benchmark numbers above still reproduce
exactly; `policy_v2` is opt-in.

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
