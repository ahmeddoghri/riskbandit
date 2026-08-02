from __future__ import annotations

# TUNING_SEEDS: used to characterize and tune the conformal-quantile fix.
# HOLDOUT_SEEDS: disjoint, evaluated exactly once after the fix was finalized.
TUNING_SEEDS = list(range(30000, 30020))
HOLDOUT_SEEDS = list(range(40000, 40015))
