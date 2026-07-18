from __future__ import annotations

from .bandit import make_log
from .policy import evaluate_policies


def main() -> None:
    rows = evaluate_policies(make_log())
    print("riskbandit benchmark: conformal risk-controlled contextual bandit")
    print("policy         avg_reward  violation_rate")
    for label, (reward, violation) in rows.items():
        print(f"{label:13s} {reward:10.3f}  {violation:14.3f}")
    print(f"violation_drop {rows['reward_greedy'][1] - rows['conformal'][1]:10.3f}")


if __name__ == "__main__":
    main()
