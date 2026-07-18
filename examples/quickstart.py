from riskbandit import evaluate_policies, make_log

metrics = evaluate_policies(make_log())
print(f"conformal_violation={metrics['conformal'][1]:.3f}")
