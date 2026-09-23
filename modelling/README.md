# Modelling

The presentation proposes LightGBM to study non-linear prepayment behaviour, refinancing incentives and rate-cycle effects. No fitted model or benchmark result is included yet.

## Decisions before training

- Specify whether the target is a loan-level prepayment event in the next month or another horizon. Resolve treatment of scheduled maturity, defaults, repurchases and other exits using the vendor guide.
- Define the population at risk and right censoring. A missing next-month row is not automatically a negative label.
- Use chronological train/validation/test windows and purge overlapping label horizons. Fit preprocessing only on training data; use validation for tuning and freeze test data.
- For forecasting existing loans, repeated loans across time splits can be intentional; document it. Add a loan-disjoint test if making claims about unseen borrowers.
- Evaluate by rate regime and cohort; incomplete source cohorts limit generalization.

## Models and metrics

Begin with an unconditional event-rate baseline and regularized logistic regression, then LightGBM. Pin model dependencies when the implementation is introduced.
Report sample/event counts, PR-AUC, ROC-AUC where both classes exist, log loss,
Brier score, calibration and cohort/regime breakdowns. Choose thresholds on validation
only. Explain predictions with held-out feature importance/SHAP where appropriate;
importance is not causal evidence.

## Valuation extension

Separate loan-level event probabilities from balance-weighted prepayment measures.
Define how predictions produce pool SMM and CPR before using them in cash-flow
scenarios. Document scheduled amortization, defaults/recoveries, discount curves and
rate paths. Benchmark duration/yield changes; do not present classification accuracy
as evidence of pricing accuracy.

See `experiment.example.json` for the fields each experiment must record. Its null
values are intentional decisions for the team, not runnable training settings.
