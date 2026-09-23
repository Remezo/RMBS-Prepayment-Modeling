# Cleaning and feature engineering

`profile_data.py` is a runnable, read-only initial audit. It does not silently impute, drop rows, or reinterpret source codes.

## Proposed workflow

1. Reconcile selected files and row counts with the partial manifest.
2. Validate `loan_identifier` and `monthly_reporting_period`; check duplicate loan-month keys before joining.
3. Inspect missing values and vendor sentinel codes using the release-specific Freddie Mac guide. Preserve leading zeros and distinguish missing codes from true zero.
4. Investigate the raw `.` numeric value that stopped the historical conversion. The accepted snapshot excludes that failed source file; do not claim this issue was repaired.
5. Ensure one origination record per loan; report unmatched performance loans and avoid many-to-many joins.
6. Define prepayment events, defaults, other liquidations, censoring and observation eligibility with reference to documented zero-balance codes.
7. Create only information available at prediction time: lagged delinquency, loan age, rate incentive and borrower/property features. Labels and future balances cannot be predictors.
8. Merge macro values using their publication/availability date, not a future revision or period-end value unavailable at prediction time.

## Output contract to approve

Local cleaned table: one row per eligible loan and prediction month. Required fields:
`loan_identifier`, `prediction_month`, `cohort_year`, `label`, `label_observed`,
`label_horizon_months`, plus versioned features. Date and target definitions must be
written before implementation. Keep raw values recoverable and log every exclusion.

Deliver a quality report containing input/output counts, duplicate counts, missingness,
join match rate, exclusions by reason, and reporting-period coverage. Full cleaning
and feature pipelines are not implemented yet.
