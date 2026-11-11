#Light GBM Model 

## Overview
This folder contains all files related to the training, validation, and testing of the LightGBM model. 

**Pulling data into a pandas is avoided when possible to mimimize memory usage.**

This pipeline was created based on discussion from the 10-23-2026 team meeting. 

The pipeline takes a sample of the training and validation data and trains multiple LightGBM models with different combinations of training parameters. The best model is selected based on performance on the validation sample data. The selected model is then used to predict prepayment behavior for every loan in the test data files. The predictions are exported to parquet and evaluated for correctness. 

## Check input files

From `lgbm_model`, audit the local `data/loan_month/loan_month_perf_{year}.parquet`
files before training:

```sh
python check_input_data.py --output data/input_validation.json
```

The checker defaults to 2016–2026; use `--first-year` and `--last-year` to
select a different range. It scans every row for numeric missing/infinite
values, target counts, and reporting dates, checks required feature types,
and exercises model preparation on a small batch. It leaves inputs unchanged
and exits with status 1 if a required file or data check fails. Missing numeric
values are reported as warnings because LightGBM supports them. Model preparation
maps `zero_balance_code = 1` to label 1 and codes `0, 2, 3, 9, 15, 16, 96` to
label 0 for training, validation, and testing. Null and unknown codes are rejected.
Code 1 includes scheduled maturity. The original Parquet values remain unchanged.
The JSON report lists optional feature compatibility and the limits of the audit.

For these filenames, running the model from `lgbm_model` requires
`python main.py --first-year 2021 --last-year 2026 --file-template 'data/loan_month/loan_month_perf_{year}.parquet'`.
The model defaults to 2021–2026, inclusive.
From the repository root, use `python -m lgbm_model.main --first-year 2021 --last-year 2026 --file-template
'lgbm_model/data/loan_month/loan_month_perf_{year}.parquet'` instead.

To compare prepayment rates across years without training a model:

```sh
python yearly_prepayment_rates.py --first-year 2016 --last-year 2025 --output data/yearly_prepayment_rates.csv
```

This scans all rows and groups by the reporting year in `period`. The percentage
is rows with `zero_balance_code = 1` divided by all loan-month rows in that year,
multiplied by 100. It is a pooled monthly event rate, not an annualized CPR or
the percentage of distinct loans prepaid during the year. Inputs are unchanged.


## User Adjustments

See [Categorical conversion](CATEGORICAL_CONVERSION.md) for the reviewed columns,
shared fixed encoder, optional feature flags, yearly filename configuration,
memory tradeoffs, and verification. New categorical and timestamp features are
explicit opt-ins. Run from the repository root with `python -m lgbm_model.main`.

`main()` creates one encoder before the rolling-window loop. Its categorical
vocabularies are predefined in `feature_encoder.py`, so categories keep the
same codes across all windows even when a training sample misses a category.
No encoder fitting or full-data vocabulary scan is needed. Each model saves
a copy of the shared `encoder.json`; older encoder files can still be loaded.

To update the features that are used by the LightGBM model ("**X**"), see the "**prepare_data_for_model**" function in "**import_data.py**" 

To update the ten initial parameter configurations evaluated on the validation set, see "**INITIAL_PARAMETER_GRID**" in "**parameter_grid.py**".

Candidate models are ranked by validation monthly rate error from
`monthly_rate_error()` in `calculate_error.py`, with log loss breaking ties.
For each month this compares `AVG(actual)` with `AVG(predicted)`, then averages
the absolute gaps equally across months. It uses probabilities, without a
classification threshold or annualization; an error of `0.001` means 0.1
percentage points. Reporting dates are metadata unless explicitly enabled as features.

Each candidate stops after 50 rounds without improvement in validation log loss.
The final fit uses the winning candidate's best round count, saved as
`parameters.num_iterations` in `window.json`. The console reports each candidate's
round count, monthly rate error and log loss. The first grid candidate uses 15
leaves, at least 500 rows per leaf, L2 regularization of 5 and up to 2,000 rounds.

## Script Functions

### main.py
The "**main.py**" script orchestrates the training, validation, and testing of the LightGBM model. 

### import_data.py
The "**import_data.py**" script contains the following functions

"**sample_data**"- Samples **num_rows_each** rows from each file and returns a combined sample as an arrow table.

"**prepare_data_for_model**"- Takes an arrow table and returns (dependent variables, target variable)

### validate.py
The "**validate.py**" script contains the following functions

"**validated_parameters**"- takes training data, validation features/labels and reporting periods; returns the configuration with the lowest validation monthly rate error, using its best round count.

### parameter_grid.py
The "**parameter_grid.py**" script contains ten reproducible LightGBM configurations covering shallow, deep, fast, conservative, regularized, and sampled models. Candidates use early stopping on validation log loss; selection uses validation monthly rate error.

### calculate_error.py
The "**calculate_error.py**" script contains the following functions
 
 "**error_summary**" - takes a file path for a file containing actual targets and prediction probabilities and returns a dictionary with key metrics. Also returns a validation key tuple which contains the key metric and the optimality direction to be used during model validation.


 ### test.py
The "**test.py**" script contains the following functions
 
 "**test_in_batches**" - takes a trained model and a list of files paths corresponding to test data and returns the file path to a parquet containing the actual targets and predicted probabilities.
