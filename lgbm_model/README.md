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
`python main.py --file-template 'data/loan_month/loan_month_perf_{year}.parquet'`.
From the repository root, use `python -m lgbm_model.main --file-template
'lgbm_model/data/loan_month/loan_month_perf_{year}.parquet'` instead.


## User Adjustments

See [Categorical conversion](CATEGORICAL_CONVERSION.md) for the reviewed columns,
training-only encoder, optional feature flags, yearly filename configuration,
memory tradeoffs, and verification. New categorical and timestamp features are
explicit opt-ins. Run from the repository root with `python -m lgbm_model.main`.

To update the features that are used by the LightGBM model ("**X**"), see the "**prepare_data_for_model**" function in "**import_data.py**" 

To update the ten initial parameter configurations evaluated on the validation set, see "**INITIAL_PARAMETER_GRID**" in "**parameter_grid.py**".

to update the metric ("**key_metric**") that is used to rank the trained models performance, see the "**error_summary**" function in "**calculate_error.py**" 

## Script Functions

### main.py
The "**main.py**" script orchestrates the training, validation, and testing of the LightGBM model. 

### import_data.py
The "**import_data.py**" script contains the following functions

"**sample_data**"- Samples **num_rows_each** rows from each file and returns a combined sample as an arrow table.

"**prepare_data_for_model**"- Takes an arrow table and returns (dependent variables, target variable)

### validate.py
The "**validate.py**" script contains the following functions

"**validated_parameters**"- takes training and validation data and returns the parameter configuration with the best performance on the validation set as a dictionary.

### parameter_grid.py
The "**parameter_grid.py**" script contains ten reproducible LightGBM configurations covering baseline, shallow, deep, fast, conservative, regularized, and sampled models. The validation process evaluates each configuration and selects the best one using the validation metric defined in "**calculate_error.py**".

### calculate_error.py
The "**calculate_error.py**" script contains the following functions
 
 "**error_summary**" - takes a file path for a file containing actual targets and prediction probabilities and returns a dictionary with key metrics. Also returns a validation key tuple which contains the key metric and the optimality direction to be used during model validation.


 ### test.py
The "**test.py**" script contains the following functions
 
 "**test_in_batches**" - takes a trained model and a list of files paths corresponding to test data and returns the file path to a parquet containing the actual targets and predicted probabilities.
