#Light GBM Model 

## Overview
This folder contains all files related to the training, validation, and testing of the LightGBM model. 

**Pulling data into a pandas is avoided when possible to mimimize memory usage.**

This pipeline was created based on discussion from the 10-23-2026 team meeting. 

The pipeline takes a sample of the training and validation data and trains multiple LightGBM models with different combinations of training parameters. The best model is selected based on performance on the validation sample data. The selected model is then used to predict prepayment behavior for every loan in the test data files. The predictions are exported to parquet and evaluated for correctness. 


## User Adjustments

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
