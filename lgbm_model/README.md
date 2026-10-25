#Light GBM Model 

## Overview
This folder contains all files related to the training, validation, and testing of the LightGBM model. Pulling data into a pandas is avoided when possible to mimimize memory usage.

The "**prepare_data_for_model**" function in "**import_data.py**" contains the relevant features that are selected for the model to use

the "**validated_parameters**" function in "**validate.py**" contains the relevant parameters for tuning the model. 

### main.py
The "**main.py**" script orchestrates the training, validation, and testing of the LightGBM model. 

### import_data.py
The "**import_data.py**" script contains the following functions
"**sample_data**"- Utilizes duckdb to sample **num_rows_each** from each of the files that it is provided. Temporary files will be created and subsequently deleted before the function returns a combined sample as an arrow table.
"**prepare_data_for_model**"- Takes an arrow table and returns (dependent variables, target variable)

### validate.py
The "**validate.py**" script contains the following functions
"**validated_parameters**"- takes training and validation data and returns the parameter configuration with the best performance on the test set.

### calculate_error.py
The "**calculate_error.py**" script contains the following functions
None 


