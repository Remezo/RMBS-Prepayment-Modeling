import lightgbm as lgb
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from import_data import sample_data,prepare_data_for_model
from validate import validated_parameters
from test import test_in_batches
from calculate_error import error_summary




first_year=2016
last_year=2025

test_years=1
train_years=3
validate_years=1

total_years=test_years+train_years+validate_years
last_start_year=last_year-total_years+1

#last_start_year=first_year

for start_train in range(first_year,last_start_year+1,test_years):
    # Identify files corresponding to train, validate, and test sets
    train_files=[]
    validate_files=[]
    test_files=[]

    start_validate=start_train+train_years
    start_test=start_train+train_years+validate_years
    end_test=start_test+test_years
    print(f"Training on {start_train} - {start_validate-1}",end =" -> ")
    print(f"Validating on {start_validate} - {start_test-1}",end =" -> ")
    print(f"Testing on {start_test} - {end_test-1}")
    
    for j in range(start_train,start_validate):
        train_files.append(f"data/loan_month/loan_month_{j}.parquet")
    for k in range(start_validate,start_test):
        validate_files.append(f"data/loan_month/loan_month_{k}.parquet")
    for l in range(start_test,end_test):
        test_files.append(f"data/loan_month/loan_month_{l}.parquet")



    #Sample training and validation data
    training_sample=sample_data(train_files,num_rows_each=1000000,verbosity=-1)
    validate_sample=sample_data(validate_files,num_rows_each=1000000,verbosity=-1)

    # #Seperate dependent and independent variables
    X_train,y_train=prepare_data_for_model(training_sample)
    X_validate,y_validate=prepare_data_for_model(validate_sample)

    # #turn training data into a dataset
    train_data=lgb.Dataset(X_train,label=y_train)

    
    # #Determine optimal LightGBM parameters based on the training and validation samples
    model_params=validated_parameters(train_data,X_validate,y_validate,verbosity=-1)

    # #train the model
    model=lgb.train(model_params,train_data)

    # #determine the importance of each feature in the trained model
    importance = pd.DataFrame({
            "feature": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain")
            })
    # #print(importance.sort_values("importance",ascending=False))

    # #Create a parquet file with actual prepayment results and predicted probabilities
    
    predictions_path=test_in_batches(model,test_files,batch_size=500000,verbosity=1)

    # #Calculate key metrics on the test set
    results,_=error_summary(predictions_path)
    print(f"Metrics on the test data from {start_test} to {end_test-1} are as follows")
    print(results)

    
