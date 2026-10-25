import lightgbm as lgb
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from import_data import sample_data,prepare_data_for_model
from validate import validated_parameters
from test import test_in_batches
from calculate_error import error_summary



# Identify files corresponding to train, validate, and test sets
for i in range(2024,2025,2):
    train_files=[]
    validate_files=[]
    test_files=[]
    for j in range(8,9):
        train_files.append(f"data/loan_month/loan_month_{i-j}.parquet")
    for k in range(1,3):
        validate_files.append(f"data/loan_month/loan_month_{i-k}.parquet")
    for l in range(0,1):
        test_files.append(f"data/loan_month/loan_month_{i-l}.parquet")



    #Sample training and validation data
    training_sample=sample_data(train_files,num_rows_each=1000000,verbosity=-1)
    validate_sample=sample_data(validate_files,num_rows_each=1000000,verbosity=-1)

    #Seperate dependent and independent variables
    X_train,y_train=prepare_data_for_model(training_sample)
    X_validate,y_validate=prepare_data_for_model(validate_sample)

    #turn training data into a dataset
    train_data=lgb.Dataset(X_train,label=y_train)

    
    #Determine optimal LightGBM parameters based on the training and validation samples
    model_params=validated_parameters(train_data,X_validate,y_validate,verbosity=-1)

    #train the model
    model=lgb.train(model_params,train_data)

    #determine the importance of each feature in the trained model
    importance = pd.DataFrame({
            "feature": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain")
            })
    #print(importance.sort_values("importance",ascending=False))

    #Create a parquet file with actual prepayment results and predicted probabilities
    predictions_path=test_in_batches(model,test_files,batch_size=100000)

    #Calculate key metrics on the test set
    results,_=error_summary(predictions_path)
    print(results)

    
