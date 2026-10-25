import lightgbm as lgb
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from import_data import sample_data,prepare_data_for_model
from validate import validated_parameters




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

    training_sample=sample_data(train_files,num_rows_each=1000,verbosity=-1)
    validate_sample=sample_data(validate_files,num_rows_each=1000,verbosity=-1)

    X_train,y_train=prepare_data_for_model(training_sample)
    X_validate,y_validate=prepare_data_for_model(validate_sample)
    train_data=lgb.Dataset(X_train,label=y_train)

    
    #optimal parameters based on the validation data
    model_params=validated_parameters(train_data,X_validate,y_validate)

    #train the model
    #model=lgb.train(model_params,train_data)

    

#print(train_data.schema)