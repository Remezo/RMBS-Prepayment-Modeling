from itertools import product
import os
from calculate_error import error_summary
import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq

def validated_parameters(train_data,X_validate,y_validate,verbosity=-1):

    parameters={
    'objective': ['binary'],
    'metric': ['binary_logloss'],
    'boosting_type': ['gbdt'],
    'num_leaves': [2,4,8,16,30,40,50,80,100],
    'verbosity': [-1]
    }
    best_params=None
    best_metrics=None
    keys=parameters.keys()
    temp_file_path="data/tmp_validation.parquet"

    #Loop through all combinations of parameters
    #Turn the predictions and actuals into a parquet file 
    #Call the error_summary function to get the error metrics
    #Keep the best parameters based on a key metric (e.g., log_loss)
    for combination in product(*parameters.values()):
        current_params=dict(zip(keys,combination))
        model=lgb.train(current_params,train_data)

        y_pred=model.predict(X_validate) #This returns a numpy array
        arrow_table=pa.table({"actual":y_validate,"predicted":y_pred})
        pq.write_table(arrow_table,temp_file_path)
        error_metrics,validation_key=error_summary(temp_file_path)
        key_metric=validation_key[0]#This is a string key of the error_metrics dictionary
        minimize=validation_key[1]#This is True if a smaller key metric is better

       
        if ( 
            best_params is None
             or (minimize and error_metrics[key_metric]<best_metrics[key_metric]) 
             or (not minimize and error_metrics[key_metric]>best_metrics[key_metric])
        ):
            best_params=current_params
            best_metrics=error_metrics
            if verbosity==1:
                print(f"New best parameters found: {best_params} with {key_metric}: {error_metrics[key_metric]}")
            

    #delete the temporary file
    try:
        os.remove(temp_file_path)
        if verbosity==1:
            print(f"Deleted: {temp_file_path}")
    except OSError as e:
        if verbosity==1:
            print(f"Failed to delete {temp_file_path}: {e.strerror}")

    return best_params


