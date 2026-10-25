import duckdb


def error_summary(file):
    #file is the filepath to a parquet file that contains the actual and predicted columns
    #the actual column contains 0s and 1s
    #the predicted column contains probabilities
    #this script should return an dictionary with appropriate error metrics for the model
    key_metric="log_loss"
    metrics={'log_loss': 0.0, 
             'accuracy': 0.0
             }
    return metrics, key_metric