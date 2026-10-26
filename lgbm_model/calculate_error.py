import duckdb


def error_summary(file):
    #File is the filepath to a parquet file that contains the actual and predicted columns
    #The actual column contains 0s and 1s
    #The predicted column contains probabilities
    #This script should return an dictionary with appropriate error metrics for the model
    #This script also returns the key metric used for validation and the direction of optimality

    threshold=.5
    con=duckdb.connect()
    confusion_matrix=con.sql(f"""
    SELECT
        SUM(CASE
            WHEN actual = 1 AND predicted >={threshold}
            THEN 1 ELSE 0
        END) AS true_positive,
        SUM(CASE
            WHEN actual = 1 AND predicted <{threshold}
            THEN 1 ELSE 0
        END) AS false_negative,
        SUM(CASE
            WHEN actual = 0 AND predicted >={threshold}
            THEN 1 ELSE 0
        END) AS false_positive,
        SUM(CASE
            WHEN actual = 0 AND predicted <{threshold}
            THEN 1 ELSE 0
        END) AS true_negative
        FROM read_parquet('{file}')
    """).fetchall()[0]

    TP=confusion_matrix[0]
    FN=confusion_matrix[1]
    FP=confusion_matrix[2]
    TN=confusion_matrix[3]
    if TP and FN and FP and TP!=0:
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
    else:
        precision=0
        recall=0

  



    
    metrics={"true_positive":TP,
             "false_negative":FN,
             "false_positive":FP,
             "true_negative":TN,
             "precision":round(precision,4),
             "recall":round(recall,4)

             }


    #Assign key_metric as a string that corresponds to the metric that should be used for validation
    key_metric="true_positive"
    #set minimize to True if you want to minimize the key metric in validation
    #set minimize to False if you want to maximize the key metric in validation
    minimize=False
    
    return metrics, (key_metric,minimize)