import duckdb


def monthly_rate_error(file):
    """Mean absolute gap between actual/predicted monthly probability averages."""
    with duckdb.connect() as con:
        error, invalid = con.execute("""
            SELECT avg(abs(actual_rate - predicted_rate)), sum(invalid_rows)
            FROM (
                SELECT date_trunc('month', period) AS month,
                       avg(actual) AS actual_rate, avg(predicted) AS predicted_rate,
                       count(*) FILTER (WHERE period IS NULL OR actual IS NULL
                           OR actual NOT IN (0, 1) OR predicted IS NULL
                           OR NOT isfinite(predicted)
                           OR predicted NOT BETWEEN 0 AND 1) AS invalid_rows
                FROM read_parquet(?)
                GROUP BY 1
            ) AS months
        """, [str(file)]).fetchone()
    if error is None or invalid:
        raise ValueError("Require nonempty predictions with valid periods, labels and probabilities")
    return error


def monthly_cpr(file, test_files, threshold=0.5):
    """Monthly loan-count fractions, assuming one row per active loan-month.

    Use the original test files in prediction order; test.py preserves row order.
    These are the requested monthly fractions, without annualization.
    """
    if not test_files or not 0 <= threshold <= 1:
        raise ValueError("Provide test files and a threshold between 0 and 1")
    with duckdb.connect() as con:
        con.execute("SET preserve_insertion_order=true")
        result = con.execute("""
            SELECT date_trunc('month', source.period)::DATE AS month,
                   count(*) AS active_loans,
                   count(*) FILTER (WHERE actual = 1) AS actual_prepayments,
                   count(*) FILTER (WHERE predicted >= ?) AS predicted_prepayments,
                   (count(*) FILTER (WHERE actual = 1))::DOUBLE / count(*) AS actual_cpr,
                   (count(*) FILTER (WHERE predicted >= ?))::DOUBLE / count(*) AS predicted_cpr,
                   count(*) FILTER (WHERE source.period IS NULL
                       OR source.zero_balance_code IS NULL
                       OR actual IS NULL OR predicted IS NULL
                       OR NOT isfinite(predicted) OR predicted NOT BETWEEN 0 AND 1
                       OR actual <> CAST(source.zero_balance_code = 1 AS INTEGER)) AS invalid_rows
            FROM read_parquet(?) AS source
            POSITIONAL JOIN read_parquet(?) AS predictions
            GROUP BY 1 ORDER BY 1
        """, [threshold, threshold, [str(path) for path in test_files], str(file)]).fetchdf()
    if result.empty or result["invalid_rows"].sum():
        raise ValueError("Predictions must match the original test rows and have valid dates/values")
    return result.drop(columns="invalid_rows")



def expected_monthly_cpr(file, threshold=0.5):
    """Monthly loan-count fractions, assuming one row per active loan-month.

    Use the original test files in prediction order; test.py preserves row order.
    These are the requested monthly fractions, without annualization.
    """
    print(file)
    con=duckdb.connect()
    results=con.sql(f"""
    SELECT
    period,
    COUNT(*) AS loan_count,
    SUM(actual) AS actual_prepayments,
    SUM(predicted) AS expected_prepayments,

        -- Monthly single mortality rate
        AVG(actual) AS actual_smm,
        AVG(predicted) AS predicted_smm,

        -- Annualized conditional prepayment rate
        1 - POWER(1 - AVG(actual), 12) AS actual_cpr,
        1 - POWER(1 - AVG(predicted), 12) AS predicted_cpr


    FROM read_parquet('{file}')
    GROUP BY period
    ORDER BY period
    """).fetchdf()
    return results

def error_summary(file):
    #File is the filepath to a parquet file that contains the actual and predicted columns
    #The actual column contains 0s and 1s
    #The predicted column contains probabilities
    #This script should return an dictionary with appropriate error metrics for the model
    #This script also returns the key metric used for validation and the direction of optimality

    threshold=.5
    epsilon = 1e-15 # To avoid 0, 1 when calculating log loss
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
        END) AS true_negative,

        AVG(
            CASE
                WHEN actual = 1 THEN
                    -1 * LN(GREATEST({epsilon}, LEAST(1 - {epsilon}, predicted)))
                WHEN actual = 0 THEN
                    -1 * LN(1 - GREATEST({epsilon}, LEAST(1 - {epsilon}, predicted)))
            END
        ) AS log_loss

        FROM read_parquet('{file}')
    """).fetchall()[0]

    TP=confusion_matrix[0]
    FN=confusion_matrix[1]
    FP=confusion_matrix[2]
    TN=confusion_matrix[3]
    log_loss = confusion_matrix[4] # Add log_loss to the metics dictionary

    if TP and FN and FP and TP!=0:
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
    else:
        precision=0
        recall=0
    
    metrics={
             "true_positive":TP,
             "false_negative":FN,
             "false_positive":FP,
             "true_negative":TN,
             "precision":round(precision,4),
             "recall":round(recall,4),
             "log_loss": log_loss 
             }


    #Assign key_metric as a string that corresponds to the metric that should be used for validation
    key_metric="log_loss" # Changed to log loss as the key metric for validation
    #set minimize to True if you want to minimize the key metric in validation
    #set minimize to False if you want to maximize the key metric in validation
    minimize= True # Set to True since we want to minimize log loss
    
    return metrics, (key_metric,minimize)
