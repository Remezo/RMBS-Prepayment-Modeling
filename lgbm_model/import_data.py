import duckdb
import glob
import os
def sample_data(files,num_rows_each=1000000,verbosity=1):

    con=duckdb.connect()
    con.execute("Set Preserve_insertion_order=false")

    #Create a sample file for each file 
    for i,file in enumerate(files, start=1):
        if verbosity==1:
            print(f"Processing file{file}")

        con.execute(f"""
            COPY (
                SELECT * 
                FROM read_parquet('{file}')
                USING SAMPLE {num_rows_each} ROWS
            )
            TO 'data/tmp_sample_{i}.parquet'
            (FORMAT PARQUET)
        """)


    #Combine all of the sample files together into the combined file
    if verbosity==1:
        print("Combining sample files")
    arrow_data=con.sql(f"""
        SELECT *
        FROM read_parquet('data/tmp_sample_*.parquet')
    """).to_arrow_table()


    #Remove all of the sample files from the directory
    pattern = "data/tmp_sample_*.parquet"
    matched_files = glob.glob(pattern)

    
    for file_path in matched_files:
        try:
            os.remove(file_path)  
            if verbosity==1:
                print(f"Deleted: {file_path}")
        except OSError as e:
            if verbosity==1:
                print(f"Failed to delete {file_path}: {e.strerror}")

    return arrow_data


def prepare_data_for_model(data):
    X=data.select([
        #"loan_id",#: string
        #"period",#: timestamp[us]
        "loan_age",#: int16
        "current_interest_rate",#: double
        #"current_upb",#: double
        "classic_fico",#: int16
        "cltv",#: int16
        "orig_upb",#: double
        #"prepay_penalty_flag",#: string
        #"property_state",#: string
        #"occupancy_status",#: string
        "dti",#: int16
        #"orig_period",#: timestamp[us]
        "hpi_orig",#: double
        "hpi",#: double
        "current_rate",#: double
        "refi",#: double
        "chg_hpi"#: double

    ])
    y=data.column('zero_balance_code')
    return X,y

