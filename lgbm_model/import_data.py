import duckdb
import glob
import os
def sample_data(files,num_rows_each=1000):

    con=duckdb.connect()
    con.execute("Set Preserve_insertion_order=false")

    #Create a sample file for each file 
    for i,file in enumerate(files, start=1):
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
    print("Combining sample files")
    sample_file="data/combined_sample.parquet"
    con.execute(f"""
        COPY(
            SELECT *
            FROM read_parquet('data/tmp_sample_*.parquet')
        )
        TO '{sample_file}'
        (FORMAT PARQUET)
    """)


    #Remove all of the sample files from the directory
    pattern = "data/tmp_sample_*.parquet"
    matched_files = glob.glob(pattern)

    # 2. Iterate and delete safely
    for file_path in matched_files:
        try:
            os.remove(file_path)  # or os.unlink(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Failed to delete {file_path}: {e.strerror}")

    return sample_file