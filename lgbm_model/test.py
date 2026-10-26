from calculate_error import error_summary
from import_data import prepare_data_for_model
import duckdb
import pyarrow.parquet as pq
import pyarrow as pa
import scipy.sparse as sp

def test_in_batches(model,files, batch_size=1000000,verbosity=-1):

    output_path="data/tmp_test.parquet"
    schema = pa.schema([("actual", pa.int16()),("predicted", pa.float64())])
    writer=pq.ParquetWriter(output_path,schema=schema)

    con=duckdb.connect()
    total_rows=0
    batch_number=0
    if verbosity==1:
        print(f"processing batch {batch_number}           ",end="")
    for file in files:
        reader=con.sql(f"SELECT *FROM read_parquet('{file}')").to_arrow_reader(batch_size=batch_size)
        
        try:
            for batch in reader:
                batch_number+=1
                if verbosity==1:
                    print(f"\rprocessing batch {batch_number}           ",end="",flush=True)
                X_test,y_test=prepare_data_for_model(batch)
                X_test=sp.csr_matrix(X_test)
                y_pred=model.predict(X_test) #This returns a numpy array
                
                arrow_table=pa.table({"actual":y_test,"predicted":y_pred})
                
                writer.write_table(arrow_table)
                total_rows+=batch.num_rows
                

        except Exception as e:
            print(f"Error occurred while processing file {file}: {e}")
        
    if writer:
        writer.close()
    if verbosity==1:
        print("\rPredictions complete               ")
    return output_path