import lightgbm as lgb
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from import_data import sample_data

for i in range(2021,2022):
    files=[]
    for j in range(1,6):
        files.append(f"data/loan_month/loan_month_{i-j}.parquet")

sample_file=sample_data(files)

train_data=lgb.Dataset(sample_file)