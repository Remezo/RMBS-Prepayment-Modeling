"""Projected yearly sampling and shared feature preparation for every split."""
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import pyarrow as pa
import pyarrow.compute as pc

NUMERIC_FEATURES = (
    "loan_age", "current_interest_rate", "classic_fico", "cltv", "orig_upb",
    "dti", "hpi_orig", "hpi", "current_rate", "refi", "chg_hpi",
)
# Candidates from the original commented selection. Opt in after confirming
# their definitions and availability at prediction time. loan_id stays excluded.
CATEGORICAL_CANDIDATES = ("prepay_penalty_flag", "property_state", "occupancy_status")
TIMESTAMP_CANDIDATES = ("period", "orig_period")
TARGET = "zero_balance_code"


def required_columns(categorical_columns=(), timestamp_columns=()):
    return [*NUMERIC_FEATURES, *categorical_columns, *timestamp_columns, TARGET]


def sample_data(files, num_rows_each=1000000, verbosity=1, columns=None, seed=42):
    """Reservoir sample each year separately, with isolated temporary files."""
    if not files or num_rows_each <= 0:
        raise ValueError("Provide files and a positive sample size")
    projection = "*" if columns is None else ", ".join(
        '"' + name.replace('"', '""') + '"' for name in columns
    )
    with TemporaryDirectory(prefix="loan-month-sample-") as temp, duckdb.connect() as con:
        con.execute("SET preserve_insertion_order=false")
        samples = []
        for i, file in enumerate(files):
            if verbosity == 1:
                print(f"Sampling {file}")
            destination = str(Path(temp) / f"sample_{i}.parquet")
            con.read_parquet(str(file)).project(projection).query(
                "source",
                f"SELECT * FROM source USING SAMPLE {int(num_rows_each)} ROWS "
                f"(reservoir, {int(seed)})",
            ).write_parquet(destination)
            samples.append(destination)
        return con.read_parquet(samples).to_arrow_table()


def prepare_data_for_model(data, encoder=None):
    """Use the same fitted encoder for training, validation, and every batch."""
    if isinstance(data, pa.RecordBatch):
        data = pa.Table.from_batches([data])
    X = data.select(NUMERIC_FEATURES) if encoder is None else encoder.transform(data)
    for field in X.schema:
        if not (pa.types.is_integer(field.type) or pa.types.is_floating(field.type)):
            raise TypeError(f"Feature {field.name} must be numeric, got {field.type}")
    y = data.column(TARGET)
    if not (pa.types.is_integer(y.type) or pa.types.is_boolean(y.type)):
        raise TypeError("zero_balance_code must already be a binary 0/1 target; no target recoding is inferred")
    if pa.types.is_boolean(y.type):
        y = pc.cast(y, pa.int16())
    if y.null_count or not pc.all(pc.is_in(y, value_set=pa.array([0, 1], type=y.type))).as_py():
        raise ValueError("zero_balance_code must contain only non-null binary 0/1 labels")
    return X, pc.cast(y, pa.int16())
