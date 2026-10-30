"""Bounded-batch prediction using the training window's frozen encoder."""
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .import_data import prepare_data_for_model, required_columns
except ImportError:
    from import_data import prepare_data_for_model, required_columns


def test_in_batches(model, files, batch_size=1000000, verbosity=-1, *, encoder=None,
                    output_path="data/tmp_test.parquet"):
    if not files or batch_size <= 0:
        raise ValueError("Provide files and a positive batch size")
    columns = required_columns(
        encoder.categorical_columns if encoder else (),
        encoder.timestamp_columns if encoder else (),
    )
    projection = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
    output = Path(output_path)
    if output.resolve() in {Path(file).resolve() for file in files}:
        raise ValueError("Prediction output must not overwrite an input file")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    schema = pa.schema([("actual", pa.int16()), ("predicted", pa.float64())])
    try:
        with duckdb.connect() as con, pq.ParquetWriter(partial, schema) as writer:
            con.execute("SET preserve_insertion_order=true")
            batch_number = 0
            for file in files:
                reader = con.read_parquet(str(file)).project(projection).to_arrow_reader(batch_size=batch_size)
                for batch in reader:
                    X_test, y_test = prepare_data_for_model(batch, encoder)
                    if X_test.column_names != model.feature_name():
                        raise ValueError("Model and test feature names/order differ")
                    predicted = model.predict(X_test)
                    writer.write_table(pa.table({"actual": y_test, "predicted": predicted}, schema=schema))
                    batch_number += 1
                    if verbosity == 1:
                        print(f"Processed batch {batch_number}")
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return str(output)
