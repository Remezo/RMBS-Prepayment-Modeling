"""Fixed, serializable categorical conversion shared across rolling windows."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

try:
    from .import_data import CATEGORICAL_CANDIDATES, NUMERIC_FEATURES, TIMESTAMP_CANDIDATES
except ImportError:  # python lgbm_model/main.py
    from import_data import CATEGORICAL_CANDIDATES, NUMERIC_FEATURES, TIMESTAMP_CANDIDATES


# Fixed domains: 50 states, DC, and US territories; Freddie Mac occupancy/penalty codes.
# No training, validation, or test rows are used to build these vocabularies.
CATEGORY_VOCABULARIES = {
    "property_state": tuple(
        "AK AL AR AS AZ CA CO CT DC DE FL GA GU HI IA ID IL IN KS KY LA MA MD ME "
        "MI MN MO MP MS MT NC ND NE NH NJ NM NV NY OH OK OR PA PR RI SC SD TN "
        "TX UT VA VI VT WA WI WV WY".split()
    ),
    "occupancy_status": ("9", "I", "P", "S"),
    "prepay_penalty_flag": ("N", "Y"),
}


def _strings(column):
    dtype = column.type.value_type if pa.types.is_dictionary(column.type) else column.type
    if not (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)):
        raise TypeError(f"Categorical columns must contain strings, got {column.type}")
    return pc.cast(column, pa.string())


class FeatureEncoder:
    """Codes 0..K-1 from fixed vocabularies; missing/unrecognized -> -1.

    Exact strings are matched without trimming or case conversion. The same
    encoder can transform every rolling window without fitting on data.
    """

    def __init__(self, categorical_columns=(), timestamp_columns=()):
        self.categorical_columns = tuple(categorical_columns)
        self.timestamp_columns = tuple(timestamp_columns)
        for selected, allowed in (
            (self.categorical_columns, CATEGORICAL_CANDIDATES),
            (self.timestamp_columns, TIMESTAMP_CANDIDATES),
        ):
            if len(set(selected)) != len(selected) or set(selected) - set(allowed):
                raise ValueError(f"Choose distinct columns from {allowed}; got {selected}")
        self.categories = {
            name: pa.array(CATEGORY_VOCABULARIES[name], type=pa.string())
            for name in self.categorical_columns
        }
        self.vocabulary_source = "fixed"
        # Retained only for loading metadata from older, training-fitted encoders.
        self.training_files = None

    @property
    def feature_names(self):
        return [*NUMERIC_FEATURES, *self.categorical_columns, *(
            output for name in self.timestamp_columns
            for output in (f"{name}_month_index", f"{name}_month")
        )]

    @staticmethod
    def _timestamp(column, name):
        if not pa.types.is_timestamp(column.type) or column.type.tz is not None:
            raise TypeError(f"{name} requires a timezone-free timestamp; confirm timezone/parser upstream")
        return column

    def transform(self, data):
        if isinstance(data, pa.RecordBatch):
            data = pa.Table.from_batches([data])
        features = data.select(NUMERIC_FEATURES)
        for name in self.categorical_columns:
            codes = pc.index_in(_strings(data.column(name)), value_set=self.categories[name])
            features = features.append_column(name, pc.cast(pc.fill_null(codes, -1), pa.int32()))
        for name in self.timestamp_columns:
            dates = self._timestamp(data.column(name), name)
            month = pc.month(dates)
            index = pc.add(pc.multiply(pc.subtract(pc.year(dates), 1970), 12), pc.subtract(month, 1))
            features = features.append_column(f"{name}_month_index", pc.cast(index, pa.int32()))
            features = features.append_column(f"{name}_month", pc.cast(month, pa.int8()))
        return features

    def save(self, path):
        payload = {
            "version": 2,
            "vocabulary_source": self.vocabulary_source,
            "categorical_columns": list(self.categorical_columns),
            "timestamp_columns": list(self.timestamp_columns),
            "feature_names": self.feature_names,
            "categories": {name: values.to_pylist() for name, values in self.categories.items()},
            "unknown_and_missing_code": -1,
            "timestamp_transform": "month_index_since_1970_and_calendar_month_v1",
            "string_normalization": "exact",
        }
        if self.training_files is not None:
            payload["training_files"] = self.training_files
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if (payload["version"] not in (1, 2) or payload["unknown_and_missing_code"] != -1
                or payload["timestamp_transform"] != "month_index_since_1970_and_calendar_month_v1"
                or payload["string_normalization"] != "exact"):
            raise ValueError("Unsupported encoder format")
        encoder = cls(payload["categorical_columns"], payload["timestamp_columns"])
        if encoder.feature_names != payload["feature_names"]:
            raise ValueError("Encoder feature order does not match this implementation")
        if set(payload["categories"]) != set(encoder.categorical_columns):
            raise ValueError("Encoder vocabulary columns do not match configuration")
        encoder.categories = {}
        for name, values in payload["categories"].items():
            if any(not isinstance(v, str) for v in values) or len(set(values)) != len(values):
                raise ValueError(f"Invalid vocabulary: {name}")
            encoder.categories[name] = pa.array(values, type=pa.string())
        encoder.training_files = payload.get("training_files")
        encoder.vocabulary_source = payload.get("vocabulary_source", "training")
        return encoder
