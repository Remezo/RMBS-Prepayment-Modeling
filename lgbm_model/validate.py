"""Select model parameters using only the validation sample."""
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .calculate_error import error_summary
    from .parameter_grid import INITIAL_PARAMETER_GRID
except ImportError:
    from calculate_error import error_summary
    from parameter_grid import INITIAL_PARAMETER_GRID


def validated_parameters(train_data, X_validate, y_validate, verbosity=-1):
    best_params = None
    best_metrics = None
    # Each run owns its temporary output, including when validation fails.
    with TemporaryDirectory(prefix="loan-month-validation-") as temp:
        temp_file_path = str(Path(temp) / "validation.parquet")
        for model_number, parameters in enumerate(INITIAL_PARAMETER_GRID, start=1):
            current_params = parameters.copy()
            model = lgb.train(current_params, train_data)
            y_pred = model.predict(X_validate)
            pq.write_table(pa.table({"actual": y_validate, "predicted": y_pred}), temp_file_path)
            error_metrics, (key_metric, minimize) = error_summary(temp_file_path)
            if (best_params is None
                    or (minimize and error_metrics[key_metric] < best_metrics[key_metric])
                    or (not minimize and error_metrics[key_metric] > best_metrics[key_metric])):
                best_params = current_params
                best_metrics = error_metrics
                if verbosity == 1:
                    print(f"New best configuration ({model_number}): "
                          f"{best_params} with {key_metric}: {error_metrics[key_metric]}")
    if best_params is None:
        raise ValueError("The parameter grid must contain at least one configuration")
    return best_params
