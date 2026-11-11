"""Select model parameters using only the validation sample."""
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .calculate_error import error_summary, monthly_rate_error
    from .parameter_grid import INITIAL_PARAMETER_GRID
except ImportError:
    from calculate_error import error_summary, monthly_rate_error
    from parameter_grid import INITIAL_PARAMETER_GRID


def validated_parameters(train_data, X_validate, y_validate, periods, verbosity=-1):
    best_params = None
    best_metrics = None
    validation_data = lgb.Dataset(X_validate, label=y_validate, reference=train_data)
    # Each run owns its temporary output, including when validation fails.
    with TemporaryDirectory(prefix="loan-month-validation-") as temp:
        temp_file_path = str(Path(temp) / "validation.parquet")
        for model_number, parameters in enumerate(INITIAL_PARAMETER_GRID, start=1):
            current_params = parameters.copy()
            if verbosity == 1:
                print(f"Training configuration {model_number}/{len(INITIAL_PARAMETER_GRID)}", flush=True)
            model = lgb.train(
                current_params, train_data,
                valid_sets=[validation_data], valid_names=["validation"],
                callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=verbosity == 1)],
            )
            current_params["num_iterations"] = model.best_iteration or model.current_iteration()
            y_pred = model.predict(X_validate)
            pq.write_table(pa.table({
                "actual": y_validate, "predicted": y_pred, "period": periods,
            }), temp_file_path)
            error_metrics, _ = error_summary(temp_file_path)
            error_metrics["monthly_rate_mae"] = monthly_rate_error(temp_file_path)
            if verbosity == 1:
                print(f"Configuration {model_number}: rounds={current_params['num_iterations']}, "
                      f"monthly_rate_mae={error_metrics['monthly_rate_mae']:.6f}, "
                      f"log_loss={error_metrics['log_loss']:.6f}", flush=True)
            if (best_params is None
                    or (error_metrics["monthly_rate_mae"], error_metrics["log_loss"])
                    < (best_metrics["monthly_rate_mae"], best_metrics["log_loss"])):
                best_params = current_params
                best_metrics = error_metrics
    if best_params is None:
        raise ValueError("The parameter grid must contain at least one configuration")
    if verbosity == 1:
        print(f"Selected configuration: {best_params}; "
              f"monthly_rate_mae={best_metrics['monthly_rate_mae']:.6f}, "
              f"log_loss={best_metrics['log_loss']:.6f}", flush=True)
    return best_params
