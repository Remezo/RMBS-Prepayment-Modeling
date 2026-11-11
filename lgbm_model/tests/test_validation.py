"""Validation selection, calendar grouping and real early stopping."""
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lgbm_model.calculate_error import monthly_rate_error
from lgbm_model import validate


def test_monthly_error_weights_months_equally_and_separates_years(tmp_path):
    path = tmp_path / "predictions.parquet"
    pq.write_table(pa.table({
        "period": [datetime(2024, 1, 1), datetime(2024, 1, 31)]
                  + [datetime(2025, 1, 1)] * 6,
        "actual": [0, 1] + [0] * 6,
        "predicted": [0.0, 0.0] + [0.1] * 6,
    }), path)
    assert monthly_rate_error(path) == pytest.approx((0.5 + 0.1) / 2)


@pytest.mark.parametrize("column,value", [
    ("period", None), ("actual", 2), ("predicted", float("nan")), ("predicted", 1.1),
])
def test_monthly_error_rejects_invalid_values(tmp_path, column, value):
    data = {"period": [datetime(2025, 1, 1)], "actual": [1], "predicted": [0.2]}
    data[column] = [value]
    if column == "period":
        data[column] = pa.array([value], type=pa.timestamp("us"))
    path = tmp_path / "predictions.parquet"
    pq.write_table(pa.table(data), path)
    with pytest.raises(ValueError, match="valid periods"):
        monthly_rate_error(path)


def test_selection_uses_monthly_error_before_log_loss_and_keeps_best_round(monkeypatch):
    grid = [{"num_iterations": 100, "num_leaves": 15},
            {"num_iterations": 100, "num_leaves": 31}]
    monkeypatch.setattr(validate, "INITIAL_PARAMETER_GRID", grid)

    class Model:
        def __init__(self, predictions, rounds):
            self.predictions = predictions
            self.best_iteration = rounds

        def predict(self, data):
            return self.predictions

    # The second model has worse log loss, but exactly matches both monthly rates.
    models = iter([Model([0.1, 0.8, 0.1, 0.8], 3),
                   Model([0.9, 0.1, 0.9, 0.1], 7)])

    def train(params, data, *, valid_sets, valid_names, callbacks):
        assert len(valid_sets) == 1 and callbacks[0].stopping_rounds == 50
        return next(models)

    monkeypatch.setattr(validate.lgb, "train", train)
    X = pa.table({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pa.array([0, 1, 0, 1])
    periods = pa.array([datetime(2024, 1, 1)] * 2 + [datetime(2024, 2, 1)] * 2)
    params = validate.validated_parameters(lgb.Dataset(X, label=y), X, y, periods)
    assert params == {"num_iterations": 7, "num_leaves": 31}
    assert grid[1]["num_iterations"] == 100


def test_real_early_stopping_returns_round_count_for_final_fit(monkeypatch):
    X = pa.table({"x": np.tile([-2.0, -1.0, 1.0, 2.0], 50)})
    y = pa.array(np.tile([0, 0, 1, 1], 50))
    validation_y = pa.array(np.tile([1, 1, 0, 0], 50))
    periods = pa.array([datetime(2024, 1, 1)] * len(y))
    monkeypatch.setattr(validate, "INITIAL_PARAMETER_GRID", [{
        "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
        "num_threads": 1, "num_iterations": 80, "num_leaves": 3,
        "min_child_samples": 1, "learning_rate": 0.2, "feature_pre_filter": False,
    }])
    dataset = lgb.Dataset(X, label=y)
    params = validate.validated_parameters(dataset, X, validation_y, periods)
    assert params["num_iterations"] == 1
    model = lgb.train(params, dataset)
    assert model.current_iteration() == 1
