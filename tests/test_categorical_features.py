"""Synthetic yearly fixtures verify encoding and the actual LightGBM test path."""
from datetime import datetime
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from lgbm_model.feature_encoder import CATEGORY_VOCABULARIES, FeatureEncoder
from lgbm_model.import_data import (
    CATEGORICAL_CANDIDATES, NUMERIC_FEATURES, TIMESTAMP_CANDIDATES,
    prepare_data_for_model, required_columns, sample_data,
)


def loan_month(states, dates=None):
    """Fixture values illustrate mechanics; they are not vendor schema evidence."""
    n = len(states)
    data = {name: pa.array([1.0] * n) for name in NUMERIC_FEATURES}
    data.update({
        "property_state": pa.array(states, type=pa.string()),
        "prepay_penalty_flag": pa.array(["N"] * n, type=pa.string()),
        "occupancy_status": pa.array(["P"] * n, type=pa.string()),
        "period": pa.array(dates if dates is not None else [datetime(2018, 2, 1)] * n,
                           type=pa.timestamp("us")),
        "orig_period": pa.array([datetime(2016, 1, 1)] * n, type=pa.timestamp("us")),
        "zero_balance_code": pa.array([int(state == "CA") for state in states], type=pa.int16()),
        "loan_id": pa.array([f"identifier-{i}" for i in range(n)]),
        "unpaid_principal_balance": pa.array([0.0] * n),
        "current_upb": pa.array([0.0] * n),
    })
    return pa.table(data)


def make_encoder(categorical=("property_state",), timestamps=()):
    return FeatureEncoder(categorical, timestamps)


def test_fixed_categories_across_splits_and_saved_reload(tmp_path):
    encoder = make_encoder()
    vocabulary = list(CATEGORY_VOCABULARIES["property_state"])
    ca, ny, tx = [vocabulary.index(state) for state in ("CA", "NY", "TX")]
    training = loan_month(["NY", "CA", None, "NY"])
    assert encoder.transform(training)["property_state"].to_pylist() == [ny, ca, -1, ny]
    # TX is known even though it never appeared in the training sample.
    validation = loan_month(["CA", "TX", None, "NY", "XX"])
    validation = validation.set_column(
        validation.schema.get_field_index("property_state"), "property_state",
        pc.dictionary_encode(validation["property_state"]),
    )
    result = encoder.transform(validation)
    assert result["property_state"].to_pylist() == [ca, tx, -1, ny, -1]
    assert result["property_state"].type == pa.int32()
    path = tmp_path / "encoder.json"
    encoder.save(path)
    loaded = FeatureEncoder.load(path)
    assert loaded.transform(validation).equals(result)
    artifact = json.loads(path.read_text())
    assert artifact["version"] == 2
    assert artifact["vocabulary_source"] == "fixed"
    assert "training_files" not in artifact
    assert "TX" in artifact["categories"]["property_state"]


def test_fixed_encoder_needs_no_training_data_and_keeps_exact_strings():
    first, second = make_encoder(), make_encoder()
    values = loan_month(["NY", "CA", "", " NY", None])
    assert first.transform(values)["property_state"].to_pylist()[2:] == [-1, -1, -1]
    assert first.transform(values.drop(["zero_balance_code"])).equals(first.transform(values))
    assert first.categories["property_state"].equals(second.categories["property_state"])
    assert len(first.categories["property_state"]) == 56
    codes = FeatureEncoder(("occupancy_status", "prepay_penalty_flag")).transform(values)
    assert codes["occupancy_status"].to_pylist() == [2] * 5
    assert codes["prepay_penalty_flag"].to_pylist() == [0] * 5


def test_all_null_input_does_not_hide_valid_categories():
    encoder = make_encoder()
    assert encoder.transform(loan_month([None, None]))["property_state"].to_pylist() == [-1, -1]
    assert encoder.transform(loan_month(["CA"]))["property_state"][0].as_py() >= 0


def test_legacy_encoder_load_preserves_original_codes(tmp_path):
    path = tmp_path / "legacy.json"
    make_encoder().save(path)
    payload = json.loads(path.read_text())
    payload["version"] = 1
    payload.pop("vocabulary_source")
    payload["categories"]["property_state"] = ["NY", "CA"]
    payload["training_files"] = [{"year": 2018, "path": "old_training.parquet"}]
    path.write_text(json.dumps(payload))
    loaded = FeatureEncoder.load(path)
    values = loan_month(["NY", "CA", "TX"])
    assert loaded.transform(values)["property_state"].to_pylist() == [0, 1, -1]
    assert loaded.training_files == payload["training_files"]
    loaded.save(path)
    assert FeatureEncoder.load(path).transform(values).equals(loaded.transform(values))


def test_fixed_timestamp_transform_nulls_and_future_dates():
    train = loan_month(["CA", "NY", "CA"], [datetime(1970, 1, 1), datetime(2018, 12, 1), None])
    encoder = make_encoder(timestamps=TIMESTAMP_CANDIDATES)
    X, _ = prepare_data_for_model(train, encoder)
    assert X["period_month_index"].to_pylist() == [0, 587, None]
    assert X["period_month"].to_pylist() == [1, 12, None]
    future = encoder.transform(loan_month(["CA"], [datetime(2025, 1, 1)]))
    assert future["period_month_index"].to_pylist() == [660]
    with pytest.raises(TypeError, match="timezone-free timestamp"):
        make_encoder(timestamps=("period",)).transform(
            train.set_column(train.schema.get_field_index("period"), "period",
                             pa.array(["2018", "2019", None])))


def test_projection_and_model_inputs_exclude_identifiers_and_leaking_balance(tmp_path):
    files = []
    for year, states in ((2017, ["NY"] * 8), (2018, ["CA"] * 8)):
        path = tmp_path / f"loan_month_{year}.parquet"
        pq.write_table(loan_month(states), path)
        files.append(str(path))
    columns = required_columns(CATEGORICAL_CANDIDATES)
    sample = sample_data(files, num_rows_each=3, verbosity=-1, columns=columns)
    assert sample.num_rows == 6
    assert sorted(sample["property_state"].to_pylist()) == ["CA"] * 3 + ["NY"] * 3
    encoder = make_encoder(categorical=CATEGORICAL_CANDIDATES)
    X, _ = prepare_data_for_model(sample, encoder)
    assert X.column_names == [*NUMERIC_FEATURES, *CATEGORICAL_CANDIDATES]
    assert not {"loan_id", "unpaid_principal_balance", "current_upb"}.intersection(sample.column_names)
    for name in ("loan_id", "unpaid_principal_balance", "current_upb"):
        with pytest.raises(ValueError, match="Choose distinct"):
            FeatureEncoder([name])


def test_target_maps_known_exit_codes_and_preserves_input():
    codes = [0, 1, 2, 3, 9, 15, 16, 96]
    data = loan_month(["CA"] * len(codes))
    column = data.schema.get_field_index("zero_balance_code")
    data = data.set_column(column, "zero_balance_code", pa.array(codes, type=pa.int16()))
    X, y = prepare_data_for_model(data)
    assert y.to_pylist() == [0, 1, 0, 0, 0, 0, 0, 0]
    assert y.type == pa.int16()
    assert "zero_balance_code" not in X.column_names
    assert data["zero_balance_code"].to_pylist() == codes
    _, encoded_y = prepare_data_for_model(data, make_encoder())
    assert encoded_y.equals(y)
    _, batch_y = prepare_data_for_model(data.to_batches()[0])
    assert batch_y.to_pylist() == y.to_pylist()


def test_target_rejects_null_unknown_and_string_codes():
    data = loan_month(["CA", "NY"])
    column = data.schema.get_field_index("zero_balance_code")
    with pytest.raises(TypeError, match="integer codes"):
        prepare_data_for_model(data.set_column(column, "zero_balance_code", pa.array(["01", "00"])))
    for codes in ([1, 99], [1, None]):
        with pytest.raises(ValueError, match="non-null codes"):
            prepare_data_for_model(data.set_column(column, "zero_balance_code", pa.array(codes)))
    _, y = prepare_data_for_model(data.set_column(column, "zero_balance_code", pa.array([True, False])))
    assert y.to_pylist() == [1, 0]


@pytest.fixture
def lgb():
    # Non-model users can run the original repository checks without LightGBM.
    return pytest.importorskip("lightgbm")


def small_parameters():
    return {"objective": "binary", "verbosity": -1, "num_threads": 1,
            "num_iterations": 5, "min_data_in_leaf": 1, "min_data_per_group": 1,
            "cat_smooth": 0, "cat_l2": 0}


def test_real_lightgbm_train_validation_and_multi_year_batched_prediction(tmp_path, lgb):
    from lgbm_model.test import test_in_batches as predict_batches

    train = loan_month(["NY", "CA", "TX", "FL"] * 40)
    encoder = make_encoder(CATEGORICAL_CANDIDATES, TIMESTAMP_CANDIDATES)
    X_train, y_train = prepare_data_for_model(train, encoder)
    dataset = lgb.Dataset(X_train, label=y_train, feature_name=encoder.feature_names,
                          categorical_feature=list(encoder.categorical_columns))
    model = lgb.train(small_parameters(), dataset)
    # Constant numeric fixture columns force the model to use categorical splits.
    trees = model.dump_model()["tree_info"]
    assert any(tree["tree_structure"].get("decision_type") == "==" for tree in trees)
    validation = loan_month(["CA", "NY", "new-category", None])
    X_valid, _ = prepare_data_for_model(validation, encoder)
    assert np.isfinite(model.predict(X_valid)).all()

    test_tables = [validation, loan_month([None, "TX", "CA", "new-category", "NY"])]
    paths = []
    for year, table in zip((2020, 2021), test_tables):
        path = tmp_path / f"loan_month_{year}.parquet"
        pq.write_table(table, path, row_group_size=3)
        paths.append(str(path))
    encoder_path = tmp_path / "encoder.json"
    encoder.save(encoder_path)
    model_path = tmp_path / "model.txt"
    model.save_model(str(model_path))
    reloaded = lgb.Booster(model_file=str(model_path))
    output = predict_batches(reloaded, paths, batch_size=2, encoder=FeatureEncoder.load(encoder_path),
                             output_path=tmp_path / "predictions.parquet")
    predictions = pq.read_table(output)
    assert predictions.num_rows == 9
    assert predictions["actual"].to_pylist() == pa.concat_tables(test_tables)["zero_balance_code"].to_pylist()
    expected = np.concatenate([model.predict(prepare_data_for_model(t, encoder)[0]) for t in test_tables])
    np.testing.assert_allclose(predictions["predicted"].to_numpy(), expected)
    assert np.all((expected >= 0) & (expected <= 1))


def test_batch_failure_does_not_publish_partial_output(tmp_path, lgb):
    from lgbm_model.test import test_in_batches as predict_batches

    table = loan_month(["CA", "NY"] * 20)
    encoder = make_encoder()
    X, y = prepare_data_for_model(table, encoder)
    model = lgb.train(small_parameters(), lgb.Dataset(X, label=y, categorical_feature=["property_state"]))
    good, bad = tmp_path / "good.parquet", tmp_path / "bad.parquet"
    pq.write_table(table, good)
    pq.write_table(table.drop(["property_state"]), bad)
    output = tmp_path / "predictions.parquet"
    with pytest.raises(Exception, match="property_state"):
        predict_batches(model, [str(good), str(bad)], encoder=encoder, output_path=output)
    assert not output.exists()
    assert not output.with_name(output.name + ".partial").exists()


def test_rolling_years_and_alternative_filenames(lgb):
    from lgbm_model.main import rolling_windows, yearly_files

    windows = list(rolling_windows(file_template="data/converted/loan_month_{year}_prepared.parquet"))
    assert len(windows) == 6
    assert [item["year"] for item in windows[0]["train"]] == [2016, 2017, 2018]
    assert windows[0]["validation"][0]["year"] == 2019
    assert windows[0]["test"] == [{"year": 2020, "path": "data/converted/loan_month_2020_prepared.parquet"}]
    assert windows[-1]["test"][0]["year"] == 2025
    with pytest.raises(ValueError, match="retain yearly"):
        yearly_files([2018, 2019], "all_years.parquet")


def test_main_window_runs_sampling_tuning_training_and_test(tmp_path, monkeypatch, lgb):
    import lgbm_model.main as main
    import lgbm_model.validate as validation

    template = str(tmp_path / "loan_month_{year}.parquet")
    for year in range(2016, 2021):
        pq.write_table(loan_month(["NY", "CA"] * 30), template.format(year=year))
    # Exercise the real tuning loop with a small grid, rather than mocking training.
    monkeypatch.setattr(validation, "INITIAL_PARAMETER_GRID", [small_parameters()])
    window = next(main.rolling_windows(2016, 2020, file_template=template))
    encoder = make_encoder(CATEGORICAL_CANDIDATES)
    output = main.run_window(window, encoder=encoder,
                             sample_rows=40, batch_size=7, output_dir=tmp_path / "runs")
    assert pq.read_table(output / "predictions.parquet").num_rows == 60
    artifact = json.loads((output / "encoder.json").read_text())
    assert artifact["vocabulary_source"] == "fixed"
    assert artifact["categories"]["property_state"] == list(CATEGORY_VOCABULARIES["property_state"])
    assert json.loads((output / "window.json").read_text())["files"] == window
    with pytest.raises(ValueError, match="chronological"):
        main.run_window({"train": window["test"], "validation": window["validation"], "test": window["train"]}, encoder=encoder)


def test_main_reuses_one_encoder_across_rolling_windows(monkeypatch, lgb):
    import lgbm_model.main as main

    seen = []
    monkeypatch.setattr(main, "run_window", lambda window, **kwargs: seen.append(kwargs["encoder"]))
    monkeypatch.setattr("sys.argv", ["main.py", "--first-year", "2016", "--last-year", "2021",
                                     "--categorical-columns", "property_state", "occupancy_status"])
    main.main()
    assert len(seen) == 2
    assert seen[0] is seen[1]
    assert seen[0].categories["property_state"].to_pylist() == list(CATEGORY_VOCABULARIES["property_state"])
