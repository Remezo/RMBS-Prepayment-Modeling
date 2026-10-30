"""Rolling yearly LightGBM training with training-only feature encoders."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

try:
    from .calculate_error import error_summary
    from .feature_encoder import FeatureEncoder
    from .import_data import (
        CATEGORICAL_CANDIDATES, TIMESTAMP_CANDIDATES,
        prepare_data_for_model, required_columns, sample_data,
    )
    from .test import test_in_batches
    from .validate import validated_parameters
except ImportError:
    from calculate_error import error_summary
    from feature_encoder import FeatureEncoder
    from import_data import (
        CATEGORICAL_CANDIDATES, TIMESTAMP_CANDIDATES,
        prepare_data_for_model, required_columns, sample_data,
    )
    from test import test_in_batches
    from validate import validated_parameters


def yearly_files(years, file_template):
    if "{year}" not in file_template:
        raise ValueError("file_template must contain {year} to retain yearly file identity")
    return [{"year": year, "path": file_template.format(year=year)} for year in years]


def rolling_windows(first_year=2016, last_year=2025, train_years=3,
                    validate_years=1, test_years=1,
                    file_template="data/loan_month/loan_month_{year}.parquet"):
    if min(train_years, validate_years, test_years) <= 0 or first_year > last_year:
        raise ValueError("Require positive window lengths and first_year <= last_year")
    total_years = train_years + validate_years + test_years
    for start in range(first_year, last_year - total_years + 2, test_years):
        validation_start = start + train_years
        test_start = validation_start + validate_years
        yield {
            "train": yearly_files(range(start, validation_start), file_template),
            "validation": yearly_files(range(validation_start, test_start), file_template),
            "test": yearly_files(range(test_start, test_start + test_years), file_template),
        }


def run_window(window, *, categorical_columns=(), timestamp_columns=(),
               sample_rows=1000000, batch_size=500000,
               output_dir="data/model_runs", seed=42):
    years = {split: [item["year"] for item in files] for split, files in window.items()}
    if (not all(years.get(split) for split in ("train", "validation", "test"))
            or max(years["train"]) >= min(years["validation"])
            or max(years["validation"]) >= min(years["test"])):
        raise ValueError("Require strictly chronological train < validation < test years")
    paths = {split: [item["path"] for item in files] for split, files in window.items()}
    resolved = [str(Path(path).resolve()) for files in paths.values() for path in files]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Each year/split must reference a distinct file")
    encoder = FeatureEncoder(categorical_columns, timestamp_columns)
    columns = required_columns(categorical_columns, timestamp_columns)
    print(f"Train {years['train']} -> validate {years['validation']} -> test {years['test']}")

    training_sample = sample_data(paths["train"], sample_rows, -1, columns, seed)
    encoder.fit(training_sample, window["train"])
    X_train, y_train = prepare_data_for_model(training_sample, encoder)
    # Release the raw sampled string columns before loading validation.
    del training_sample
    validation_sample = sample_data(paths["validation"], sample_rows, -1, columns, seed)
    X_validate, y_validate = prepare_data_for_model(validation_sample, encoder)
    del validation_sample

    train_data = lgb.Dataset(
        X_train, label=y_train,
        feature_name=encoder.feature_names,
        categorical_feature=list(encoder.categorical_columns),
    )
    params = validated_parameters(train_data, X_validate, y_validate, verbosity=-1)
    model = lgb.train(params, train_data)
    del X_train, y_train, train_data, X_validate, y_validate

    name = f"train_{min(years['train'])}_{max(years['train'])}_test_{min(years['test'])}_{max(years['test'])}"
    destination = Path(output_dir) / name
    destination.mkdir(parents=True, exist_ok=True)
    encoder.save(destination / "encoder.json")
    model.save_model(str(destination / "model.txt"))
    (destination / "window.json").write_text(json.dumps({
        "files": window, "sample_rows_per_year": sample_rows,
        "sample_seed": seed, "parameters": params,
    }, indent=2) + "\n")
    pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }).to_csv(destination / "importance.csv", index=False)

    predictions = test_in_batches(
        model, paths["test"], batch_size, verbosity=1,
        encoder=encoder, output_path=destination / "predictions.parquet",
    )
    results, _ = error_summary(predictions)
    (destination / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(results)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=2016)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--validate-years", type=int, default=1)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--file-template", default="data/loan_month/loan_month_{year}.parquet")
    parser.add_argument("--categorical-columns", nargs="*", choices=CATEGORICAL_CANDIDATES, default=[],
                        help="Opt in only after confirming prediction-time availability")
    parser.add_argument("--timestamp-columns", nargs="*", choices=TIMESTAMP_CANDIDATES, default=[],
                        help="Opt in only after confirming definitions and timestamp semantics")
    parser.add_argument("--sample-rows", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=500000)
    parser.add_argument("--output-dir", default="data/model_runs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for window in rolling_windows(
        args.first_year, args.last_year, args.train_years, args.validate_years,
        args.test_years, args.file_template,
    ):
        run_window(
            window, categorical_columns=args.categorical_columns,
            timestamp_columns=args.timestamp_columns, sample_rows=args.sample_rows,
            batch_size=args.batch_size, output_dir=args.output_dir, seed=args.seed,
        )


if __name__ == "__main__":
    main()
