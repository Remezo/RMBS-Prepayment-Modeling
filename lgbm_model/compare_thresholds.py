"""Compare threshold-based monthly rates using saved loan-level predictions."""
import argparse
from pathlib import Path

import duckdb
import pandas as pd


def compare_thresholds(file, thresholds):
    thresholds = sorted(set(thresholds))
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Provide thresholds between 0 and 1")
    counts = ", ".join(
        f"count(*) FILTER (WHERE predicted >= {value}) AS predicted_{i}"
        for i, value in enumerate(thresholds)
    )
    # All thresholds are counted in one scan; loan-level rows stay in DuckDB.
    with duckdb.connect() as con:
        rates = con.execute(f"""
            SELECT date_trunc('month', period)::DATE AS period,
                   count(*) AS loan_count,
                   count(*) FILTER (WHERE actual = 1) AS actual_prepayments,
                   avg(predicted) AS probability_smm,
                   count(*) FILTER (WHERE period IS NULL OR actual IS NULL
                       OR actual NOT IN (0, 1) OR predicted IS NULL
                       OR NOT isfinite(predicted)
                       OR predicted NOT BETWEEN 0 AND 1) AS invalid_rows,
                   {counts}
            FROM read_parquet(?)
            GROUP BY 1 ORDER BY 1
        """, [str(file)]).fetchdf()
    if rates.empty or rates["invalid_rows"].sum():
        raise ValueError("Require nonempty predictions with valid periods, labels and probabilities")

    monthly = pd.concat([
        rates[["period", "loan_count", "actual_prepayments", "probability_smm"]].assign(
            threshold=value, predicted_prepayments=rates[f"predicted_{i}"],
        ) for i, value in enumerate(thresholds)
    ], ignore_index=True)
    monthly["actual_smm"] = monthly["actual_prepayments"] / monthly["loan_count"]
    monthly["predicted_smm"] = monthly["predicted_prepayments"] / monthly["loan_count"]
    monthly["absolute_error"] = (monthly["actual_smm"] - monthly["predicted_smm"]).abs()
    summary = monthly.groupby("threshold", as_index=False).agg(
        monthly_rate_mae=("absolute_error", "mean"),
        actual_prepayments=("actual_prepayments", "sum"),
        predicted_prepayments=("predicted_prepayments", "sum"),
    ).sort_values(["monthly_rate_mae", "threshold"])
    return summary, monthly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Parquet with actual, predicted and period")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.predictions.is_file():
        parser.error(f"Missing predictions file: {args.predictions}")
    print("Comparing thresholds using all saved prediction rows...", flush=True)
    summary, monthly = compare_thresholds(args.predictions, args.thresholds)
    output = args.output_dir or args.predictions.parent / "threshold_comparison"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    for threshold, rows in monthly.groupby("threshold"):
        rows.to_csv(output / f"monthly_rates_{threshold:g}.csv", index=False)
    probability_mae = (monthly["actual_smm"] - monthly["probability_smm"]).abs().mean()
    print(f"Probability-based monthly rate MAE: {probability_mae:.6f}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved comparison to {output}")
    print("Choose thresholds using validation data; test-data comparisons are exploratory.")


if __name__ == "__main__":
    main()
