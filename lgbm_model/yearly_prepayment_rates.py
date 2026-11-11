"""Count prepayments by reporting year across the supplied Parquet files."""
import argparse
from pathlib import Path

import duckdb

try:
    from .import_data import ZERO_BALANCE_CODES
except ImportError:
    from import_data import ZERO_BALANCE_CODES


def yearly_prepayment_rates(files):
    codes = ", ".join(map(str, ZERO_BALANCE_CODES))
    with duckdb.connect() as con:
        result = con.execute(f"""
            SELECT year(period) AS year,
                   count(*) AS loan_month_rows,
                   count(*) FILTER (WHERE zero_balance_code = 1) AS prepayments,
                   100.0 * count(*) FILTER (WHERE zero_balance_code = 1)
                       / count(*) AS prepayment_rate_pct,
                   count(*) FILTER (WHERE period IS NULL
                       OR zero_balance_code IS NULL
                       OR zero_balance_code NOT IN ({codes})) AS invalid_rows
            FROM read_parquet(?)
            GROUP BY 1 ORDER BY 1
        """, [[str(path) for path in files]]).fetchdf()
    if result.empty or result["invalid_rows"].sum():
        raise ValueError("Require nonempty data with valid periods and zero balance codes")
    return result.drop(columns="invalid_rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-template", default="data/loan_month/loan_month_perf_{year}.parquet")
    parser.add_argument("--first-year", type=int, default=2016)
    parser.add_argument("--last-year", type=int, default=2026)
    parser.add_argument("--output", type=Path, help="Optional CSV report path")
    args = parser.parse_args()
    if args.first_year > args.last_year or "{year}" not in args.file_template:
        parser.error("Require first-year <= last-year and {year} in file-template")
    files = [Path(args.file_template.format(year=year))
             for year in range(args.first_year, args.last_year + 1)]
    for path in files:
        if not path.is_file():
            parser.error(f"Missing input file: {path}")
    print("Scanning all rows; grouping by the reporting year in period.", flush=True)
    result = yearly_prepayment_rates(files)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
