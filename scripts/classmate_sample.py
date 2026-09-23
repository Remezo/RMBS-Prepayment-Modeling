#!/usr/bin/env python3
"""Run from the repository root after obtaining your own read-only AWS login."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfld.access import DEFAULT_DATASET, load_month_sample

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--profile', help='Your own AWS CLI profile; omitted on EC2 with an IAM role')
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--max-download-mib', type=int, default=256)
    args = parser.parse_args()
    table = load_month_sample(args.year, args.month, dataset=args.dataset, profile=args.profile,
                             limit=args.limit, max_download_mib=args.max_download_mib)
    print(f'Loaded {table.num_rows:,} rows from reporting month {args.year}-{args.month:02d}.')
    print('Deterministic preview; not a representative statistical sample.')
    for row in table.slice(0,5).to_pylist():
        print(row)
    # Optional, once pandas is installed: df = table.to_pandas()
    # Use table directly with PyArrow, or duckdb.sql('SELECT count(*) FROM table').

if __name__ == '__main__':
    main()
