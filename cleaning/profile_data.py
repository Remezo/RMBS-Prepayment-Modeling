"""Read-only initial quality report for local performance Parquet files."""
import argparse
import glob
import json
from pathlib import Path
import duckdb

def profile(pattern):
    files=sorted(glob.glob(pattern,recursive=True))
    if not files:
        raise ValueError('No Parquet files match the pattern')
    with duckdb.connect() as con:
        con.execute("SET memory_limit='2GB'")
        Path('work/duckdb-temp').mkdir(parents=True,exist_ok=True)
        con.execute("SET temp_directory='work/duckdb-temp'")
        con.read_parquet(files,hive_partitioning=True).create_view('records')
        rows,missing_ids,missing_dates,bad_dates=con.sql("""
            SELECT count(*),
            count(*) FILTER(WHERE loan_identifier IS NULL OR trim(loan_identifier)=''),
            count(*) FILTER(WHERE monthly_reporting_period IS NULL),
            count(*) FILTER(WHERE monthly_reporting_period IS NOT NULL AND
                NOT regexp_full_match(CAST(monthly_reporting_period AS VARCHAR), '[12][0-9]{3}(0[1-9]|1[0-2])'))
            FROM records
        """).fetchone()
        duplicates=con.sql("""SELECT coalesce(sum(n-1),0) FROM (
            SELECT count(*) n FROM records
            GROUP BY loan_identifier,monthly_reporting_period HAVING count(*)>1)
        """).fetchone()[0]
        coverage=con.sql('SELECT min(monthly_reporting_period),max(monthly_reporting_period) FROM records').fetchone()
    return dict(files=len(files),rows=rows,missing_identifiers=missing_ids,
                missing_reporting_dates=missing_dates,malformed_reporting_dates=bad_dates,
                duplicate_key_excess_rows=int(duplicates),reporting_period_min=coverage[0],
                reporting_period_max=coverage[1],scope='initial_key_date_audit_only')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glob',required=True)
    parser.add_argument('--output',default='work/quality.json')
    args=parser.parse_args()
    result=profile(args.glob)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps(result,indent=2,default=str))
