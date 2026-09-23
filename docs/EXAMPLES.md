# Classmate examples — accepted partial dataset

The available dataset is a deliberately partial snapshot: 158 verified source files,
1,952,902,609 rows, approximately 33.35 GB. Source cohorts span 2006 Q1–2026 Q1.
Reporting years/months contain only loans from included source files. They are not
complete coverage of every loan in that reporting period. Uncommitted files are excluded.

Each person needs their own AWS identity with the dataset read-only permissions.
GitHub access does not grant S3 access. Do not share AWS passwords or keys.
Install the repository requirements and configure your own AWS profile named `sfld`.
Use your institution/account's supplied sign-in instructions; `--profile sfld` selects
that identity. On EC2 with an authorized role, omit `--profile sfld`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Small working monthly sample

```bash
python scripts/classmate_sample.py --year 2020 --month 1 --limit 1000 --profile sfld --allow-partial
```

This returns up to 1,000 rows, not the entire month. It is a deterministic preview,
not a representative random sample. The default download cap is 256 MiB.

## All records for one reporting year

```bash
python scripts/download_data.py --year 2020 --profile sfld --allow-partial
```

This first shows the exact selection size without downloading data. Add
`--download --max-download-gb 10` after reviewing that size. The cap must cover
the selected compressed files; nothing downloads if the selection exceeds it.

## All records for one reporting month

```bash
python scripts/download_data.py --year 2020 --month 1 --profile sfld --allow-partial --download --max-download-gb 2
```

## A reporting-year range (inclusive)

```bash
python scripts/download_data.py --year 2018 --end-year 2020 --profile sfld --allow-partial
```

Add `--download --max-download-gb 20` to download after checking the reported size.
No fixed cap is a prediction of dataset size; raise it only as needed.

## Original loan information (origination records)

```bash
python scripts/download_data.py --kind origination --year 2020 --profile sfld --allow-partial --download --max-download-gb 2
```

This is original loan information for the **2020 loan cohort**, not every loan
reported in 2020. A 2020 performance month also contains loans from older cohorts.
For matching original information, download the cohorts referenced by the selected
performance rows' `cohort_year` field, then join on `loan_identifier`.

```python
from sfld.access import load_month_sample, select_files, download_selection
import duckdb

sample = load_month_sample(2020, 1, profile='sfld', allow_partial=True, limit=1000)
cohorts = sorted(set(sample['cohort_year'].to_pylist()))
files = []
for year in cohorts:
    files.extend(select_files(year, kind='origination', profile='sfld', allow_partial=True))
paths = download_selection(files, 'data', profile='sfld', max_download_gb=2)
con = duckdb.connect()
con.register('performance_sample', sample)
con.read_parquet([str(p) for p in paths], hive_partitioning=True).create_view('originals')
result = con.sql('''
    SELECT p.*, o.original_upb, o.original_interest_rate, o.property_state
    FROM performance_sample p
    LEFT JOIN originals o USING (loan_identifier)
''').fetch_arrow_table()
print(result.slice(0, 5).to_pylist())
```

The raw vendor ZIP is separate from the origination Parquet and is not included in
classmate read-only permissions. These examples use Parquet to avoid downloading
the entire 43 GB ZIP.

## Query downloaded files without loading everything into memory

```python
import duckdb
con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET temp_directory='data/duckdb-temp'")
print(con.sql("""
    SELECT year, month, count(*) AS observations
    FROM read_parquet('data/performance/year=*/month=*/*.parquet', hive_partitioning=true)
    WHERE year BETWEEN 2018 AND 2020
    GROUP BY year, month ORDER BY year, month
""").fetchall())
```

This queries only what you downloaded. Use separate destination folders if you want
to keep downloads for different analyses distinct. S3 downloads may incur transfer
and request charges to the bucket owner.
