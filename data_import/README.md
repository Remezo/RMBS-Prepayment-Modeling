# Data Import

Use `scripts/download_data.py` for complete selections and `scripts/classmate_sample.py` for a bounded preview. Both use `sfld/access.py`; do not fork separate download logic in notebooks.

## Inputs and outputs

Input: private S3 accepted partial manifest plus the listed Parquet files.
Output: local `data/performance/year=YYYY/month=MM/` and `data/origination/cohort_year=YYYY/` directories. Local data are ignored by Git.

Before every research run, record the Git commit, manifest SHA256, selected date window, cohort coverage, columns and file/row counts. Retain identifiers as strings. The access library verifies downloaded file hashes.

## Acceptance criteria

- Each file is in the verified manifest and matches its checksum.
- Coverage is described as partial, including omitted source cohorts.
- Dates are reporting dates for performance; cohort years for origination.
- Macroeconomic files have a source, license, units, frequency, release dates and revision policy.
- No credentials, borrower records, vendor ZIPs or licensed Bloomberg extracts are committed.

The conversion utilities remain available under `sfld/convert.py` and `scripts/run_ec2.sh`. Do not restart the full conversion or overwrite the accepted snapshot without agreeing a new version and destination.
