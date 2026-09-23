# Validation evidence

Validated locally on 2026-09-23 using PyArrow 25.0.1 and DuckDB 1.5.5.

The real `historical_data_2026Q1.zip` nested inside the user's full Standard historical archive was converted into:

- 143,965 origination records, partitioned by cohort year 2026.
- 203,412 monthly performance records, partitioned by reporting year/month.
- 347,377 total records reconciled to source line counts.

All pilot outputs were decoded and monthly placement checked. This pilot validates the 2026 cohort only; the full historical conversion performs the same validation per source file as it runs.

Ten synthetic tests passed covering cross-year reporting partitions, exact decimal amounts, leading-zero codes and postal prefixes, nulls, preserved vendor sentinels, a last line without newline, invalid reporting dates, wrong column counts, non-empty output protection, unchanged-file resume, source-change rejection, and the classmate reader's row limit and manifest checksum.

No AWS student identity has been provisioned as part of this test. Actual classmate access requires the separate permissions described in ACCESS.md.
