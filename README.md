# SFLD Parquet for classmates

Convert Freddie Mac's Standard Single-Family Loan-Level Dataset (Release 47 / July 2026 layout) into compressed Parquet and read a selected reporting month from a private S3 bucket.

**Data stays in S3. This repository contains code, documentation, and synthetic tests only.** GitHub membership does not grant S3 access. Each classmate needs their own AWS identity with read-only dataset permissions.

## Layout and meaning

```text
s3://s3sfld/parquet/release47-monthly/
  origination/cohort_year=1999/orig_1999Q1-0.parquet
  performance/year=2020/month=01/perf_1999Q1-0.parquet
  performance/year=2020/month=02/perf_1999Q1-0.parquet
  ...
  schema.json
  manifest.json
  _SUCCESS
```

- `performance/year/month` is the **monthly reporting date**, not loan origination. Every performance row also retains `cohort_year` and `monthly_reporting_period`.
- `origination/cohort_year` is the annual source-file cohort. The source supplies origination quarters, not an exact origination month. First-payment month is a different concept and is not substituted for origination month.
- Join the tables on `loan_identifier`. Origination rows describe loans; performance rows describe loan-months. Avoid treating repeated performance records as independent loans.
- Multiple quarterly cohorts can contribute files to the same reporting month. Read all month files for complete monthly analysis; do not choose one file and call it the whole month.
- `_SUCCESS` is published last. Until it exists, conversion is incomplete and the sample reader refuses to load the dataset.
- The completion manifest states whether the run covers the full archive or a selected set of cohorts.

## Classmates: install and read a sample

Use Python 3.11 or newer. Accept your GitHub invitation, then:

```bash
git clone https://github.com/Remezo/sfld-parquet.git
cd sfld-parquet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead.

Authenticate using the individual AWS access provided by the project owner. For IAM Identity Center, use `aws configure sso --profile sfld-class` followed by `aws sso login --profile sfld-class`. If your administrator instead provides a compatible console login with local-development permission, use `aws login --profile sfld-class --region us-east-2`.

**Do not use the owner's account, private key, or credentials.** See [private access setup](docs/ACCESS.md).

Once the dataset has a completion marker:

```bash
python scripts/classmate_sample.py \
  --profile sfld-class --year 2026 --month 1 --limit 10000
```

This downloads files from just the selected month, validates their SHA-256 checksums, selects five useful columns, and returns at most 10,000 rows. The default download cap is 256 MiB; increase `--max-download-mib` deliberately for larger samples. The preview is deterministic, **not random or representative**. Monthly partitions may contain many millions of records.

To use it from a notebook launched in the repository root:

```python
from sfld.access import load_month_sample

table = load_month_sample(
    2026, 1,
    profile="sfld-class",
    limit=10_000,
    columns=["loan_identifier", "monthly_reporting_period",
             "current_actual_upb", "current_loan_delinquency_status", "cohort_year"],
)
print(table.slice(0, 5).to_pylist())
# If pandas is installed: df = table.to_pandas()
```

S3's request and internet download charges apply; repeated full-month downloads can cost money. The sample downloads selected files rather than the entire archive. [S3 pricing](https://aws.amazon.com/s3/pricing/).

## Owner: one-time conversion on EC2

Use the EC2 IAM role with read access to the ZIP and read/write access under `s3sfld/parquet/`. No access keys belong in the script. An 8 GiB Linux instance is the intended baseline; default sorting memory is 3 GB and two threads. Allow disk space for the 40 GiB ZIP plus one source file's typed Parquet, sorted output, and temporary sort spill. The existing 300 GiB volume is suitable for this staged approach; **do not unzip the full 335 GB archive**.

Copy/clone this repository to EC2. On Ubuntu, install `python3-venv` if it is absent. From the repository root:

```bash
nohup bash scripts/run_ec2.sh > conversion.log 2>&1 < /dev/null &
tail -f conversion.log
```

The script downloads the original ZIP directly from S3 onto EC2, converts and uploads one source file at a time, and deletes only its own verified scratch output. The downloaded source ZIP is retained. Keep `work/full-conversion/` to resume safely after interruption; rerun the same command. Completed source files are checked rather than duplicated.

For a smaller pilot, use a **different output prefix and work directory**:

```bash
python -m sfld.convert \
  --source s3://s3sfld/full_set_standard_historical_data.zip \
  --output s3://s3sfld/parquet/pilot-2026-monthly \
  --work-dir work/pilot-2026 \
  --cohorts 2026
```

Do not add `--cohorts` to the full-run launcher: a partial cohort selection should never be published under the full dataset's prefix. To perform a full conversion without AWS, both `--source` and `--output` can be local paths. The converter uses Unix file locking and is intended for Linux/macOS; the classmate reader also works on Windows.

The full command refuses a non-empty destination unless its matching local checkpoint exists. Use a new output prefix for a new release or changed configuration. This is a single-publisher tool; do not run competing instances into the same prefix. On failure, uncommitted files may remain under that prefix, but `_SUCCESS` is absent. Resume from the same work directory so deterministic filenames are rewritten and validation completes.

Check completion with:

```bash
aws s3 ls s3://s3sfld/parquet/release47-monthly/_SUCCESS --region us-east-2
```

After completion, confirm classmates can read the sample before shutting down the EC2 instance. Stopping ends compute charges, but attached EBS storage remains billable. Terminating and removing unneeded volumes is a separate owner decision. [EC2 pricing](https://aws.amazon.com/ec2/pricing/on-demand/).

## Validation and data conventions

- Explicit column order follows the [July 2026 Freddie Mac guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf): 31 origination fields and 35 performance fields.
- Monetary amounts use `decimal128(18,2)` and interest rates use `decimal128(7,3)`; identifiers and codes remain strings. Leading-zero ZIP prefixes and status codes are preserved.
- Empty values become nulls. Vendor sentinel values such as 9999 or 999 remain unchanged; users must interpret them with the data dictionary. Negative recovery values are not rewritten.
- Source ZIP member CRCs, byte counts, and line counts are checked while reading. Incorrect field counts or invalid reporting dates fail the run.
- Every output file is decoded, monthly placement checked, and total rows reconciled before publication.
- SHA-256 values are recorded for each output and verified by the sample reader. S3 publication checks size and stored checksum metadata; this is not a full remote byte-for-byte reread.
- Sorting uses disk spill to avoid reading the complete source file into RAM. Output files contain at most two million rows; compressed sizes vary. Quarterly cohort boundaries may produce small files in sparse months. A later compaction job can reduce file counts if query performance warrants it.
- No Athena/Glue catalog is created automatically. The folders use Hive partition names compatible with those tools after table registration.

Run the synthetic correctness and failure-path tests:

```bash
python -m pip install pytest
python -m pytest -q
```

See [validation evidence](docs/VALIDATION.md) for the real-data pilot results. No Freddie Mac data or credentials are included in tests or Git history.

## Invite classmates

Go to this repository's **Settings → Collaborators → Add people** and invite their GitHub usernames. Separately grant each person read-only AWS access as described in [ACCESS.md](docs/ACCESS.md). No invitations or AWS grants are made automatically by these scripts.

Freddie Mac data remains subject to its [dataset terms](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset). Keep the bucket and repository private and ensure participants have the required dataset access. This repository does not grant a license to redistribute the underlying data.
