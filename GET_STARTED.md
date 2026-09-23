# Get started in VS Code

## 1. Get the code

You need access to this private GitHub repository. In a terminal:

```bash
git clone https://github.com/Remezo/mortgage-prepayment-research.git
cd mortgage-prepayment-research
```

Already cloned it? Open its folder and run `git pull`. In VS Code use **File → Open Folder**, select `mortgage-prepayment-research`, then **Terminal → New Terminal**.

## 2. Install the Python environment

Use Python 3.11 or later. On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `py -m venv .venv` then `.venv\Scripts\Activate.ps1`. If activation is restricted, run `.venv\Scripts\python.exe` directly instead of `python`.

Install Microsoft’s Python extension in VS Code and choose **Python: Select Interpreter → .venv**.

## 3. Connect to S3

**Mike:** use the existing AWS CLI login. Check `aws sts get-caller-identity`. If expired, run `aws login --region us-east-2` and select Sky is the Limit. On this Mac, `export PATH="$HOME/.local/bin:$PATH"` makes the installed CLI available.

**Classmates:** request your individual S3-only credentials from Mike. They have not yet been issued. Once issued, run `aws configure --profile sfld`, enter your own credentials privately, and use region `us-east-2`. Add `--profile sfld` to each command below. Do not use Mike’s administrator login or commit credentials.

## 4. Try 1,000 rows

```bash
python scripts/classmate_sample.py --year 2020 --month 1 --limit 1000 --allow-partial
```

Expected: a row count and five preview records. This is a deterministic sample, not a representative training dataset.

## 5. Choose your data

```bash
# Preview a full reporting year, without downloading it
python scripts/download_data.py --year 2020 --allow-partial

# Download one reporting month (~177 MB for January 2020)
python scripts/download_data.py --year 2020 --month 1 --allow-partial --download --max-download-gb 2

# Preview an inclusive range (~6.24 GB for 2018–2020)
python scripts/download_data.py --year 2018 --end-year 2020 --allow-partial

# Download original loan records for the 2020 cohort
python scripts/download_data.py --kind origination --year 2020 --allow-partial --download --max-download-gb 2
```

Reporting year and loan-cohort year mean different things. Use the monthly data’s `cohort_year` values to select origination files and join on `loan_identifier`. See the complete [join example](docs/EXAMPLES.md).

## 6. Run the initial quality report

After downloading performance files:

```bash
python cleaning/profile_data.py --glob 'data/performance/year=*/month=*/*.parquet' --output work/quality.json
```

The report does not change data. It checks key completeness, duplicates and reporting-date shape. It is an initial audit, not a complete cleaning procedure.

## 7. Pick a project task

Read [ROADMAP.md](ROADMAP.md). Create an issue, agree an owner, and use a branch such as `cleaning/date-validation`. Follow the [contribution guide](CONTRIBUTING.md). Models and final research results are still to be built.

## Common problems

| Message/problem | What to do |
|---|---|
| GitHub 404 | Sign into the invited GitHub account; ask Mike for repo access |
| Missing credentials / AccessDenied | Check your own AWS profile and request dataset permissions |
| `_SUCCESS` missing | Add `--allow-partial`; this is intentionally a partial dataset |
| Download cap exceeded | Preview selection size, narrow the selection or raise the cap deliberately |
| No matching files | Check coverage; not every source cohort is included |
| Package import fails | Select the `.venv` interpreter and reinstall requirements |
