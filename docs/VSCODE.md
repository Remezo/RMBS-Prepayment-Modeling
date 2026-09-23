# Try the dataset in VS Code (Mac)

## Clone and open

In a terminal:

```bash
git clone https://github.com/Remezo/mortgage-prepayment-research.git
cd mortgage-prepayment-research
```

In VS Code choose **File → Open Folder** and select `mortgage-prepayment-research`.
Open **Terminal → New Terminal** and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use **Python: Select Interpreter** in the Command Palette and select `.venv/bin/python`.
Install Microsoft's Python extension if that command is unavailable.

## Mike: use your existing AWS login

```bash
export PATH="$HOME/.local/bin:$PATH"
aws sts get-caller-identity
```

If the session has expired, run `aws login --region us-east-2` and select **Sky is the Limit**
in the browser. The account ID should be `155980438952`.
The examples below use your default AWS profile, so do not add `--profile sfld`.

## Run a small sample first

```bash
python scripts/classmate_sample.py --year 2020 --month 1 --limit 1000 --allow-partial
```

This prints five preview records from a sample of up to 1,000 rows. It downloads only
as many complete Parquet files as needed, within a 256 MiB cap. It does not download
the entire dataset. The sample is deterministic, not statistically representative.

## Preview selections and their download sizes

```bash
# Reporting year
python scripts/download_data.py --year 2020 --allow-partial

# Reporting month
python scripts/download_data.py --year 2020 --month 1 --allow-partial

# Reporting years, inclusive
python scripts/download_data.py --year 2018 --end-year 2020 --allow-partial

# Original loan information for the 2020 cohort
python scripts/download_data.py --kind origination --year 2020 --allow-partial
```

These four commands list the selection size without downloading Parquet. To download
a selection, add `--download --max-download-gb 10`; adjust the cap to the reported size.
Files are saved under the ignored `data/` folder.

For Python examples and matching monthly observations to origination records, see
[EXAMPLES.md](EXAMPLES.md). Omit `profile='sfld'` in those examples to use Mike's default login.

## Classmates: individual S3 credentials

Classmate credentials have not been issued yet. Once each person receives their own
scoped access key privately, they can configure it interactively:

```bash
aws configure --profile sfld
```

Enter their own key ID and secret at the prompts, region `us-east-2`, and output `json`.
Then add `--profile sfld` to the example commands. Never paste keys into source code,
chat, notebooks, or GitHub. GitHub repository access is separate from AWS access.

## Dataset status

This is the accepted partial snapshot (about 33.35 GB), not the full archive.
`--allow-partial` explicitly selects its verified manifest. Performance records are
partitioned by reporting year/month; origination records by source cohort year.
Reporting periods exclude loans from omitted source files. The original ZIP and
Parquet data stay in private S3, not in GitHub.
