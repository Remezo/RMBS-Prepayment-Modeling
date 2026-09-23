# RMBS Prepayment Modeling

## This is how you can read the data

1. Download the shared access-key CSV from our [team access-key folder](https://drive.google.com/drive/u/0/folders/1yqB9Pjfqd_yNYSHx1I_H1BjJtEETLeKH). If Drive asks for access, contact Mike. Keep the CSV outside this repository and never paste keys into code or GitHub.
2. Install the [AWS CLI](https://aws.amazon.com/cli/), clone this repository and install the Python requirements (see [GET_STARTED.md](GET_STARTED.md)).
3. In your terminal, run `aws configure --profile sfld`. Enter the **Access key ID** and **Secret access key** from the CSV, region **us-east-2**, and output format **json**.
4. From the repository folder, run this Python example using your project environment:

```python
from sfld.access import load_month_sample

# Read up to 1,000 records from January 2020.
data = load_month_sample(
    2020, 1, profile="sfld", limit=1000, allow_partial=True
)
print(data.slice(0, 5).to_pylist())
```

Change `2020, 1` to your reporting year and month. This is a small deterministic preview,
not the entire month or a representative training sample. The dataset is the accepted
partial archive. [Full year, month, year-range and origination examples](docs/EXAMPLES.md).

The shared `sfld-class-readonly` credentials were tested successfully against S3.
GitHub access and access to the Drive folder are separate; the folder's sharing
permissions have not been verified by this repository. Only authorized teammates
should receive the credentials.

A shared research project on mortgage prepayment prediction and structured credit, based on Isha Wadekar’s project presentation. The repository combines the existing Freddie Mac SFLD import tools with a documented workflow for cleaning, feature engineering, LightGBM modelling, validation, and RMBS cash-flow analysis.

**New here? Read [GET_STARTED.md](GET_STARTED.md).**

Track team work on the private [GitHub project board](https://github.com/users/Remezo/projects/6).

## Project stages

| Stage | Start here | Current status |
|---|---|---|
| Data Import | [data_import](data_import/README.md) | Working S3 selection, download and sample tools |
| Cleaning | [cleaning](cleaning/README.md) | Working initial quality report; cleaning rules need team approval |
| Modelling | [modelling](modelling/README.md) | Research protocol and experiment configuration; models not yet trained |
| Results | [results](results/README.md) | Reporting template; no research results claimed |

## Data available now

The accepted partial dataset contains **1,952,902,609 rows**, about **33.35 GB**, from **158 verified source files**. Source cohorts span 2006 Q1–2026 Q1. Performance folders use reporting year/month; origination folders use cohort year. Reporting periods omit loans from excluded source files, so coverage is not the complete Freddie Mac archive.

Location: `s3://s3sfld/parquet/release47-monthly/`.
Use `--allow-partial` to acknowledge this snapshot. The partial manifest lists verified files; uncommitted files must not be used. Raw data and credentials never belong in this repository.

## Research workflow

1. Record the input manifest, reporting window, and selected cohorts.
2. Audit identifiers, dates, missing values, duplicates, source codes, and join coverage.
3. Define prepayment labels and the population at risk before training.
4. Add loan and borrower features, then licensed macro data using publication dates.
5. Compare a simple baseline with LightGBM using chronological validation.
6. Report calibration and discrimination across rate regimes.
7. Translate validated probabilities into clearly specified cash-flow scenarios.

The deck proposes Bloomberg macroeconomic inputs; no Bloomberg data has been acquired or included. Valuation outputs are a later research stage, not an implemented production pricing model.

## Working together

Team: Mike, Isha, Devansh, Shivam, Carl, and Hyumin. Workstream ownership is unassigned until the team agrees. Use an issue for each task and a branch/pull request for changes. See [CONTRIBUTING.md](CONTRIBUTING.md), [ROADMAP.md](ROADMAP.md), and [the experiment template](results/EXPERIMENT_TEMPLATE.md).

## Repository map

```text
GET_STARTED.md           First successful VS Code run
sfld/                    Tested archive conversion and S3 access library
scripts/                 Command-line download and sample entry points
data_import/             Source selection, provenance and access guidance
cleaning/                Initial quality report and cleaning specification
modelling/               Leakage controls, model plan and experiment config
results/                 Metrics/report templates; no fabricated results
docs/                    Detailed examples, permissions and validation notes
tests/                   Automated converter and access tests
```

[Year/month/range examples](docs/EXAMPLES.md) · [VS Code details](docs/VSCODE.md) · [AWS access](docs/ACCESS.md) · [Validation](docs/VALIDATION.md)

GitHub invitations and AWS permissions are separate. Shared classmate AWS credentials are available through the team folder linked above. Do not use AWS project invitations for S3-only access: those grant project administration.
