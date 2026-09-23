# Decoding Mortgage Prepayments



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


