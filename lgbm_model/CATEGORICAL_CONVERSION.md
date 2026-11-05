# Categorical conversion: review and implementation

## What the existing code establishes

`import_data.py` originally selected 11 numeric predictors and the target
`zero_balance_code`. Its commented strings were `loan_id`,
`prepay_penalty_flag`, `property_state`, and `occupancy_status`; its commented
timestamps were `period` and `orig_period` (annotated `timestamp[us]`).
`current_upb` was also commented out. These are code annotations, not verified
Parquet schemas. No yearly loan-month files are present in this checkout.

`main.py` used separate `data/loan_month/loan_month_{year}.parquet` files for
three training years, one validation year, and one test year. Defaults cover
2016–2025: the first window trains on 2016–2018, validates on 2019, and tests
on 2020; the last tests on 2025. Each training/validation file contributes up
to one million sampled rows. Main requests 500,000-row test batches.

`validate.py` tries the parameter grid and uses `calculate_error.py` to rank
models. The existing criterion is true-positive count, maximized; this task
does not change the criterion. `test.py` writes actual/predicted columns for
evaluation. Its old exception handler could skip a failed test file and
return incomplete output; the implementation now propagates the error.

The `sfld/` library handles archive conversion and S3 access. Its source schema
is distinct from the derived loan-month schema expected by this model. In
particular, source zero-balance codes are strings; this model expects integer
codes and maps code 1 to label 1, with codes 0, 2, 3, 9, 15, 16, 96 mapped to 0.

## Recommendation and column decisions

LightGBM supports native categorical splits. Give it integer category codes
and explicitly set `categorical_feature`; merely converting strings to
integers without that declaration would allow ordinal numeric splits.
See the [official LightGBM documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Advanced-Topics.html#categorical-feature-support).

This is a representation choice, not a pre-existing team decision. One-hot
encoding is another valid option with a training-fitted schema, but expands
the feature matrix. Pandas categoricals can also work if every split uses the
same training-derived category dtype; converting each split independently is
unsafe. Native integer categoricals fit the existing Arrow path without either
matrix expansion or conversion to pandas. Actual cardinality measurements are
still needed to assess whether a candidate should be excluded or regularized.

| Actual commented column | Proposed representation | Eligibility/status |
| --- | --- | --- |
| `loan_id` | Exclude from predictors | Recommended identifier exclusion; meaning unconfirmed |
| `prepay_penalty_flag` | Fixed `N/Y` vocabulary, `int32`, declared categorical | Available by explicit opt-in |
| `property_state` | Fixed state/territory vocabulary, `int32`, declared categorical | Available by explicit opt-in |
| `occupancy_status` | Fixed `9/I/P/S` vocabulary, `int32`, declared categorical | Available by explicit opt-in |
| `period` | Numeric month index since January 1970 plus month 1–12 | Conditional on meaning and prediction-time availability; disabled by default |
| `orig_period` | Numeric month index since January 1970 plus month 1–12 | Conditional on meaning and prediction-time availability; disabled by default |

All new columns default to disabled, retaining the original numeric baseline.
No timestamp is passed raw to LightGBM or assigned a category per date. The
timestamp option is a generic calendar representation, not an assertion that
either column is a reporting date or origination date. If that interpretation
is wrong, the option should remain disabled. Existing `loan_age` is retained;
no new age calculation is inferred. Numeric month splits can express
seasonality but do not impose a cyclic relationship between December/January.

`unpaid_principal_balance` is excluded. So is `current_upb`, which the original
selection already excluded. The existing `orig_upb` is retained as requested
by the original numeric feature selection; it is a different column.

`CATEGORY_VOCABULARIES` in `feature_encoder.py` defines the fixed domains:
50 states, DC, and AS/GU/MP/PR/VI; occupancy `9/I/P/S`; penalty flags `N/Y`.
The abbreviations follow [USPS](https://pe.usps.com/text/pub28/28apb.htm);
occupancy and penalty values follow the
[Freddie Mac guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf).
Strings are matched exactly, with no trimming or case conversion. Occupancy
`9` remains a category; nulls, blanks, and values outside the fixed domain map
to `-1`. Review unexpected values rather than deriving new codes from test data.

## How consistency and time separation work

1. `main()` creates one `FeatureEncoder` before the rolling-window loop.
2. Category codes are positions in the fixed vocabulary lists. No rows or
   labels are inspected and there is no `fit()` step.
3. Pass that same encoder to every `run_window()`. Reuse it for training,
   validation, and every test batch. Local Arrow/Parquet dictionary codes
   are decoded to strings and never reused as model codes.
4. Encode nulls and strings outside the fixed vocabulary as `-1`. LightGBM treats
   negative categorical values as missing. Code `0` is a valid category.
   Unknowns share missing-value handling; this does not teach a distinct
   unknown-category effect without training observations of that group.
5. Null timestamps remain null. Valid dates use a fixed epoch and calendar
   extraction, so future dates need no fitted dictionary or training range.
   Timezone-aware timestamps and string dates fail with an explicit error;
   parsing/timezone rules need to be defined upstream.
6. Save a copy of the shared `encoder.json` alongside each `model.txt` and
   `window.json`. Version 2 encoder files record `vocabulary_source: fixed`.
   The loader also accepts older version 1 files and preserves their original
   mappings for existing models. Feature order is fixed and checked at prediction.

Codes stay identical across all new rolling models. A valid category absent
from a training sample still has its predefined code in validation and test.
The model still has no training observations from which to learn that category's
effect. No training/validation/test vocabulary scans or target-based encodings
are performed. Validation outcomes are used for parameter selection. Prior test
years can become training data in later windows, as in the original setup.

These controls do not establish whether an input column itself contains
post-event information. That depends on its definition and prediction horizon.
`zero_balance_code` is checked for known, non-null integer codes and mapped to
a binary code-1 target. Code 1 includes scheduled maturity.

## Pipeline conversion versus upstream yearly preprocessing

| Approach | Runtime and memory | Practical implications |
| --- | --- | --- |
| **Recommended: convert inside the pipeline** | Create fixed dictionaries once; convert each training/validation sample and test batch as needed. Avoids writing all years again. | Original yearly Parquet files remain unchanged. Arrow lookup uses the same fixed vocabulary across all windows. |
| Upstream, stateless preparation | A one-time scan/rewrite of every yearly file; streaming DuckDB `COPY` can keep application memory bounded. Uses extra disk and I/O; repeated experiments may reuse the result. | Useful for agreed parsing, normalization, and fixed calendar features. Keep strings or apply an approved fixed external codebook. |
| Upstream, training-derived integer encoding | Fit only on each window's training data, then stream conversion with that frozen dictionary. Requires a separate cache/version per rolling model, or careful artifact keys. | Independent yearly dictionaries are invalid. Encoded output must carry the exact encoder identity; current pipeline expects raw strings and must not double-encode integer inputs. |

For upstream conversion, DuckDB can register a frozen `{value, code}` table,
LEFT JOIN each selected column, and use `COALESCE(code, -1)` before streaming
`COPY ... TO` a new yearly Parquet file. The dictionary must be frozen before
processing validation/test; do not use `DENSE_RANK()` independently per file.
Keep the year in every output filename and record the dictionary/window version.

For a 500,000-row test batch, the three new int32 code arrays alone occupy
about 6 MB; raw input buffers, timestamp outputs, DuckDB, and LightGBM prediction
workspaces consume additional memory. The implementation projects unused
columns away and avoids pandas/CSR conversion of model matrices. This does
not imply zero-copy processing throughout LightGBM. Training still holds its
sample and LightGBM dataset in memory. Reservoir sampling scans the selected
yearly files and may use substantial memory for a million-row reservoir;
temporary projected samples also consume disk.

Additional compute is optional. Measure actual year-file scan time and peak
memory before deciding whether upstream caching, a smaller batch, or more
hardware is worthwhile. No 20-million-row performance benchmark is claimed.

## Run and filename changes

Install from the repository root:

```sh
python -m pip install -r lgbm_model/requirements.txt
```

The `arrow` extra includes `cffi`, which LightGBM needs for Arrow input. On
macOS the LightGBM wheel also needs the OpenMP runtime (`brew install libomp`).

Run categorical support with the three candidate fields explicitly selected:

```sh
python -m lgbm_model.main \
  --categorical-columns prepay_penalty_flag property_state occupancy_status
```

This uses the existing `data/loan_month/loan_month_{year}.parquet` paths. No new
yearly filenames are necessary. For string-preserving upstream prepared files
with the same required column names/types, change the template:

```sh
python -m lgbm_model.main \
  --file-template 'data/loan_month_prepared/loan_month_{year}_prepared.parquet' \
  --categorical-columns prepay_penalty_flag property_state occupancy_status
```

For 2016 this resolves to
`data/loan_month_prepared/loan_month_2016_prepared.parquet`, and likewise for
each year. Those prepared files are not generated by this task. Do not point
this command at categoricals that have already been replaced with integers.

Only after confirming the timestamp definitions, add
`--timestamp-columns period orig_period`. The file template must contain
`{year}`; manifests retain the year separately from the path. The code checks
chronological split ordering and distinct paths; it does not establish that
rows inside each file match its filename year without a confirmed date column.

Artifacts go under, for example,
`data/model_runs/train_2016_2018_test_2020_2020/`: `model.txt`, `encoder.json`,
`window.json`, `importance.csv`, `predictions.parquet`, and `metrics.json`.
Each test prediction file includes all test years for that window, with its
input year/path list in `window.json`. Existing successful output is replaced
only after all batches finish; failed partial output is removed and the error
propagates. Use a distinct output directory for concurrent/retained runs.

## Verification and remaining real-data inputs

```sh
python -m pip install pytest
python -m pytest -q
```

`tests/test_categorical_features.py` uses synthetic yearly Parquet fixtures to
verify stable codes across splits, local Arrow dictionary differences,
missing/unrecognized values, all-null input, exact-string handling, construction
without training data, legacy artifact reload, fixed date extraction, excluded balance
columns, projected per-year sampling, the rolling filename logic, and failure
cleanup. A real LightGBM model must make categorical tree splits; reloaded
model/encoder predictions from multiple test files and batches must agree with
direct predictions. Another check exercises the real main-window sampling,
validation, training, and prediction flow with a reduced parameter grid.

Local verification completed with **22 tests passed**, including all ten new
categorical/conversion checks and the existing repository tests, using LightGBM
4.6.0, DuckDB 1.5.5, and PyArrow 25.0.1. Both module and script CLI help paths
work, and `git diff --check` passed. These checks use synthetic loan-month
fixtures rather than the absent yearly production data.

Real yearly file schemas, cardinalities, sentinel values, and timestamp meanings
remain unverified. Before a production run, obtain names/types from at least
one training, validation, and test year, and check schema drift across all
selected years. Obtain a few non-sensitive values and null/distinct counts for
the candidate columns. Confirm which values are available at prediction time.
No access credentials or full loan records are needed for that review.

## Change inventory and rationale

| File | Change and purpose |
| --- | --- |
| `feature_encoder.py` | Shared fixed Arrow encoder and JSON persistence; stable codes across all rolling windows, with legacy artifact loading. |
| `import_data.py` | Explicit existing feature allowlist, projected per-year sampling, isolated temporary files, shared conversion, and binary-target validation. |
| `main.py` | Import-safe entry point, year/path window records, optional features/file template, encoder wiring, and per-window artifacts. Preserves default split sizes/years. |
| `test.py` | Same frozen encoder per batch, direct numeric Arrow prediction, feature-order checks, and complete-output publication with propagated errors. |
| `validate.py` | Module/script imports and isolated temporary prediction output; parameter grid and selection metric preserved. |
| `requirements.txt` in this folder | Model dependencies, including LightGBM's Arrow extra; root dependency file preserved. |
| `README.md` | Link to this design and usage guide. |
| `tests/test_categorical_features.py` | Contract and actual model integration checks described above. |
| This document | Confirmed facts, conditional choices, alternatives, usage, limits, and change inventory. |

For local verification, dependencies were installed in
`/private/tmp/rmbs-categorical-venv`. The approved Homebrew installation added
`libomp`; Homebrew also performed its automatic update and cache cleanup.
No yearly source files were rewritten and no AWS data was downloaded.
Remaining work before enabling unconfirmed predictors on
real data is the schema/meaning review and a representative runtime/memory run.
