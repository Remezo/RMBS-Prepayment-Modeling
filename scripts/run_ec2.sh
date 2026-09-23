#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
exec .venv/bin/python -u -m sfld.convert \
  --source s3://s3sfld/full_set_standard_historical_data.zip \
  --output s3://s3sfld/parquet/release47-monthly \
  --work-dir work/full-conversion "$@"
