"""Read a bounded sample from one published reporting month using private S3 access."""
import hashlib
import json
import tempfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from .cloud import client, parse_uri, sha256, TRANSFER

DEFAULT_DATASET = 's3://s3sfld/parquet/release47-monthly'
DEFAULT_COLUMNS = ['loan_identifier', 'monthly_reporting_period', 'current_actual_upb',
                   'current_loan_delinquency_status', 'cohort_year']

def get_json(s3, bucket, key):
    with s3.get_object(Bucket=bucket, Key=key)['Body'] as body:
        data = body.read()
    return data, json.loads(data)

def month_files(s3, uri, year, month):
    if not 1 <= month <= 12 or not 1900 <= year <= 2999:
        raise ValueError('Use a valid reporting year and month')
    bucket, prefix = parse_uri(uri)
    _, ready = get_json(s3, bucket, prefix+'/_SUCCESS')
    raw, manifest = get_json(s3, bucket, prefix+'/manifest.json')
    if hashlib.sha256(raw).hexdigest() != ready['manifest_sha256']:
        raise RuntimeError('Dataset publication is incomplete or the manifest changed')
    path = f'performance/year={year}/month={month:02d}/'
    files = [f for f in manifest['files'] if f['path'].startswith(path) and f['path'].endswith('.parquet')]
    if not files:
        raise ValueError(f'No published performance records for {year}-{month:02d}')
    return bucket, prefix, sorted(files, key=lambda f: f['path'])

def load_month_sample(year, month, *, dataset=DEFAULT_DATASET, profile=None,
                      limit=10000, max_download_mib=256, columns=None, s3=None):
    """Return at most limit rows, reading at most max_download_mib compressed MiB.

    This is a deterministic preview, NOT a random or representative sample.
    Files are downloaded one at a time and removed when finished.
    """
    if limit < 1 or max_download_mib < 1:
        raise ValueError('Row and download limits must be positive')
    s3 = s3 or client(profile)
    bucket, prefix, files = month_files(s3, dataset, year, month)
    columns = columns or DEFAULT_COLUMNS
    batches = []
    rows = downloaded = 0
    with tempfile.TemporaryDirectory(prefix='sfld-sample-') as folder:
        for part in files:
            if downloaded + part['bytes'] > max_download_mib*1024**2:
                raise RuntimeError('Download cap reached before requested row count. Lower --limit or increase --max-download-mib.')
            path = Path(folder)/'part.parquet'
            s3.download_file(bucket, prefix+'/'+part['path'], str(path), Config=TRANSFER)
            if path.stat().st_size != part['bytes'] or sha256(path) != part['sha256']:
                raise RuntimeError('Downloaded Parquet checksum mismatch')
            downloaded += part['bytes']
            for batch in pq.ParquetFile(path).iter_batches(batch_size=min(limit,64000), columns=columns):
                take = min(batch.num_rows, limit-rows)
                batches.append(batch.slice(0,take))
                rows += take
                if rows >= limit:
                    break
            path.unlink()
            if rows >= limit:
                break
    if not batches:
        raise ValueError('Month has no rows')
    return pa.Table.from_batches(batches)
