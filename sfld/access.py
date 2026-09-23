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

def month_files(s3, uri, year, month, allow_partial=False):
    if not 1 <= month <= 12 or not 1900 <= year <= 2999:
        raise ValueError('Use a valid reporting year and month')
    bucket, prefix = parse_uri(uri)
    marker = '_PARTIAL_SUCCESS' if allow_partial else '_SUCCESS'
    name = 'manifest.partial.json' if allow_partial else 'manifest.json'
    _, ready = get_json(s3, bucket, prefix+'/'+marker)
    raw, manifest = get_json(s3, bucket, prefix+'/'+name)
    if hashlib.sha256(raw).hexdigest() != ready['manifest_sha256']:
        raise RuntimeError('Dataset publication is incomplete or the manifest changed')
    path = f'performance/year={year}/month={month:02d}/'
    files = [f for f in manifest['files'] if f['path'].startswith(path) and f['path'].endswith('.parquet')]
    if not files:
        raise ValueError(f'No published performance records for {year}-{month:02d}')
    return bucket, prefix, sorted(files, key=lambda f: f['path'])

def load_month_sample(year, month, *, dataset=DEFAULT_DATASET, profile=None,
                      limit=10000, max_download_mib=256, columns=None, s3=None, allow_partial=False):
    """Return at most limit rows, reading at most max_download_mib compressed MiB.

    This is a deterministic preview, NOT a random or representative sample.
    Files are downloaded one at a time and removed when finished.
    """
    if limit < 1 or max_download_mib < 1:
        raise ValueError('Row and download limits must be positive')
    s3 = s3 or client(profile)
    bucket, prefix, files = month_files(s3, dataset, year, month, allow_partial)
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


def select_files(start_year, end_year=None, *, month=None, kind='performance',
                 dataset=DEFAULT_DATASET, profile=None, allow_partial=False, s3=None):
    """Select all verified files. Performance years are reporting years;
    origination years are loan-cohort years. Partial access must be explicit.
    """
    end_year = start_year if end_year is None else end_year
    if not 1900 <= start_year <= end_year <= 2999:
        raise ValueError('Invalid year range')
    if kind not in ('performance', 'origination'):
        raise ValueError('kind must be performance or origination')
    if month is not None and (kind != 'performance' or not 1 <= month <= 12):
        raise ValueError('month must be 1-12 and is only valid for performance')
    s3 = s3 or client(profile)
    bucket, prefix = parse_uri(dataset)
    marker = '_PARTIAL_SUCCESS' if allow_partial else '_SUCCESS'
    name = 'manifest.partial.json' if allow_partial else 'manifest.json'
    _, ready = get_json(s3, bucket, prefix+'/'+marker)
    raw, manifest = get_json(s3, bucket, prefix+'/'+name)
    if hashlib.sha256(raw).hexdigest() != ready['manifest_sha256']:
        raise RuntimeError('Manifest checksum mismatch')
    selected = []
    for f in manifest['files']:
        parts = f['path'].split('/')
        if parts[0] != kind:
            continue
        tags = dict(p.split('=', 1) for p in parts[1:-1] if '=' in p)
        year = int(tags['year' if kind == 'performance' else 'cohort_year'])
        if start_year <= year <= end_year and (month is None or int(tags['month']) == month):
            selected.append(f)
    if not selected:
        raise ValueError('No verified files match this selection')
    return sorted(selected, key=lambda f: f['path'])


def download_selection(files, destination, *, dataset=DEFAULT_DATASET, profile=None,
                       max_download_gb=2, s3=None):
    """Download an entire selection to disk, with a size cap and SHA256 checks.
    Keeps the partition directory structure; reruns reuse verified local files.
    """
    from pathlib import PurePosixPath
    if max_download_gb <= 0 or sum(f['bytes'] for f in files) > max_download_gb*1e9:
        raise ValueError('Selection exceeds max_download_gb; narrow selection or explicitly raise cap')
    s3 = s3 or client(profile)
    bucket, prefix = parse_uri(dataset)
    root = Path(destination).resolve()
    for f in files:
        relative = PurePosixPath(f['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        target = root.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(root):
            raise ValueError('Destination escapes output folder')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == f['bytes'] and sha256(target) == f['sha256']:
            continue
        temporary = target.with_suffix('.download')
        s3.download_file(bucket, prefix+'/'+f['path'], str(temporary), Config=TRANSFER)
        if temporary.stat().st_size != f['bytes'] or sha256(temporary) != f['sha256']:
            raise RuntimeError('Downloaded Parquet checksum mismatch')
        temporary.replace(target)
    return [root/f['path'] for f in files]
