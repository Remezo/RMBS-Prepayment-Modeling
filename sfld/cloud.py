"""Managed S3 transfers using the caller's own AWS credentials."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

TRANSFER = TransferConfig(multipart_threshold=64*1024**2, multipart_chunksize=64*1024**2,
                          max_concurrency=4)

def client(profile=None, region='us-east-2'):
    return boto3.Session(profile_name=profile, region_name=region).client(
        's3', config=Config(retries={'mode': 'standard', 'total_max_attempts': 5},
                           connect_timeout=15, read_timeout=120, max_pool_connections=8))

def parse_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != 's3' or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError('Expected s3://bucket/key or prefix')
    return parsed.netloc, parsed.path.lstrip('/').rstrip('/')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def download_source(s3, uri, folder):
    bucket, key = parse_uri(uri)
    if not key:
        raise ValueError('The source URI must identify a ZIP object')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder/'source.zip'
    receipt = folder/'source-receipt.json'
    head = s3.head_object(Bucket=bucket, Key=key)
    identity = {'uri': uri, 'bytes': head['ContentLength'], 'etag': head['ETag'],
                'version': head.get('VersionId')}
    if receipt.exists() and destination.exists():
        saved = json.loads(receipt.read_text())
        if saved['identity'] == identity and destination.stat().st_size == identity['bytes']:
            if sha256(destination) != saved['sha256']:
                raise RuntimeError('Cached source checksum mismatch')
            return destination
        raise RuntimeError('Source changed; use a new work directory')
    partial = destination.with_suffix('.download')
    extra = {'VersionId': identity['version']} if identity['version'] else {}
    s3.download_file(bucket, key, str(partial), ExtraArgs=extra, Config=TRANSFER)
    after = s3.head_object(Bucket=bucket, Key=key)
    if partial.stat().st_size != identity['bytes'] or after['ETag'] != identity['etag']:
        raise RuntimeError('Source size or version changed during download')
    digest = sha256(partial)
    partial.replace(destination)
    receipt.write_text(json.dumps({'identity': identity, 'sha256': digest}, indent=2))
    return destination

def upload_verified(s3, path, bucket, key):
    digest = sha256(path)
    s3.upload_file(str(path), bucket, key, Config=TRANSFER,
                   ExtraArgs={'Metadata': {'sha256': digest}})
    head = s3.head_object(Bucket=bucket, Key=key)
    if head['ContentLength'] != Path(path).stat().st_size or head.get('Metadata', {}).get('sha256') != digest:
        raise RuntimeError('S3 verification failed: ' + key)
    return digest
