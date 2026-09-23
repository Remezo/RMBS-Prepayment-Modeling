"""Convert Release 47 nested ZIPs; publish only validated, complete datasets."""
import argparse
import fcntl
import json
import re
import shutil
import time
from pathlib import Path
import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from .archive import CountedStream, inventory, open_member
from .cloud import client, download_source, parse_uri, sha256, upload_verified
from .schema import SCHEMAS, GUIDE


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def check_space(folder, minimum_gib=25):
    if shutil.disk_usage(folder).free < minimum_gib*1024**3:
        raise RuntimeError(f'Less than {minimum_gib} GiB free; free space and resume the same command')


def stage_source(source, item, destination):
    schema = SCHEMAS[item['kind']]
    output_schema = schema.append(pa.field('cohort_year', pa.int32()))
    rows = 0
    with open_member(source, item['chain']) as raw:
        counted = CountedStream(raw)
        reader = csv.open_csv(counted,
            read_options=csv.ReadOptions(column_names=schema.names, block_size=16*1024**2, use_threads=False),
            parse_options=csv.ParseOptions(delimiter='|', quote_char=False, ignore_empty_lines=False),
            convert_options=csv.ConvertOptions(column_types=schema, null_values=[''], strings_can_be_null=True))
        with pq.ParquetWriter(destination, output_schema, compression='zstd') as writer:
            for batch in reader:
                check_space(destination.parent)
                table = pa.Table.from_batches([batch])
                if item['kind'] == 'performance':
                    dates = table['monthly_reporting_period']
                    valid = pc.match_substring_regex(dates, r'^[12][0-9]{3}(0[1-9]|1[0-2])$')
                    if dates.null_count or not pc.all(valid).as_py():
                        raise ValueError(f'Invalid reporting month in {item["name"]}; no rows silently dropped')
                table = table.append_column('cohort_year', pa.repeat(pa.scalar(item['cohort_year'], pa.int32()), len(table)))
                writer.write_table(table, row_group_size=128000)
                rows += len(table)
        if counted.byte_count != item['bytes'] or counted.rows != rows:
            raise RuntimeError('Source byte/row count mismatch: ' + item['name'])
    return rows


def partition_source(staged, item, destination, memory='3GB', threads=2):
    destination.mkdir(parents=True, exist_ok=True)
    partition_fields = ['cohort_year'] if item['kind'] == 'origination' else ['year', 'month']
    with duckdb.connect(config={'memory_limit': memory, 'threads': threads,
                                'temp_directory': str(staged.parent/'spill'),
                                'max_temp_directory_size': '80GB', 'preserve_insertion_order': False}) as db:
        relation = db.read_parquet(str(staged))
        if item['kind'] == 'performance':
            relation = relation.project("*, CAST(substr(monthly_reporting_period,1,4) AS INTEGER) AS year, substr(monthly_reporting_period,5,2) AS month")
        # External sorting bounds memory and avoids hundreds of simultaneously open writers.
        reader = relation.order(', '.join(partition_fields)).arrow(batch_size=128000)
        partition_schema = pa.schema([reader.schema.field(name) for name in partition_fields])
        ds.write_dataset(reader, str(destination), format='parquet',
            partitioning=ds.partitioning(partition_schema, flavor='hive'),
            basename_template=Path(item['name']).stem+'-{i}.parquet',
            file_options=ds.ParquetFileFormat().make_write_options(compression='zstd', compression_level=3),
            use_threads=False, preserve_order=True, max_open_files=8,
            max_rows_per_file=2000000, min_rows_per_group=64000, max_rows_per_group=128000,
            existing_data_behavior='error')
    rows = 0
    files = []
    for path in sorted(destination.rglob('*.parquet')):
        parquet = pq.ParquetFile(path)
        rows += parquet.metadata.num_rows
        # Every output row is decoded and its partition is independently checked.
        for batch in parquet.iter_batches(batch_size=128000):
            if item['kind'] == 'performance':
                match = re.search(r'year=(\d{4})/month=(\d{2})/', path.relative_to(destination).as_posix())
                if not match or not pc.all(pc.equal(batch.column('monthly_reporting_period'), match[1]+match[2])).as_py():
                    raise RuntimeError('Output month mismatch: '+str(path))
        files.append({'path': path.relative_to(destination).as_posix(), 'bytes': path.stat().st_size,
                      'rows': parquet.metadata.num_rows, 'sha256': sha256(path)})
    return rows, files


def run(source, output, work, cohorts=None, profile=None, memory='3GB', threads=2):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work/'conversion.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remote = output.startswith('s3://')
    s3 = client(profile) if remote or source.startswith('s3://') else None
    local_source = download_source(s3, source, work/'input') if source.startswith('s3://') else Path(source).resolve()
    source_hash = sha256(local_source)
    items = inventory(local_source)
    selected = [item for item in items if cohorts is None or item['cohort_year'] in cohorts]
    if not selected or (cohorts is not None and set(cohorts)-{i['cohort_year'] for i in selected}):
        raise ValueError('Requested cohorts are not present in the archive')
    identity = {'source_sha256': source_hash, 'output': output, 'cohorts': sorted(cohorts) if cohorts else None,
                'schema_version': 'release47-reporting-month-v1'}
    state_path = work/'state.json'
    if remote:
        bucket, prefix = parse_uri(output)
        if not prefix:
            raise ValueError('Use a dedicated S3 output prefix, not the bucket root')
    else:
        root = Path(output).resolve()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['identity'] != identity:
            raise ValueError('Run parameters or source changed; use a new work directory and output prefix')
    else:
        if remote:
            if s3.list_objects_v2(Bucket=bucket, Prefix=prefix+'/', MaxKeys=1).get('KeyCount', 0):
                raise ValueError('S3 output prefix is not empty; use a new prefix')
        elif root.exists() and any(root.iterdir()):
            raise ValueError('Output directory is not empty')
        state = {'identity': identity, 'completed': {}}
        save_json(state_path, state)
    for item in selected:
        if item['name'] in state['completed']:
            # Resume only when every committed object is still present and unchanged.
            for part in state['completed'][item['name']]['files']:
                if remote:
                    head = s3.head_object(Bucket=bucket, Key=prefix+'/'+part['path'])
                    if head['ContentLength'] != part['bytes'] or head.get('Metadata', {}).get('sha256') != part['sha256']:
                        raise RuntimeError('Committed S3 output changed: '+part['path'])
                elif sha256(root/part['path']) != part['sha256']:
                    raise RuntimeError('Committed local output changed: '+part['path'])
            print('Already verified:', item['name'], flush=True)
            continue
        check_space(work)
        stage = work/'staging'/Path(item['name']).stem
        if stage.exists():
            shutil.rmtree(stage)  # Our uncommitted scratch directory, never source data.
        stage.mkdir(parents=True)
        start = time.monotonic()
        print('Converting:', item['name'], flush=True)
        staged = stage/'typed.parquet'
        input_rows = stage_source(local_source, item, staged)
        output_rows, files = partition_source(staged, item, stage/'partitioned', memory, threads)
        if input_rows != output_rows:
            raise RuntimeError('Output row count mismatch: '+item['name'])
        for part in files:
            path = stage/'partitioned'/part['path']
            relative = item['kind']+'/'+part['path']
            if remote:
                digest = upload_verified(s3, path, bucket, prefix+'/'+relative)
                if digest != part['sha256']:
                    raise RuntimeError('Local file changed during publication')
            else:
                target = root/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)
            part['path'] = relative
        state['completed'][item['name']] = {'rows': input_rows, 'files': files, 'cohort_year': item['cohort_year'],
                                             'kind': item['kind'], 'seconds': round(time.monotonic()-start, 2)}
        save_json(state_path, state)
        shutil.rmtree(stage)
        print('Verified and committed:', item['name'], input_rows, 'rows', flush=True)
    manifest = {'schema_reference': GUIDE, 'source_sha256': source_hash,
                'performance_partition': 'monthly_reporting_period year/month',
                'origination_partition': 'cohort_year',
                'complete_for_requested_cohorts': True, 'full_archive': len(selected) == len(items),
                'cohorts': sorted({item['cohort_year'] for item in selected}),
                'source_files': len(selected), 'rows': sum(x['rows'] for x in state['completed'].values()),
                'files': [part for value in state['completed'].values() for part in value['files']]}
    save_json(work/'manifest.json', manifest)
    schema = {kind: [{'name': f.name, 'type': str(f.type)} for f in value] for kind, value in SCHEMAS.items()}
    save_json(work/'schema.json', schema)
    save_json(work/'_SUCCESS', {'manifest_sha256': sha256(work/'manifest.json'), 'rows': manifest['rows']})
    for name in ['schema.json', 'manifest.json', '_SUCCESS']:
        if remote:
            upload_verified(s3, work/name, bucket, prefix+'/'+name)
        else:
            root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work/name, root/name)
    print('Dataset ready:', output, 'rows:', manifest['rows'], flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Local ZIP path or s3://bucket/key')
    parser.add_argument('--output', required=True, help='Empty local directory or dedicated s3://bucket/prefix')
    parser.add_argument('--work-dir', default='work/conversion')
    parser.add_argument('--cohorts', type=int, nargs='+', help='Optional source cohort years; omit for full archive')
    parser.add_argument('--profile')
    parser.add_argument('--memory', default='3GB')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    run(args.source, args.output, args.work_dir, args.cohorts, args.profile, args.memory, args.threads)

if __name__ == '__main__':
    main()
