import importlib.util
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def test_quality_counts(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('quality',Path(__file__).parents[1]/'cleaning/profile_data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    pq.write_table(pa.table({'loan_identifier':['a','a',None,'b'],
                            'monthly_reporting_period':['202001','202001',None,'202013']}),tmp_path/'test.parquet')
    report=module.profile(str(tmp_path/'*.parquet'))
    assert report['rows']==4
    assert report['duplicate_key_excess_rows']==1
    assert report['missing_identifiers']==1
    assert report['missing_reporting_dates']==1
    assert report['malformed_reporting_dates']==1
