import io
import zipfile
from decimal import Decimal
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from sfld.convert import run
from sfld.schema import ORIG, PERF
from sfld.access import load_month_sample, month_files

def archive(path, periods=('202001','202002','202001'), bad_width=False):
    orig=['']*len(ORIG)
    for name,value in {'classic_fico':'9999','first_payment_date':'201902','loan_identifier':'F19Q10000001','postal_code':'005','original_upb':'123000','original_interest_rate':'3.125'}.items():
        orig[ORIG.index(name)]=value
    perf=[]
    for i,period in enumerate(periods):
        row=['']*len(PERF)
        for name,value in {'loan_identifier':'F19Q10000001','monthly_reporting_period':period,'current_actual_upb':'12345.67','current_loan_delinquency_status':'00','current_interest_rate':'3.125','loan_age':str(i),'zero_balance_code':'01'}.items():
            row[PERF.index(name)]=value
        if bad_width:row.pop()
        perf.append('|'.join(row))
    quarter=io.BytesIO()
    with zipfile.ZipFile(quarter,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('orig_2019Q1.txt','|'.join(orig)+'\n')
        z.writestr('perf_2019Q1.txt','\n'.join(perf))
    annual=io.BytesIO()
    with zipfile.ZipFile(annual,'w') as z:z.writestr('historical_data_2019Q1.zip',quarter.getvalue())
    with zipfile.ZipFile(path,'w') as z:z.writestr('historical_data_2019.zip',annual.getvalue())

def test_reporting_months_types_and_resume(tmp_path):
    source=tmp_path/'source.zip';archive(source)
    output=tmp_path/'data';work=tmp_path/'work'
    manifest=run(str(source),str(output),work)
    assert manifest['rows']==4 and manifest['full_archive']
    table=pq.ParquetFile(next((output/'performance/year=2020/month=01').glob('*.parquet'))).read()
    assert table.num_rows==2
    assert table['cohort_year'].to_pylist()==[2019,2019]
    assert table['current_actual_upb'][0].as_py()==Decimal('12345.67')
    assert table['current_loan_delinquency_status'][0].as_py()=='00'
    assert table['zero_balance_code'][0].as_py()=='01'
    assert table['modification_flag'].null_count==2
    orig=pq.ParquetFile(next((output/'origination/cohort_year=2019').glob('*.parquet'))).read()
    assert orig['postal_code'][0].as_py()=='005'
    assert orig['classic_fico'][0].as_py()==9999
    before={p:p.stat().st_mtime_ns for p in output.rglob('*.parquet')}
    run(str(source),str(output),work)
    assert before=={p:p.stat().st_mtime_ns for p in output.rglob('*.parquet')}
    assert (output/'_SUCCESS').exists()

@pytest.mark.parametrize('period',['202013','202000','2020','', '20XX01'])
def test_invalid_month_never_publishes_success(tmp_path,period):
    source=tmp_path/'source.zip';archive(source,(period,))
    with pytest.raises(ValueError,match='Invalid reporting month'):
        run(str(source),str(tmp_path/'data'),tmp_path/'work')
    assert not (tmp_path/'data/_SUCCESS').exists()

def test_bad_column_count_fails(tmp_path):
    source=tmp_path/'source.zip';archive(source,bad_width=True)
    with pytest.raises(pa.ArrowInvalid):run(str(source),str(tmp_path/'data'),tmp_path/'work')
    assert not (tmp_path/'data/_SUCCESS').exists()

def test_nonempty_output_protected(tmp_path):
    source=tmp_path/'source.zip';archive(source)
    output=tmp_path/'data';output.mkdir();(output/'existing').write_text('keep')
    with pytest.raises(ValueError,match='not empty'):run(str(source),str(output),tmp_path/'work')
    assert (output/'existing').read_text()=='keep'

class FakeS3:
    def __init__(self,root):self.root=root
    def get_object(self,Bucket,Key):return {'Body':io.BytesIO((self.root/Key).read_bytes())}
    def download_file(self,Bucket,Key,Filename,Config):Path(Filename).write_bytes((self.root/Key).read_bytes())

def test_classmate_limit_and_integrity(tmp_path):
    source=tmp_path/'source.zip';archive(source)
    run(str(source),str(tmp_path/'dataset'),tmp_path/'work')
    s3=FakeS3(tmp_path)
    table=load_month_sample(2020,1,dataset='s3://test/dataset',s3=s3,limit=1)
    assert table.num_rows==1 and table['cohort_year'][0].as_py()==2019
    manifest=tmp_path/'dataset/manifest.json';manifest.write_bytes(manifest.read_bytes()+b' ')
    with pytest.raises(RuntimeError,match='manifest'):month_files(s3,'s3://test/dataset',2020,1)

def test_modified_source_cannot_resume(tmp_path):
    source=tmp_path/'source.zip';archive(source)
    run(str(source),str(tmp_path/'data'),tmp_path/'work')
    archive(source,('202003',))
    with pytest.raises(ValueError,match='changed'):run(str(source),str(tmp_path/'data'),tmp_path/'work')


def test_partial_selection_and_year_range():
    import io
    import json
    import hashlib
    from sfld.access import select_files
    files = [
        {'path': 'performance/year=2019/month=01/a.parquet'},
        {'path': 'performance/year=2020/month=01/b.parquet'},
        {'path': 'performance/year=2020/month=02/c.parquet'},
        {'path': 'origination/cohort_year=2018/d.parquet'},
    ]
    raw=json.dumps({'files':files,'full_archive':False}).encode()
    class S3:
        def get_object(self, Bucket, Key):
            assert Key.endswith(('_PARTIAL_SUCCESS','manifest.partial.json'))
            data=raw if Key.endswith('manifest.partial.json') else json.dumps({'manifest_sha256':hashlib.sha256(raw).hexdigest()}).encode()
            return {'Body':io.BytesIO(data)}
    s3=S3()
    assert len(select_files(2019,2020,allow_partial=True,s3=s3)) == 3
    assert len(select_files(2020,month=1,allow_partial=True,s3=s3)) == 1
    assert len(select_files(2018,kind='origination',allow_partial=True,s3=s3)) == 1
    import pytest
    with pytest.raises(ValueError):
        select_files(2018,kind='origination',month=1,allow_partial=True,s3=s3)
