#!/usr/bin/env python3
"""List a selection first; add --download to retrieve verified files."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfld.access import select_files, download_selection
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--year', type=int, required=True)
p.add_argument('--end-year', type=int)
p.add_argument('--month', type=int)
p.add_argument('--kind', choices=['performance','origination'], default='performance')
p.add_argument('--profile')
p.add_argument('--allow-partial', action='store_true')
p.add_argument('--download', action='store_true')
p.add_argument('--destination', default='data')
p.add_argument('--max-download-gb', type=float, default=2)
a=p.parse_args()
files=select_files(a.year,a.end_year,month=a.month,kind=a.kind,profile=a.profile,allow_partial=a.allow_partial)
print(f"Selection: {len(files):,} files, {sum(f['bytes'] for f in files)/1e9:.3f} GB")
if a.download:
    paths=download_selection(files,a.destination,profile=a.profile,max_download_gb=a.max_download_gb)
    print(f'Downloaded and verified {len(paths)} files in {a.destination}')
else:
    print('Preview only. Add --download to retrieve these files.')
