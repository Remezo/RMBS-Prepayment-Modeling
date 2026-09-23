"""Read nested vendor archives without expanding the complete dataset."""
import contextlib
import io
import re
import zipfile

class CountedStream(io.RawIOBase):
    def __init__(self, source):
        self.source = source
        self.byte_count = self.newlines = 0
        self.last = b''
    def readable(self):
        return True
    def read(self, size=-1):
        data = self.source.read(size)
        self.byte_count += len(data)
        self.newlines += data.count(b'\n')
        if data:
            self.last = data[-1:]
        return data
    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)
    @property
    def rows(self):
        return self.newlines + int(bool(self.byte_count) and self.last != b'\n')

@contextlib.contextmanager
def open_member(source, chain):
    with contextlib.ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(source))
        for name in chain[:-1]:
            stream = stack.enter_context(archive.open(name))
            archive = stack.enter_context(zipfile.ZipFile(stream))
        yield stack.enter_context(archive.open(chain[-1]))

def inventory(source):
    result = []
    def walk(archive, chain):
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = chain + [info.filename]
            if info.filename.lower().endswith('.zip'):
                # Vendor outer layers are ZIP_STORED; seeking is cheap.
                if info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError('Nested ZIP must be stored without compression: ' + info.filename)
                with archive.open(info) as stream, zipfile.ZipFile(stream) as child:
                    walk(child, path)
            else:
                match = re.fullmatch(r'(orig|perf)_(\d{4})Q([1-4])\.txt', info.filename)
                if not match:
                    raise ValueError('Unsupported member: ' + info.filename)
                result.append(dict(chain=path, name=info.filename, bytes=info.file_size,
                                   crc=info.CRC, cohort_year=int(match[2]), quarter=int(match[3]),
                                   kind='origination' if match[1] == 'orig' else 'performance'))
    with zipfile.ZipFile(source) as archive:
        walk(archive, [])
    names = [item['name'] for item in result]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate source filenames')
    return sorted(result, key=lambda item: (-item['cohort_year'], item['quarter'], item['kind']))
