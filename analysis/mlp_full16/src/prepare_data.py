"""Safely unpack the supplied archive into the repository's data directory."""
import argparse
from pathlib import Path
import zipfile

def extract(archive,dest):
    dest=Path(dest).resolve();dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for i in z.infolist():
            if not (dest/i.filename).resolve().is_relative_to(dest):
                raise ValueError(f'Unsafe archive member: {i.filename}')
        z.extractall(dest)
    print(dest/'Student folder')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--archive',default='inputs/Student folder.zip');a.add_argument('--dest',default='data')
    args=a.parse_args();extract(args.archive,args.dest)
