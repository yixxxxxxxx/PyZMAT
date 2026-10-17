#!/usr/bin/env python3
"""Extract only raw Gaussian COM/FCHK/log evidence from hess_test.tar."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    manifest=[]
    with tarfile.open(args.archive,'r|*') as archive:
        for member in archive:
            parts=PurePosixPath(member.name).parts
            if len(parts)!=5 or parts[0]!='hess_debug' or parts[1] not in {'roy','vusdix'} or parts[2] not in {'new_lam_00001','new_lam_00050'} or parts[3] not in {'cart-2-z','z'}:
                continue
            if not member.isfile() or Path(parts[4]).suffix not in {'.com','.fchk','.log'}:
                continue
            path=Path(parts[1]+'_'+parts[2])/parts[3]/parts[4]
            target=args.out/path
            if target.exists():raise ValueError('Duplicate archive member: '+member.name)
            data=archive.extractfile(member).read()
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            manifest.append(dict(source=member.name,path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    if len(manifest)!=18:raise ValueError('Expected 18 raw reference members')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
