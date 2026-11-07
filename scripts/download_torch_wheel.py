"""Bounded, resumable download of the official PyPI CPU-compatible Windows wheel."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def main():
    info=json.load(urllib.request.urlopen('https://pypi.org/pypi/torch/2.6.0/json',timeout=30))
    wheel=next(f for f in info['urls'] if 'cp312-cp312-win_amd64' in f['filename'])
    folder=Path(__file__).resolve().parents[1]/'.tools/packages';folder.mkdir(parents=True,exist_ok=True)
    target=folder/wheel['filename'];expected=wheel['digests']['sha256']
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:print(target);return
    chunk=4*1024**2;parts=folder/'torch_parts';parts.mkdir(exist_ok=True);count=(wheel['size']+chunk-1)//chunk
    def fetch(i):
        start=i*chunk;end=min(wheel['size']-1,start+chunk-1);path=parts/f'{i:04d}.part'
        if path.exists() and path.stat().st_size==end-start+1:return path
        for attempt in range(3):
            try:
                req=urllib.request.Request(wheel['url'],headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=45) as response:
                    if response.status!=206 or not response.headers['Content-Range'].startswith(f'bytes {start}-{end}/'):raise ValueError('Unexpected range response')
                    data=response.read(end-start+2)
                if len(data)!=end-start+1:raise ValueError('Truncated wheel part')
                path.write_bytes(data);print(f'[PyTorch] verified range {i+1}/{count}',flush=True);return path
            except Exception:
                if attempt==2:raise
                time.sleep(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:result=list(pool.map(fetch,range(count)))
    pending=target.with_suffix('.pending');digest=hashlib.sha256()
    with pending.open('wb') as stream:
        for p in result:
            data=p.read_bytes();digest.update(data);stream.write(data)
    if digest.hexdigest()!=expected:raise ValueError('Official PyPI wheel SHA256 mismatch')
    pending.replace(target);print('PIN VERIFIED',expected,target,flush=True)


if __name__=='__main__':main()
