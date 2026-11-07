"""Source-only versioned bundle for MVP development and optional private Kaggle Re-ID."""
import shutil
import zipfile
from pathlib import Path

from shadowtrack.probe import sha256
from shadowtrack.detection.common import atomic_json,canonical,object_hash,read_json


def main():
    root=Path(__file__).resolve().parents[1];output=root/'artifacts/mvp/portable';output.mkdir(parents=True,exist_ok=True)
    files=list((root/'src').rglob('*.py'))+list((root/'src/shadowtrack/mvp/vendor').glob('*.txt'))
    for folder,patterns in [('configs',('*.json','*.toml')),('schemas',('*.json',)),('licenses',('*.txt',)),('scripts',('*.py','*.ps1')),('tests',('*.py',))]:
        for pattern in patterns:files+=list((root/folder).glob(pattern))
    for folder in ('frontend/src','frontend/public','frontend/e2e'):
        files+=[p for p in (root/folder).rglob('*') if p.is_file()]
    files+=list((root/'frontend').glob('*.json'))+list((root/'frontend').glob('*.js'))+list((root/'frontend').glob('*.ts'))
    for name in ('frontend/Dockerfile','frontend/index.html','frontend/nginx.conf','backend/Dockerfile','docker-compose.yml','.dockerignore','.env.example','.gitignore',
                 'pyproject.toml','AGENTS.md','README.md','MVP_SETUP.md','MVP_REPORT.md','DEMO_SCRIPT.md','DATA_AUDIT.md','TIME_ALIGNMENT.md','DETECTION_REPORT.md','KAGGLE_DETECTION_V2.md','KAGGLE_SETUP.md'):
        if (root/name).exists():files.append(root/name)
    for name in ('kaggle_detection.ipynb','kaggle_detection_v2.ipynb','kaggle_reid.ipynb'):
        path=root/'notebooks'/name;nb=read_json(path)
        assert all(not c.get('outputs') and c.get('execution_count') is None and not c.get('attachments') for c in nb['cells'])
        files.append(path)
    sources={p.relative_to(root).as_posix():p for p in files if p.is_file() and not p.name.endswith('.tsbuildinfo')}
    hashes={name:sha256(path) for name,path in sorted(sources.items())};code_hash=object_hash(hashes)
    manifest={'schema_version':1,'artifact_kind':'shadowtrack_mvp_code','code_hash':code_hash,'files':hashes,
              'contains_source_recordings':False,'contains_office_frames':False,'contains_weights_or_embeddings':False,
              'data_requirements':'Existing source MKV, saved audit evidence and alignment arrays are separate private inputs; outputs remain in artifacts/mvp.'}
    stem='shadowtrack_mvp_code_'+code_hash[:16];archive=output/(stem+'.zip');folder=output/stem
    if archive.exists() or folder.exists():raise ValueError('Snapshot already exists; use its existing manifest')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(sources.items()):z.write(path,name)
        z.writestr('mvp_code_manifest.json',canonical(manifest))
    folder.mkdir()
    for name,path in sources.items():
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    atomic_json(folder/'mvp_code_manifest.json',manifest)
    result={'filename':archive.name,'directory':folder.name,'sha256':sha256(archive),'code_hash':code_hash,'files':len(sources),'source_only':True}
    atomic_json(output/'code_artifact.json',result)
    (output/(archive.name+'.sha256')).write_text(result['sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(result)


if __name__=='__main__':main()
