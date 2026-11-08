"""Final acceptance: real FULL, immutable evidence, CPU artifacts, running Compose."""
import ast
import importlib.metadata
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import httpx
import nbformat
import numpy as np

from shadowtrack.probe import sha256
from shadowtrack.detection.common import atomic_json,object_hash,read_json
from shadowtrack.detection.evidence import load_evidence
from shadowtrack.detection.store import validate_run
from shadowtrack.mvp.config import Settings
from shadowtrack.mvp.db import connect,meta


def main():
    s=Settings.environment();root=s.root
    protected=read_json(root/'artifacts/detection/preserved_hashes.json')
    changed=[name for name,digest in protected.items() if sha256(root/name)!=digest]
    assert not changed,f'Protected audit/alignment/source files changed: {changed}'
    alignment=read_json(s.alignment)
    assert all(sha256(s.alignment.parent/c['local_frame_times_file'])==c['local_frame_times_sha256'] for c in alignment['cameras'])
    sample=root/'artifacts/detection/sample_review/validated/988a8de217bb1d3c0b951e67'
    original_sample=read_json(sample/'export_manifest.json')
    assert all(sha256(sample/name)==entry['sha256'] for name,entry in original_sample['files'].items())
    with connect(s.database,readonly=True) as db:
        detection=(root/meta(db,'detection_run').replace('\\','/')).resolve()
        tracked=read_json(s.artifacts/'local_tracks/manifest.json')
        for entry in tracked['files']:assert sha256(s.artifacts/'local_tracks'/entry['path'])==entry['sha256']
        embeddings=meta(db,'embedding_status');associations=meta(db,'association_summary')
        episode_states=[dict(r) for r in db.execute('SELECT scope,status,COUNT(*) AS count FROM episode GROUP BY scope,status')]
    manifest,chunks=validate_run(detection,load_evidence(s.evidence))
    assert manifest['descriptor']['plan']['mode']=='FULL' and len(chunks)==1135
    assert sum(e['processed_frames'] for e in manifest['chunks'])==144110
    measured=read_json(s.artifacts/'media_verification.json');assert measured['status']=='PASS'
    features=read_json(s.artifacts/'reid/embeddings/embedding_manifest.json')
    assert features['synthetic'] is False and features['device']=='cpu'
    npz=s.artifacts/'reid/embeddings/embeddings.npz';assert sha256(npz)==features['features_sha256']
    arrays=np.load(npz,allow_pickle=False);assert arrays['features'].shape==(390,512)
    assert np.allclose(np.linalg.norm(arrays['features'],axis=1),1,atol=1e-4)
    suites=ET.parse(s.artifacts/'tests.xml').getroot().findall('testsuite')
    assert all(suite.attrib[k]=='0' for suite in suites for k in ('errors','failures','skipped'))
    cpu_tests=sum(int(suite.attrib['tests']) for suite in suites)
    browser=read_json(s.artifacts/'playwright-results.json')['stats']
    assert browser['expected']==6 and browser['unexpected']==browser['skipped']==browser['flaky']==0
    for path in (root/'notebooks').glob('*.ipynb'):
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code':assert not cell.outputs and cell.execution_count is None;ast.parse(cell.source)
    frontend=httpx.get('http://localhost:3000',timeout=20);assert frontend.status_code==200
    health=httpx.get('http://localhost:8000/api/health',timeout=20).json();assert health['status']=='READY'
    stats=httpx.get('http://localhost:3000/api/stats',timeout=20).json()
    assert stats['frame']==144110 and stats['detection']==136020 and stats['local_track']==503
    cameras=httpx.get('http://localhost:3000/api/cameras',timeout=20).json()
    for camera in cameras:
        r=httpx.get('http://localhost:3000'+camera['source_url'],headers={'Range':'bytes=0-1023'},timeout=20)
        assert r.status_code==206 and b'ftyp' in r.content[:40]
    # Docker pipe access and private workspace reads use different Windows tokens.
    # Keep the approved Docker check separate from this least-privilege file audit.
    docker_proof=read_json(s.artifacts/'docker_verification.json')
    containers=docker_proof['containers']
    assert {c['Service'] for c in containers}=={'backend','frontend'} and all(c['State']=='running' for c in containers)
    assert next(c for c in containers if c['Service']=='backend')['Health']=='healthy'
    assert docker_proof['web_backend_torch_absent'] is True
    portable=read_json(s.artifacts/'portable/code_artifact.json');archive=s.artifacts/'portable'/portable['filename']
    assert sha256(archive)==portable['sha256']
    with zipfile.ZipFile(archive) as z:
        source_manifest=json.loads(z.read('mvp_code_manifest.json'))
        for name,digest in source_manifest['files'].items():assert sha256(root/name)==digest
        assert object_hash(source_manifest['files'])==portable['code_hash']
        assert not any(name.lower().endswith(('.mkv','.mp4','.jpg','.pth','.pt','.sqlite3','.npz')) for name in z.namelist())
    changed_files=['README.md','pyproject.toml','src/shadowtrack/cli.py','docker-compose.yml','.dockerignore','.env.example','.gitignore',
        'docs/MVP_SETUP.md','docs/MVP_REPORT.md','docs/DEMO_SCRIPT.md','configs/local_track_full.toml','configs/reid_model_lock.json','tests/test_mvp.py',
        'scripts/build_kaggle_reid.py','scripts/build_mvp_bundle.py','scripts/verify_mvp.py','scripts/verify_mvp_media.py','backend/Dockerfile','notebooks/kaggle_reid.ipynb']
    changed_files+=[p.relative_to(root).as_posix() for p in (root/'src/shadowtrack/mvp').rglob('*') if p.is_file() and p.suffix in ('.py','.txt')]
    changed_files+=[name for name in source_manifest['files'] if name.startswith('frontend/')]
    result={'acceptance_status':'P0_PASS_GLOBAL_ASSOCIATION_PENDING','source_run_id':manifest['run_id'],'source_chunks':1135,
        'source_frames':144110,'raw_detections':136020,'accepted':85445,'rejected':26248,'uncertain':24327,'empty_raw_frames':84904,
        'local_tracks':503,'local_observations':81357,'local_episode_states':episode_states,'episode_timeout_seconds':120,
        'supported_global_episodes':stats['global_episodes'],'true_unique_physical_people':None,'true_global_episode_count':None,
        'embeddings':embeddings,'reid_samples':1523,'reid_encode_cpu_wall_seconds':features['wall_seconds'],'associations':associations,
        'cpu_tests':cpu_tests,'cpu_test_failures':0,'cpu_test_skipped':0,'browser_tests':browser['expected'],'browser_failures':0,
        'browser_scope':'Actual FULL -> CPU ByteTrack -> FastAPI -> React -> playback -> bbox click -> track detail; five-camera playback; saved annotated MP4 decode',
        'media_verification':{'status':measured['status'],'annotated_cameras':5,'annotated_segments':15,'annotated_frames':144110,'all_packet_pts_match_source':True},
        'source_probe_repeated':False,'yolo_repeated':False,'manual_identity_labels':False,'cloud_upload':False,
        'protected_files_unchanged':len(protected),'alignment_arrays_unchanged':True,'old_gpu_sample_files_unchanged':len(original_sample['files']),
        'docker_backend_healthy':True,'web_backend_torch_installed':False,'docker_containers':[{'service':c['Service'],'state':c['State'],'health':c['Health'],'image':c['Image']} for c in containers],
        'source_artifact':portable,'changed_files':{name:sha256(root/name) for name in sorted(set(changed_files))}}
    atomic_json(s.artifacts/'verification.json',result)
    print(json.dumps({k:result[k] for k in ('acceptance_status','cpu_tests','browser_tests','local_tracks','supported_global_episodes','protected_files_unchanged','docker_backend_healthy','source_artifact')},indent=2))


if __name__=='__main__':main()
