"""Verify the new portable snapshot without changing audit, sync or GPU SAMPLE."""

import ast
import hashlib
import importlib.metadata
import json
import platform
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import nbformat

from shadowtrack.probe import sha256
from shadowtrack.detection.common import atomic_json,code_fingerprint,read_json
from shadowtrack.detection.evidence import load_evidence
from shadowtrack.detection.plan import detection_plan,evidence_identity
from shadowtrack.detection.portable import verify_code_tree


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'artifacts/detection/v2'
    old=root/'artifacts/detection/portable'
    protected=read_json(root/'artifacts/detection/preserved_hashes.json')
    changed=[name for name,digest in protected.items() if sha256(root/name)!=digest]
    assert not changed,f'Protected files changed: {changed}'
    alignment=read_json(root/'artifacts/sync/time_alignment.json')
    for camera in alignment['cameras']:
        assert sha256(root/'artifacts/sync'/camera['local_frame_times_file'])==camera['local_frame_times_sha256']
    previous=read_json(old/'code_artifact.json')
    assert sha256(old/previous['filename'])==previous['sha256']
    with zipfile.ZipFile(old/previous['filename']) as archive:
        old_manifest=json.loads(archive.read('code_manifest.json'))
        frozen={name:d for name,d in old_manifest['files'].items() if name.startswith('input_evidence/')}
        for name,digest in frozen.items():
            assert sha256(root/'artifacts/detection'/name)==digest
    sample=root/'artifacts/detection/sample_review/validated/988a8de217bb1d3c0b951e67'
    sample_manifest=read_json(sample/'export_manifest.json')
    for name,entry in sample_manifest['files'].items():
        assert sha256(sample/name)==entry['sha256'],f'GPU SAMPLE changed: {name}'

    evidence=load_evidence(root/'artifacts/detection/input_evidence')
    settings=read_json(root/'configs/detection_v2.json')
    plans={}
    for mode in ('BENCHMARK','DEMO','FULL'):
        p=detection_plan(evidence,mode,settings['raw_starts'],settings['benchmark_seconds'] if mode=='BENCHMARK' else settings['demo_seconds'],settings['chunk_frames'])
        plans[mode]={'total_frames_per_configuration':sum(p['selected_frame_counts'].values()),
                     'frames_per_camera':p['selected_frame_counts'],'chunks_per_configuration':len(p['chunks']),
                     'windows':p['windows'],'decode_frames_one_pass':sum(c['frame_count'] for c in evidence['cameras'].values())}
    assert plans['FULL']['frames_per_camera']=={cid:c['frame_count'] for cid,c in evidence['cameras'].items()}
    atomic_json(output/'plan_summary.json',plans)

    notebook=nbformat.read(root/'notebooks/kaggle_detection_v2.ipynb',as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert not cell.outputs and cell.execution_count is None
            ast.parse(cell.source)
    lock=read_json(root/'configs/detection_v2_lock.json')
    for model in lock['models'].values():
        weights=root/'.tools/models'/model['weights_filename']
        assert weights.stat().st_size==model['weights_bytes'] and sha256(weights)==model['weights_sha256']
    suites=ET.parse(output/'tests.xml').getroot().findall('testsuite')
    assert suites and all(s.attrib[k]=='0' for s in suites for k in ('errors','failures','skipped'))
    total=sum(int(s.attrib['tests']) for s in suites)

    portable=output/'portable'
    artifact=read_json(portable/'code_artifact.json')
    code_zip=portable/artifact['filename']
    assert sha256(code_zip)==artifact['sha256'] and artifact['code_hash']==code_fingerprint(root)
    assert verify_code_tree(portable/artifact['directory'])==artifact['code_hash']
    with zipfile.ZipFile(code_zip) as archive:
        manifest=json.loads(archive.read('code_manifest.json'))
        assert not any(name.lower().endswith(('.mkv','.jpg','.png','.gif','.mp4','.pt')) for name in archive.namelist())
        for name,digest in manifest['files'].items():
            source=root/'artifacts/detection'/name if name.startswith('input_evidence/') else root/name
            assert sha256(source)==digest,f'Snapshot stale: {name}'
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    (portable/(artifact['filename']+'.sha256')).write_text(artifact['sha256']+'  '+artifact['filename']+'\n',encoding='ascii')
    sources=['README.md','KAGGLE_SETUP.md','KAGGLE_DETECTION_V2.md','DETECTION_REPORT.md','.gitignore','pyproject.toml',
             'src/shadowtrack/cli.py','notebooks/kaggle_detection_v2.ipynb','configs/detection_v2.json',
             'configs/detection_v2_lock.json','configs/scene_filter_v1.json','configs/local_track_v2.toml',
             'schemas/detection_frame_v2.schema.json','scripts/build_kaggle_detection_v2.py',
             'scripts/verify_detection_v2.py','tests/test_detection_v2.py']
    sources+=[p.relative_to(root).as_posix() for p in (root/'src/shadowtrack/detection').glob('*.py')
              if old_manifest['files'].get(p.relative_to(root).as_posix())!=sha256(p)]
    result={'acceptance_status':'LOCAL_VALIDATED_GPU_PENDING','tests':total,'failures':0,'errors':0,'skipped':0,
            'original_audit_sync_tests':45,'legacy_detection_tests':57,'new_detection_v2_tests':total-102,
            'preserved_files_unchanged':len(protected),'alignment_arrays_unchanged':True,
            'source_evidence_files_unchanged':len(frozen),'old_gpu_sample_files_unchanged':len(sample_manifest['files']),
            'old_code_artifact_unchanged':True,'source_mkv_unchanged':True,
            'evidence_identity':evidence_identity(evidence),'source_evidence_producer':evidence['manifest']['producer_code_hash'],
            'plans':plans,'portable_artifact':artifact,'weights_pin_verified':True,
            'integration_scope':'Five real synthetic MKV -> synthetic predictor -> schema v2 export -> import -> real CPU ByteTrack -> real local-ID MP4; no YOLO re-inference',
            'actual_new_office_yolo_run':False,'new_gpu_benchmark_run':False,'actual_new_office_mp4':False,
            'quality_vs_previous_or_competitors':'UNMEASURED','full_office_run':False,'cloud_upload':False,
            'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ('av','supervision','numpy','opencv-python','pytest')},
            'changed_files':{name:sha256(root/name) for name in sorted(set(sources))
                             if old_manifest['files'].get(name)!=sha256(root/name)}}
    atomic_json(output/'verification.json',result)
    print(json.dumps({k:result[k] for k in ('acceptance_status','tests','preserved_files_unchanged','source_evidence_files_unchanged','old_gpu_sample_files_unchanged','portable_artifact')},indent=2))


if __name__=='__main__':
    main()
