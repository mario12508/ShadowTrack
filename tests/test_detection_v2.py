import copy
from fractions import Fraction
import gzip
import json
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
import pytest

from shadowtrack.probe import sha256,probe_video
from shadowtrack.detection.common import DetectionError,atomic_bytes,atomic_json,canonical,object_hash,read_json
from shadowtrack.detection.decoder import pyav_frames
from shadowtrack.detection.engine import run_detection
from shadowtrack.detection.evidence import load_evidence
from shadowtrack.detection.plan import detection_plan,evidence_identity
from shadowtrack.detection.schema import detection_record,validate_frame
from shadowtrack.detection.scene_filter import SceneFilter,translate_crop
from shadowtrack.detection.store import validate_run
from shadowtrack.detection.transport import export_run,import_run,restore_checkpoint
from shadowtrack.detection.tracking import local_track,replay
from shadowtrack.detection.video import render_local_tracks

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT/"configs/detection_v2_lock.json"
TRACKER = {"track_activation_threshold":.25,"lost_track_buffer":30,"minimum_matching_threshold":.8,"minimum_consecutive_frames":1}
RUNTIME = {"selected_device":"cpu","torch_cuda_is_available":False,"packages":{},"gpu":None,"nvidia_smi":{"status":"UNAVAILABLE"}}


class Frame:
    width,height = 160,120
    def __init__(self,i):self.i=i
    def to_ndarray(self,format):return np.full((self.height,self.width,3),self.i,dtype=np.uint8)


class BatchPredictor:
    calls = 0
    batch_sizes = []
    def predict_batch(self,images):
        BatchPredictor.calls += 1
        BatchPredictor.batch_sizes.append(len(images))
        rows=[]
        for rgb in images:
            h,w=rgb.shape[:2]
            rows.append([] if int(rgb[0,0,0])%7==0 else [{"bbox_xyxy":[w*.2,h*.4,w*.45,h*.9],"confidence":.9,"class_id":0}])
        return rows,{"inference_seconds":.001*len(images),"effective_half":False}
    def close(self):pass


def evidence_files(directory,videos,camera_frames):
    manifest={"schema_version":1,"package_kind":"shadowtrack_source_evidence","producer_code_hash":"0"*64,
              "audit_stats_sha256":"1"*64,"time_alignment_sha256":"2"*64,"cameras":{}}
    for cid,fps,frames in camera_frames:
        segments=[]
        for sid in dict.fromkeys(r[2] for r in frames):
            ids=[r[0] for r in frames if r[2]==sid]
            segments.append({"id":sid,"frame_start":ids[0],"frame_end":ids[-1],"raw_start":frames[ids[0]][1],"raw_end":frames[ids[-1]][1]})
        filename=f'camera_{cid}_frames.json'
        atomic_bytes(directory/filename,canonical(frames))
        manifest["cameras"][cid]={"camera_id":cid,"source_name":f"camera_{cid}.mkv","source_sha256":sha256(videos/f"camera_{cid}.mkv"),
            "frame_count":len(frames),"nominal_fps":fps,"width":160,"height":120,"frames_file":filename,
            "frames_sha256":sha256(directory/filename),"frames_content_sha256":sha256(directory/filename),
            "audited_frames_sha256":object_hash(frames),"segments":segments}
    atomic_json(directory/"source_manifest.json",manifest)
    return load_evidence(directory)


@pytest.fixture
def inputs(tmp_path):
    videos=tmp_path/"videos";videos.mkdir()
    camera_frames=[]
    for cid,fps in (("1",25),("4",20)):
        (videos/f'camera_{cid}.mkv').write_bytes(b'synthetic v2 source '+cid.encode())
        frames=[[i,round(5+i/fps+(10 if i>=fps else 0),3),f'{cid}:s{int(i>=fps)}'] for i in range(fps*2)]
        camera_frames.append((cid,fps,frames))
    evidence=evidence_files(tmp_path/"evidence",videos,camera_frames)
    scene=tmp_path/"scene.json"
    atomic_json(scene,{"schema_version":1,"version":"fixture_v1","cameras":{},"duplicate_iou":.65})
    calls={}
    def decoder(path):
        cid=path.stem.split('_')[-1];calls[cid]=calls.get(cid,0)+1
        for i,pts,sid in evidence['cameras'][cid]['frames']:
            yield round(pts*1000),Fraction(1,1000),Frame(i)
    return {"tmp":tmp_path,"videos":videos,"evidence":evidence,"scene":scene,"decoder":decoder,"decode_calls":calls,
            "settings":{"benchmark_seconds":.4,"demo_seconds":.6,"chunk_frames":3,"raw_starts":{"1":5.,"4":5.}}}


def run(inputs,name="run",**kwargs):
    return run_detection(inputs["evidence"]["directory"],inputs["videos"],Path("unused"),LOCK,inputs["scene"],inputs["tmp"]/name,ROOT,
        settings=inputs["settings"],batch_size=4,predictor_factory=lambda cfg:BatchPredictor(),decoder=inputs["decoder"],runtime_override=RUNTIME,progress=False,**kwargs)


def test_benchmark_single_decode_two_models_all_empty_frames_preserved(inputs):
    manifest=run(inputs)
    assert inputs['decode_calls']=={'1':1,'4':1}
    assert manifest['schema_version']==2 and len(manifest['descriptor']['configurations'])==2
    _,chunks=validate_run(inputs['tmp']/'run',inputs['evidence'],collect_records=True)
    rows=[r for h,c,records in chunks for r in records]
    assert any(r['frame_status']=='empty' for r in rows)
    assert all(r['task']=='detect' and 'raw_media_pts' in r for r in rows)
    assert all('mask_rle' not in d for r in rows for d in r['detections'])
    from jsonschema import Draft202012Validator
    validator=Draft202012Validator(read_json(ROOT/'schemas/detection_frame_v2.schema.json'))
    validator.validate(rows[0])
    validator.validate(next(r for r in rows if r['detections']))
    assert read_json(inputs['tmp']/'run/benchmark.json')['status']=='SYNTHETIC_TEST_ONLY'
    assert read_json(inputs['tmp']/'run/benchmark.json')['decode_frames_all_sessions']==90


def test_evidence_producer_does_not_change_v2_source_identity(inputs):
    run(inputs)
    path=inputs['evidence']['directory']/'source_manifest.json'
    manifest=read_json(path);manifest['producer_code_hash']='9'*64;atomic_json(path,manifest)
    updated=load_evidence(path.parent)
    assert evidence_identity(updated)==evidence_identity(inputs['evidence'])
    validate_run(inputs['tmp']/'run',updated)


def test_v2_checkpoint_resume_and_corrupt_upload(inputs):
    with pytest.raises(InterruptedError):run(inputs,stop_after_chunks=1)
    archive=export_run(inputs['tmp']/'run',inputs['tmp']/'partial.zip',progress=False)
    restored=restore_checkpoint(archive,inputs['tmp']/'resumed',progress=False)
    with pytest.raises(DetectionError,match='Incomplete'):
        import_run(archive,inputs['evidence']['directory'],inputs['tmp']/'badimport',LOCK,allow_synthetic=True,raw_dir=inputs['videos'],progress=False)
    run(inputs,name='resumed')
    validate_run(restored,inputs['evidence'])
    first=read_json(restored/'run_manifest.json')['chunks'][0]
    (restored/first['path']/'frames.jsonl.gz').write_bytes(b'broken')
    with pytest.raises(DetectionError,match='SHA256'):validate_run(restored,inputs['evidence'])


def test_v2_rejects_wrong_scene_hash_and_shuffled_frame(inputs):
    scene=SceneFilter(read_json(inputs['scene']));cam=inputs['evidence']['cameras']['1']
    record=detection_record(cam,1,scene.apply(cam,[{'bbox_xyxy':[20.,45.,50.,100.],'confidence':.8,'class_id':0}]),'a'*64,'b'*64,scene.hash)
    validate_frame(record,cam,'a'*64,'b'*64)
    bad=copy.deepcopy(record);bad['detections'][0]['scene_filter']['config_hash']='c'*64
    with pytest.raises(DetectionError,match='scene filter'):validate_frame(bad,cam,'a'*64,'b'*64)
    bad=copy.deepcopy(record);bad['source_frame_index']=2
    with pytest.raises(DetectionError,match='PTS'):validate_frame(bad,cam,'a'*64,'b'*64)


def test_scene_filters_lamp_retains_people_stationary_and_occluded():
    scene=SceneFilter(read_json(ROOT/'configs/scene_filter_v1.json'))
    config=scene.config['cameras']['2']
    info={'camera_id':'2','source_sha256':config['source_sha256'],'width':config['resolution'][0],'height':config['resolution'][1]}
    raw=[{'bbox_xyxy':[1357.95,636.85,1459.56,836.07],'confidence':.286,'class_id':0},
         {'bbox_xyxy':[2373.,578.,2535.,932.],'confidence':.825,'class_id':0},
         {'bbox_xyxy':[1497.,190.,1653.,510.],'confidence':.83,'class_id':0}]
    before=copy.deepcopy(raw);out=scene.apply(info,raw)
    assert raw==before and len(out)==3
    assert out[0]['scene_filter']['decision']=='rejected'
    assert all(r['scene_filter']['decision']=='accepted' for r in out[1:])
    assert scene.apply(info,raw)==out  # No movement condition or identity annotation.


def test_crop_translation_and_duplicate_raw_candidates_preserved():
    crop=translate_crop([{'bbox_xyxy':[10.,20.,30.,80.],'confidence':.8,'class_id':0}],[100,50,200,150],300,200)
    assert crop[0]['bbox_xyxy']==[110.,70.,130.,130.]
    scene=SceneFilter({'schema_version':1,'version':'test','cameras':{}})
    out=scene.apply({'camera_id':'5','source_sha256':'x','width':300,'height':200},crop+copy.deepcopy(crop))
    assert len(out)==2 and out[1]['scene_filter']['decision']=='rejected'
    with pytest.raises(DetectionError,match='outside'):
        translate_crop([{'bbox_xyxy':[0.,0.,400.,50.]}],[100,50,200,150],300,200)


def test_full_plan_every_source_frame_gaps_and_different_fps(inputs):
    plan=detection_plan(inputs['evidence'],'FULL',{},chunk_frames=8)
    assert plan['selected_frame_counts']=={'1':50,'4':40}
    for cid,camera in inputs['evidence']['cameras'].items():
        ids=[i for c in plan['chunks'] if c['camera_id']==cid for i in range(c['start'],c['stop'])]
        assert ids==list(range(camera['frame_count']))
        assert all(len({camera['frames'][i][2] for i in range(c['start'],c['stop'])})==1 for c in plan['chunks'] if c['camera_id']==cid)
    with pytest.raises(DetectionError,match='segment'):
        detection_plan(inputs['evidence'],'DEMO',{'1':5.5,'4':5.5},seconds=11.)


def test_full_requires_demo_and_saves_all_frames_after_demo(inputs):
    with pytest.raises(DetectionError,match='DEMO'):
        run(inputs,mode='FULL',selected_configuration='nano_640')
    run(inputs,name='demo',mode='DEMO',selected_configuration='nano_640')
    assert read_json(inputs['tmp']/'demo/demo_videos.json')['status']=='COMPLETE'
    full=run(inputs,name='full',mode='FULL',selected_configuration='nano_640',demo_source=inputs['tmp']/'demo')
    assert sum(c['processed_frames'] for c in full['chunks'])==90
    _,chunks=validate_run(inputs['tmp']/'full',inputs['evidence'],collect_records=True)
    assert all([r['source_frame_index'] for h,c,rs in chunks if c['camera_id']==cid for r in rs]==list(range(cam['frame_count'])) for cid,cam in inputs['evidence']['cameras'].items())


def test_derived_mp4_failure_does_not_lose_machine_records(inputs,monkeypatch):
    def fail(*a,**k):raise RuntimeError('synthetic encoder failure')
    monkeypatch.setattr('shadowtrack.detection.engine.AnnotatedVideo',fail)
    result=run(inputs,mode='DEMO',selected_configuration='nano_640')
    assert result['status']=='COMPLETE'
    assert read_json(inputs['tmp']/'run/demo_videos.json')['status']=='FAILED'
    validate_run(inputs['tmp']/'run',inputs['evidence'])


@pytest.mark.parametrize('damage',['missing_mp4','missing_sidecar','corrupt_sidecar','missing_camera','bad_frame_count','empty_video_list'])
def test_video_resume_reuses_records_without_inference(inputs,damage):
    run(inputs,mode='DEMO',selected_configuration='nano_640')
    saved=BatchPredictor.calls
    manifest_path=inputs['tmp']/'run/demo_videos.json'
    videos=read_json(manifest_path)
    first=videos['videos'][0]
    if damage=='missing_mp4':
        (inputs['tmp']/'run/demo'/first['path']).unlink()
    elif damage=='missing_sidecar':
        (inputs['tmp']/'run/demo'/first['sidecar']).unlink()
    elif damage=='corrupt_sidecar':
        (inputs['tmp']/'run/demo'/first['sidecar']).write_bytes(b'corrupt upload')
    else:
        if damage=='missing_camera':videos['videos'].pop()
        elif damage=='empty_video_list':videos['videos']=[]
        else:first['frames']+=1
        atomic_json(manifest_path,videos)
    run(inputs,mode='DEMO',selected_configuration='nano_640')
    assert BatchPredictor.calls==saved
    from shadowtrack.detection.engine import _demo_videos_valid
    result=read_json(inputs['tmp']/'run/run_manifest.json')
    assert _demo_videos_valid(inputs['tmp']/'run',inputs['evidence'],result['descriptor']['plan'])


def test_encoder_add_and_cleanup_failure_preserves_all_machine_records(inputs,monkeypatch):
    class BrokenEncoder:
        def __init__(self,*a,**k):pass
        def add(self,*a,**k):raise RuntimeError('synthetic add failure')
        def close(self,*a,**k):raise RuntimeError('synthetic cleanup failure')
    monkeypatch.setattr('shadowtrack.detection.engine.AnnotatedVideo',BrokenEncoder)
    result=run(inputs,mode='DEMO',selected_configuration='nano_640')
    assert result['status']=='COMPLETE'
    assert read_json(inputs['tmp']/'run/demo_videos.json')['status']=='FAILED'
    assert sum(c['processed_frames'] for c in result['chunks'])==sum(result['descriptor']['plan']['selected_frame_counts'].values())
    validate_run(inputs['tmp']/'run',inputs['evidence'])


def test_v2_tracker_original_detection_indices_and_no_masks(inputs):
    cam=inputs['evidence']['cameras']['1'];scene=SceneFilter(read_json(inputs['scene']))
    rows=[]
    for i in range(1,6):
        ds=scene.apply(cam,[{'bbox_xyxy':[20.,45.,50.,100.],'confidence':.9,'class_id':0}]*2)
        rows.append(detection_record(cam,i,ds,'a'*64,'b'*64,scene.hash))
    first=list(replay(rows,cam,TRACKER));second=list(replay(rows,cam,TRACKER))
    assert first==second
    assert all(t['mask_ref'] is None and t['detection_index']==0 for r in first for t in r['local_tracks'])
    assert all(r['rejected_detection_indices']==[1] for r in first)


@pytest.mark.integration
def test_kaggle_detection_export_import_bytetrack_annotated_video_five_cameras(tmp_path,monkeypatch):
    ffmpeg,ffprobe=ROOT/'.tools/ffmpeg.exe',ROOT/'.tools/ffprobe.exe'
    if not ffmpeg.exists():pytest.skip('FFmpeg unavailable')
    videos=tmp_path/'videos';videos.mkdir();camera_frames=[]
    for cid in map(str,range(1,6)):
        fps=20 if cid=='4' else 25
        video=videos/f'camera_{cid}.mkv'
        subprocess.run([str(ffmpeg),'-v','error','-f','lavfi','-i',f'nullsrc=size=160x120:rate={fps}:duration=0.8,geq=lum=N*5+16:cb=128:cr=128',
            '-vf','setpts=PTS+5/TB','-c:v','ffv1','-fps_mode','passthrough',str(video)],check=True,capture_output=True,timeout=30)
        probe_video(video,tmp_path/f'probe_{cid}',str(ffprobe),30)
        frames=read_json(tmp_path/f'probe_{cid}/frames.json')['frames']
        camera_frames.append((cid,fps,[[i,float(f['best_effort_timestamp_time']),f'{cid}:s0'] for i,f in enumerate(frames)]))
    evidence=evidence_files(tmp_path/'evidence',videos,camera_frames)
    scene=tmp_path/'scene.json';atomic_json(scene,{'schema_version':1,'version':'integration_v1','cameras':{}})
    settings={'benchmark_seconds':.4,'demo_seconds':.6,'chunk_frames':4,'raw_starts':{cid:5. for cid in evidence['cameras']}}
    result=run_detection(evidence['directory'],videos,Path('unused'),LOCK,scene,tmp_path/'kaggle',ROOT,
        mode='DEMO',selected_configuration='nano_640',settings=settings,batch_size=4,
        predictor_factory=lambda cfg:BatchPredictor(),runtime_override=RUNTIME,progress=False)
    assert result['status']=='COMPLETE'
    assert len(read_json(tmp_path/'kaggle/demo_videos.json')['videos'])==5
    archive=export_run(tmp_path/'kaggle',tmp_path/'shadowtrack_detection_demo.zip',progress=False)
    calls=BatchPredictor.calls
    def no_inference(*a,**k):raise AssertionError('Local processing must never infer YOLO')
    monkeypatch.setattr(BatchPredictor,'predict_batch',no_inference)
    imported=import_run(archive,evidence['directory'],tmp_path/'imported',allow_synthetic=True,raw_dir=videos,progress=False)
    config=tmp_path/'track.toml'
    paths={'detection_run':str(imported),'evidence_dir':str(evidence['directory']),'raw_dir':str(videos),'output_dir':str(tmp_path/'tracks')}
    text='[paths]\n'+'\n'.join(f'{k}={json.dumps(v.replace(chr(92),"/"))}' for k,v in paths.items())
    text+='\n[tracking]\nconfiguration_name="nano_640"\n[bytetrack]\n'+'\n'.join(f'{k}={v}' for k,v in TRACKER.items())
    config.write_text(text,encoding='utf-8')
    tracked=local_track(config,progress=False)
    rendered=render_local_tracks(tmp_path/'tracks',evidence,videos,tmp_path/'local_videos',progress=False)
    assert len(tracked['files'])==len(rendered['videos'])==5 and BatchPredictor.calls==calls
    assert all(v['local_id_count']>0 for v in rendered['videos'])
    empty_count=0
    for entry in tracked['files']:
        rows=[json.loads(l) for l in gzip.decompress((tmp_path/'tracks'/entry['path']).read_bytes()).splitlines()]
        empty_count+=sum(r['input_empty'] for r in rows)
        assert all(r['global_time_status']=='NOT_REQUESTED' for r in rows)
        assert all(t['mask_ref'] is None for r in rows for t in r['local_tracks'])
    assert empty_count>0
    for video in rendered['videos']:
        rows=[json.loads(l) for l in (tmp_path/'local_videos'/video['sidecar']).read_text(encoding='utf-8').splitlines()]
        with av.open(str(tmp_path/'local_videos'/video['path'])) as container:
            frames=list(container.decode(video=0))
        assert len(rows)==len(frames)==video['frames']
        assert any(r['local_track_ids'] for r in rows)
        assert all(abs(float(f.pts*f.time_base)-r['output_relative_seconds'])<.00001 for f,r in zip(frames,rows))
    assert 'torch' not in sys.modules


def test_v2_notebook_is_clean_and_uses_shared_detection_engine():
    import ast,nbformat
    nb=nbformat.read(ROOT/'notebooks/kaggle_detection_v2.ipynb',as_version=4)
    nbformat.validate(nb)
    code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    assert 'MODE = "BENCHMARK"' in code and 'EXECUTE = False' in code
    assert 'from shadowtrack.detection.engine import run_detection' in code
    assert 'from shadowtrack.detection.notebook_support import pinned_detection_weights' in code
    assert 'YOLO(' not in code and 'frame_index/FPS' not in code
    for cell in nb.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is None and not cell.outputs
            ast.parse(cell.source)


def test_full_export_records_not_video_and_wrong_model_hash_rejected(inputs):
    run(inputs,name='demo',mode='DEMO',selected_configuration='nano_640')
    run(inputs,name='full',mode='FULL',selected_configuration='nano_640',demo_source=inputs['tmp']/'demo')
    archive=export_run(inputs['tmp']/'full',inputs['tmp']/'full.zip',progress=False)
    imported=import_run(archive,inputs['evidence']['directory'],inputs['tmp']/'full_import',LOCK,
                        allow_synthetic=True,raw_dir=inputs['videos'],progress=False)
    manifest,chunks=validate_run(imported,inputs['evidence'],collect_records=True)
    assert sum(len(rs) for h,c,rs in chunks)==90
    assert any(r['frame_status']=='empty' for h,c,rs in chunks for r in rs)
    assert not list(imported.glob('**/*.mp4'))
    wrong=read_json(LOCK);wrong['models']['nano']['weights_sha256']='f'*64
    wrong_path=inputs['tmp']/'wrong_lock.json';atomic_json(wrong_path,wrong)
    with pytest.raises(DetectionError,match='lock differs'):
        import_run(archive,inputs['evidence']['directory'],inputs['tmp']/'wrong_import',wrong_path,
                    allow_synthetic=True,raw_dir=inputs['videos'],progress=False)


def test_v2_empty_frame_without_mask_is_valid_but_wrong_identity_is_not(inputs):
    cam=inputs['evidence']['cameras']['1'];scene=SceneFilter(read_json(inputs['scene']))
    row=detection_record(cam,0,[],'a'*64,'b'*64,scene.hash)
    validate_frame(row,cam,'a'*64,'b'*64)
    assert row['processed'] and row['frame_status']=='empty'
    row['raw_media_pts']=0.
    with pytest.raises(DetectionError,match='PTS'):validate_frame(row,cam,'a'*64,'b'*64)


def test_local_videos_split_segments_instead_of_concatenating_them(inputs,monkeypatch):
    scene=SceneFilter(read_json(inputs['scene']));cam=inputs['evidence']['cameras']['1']
    rows=[]
    for i in (23,24,25,26):
        d=scene.apply(cam,[{'bbox_xyxy':[20.,45.,50.,100.],'confidence':.9,'class_id':0}])
        rows.append(detection_record(cam,i,d,'a'*64,'b'*64,scene.hash))
    tracked=list(replay(rows,cam,TRACKER))
    left=tracked[0]['local_tracks'][0]['local_track_id']
    right=tracked[2]['local_tracks'][0]['local_track_id']
    assert left!=right and '/1:s0/' in left and '/1:s1/' in right
    path=inputs['tmp']/'tracks/camera_1_tracks.jsonl.gz'
    atomic_bytes(path,gzip.compress(b'\n'.join(canonical(r) for r in tracked)+b'\n',mtime=0))
    atomic_json(path.parent/'manifest.json',{'source_run_id':'synthetic','files':[{'camera_id':'1','path':path.name,'sha256':sha256(path),'processed_frames':4}]})
    def decode_fixture(path,camera,*args,**kwargs):
        for i,pts,sid in camera['frames']:yield i,Frame(i)
    monkeypatch.setattr('shadowtrack.detection.video.checked_frames',decode_fixture)
    videos=render_local_tracks(path.parent,inputs['evidence'],inputs['videos'],inputs['tmp']/'split_videos',progress=False)
    assert len(videos['videos'])==2 and all(v['frames']==2 for v in videos['videos'])
    ranges=[]
    for video in videos['videos']:
        records=[json.loads(l) for l in (inputs['tmp']/'split_videos'/video['sidecar']).read_text(encoding='utf-8').splitlines()]
        ranges.append([r['source_frame_index'] for r in records])
        assert records[0]['output_relative_seconds']==0.
    assert ranges==[[23,24],[25,26]]


def test_optional_960_batch_config_and_no_duplicate_model_pass(inputs):
    result=run(inputs,benchmark_configurations=['nano_960','small_960'])
    assert inputs['decode_calls']=={'1':1,'4':1}
    assert all(c['imgsz']==960 and c['batch_size']==4 for c in result['descriptor']['configurations'].values())


def test_no_measurements_means_no_gpu_benchmark_claim(inputs):
    from types import SimpleNamespace
    from shadowtrack.detection.engine import _benchmark
    cfg=read_json(LOCK)['configurations'][0]
    # Hypothetical environment metadata is not evidence of an executed prediction.
    store=SimpleNamespace(root=inputs['tmp'],manifest={
        'descriptor':{'backend_kind':'ultralytics','configurations':{object_hash(cfg):cfg}},
        'chunks':[],'runtimes':{'not_used':{'selected_device':'cuda:0'}},'sessions':[]})
    result=_benchmark(store)
    assert result['status']=='NOT_RUN' and result['gpu_benchmark_status']!='MEASURED'


def test_camera5_crop_batch_keeps_frame_identity_and_source_coordinates(inputs):
    camera_frames=[]
    for cid,camera in inputs['evidence']['cameras'].items():
        camera_frames.append((cid,camera['nominal_fps'],camera['frames']))
    frames=[[i,p,'5:s'+sid.split(':s')[-1]] for i,p,sid in camera_frames[0][2]]
    (inputs['videos']/'camera_5.mkv').write_bytes(b'synthetic crop source')
    camera_frames.append(('5',25,frames))
    inputs['evidence']=evidence_files(inputs['tmp']/'crop_evidence',inputs['videos'],camera_frames)
    def crop_decoder(path):
        cid=path.stem.split('_')[-1]
        inputs['decode_calls'][cid]=inputs['decode_calls'].get(cid,0)+1
        for i,pts,sid in inputs['evidence']['cameras'][cid]['frames']:
            yield round(pts*1000),Fraction(1,1000),Frame(i)
    inputs['decoder']=crop_decoder
    inputs['settings']['raw_starts']['5']=5.
    atomic_json(inputs['scene'],{'schema_version':1,'version':'crop_fixture','cameras':{'5':{'far_crop':[20,20,100,100]}}})
    result=run(inputs)
    assert inputs['decode_calls']['5']==1
    _,chunks=validate_run(inputs['tmp']/'run',inputs['evidence'],collect_records=True)
    rows=[r for h,c,rs in chunks if c['camera_id']=='5' for r in rs]
    assert any(r['frame_status']=='empty' for r in rows)
    crop_boxes=[(r,d) for r in rows for d in r['detections'] if d['inference_source']=='far_crop']
    assert crop_boxes
    for row,d in crop_boxes:
        assert d['source_frame_index']==row['source_frame_index'] and d['raw_media_pts']==row['raw_media_pts']
        assert d['crop_xyxy']==[20,20,100,100] and 20<=d['bbox_xyxy'][0]<d['bbox_xyxy'][2]<=100
