"""Pinned pretrained OSNet. Explicit offline command; never invoked by playback."""
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image

from ..probe import sha256
from ..detection.common import atomic_bytes,atomic_json,canonical,object_hash,read_json
from .db import connect,meta,rows,set_meta


def select_crops(settings,samples=4):
    if not 3<=samples<=8:raise ValueError('Use 3–8 independent crops per track')
    with connect(settings.database,readonly=True) as db:
        tracks=rows(db,'SELECT * FROM local_track WHERE observations>=12 AND end_time-start_time>=1.0 ORDER BY camera_id,start_time')
        chosen=[]
        for t in tracks:
            candidates=rows(db,'SELECT * FROM observation WHERE track_id=? ORDER BY confidence DESC,frame_index',(t['id'],))
            scored=[]
            for c in candidates:
                x1,y1,x2,y2=json.loads(c['bbox']);height=y2-y1;width=x2-x1
                if height<40 or width<12:continue
                scored.append((c['confidence']*np.sqrt(width*height),c))
            selected=[]
            for score,c in sorted(scored,key=lambda p:(-p[0],p[1]['frame_index'])):
                if all(abs(c['media_time']-other['media_time'])>=.5 for other in selected):selected.append(c)
                if len(selected)>=samples:break
            if len(selected)>=3:chosen.extend(selected)
        return sorted(chosen,key=lambda c:(c['camera_id'],c['frame_index'],c['track_id']))


def make_crop_package(settings,samples=4):
    selected=select_crops(settings,samples);output=settings.artifacts/'reid/crops';output.mkdir(parents=True,exist_ok=True)
    with connect(settings.database,readonly=True) as db:
        cameras={r['id']:dict(r) for r in db.execute('SELECT * FROM camera')};source_run=meta(db,'source_run_id');source_manifest=meta(db,'source_manifest_sha256')
    entries=[];current=None;container=None
    try:
        for n,row in enumerate(selected):
            cid=row['camera_id'];camera=cameras[cid]
            target=output/f"{row['track_id']}_{row['frame_index']:08d}.jpg"
            if not target.exists():
                if cid!=current:
                    if container:container.close()
                    if sha256(settings.raw/camera['source_name'])!=camera['source_sha256']:raise ValueError('Wrong crop source video')
                    container=av.open(str(settings.raw/camera['source_name']));stream=container.streams.video[0];stream.codec_context.thread_count=2;current=cid
                container.seek(round(row['raw_pts']/float(stream.time_base)),stream=stream,backward=True)
                found=None
                for frame in container.decode(stream):
                    actual=float(frame.pts*frame.time_base)
                    if abs(actual-row['raw_pts'])<.000001:found=frame;break
                    if actual>row['raw_pts']+.001:break
                if found is None:raise ValueError('Exact crop PTS unavailable; do not substitute neighbouring frame')
                x1,y1,x2,y2=json.loads(row['bbox']);image=found.to_image().crop((round(x1),round(y1),round(x2),round(y2)))
                image.thumbnail((256,512));image.save(target,quality=92)
            entries.append({'track_id':row['track_id'],'camera_id':cid,'source_frame_index':row['frame_index'],'raw_media_pts':row['raw_pts'],
                'bbox_xyxy':json.loads(row['bbox']),'source_video_sha256':camera['source_sha256'],'path':target.name,'sha256':sha256(target),'confidence':row['confidence']})
            if n%80==0:print(f'[MVP ReID] crops {n+1}/{len(selected)}',flush=True)
    finally:
        if container:container.close()
    manifest={'schema_version':1,'task':'person_reidentification_crops','source_run_id':source_run,'source_manifest_sha256':source_manifest,
        'samples_per_track_requested':samples,'entries':entries,'manual_identity_labels':False,'private_only':True}
    atomic_json(output/'crop_manifest.json',manifest);return output


def encode_crop_package(crop_dir,weights,lock_path,output,device='cpu',batch_size=32):
    """Shared by local CPU and optional Kaggle notebook. No YOLO imports."""
    crop_dir,weights,output=Path(crop_dir),Path(weights),Path(output)
    lock=read_json(lock_path);manifest=read_json(crop_dir/'crop_manifest.json')
    if weights.stat().st_size!=lock['weights_bytes'] or sha256(weights)!=lock['weights_sha256']:raise ValueError('OSNet weights pin mismatch')
    if sha256(Path(__file__).parent/'vendor/osnet.py')!=lock['source_file_sha256']:raise ValueError('OSNet architecture source changed')
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('Requested CUDA is unavailable')
    torch.set_num_threads(2)
    from .vendor.osnet import osnet_x0_25
    started=time.monotonic();state=torch.load(weights,map_location='cpu',weights_only=True)
    state=state.get('state_dict',state);state={k.removeprefix('module.'):v for k,v in state.items()}
    model=osnet_x0_25(num_classes=1,pretrained=False)
    # Classifier belongs to training identities; inference uses the feature head.
    state={k:v for k,v in state.items() if not k.startswith('classifier.')}
    missing,unexpected=model.load_state_dict(state,strict=False)
    if unexpected or set(missing)!={'classifier.weight','classifier.bias'}:raise ValueError(f'Incomplete pretrained feature encoder: {missing}, {unexpected}')
    model.eval().to(device);by_track=defaultdict(list);evidence=defaultdict(list)
    mean=np.asarray(lock['rgb_mean'],np.float32).reshape(1,1,3);std=np.asarray(lock['rgb_std'],np.float32).reshape(1,1,3)
    entries=manifest['entries']
    for offset in range(0,len(entries),batch_size):
        batch=entries[offset:offset+batch_size];images=[]
        for e in batch:
            p=crop_dir/e['path']
            if not p.resolve().is_relative_to(crop_dir.resolve()) or sha256(p)!=e['sha256']:raise ValueError('Crop SHA256/path mismatch')
            with Image.open(p) as im:array=np.asarray(im.convert('RGB').resize((128,256),Image.Resampling.BILINEAR),dtype=np.float32)/255.
            images.append(np.ascontiguousarray(((array-mean)/std).transpose(2,0,1)))
        with torch.inference_mode():vectors=model(torch.from_numpy(np.stack(images)).to(device)).float().cpu().numpy()
        vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
        for e,v in zip(batch,vectors):by_track[e['track_id']].append(v);evidence[e['track_id']].append(e)
        print(f'[MVP ReID] genuine OSNet crops {min(offset+len(batch),len(entries))}/{len(entries)} device={device}',flush=True)
    keys=[];features=[];records=[]
    for key,vs in sorted(by_track.items()):
        vs=np.vstack(vs);feature=vs.mean(axis=0);norm=np.linalg.norm(feature)
        if norm<1e-8:continue
        feature/=norm;keys.append(key);features.append(feature)
        records.append({'track_id':key,'samples':len(vs),'sample_coherence':float(np.min(vs@feature)),'evidence':evidence[key]})
    output.mkdir(parents=True,exist_ok=True);pending=output/'embeddings.pending.npz'
    with pending.open('wb') as f:np.savez_compressed(f,features=np.asarray(features,dtype=np.float32).reshape(-1,512),track_ids=np.asarray(keys,dtype='U20'))
    os.replace(pending,output/'embeddings.npz')
    result={'schema_version':1,'task':'person_reidentification','source_run_id':manifest['source_run_id'],'source_manifest_sha256':manifest['source_manifest_sha256'],
        'crop_manifest_sha256':sha256(crop_dir/'crop_manifest.json'),'weights_sha256':lock['weights_sha256'],'source_commit':lock['source_commit'],
        'features_sha256':sha256(output/'embeddings.npz'),'dimensions':512,'records':records,'synthetic':False,'device':device,'torch_version':torch.__version__,'wall_seconds':time.monotonic()-started,'license':lock['license']}
    atomic_json(output/'embedding_manifest.json',result);return output


def extract_embeddings(settings,weights,device='cpu',samples=4):
    crops=make_crop_package(settings,samples)
    return encode_crop_package(crops,weights,settings.root/'configs/reid_model_lock.json',settings.artifacts/'reid/embeddings',device)


def import_embeddings(settings,source):
    source=Path(source);m=read_json(source/'embedding_manifest.json');lock=read_json(settings.root/'configs/reid_model_lock.json')
    if m.get('synthetic') is not False or m['weights_sha256']!=lock['weights_sha256'] or m['dimensions']!=512:raise ValueError('Untrusted/synthetic Re-ID model output')
    if sha256(source/'embeddings.npz')!=m['features_sha256']:raise ValueError('Embedding file checksum mismatch')
    arrays=np.load(source/'embeddings.npz',allow_pickle=False);features=arrays['features'];keys=arrays['track_ids'].tolist()
    if features.shape!=(len(keys),512) or len(set(keys))!=len(keys) or not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-4):raise ValueError('Invalid embedding dimensions/normalization/duplicates')
    if [r['track_id'] for r in m['records']]!=keys:raise ValueError('Embedding records are shuffled or incomplete')
    count=0
    with connect(settings.database) as db:
        if m['source_run_id']!=meta(db,'source_run_id') or m['source_manifest_sha256']!=meta(db,'source_manifest_sha256'):raise ValueError('Embedding source run mismatch')
        for r,v in zip(m['records'],features):
            if len({e['source_frame_index'] for e in r['evidence']})!=r['samples'] or r['samples']<3:raise ValueError('At least three distinct genuine observations required')
            for e in r['evidence']:
                observed=db.execute('SELECT o.*,c.source_sha256 FROM observation o JOIN camera c ON c.id=o.camera_id WHERE o.track_id=? AND o.frame_index=?',(r['track_id'],e['source_frame_index'])).fetchone()
                if not observed or observed['camera_id']!=e['camera_id'] or observed['raw_pts']!=e['raw_media_pts'] or observed['source_sha256']!=e['source_video_sha256'] or json.loads(observed['bbox'])!=e['bbox_xyxy']:raise ValueError('Embedding evidence differs from indexed source observation')
            if r['sample_coherence']<.60:continue
            db.execute('INSERT OR REPLACE INTO embedding VALUES(?,?,?,?,?,?)',(r['track_id'],np.asarray(v,dtype='<f4').tobytes(),512,r['samples'],m['weights_sha256'],json.dumps({'sample_coherence':r['sample_coherence'],'source_commit':m['source_commit'],'samples':r['evidence']})))
            count+=1
        set_meta(db,'embedding_status',{'status':'GENUINE_OSNET_IMPORTED','tracks':count,'model_hash':m['weights_sha256']})
        set_meta(db,'global_association_status','GLOBAL_ASSOCIATION_PENDING')
    return {'status':'GENUINE_OSNET_IMPORTED','tracks':count,'global_association':'RECOMPUTE_REQUIRED'}
