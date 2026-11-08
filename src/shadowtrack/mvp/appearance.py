"""Temporal appearance cache over saved detections, using the existing OSNet encoder."""
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict

import av
import cv2
import numpy as np
from PIL import Image

from ..detection.common import atomic_json,object_hash,read_json
from ..probe import sha256
from .db import connect,meta,rows,set_meta
from .reid import encode_crop_package


def clothing_histogram(image):
    """Central torso/legs HSV histograms; auxiliary evidence, never identity alone."""
    image=np.asarray(image.convert('RGB').resize((64,128)))
    parts=[]
    for top,bottom in ((20,76),(76,120)):
        hsv=cv2.cvtColor(image[top:bottom,14:50],cv2.COLOR_RGB2HSV)
        # Low saturation goes into separate achromatic bins, avoiding arbitrary
        # hue labels for white/grey/black clothing under different illuminants.
        hue=hsv[:,:,0]//15;sat=hsv[:,:,1];value=hsv[:,:,2]
        bins=np.where(sat<45,12+np.minimum(value//64,3),hue)
        hist=np.bincount(bins.ravel(),minlength=16).astype(float);hist/=max(hist.sum(),1);parts.extend(hist.tolist())
    return parts


def sample_quality(observation,width,height,other_boxes=()):
    box=np.asarray(json.loads(observation['bbox']) if isinstance(observation['bbox'],str) else observation['bbox'],float)
    x1,y1,x2,y2=box;area=max(1,(x2-x1)*(y2-y1));covered=0.
    for other in other_boxes:
        a,b,c,d=other;overlap=max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b));covered=max(covered,overlap/area)
    clipping=(x1<2)+(y1<2)+(x2>width-2)+(y2>height-2)
    visibility=max(.15,1-covered*.8)*(0.8 if clipping else 1)
    resolution=min(1,(y2-y1)/180)
    return float(max(.03,observation['confidence']*visibility*resolution)),{'overlap_fraction':covered,'clipped_edges':int(clipping),'height_pixels':float(y2-y1)}


def prepare_temporal_features(settings,interval=4.):
    """One explicit CPU preparation; every later association run reads the cache."""
    out=settings.artifacts/'mtmc/appearance';cropdir=out/'crops';cropdir.mkdir(parents=True,exist_ok=True)
    original=read_json(settings.artifacts/'reid/crops/crop_manifest.json');known={(e['track_id'],e['source_frame_index']):e for e in original['entries']}
    selected=[]
    with connect(settings.database) as db:
        cameras={r['id']:dict(r) for r in db.execute('SELECT * FROM camera')}
        sources={r['camera_id']:dict(r) for r in db.execute('SELECT * FROM media_segment WHERE kind="source"')}
        for t in rows(db,'SELECT * FROM local_track ORDER BY camera_id,start_time'):
            obs=rows(db,'SELECT * FROM observation WHERE track_id=? ORDER BY frame_index',(t['id'],))
            # Quality candidate per temporal bin, plus first/end context. Unlike
            # the old top-4 confidence rule this covers the whole track duration.
            buckets=defaultdict(list)
            for o in obs:buckets[int((o['media_time']-t['start_time'])/interval)].append(o)
            chosen={obs[0]['frame_index']:obs[0],obs[-1]['frame_index']:obs[-1]}
            camera=cameras[t['camera_id']]
            for bucket in buckets.values():
                best=max(bucket,key=lambda o:sample_quality(o,camera['width'],camera['height'])[0]);chosen[best['frame_index']]=best
            for key,e in known.items():
                if key[0]==t['id']:
                    o=next((o for o in obs if o['frame_index']==key[1]),None)
                    if o:chosen[o['frame_index']]=o
            for o in chosen.values():
                other=[json.loads(r[0]) for r in db.execute('SELECT bbox FROM observation WHERE camera_id=? AND frame_index=? AND track_id<>?',(o['camera_id'],o['frame_index'],o['track_id']))]
                quality,detail=sample_quality(o,camera['width'],camera['height'],other)
                selected.append({**o,'quality':quality,'quality_evidence':detail})
        run=meta(db,'source_run_id');source_manifest=meta(db,'source_manifest_sha256')
    entries=[];container=None;current=None;started=time.monotonic()
    try:
        for n,o in enumerate(sorted(selected,key=lambda o:(o['camera_id'],o['frame_index'],o['track_id']))):
            parent=o['track_id'];cid=o['camera_id'];filename=f"{parent}_{o['frame_index']:08d}.jpg";target=cropdir/filename
            if not target.exists():
                old=known.get((parent,o['frame_index']))
                if old:
                    source=settings.artifacts/'reid/crops'/old['path']
                    if sha256(source)!=old['sha256']:raise ValueError('Old genuine crop changed')
                    shutil.copyfile(source,target)
                else:
                    if current!=cid:
                        if container:container.close()
                        media=sources[cid];source=settings.media/media['path']
                        if sha256(source)!=media['sha256']:raise ValueError('Browser source changed')
                        container=av.open(str(source));stream=container.streams.video[0];stream.codec_context.thread_count=2;current=cid
                    container.seek(round(o['media_time']/float(stream.time_base)),stream=stream,backward=True)
                    found=None
                    for frame in container.decode(stream):
                        pts=float(frame.pts*frame.time_base)
                        if abs(pts-o['media_time'])<1e-5:found=frame;break
                        if pts>o['media_time']+.001:break
                    if found is None:raise ValueError('Temporal crop does not match audited source frame PTS')
                    image=found.to_image();scale=image.width/cameras[cid]['width'];bbox=[round(v*scale) for v in json.loads(o['bbox'])]
                    crop=image.crop(tuple(bbox));crop.thumbnail((256,512));crop.save(target,quality=92)
            # A unique crop key makes the shared existing encoder return one
            # vector per sample. Its preprocessing/model/weights are unchanged.
            key=hashlib.sha256(f'{parent}/{o["frame_index"]}'.encode()).hexdigest()[:20]
            with Image.open(target) as im:color=clothing_histogram(im)
            entries.append({'track_id':key,'source_track_id':parent,'camera_id':cid,'source_frame_index':o['frame_index'],'raw_media_pts':o['raw_pts'],
                'bbox_xyxy':json.loads(o['bbox']),'confidence':o['confidence'],'quality':o['quality'],'quality_evidence':o['quality_evidence'],'color':color,
                'source_video_sha256':cameras[cid]['source_sha256'],'path':filename,'sha256':sha256(target)})
            if n%100==0:print(f'[appearance] temporal crops {n+1}/{len(selected)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    finally:
        if container:container.close()
    manifest={'schema_version':1,'task':'temporal_reid_samples','source_run_id':run,'source_manifest_sha256':source_manifest,'entries':entries,'interval_seconds':interval,'manual_identity_labels':False}
    atomic_json(cropdir/'crop_manifest.json',manifest)
    lock=read_json(settings.root/'configs/reid_model_lock.json');encoded=out/'encoded'
    previous=read_json(encoded/'embedding_manifest.json') if (encoded/'embedding_manifest.json').exists() else None
    if not previous or previous['crop_manifest_sha256']!=sha256(cropdir/'crop_manifest.json'):
        encode_crop_package(cropdir,settings.root/'.tools/models'/lock['weights_filename'],settings.root/'configs/reid_model_lock.json',encoded,'cpu')
    em=read_json(encoded/'embedding_manifest.json')
    if em['synthetic'] is not False or em['weights_sha256']!=lock['weights_sha256'] or sha256(encoded/'embeddings.npz')!=em['features_sha256']:raise ValueError('Temporal encoder provenance mismatch')
    arrays=np.load(encoded/'embeddings.npz',allow_pickle=False);vectors={k:v for k,v in zip(arrays['track_ids'].tolist(),arrays['features'])}
    with connect(settings.database) as db:
        db.execute('DELETE FROM appearance_sample')
        for e in entries:
            db.execute('INSERT INTO appearance_sample VALUES(?,?,?,?,?,?,?)',(e['source_track_id'],e['source_frame_index'],e['raw_media_pts'],vectors[e['track_id']].astype('<f4').tobytes(),e['quality'],json.dumps(e['color']),json.dumps(e)))
        status={'samples':len(entries),'tracks':len({e['source_track_id'] for e in entries}),'model_hash':lock['weights_sha256'],'features_sha256':em['features_sha256'],'source_run_id':run,'wall_seconds':time.monotonic()-started,'manual_identity_labels':False}
        set_meta(db,'temporal_appearance',status)
    atomic_json(out/'manifest.json',status);print('[appearance] '+json.dumps(status),flush=True);return status
