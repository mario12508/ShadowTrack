"""Versioned scene-balanced Re-ID and clothing cache over existing private crops.

Only five reference frames are decoded. YOLO, ByteTrack, source PTS and original
embeddings are untouched. Normal MTMC runs import no torch and read this cache.
"""
import time
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from ..detection.common import atomic_json,read_json,object_hash
from ..probe import sha256
from .appearance import clothing_histogram
from .db import connect,rows,meta,set_meta
from .reid import encode_crop_package


NEUTRAL_FLOOR_ROIS={'1':[.77,.50,.89,.68],'2':[.4,.75,.58,.90],
                   '3':[.32,.27,.39,.52],'4':[.30,.77,.50,.90],'5':[.49,.73,.62,.90]}


def scene_gain(image,roi):
    array=np.asarray(image.convert('RGB'));h,w=array.shape[:2];a,b,c,d=roi
    pixels=array[round(b*h):round(d*h),round(a*w):round(c*w)].reshape(-1,3).astype(float)
    pixels=pixels[(pixels.min(axis=1)>35)&(pixels.max(axis=1)<235)]
    if len(pixels)<100:raise ValueError('Neutral reference has insufficient unsaturated scene pixels')
    rgb=np.median(pixels,axis=0)
    return np.clip(rgb.mean()/np.maximum(rgb,1),.65,1.6),rgb


def balanced(image,gain):
    return Image.fromarray(np.clip(np.asarray(image.convert('RGB'),dtype=float)*gain,0,255).astype('uint8'))


def head_signature(image):
    """Small same-view gradient signature; never a stand-alone identity model."""
    array=np.asarray(image.convert('RGB'));h,w=array.shape[:2]
    patch=array[:max(4,min(h,round(w*.8))),round(w*.15):max(round(w*.85),round(w*.15)+1)]
    gray=cv2.resize(cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY),(32,32)).astype('float32')
    dx=cv2.Sobel(gray,cv2.CV_32F,1,0);dy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    magnitude,angle=cv2.cartToPolar(dx,dy);bins=(angle*4/np.pi).astype(int)%8
    result=[]
    for y in range(0,32,8):
        for x in range(0,32,8):
            hist=np.bincount(bins[y:y+8,x:x+8].ravel(),weights=magnitude[y:y+8,x:x+8].ravel(),minlength=8)
            result.extend(hist)
    v=np.sqrt(np.asarray(result,dtype='float32')/max(sum(result),1e-9))
    return v/max(float(np.linalg.norm(v)),1e-9)


def validate_cache(out,signature,model_hash):
    """Validate a complete immutable cache before replacing any DB features."""
    status=read_json(out/'manifest.json');cm=read_json(out/'crops/crop_manifest.json')
    em=read_json(out/'encoded/embedding_manifest.json')
    if status.get('signature')!=signature or cm.get('normalization_signature')!=signature:raise ValueError('Normalized cache signature mismatch')
    if status.get('model_hash')!=model_hash or em.get('weights_sha256')!=model_hash:raise ValueError('Normalized cache model mismatch')
    if not status.get('source_run_id') or status['source_run_id']!=cm.get('source_run_id'):raise ValueError('Normalized cache source mismatch')
    digest=sha256(out/'crops/crop_manifest.json')
    if status['crop_manifest_sha256']!=digest or em['crop_manifest_sha256']!=digest:raise ValueError('Normalized crop manifest checksum mismatch')
    digest=sha256(out/'encoded/embeddings.npz')
    if status['features_sha256']!=digest or em['features_sha256']!=digest:raise ValueError('Normalized cache checksum mismatch')
    with np.load(out/'encoded/embeddings.npz',allow_pickle=False) as arrays:
        names=arrays['track_ids'].tolist();features=arrays['features']
        expected=[e['track_id'] for e in cm['entries']]
        if len(names)!=len(set(names)) or len(names)!=len(expected) or set(names)!=set(expected) or len(expected)!=status['samples']:raise ValueError('Normalized cache sample coverage mismatch')
        if features.ndim!=2 or features.shape[0]!=len(names) or not np.isfinite(features).all():raise ValueError('Invalid normalized features')
        vectors=dict(zip(names,features.copy()))
    return status,cm,vectors


def prepare(settings):
    started=time.monotonic();source=settings.artifacts/'mtmc/appearance/crops'
    manifest=read_json(source/'crop_manifest.json');lock=read_json(settings.root/'configs/reid_model_lock.json')
    signature=object_hash({'input':sha256(source/'crop_manifest.json'),'rois':NEUTRAL_FLOOR_ROIS,'version':1,'model':lock['weights_sha256']})
    out=settings.artifacts/'mtmc/normalized_appearance'/signature[:16];cropdir=out/'crops';encoded=out/'encoded'
    marker=out/'manifest.json'
    # A completed, hash-verified encoding can be imported after an interrupted
    # final metadata/DB write without recomputing crops or model inference.
    if not marker.exists() and (encoded/'embedding_manifest.json').exists() and (cropdir/'crop_manifest.json').exists():
        cm=read_json(cropdir/'crop_manifest.json');em=read_json(encoded/'embedding_manifest.json')
        if cm.get('normalization_signature')==signature and em['crop_manifest_sha256']==sha256(cropdir/'crop_manifest.json') and em['weights_sha256']==lock['weights_sha256'] and sha256(encoded/'embeddings.npz')==em['features_sha256']:
            atomic_json(marker,{'schema_version':1,'signature':signature,'source_run_id':manifest['source_run_id'],
                'samples':len(cm['entries']),'source_reference_frames_decoded':5,'model_hash':lock['weights_sha256'],
                'features_sha256':em['features_sha256'],'crop_manifest_sha256':sha256(cropdir/'crop_manifest.json'),
                'module_sha256':sha256(Path(__file__)),'scene_references':cm['scene_references'],
                'manual_identity_labels':False,'wall_seconds':None,'encoder_wall_seconds':em['wall_seconds'],
                'resumed_completed_encoding':True,'device':'cpu'})
    if marker.exists():
        status=read_json(marker)
        if sha256(encoded/'embeddings.npz')!=status['features_sha256']:raise ValueError('Normalized cache checksum mismatch')
        print('[appearance-normalize] reuse complete cache',flush=True)
    else:
        cropdir.mkdir(exist_ok=True,parents=True)
        with connect(settings.database,readonly=True) as db:
            cameras={r['id']:r for r in rows(db,'SELECT * FROM camera')}
            videos={r['camera_id']:r for r in rows(db,'SELECT * FROM media_segment WHERE kind="source"')}
        gains={};references={}
        for cid,roi in NEUTRAL_FLOOR_ROIS.items():
            with av.open(str(settings.media/videos[cid]['path'])) as container:
                frame=next(container.decode(video=0));gain,rgb=scene_gain(frame.to_image(),roi)
            gains[cid]=gain
            references[cid]={'source_sha256':cameras[cid]['source_sha256'],'source_frame_index':0,
                'raw_media_pts':cameras[cid]['first_pts'],'normalized_roi':roi,'neutral_median_rgb':rgb.tolist(),'gain_rgb':gain.tolist(),
                'source':'Visible neutral floor patch in the original static scene; approximate illuminant, no person labels'}
        entries=[]
        for i,original in enumerate(manifest['entries']):
            path=source/original['path']
            if sha256(path)!=original['sha256']:raise ValueError('Original crop checksum mismatch')
            with Image.open(path) as image:corrected=balanced(image,gains[original['camera_id']])
            target=cropdir/original['path'];corrected.save(target,quality=95)
            entries.append({**original,'original_crop_sha256':original['sha256'],'sha256':sha256(target),
                'color':clothing_histogram(corrected),'head_signature':head_signature(corrected).tolist(),
                'normalization':{'version':signature,'gain_rgb':gains[original['camera_id']].tolist()}})
            if i%128==0:print(f'[appearance-normalize] balanced crops {i+1}/{len(manifest["entries"])} elapsed={time.monotonic()-started:.1f}s',flush=True)
        atomic_json(cropdir/'crop_manifest.json',{**manifest,'entries':entries,'scene_references':references,'normalization_signature':signature})
        encode_crop_package(cropdir,settings.root/'.tools/models'/lock['weights_filename'],settings.root/'configs/reid_model_lock.json',encoded,'cpu',batch_size=32)
        result=read_json(encoded/'embedding_manifest.json')
        status={'schema_version':1,'signature':signature,'source_run_id':manifest['source_run_id'],
            'samples':len(entries),'source_reference_frames_decoded':5,'model_hash':lock['weights_sha256'],
            'features_sha256':result['features_sha256'],'crop_manifest_sha256':sha256(cropdir/'crop_manifest.json'),
            'module_sha256':sha256(Path(__file__)),'scene_references':references,'manual_identity_labels':False,
            'wall_seconds':time.monotonic()-started,'device':'cpu'}
        atomic_json(marker,status)
    status,normalized,vectors=validate_cache(out,signature,lock['weights_sha256'])
    with connect(settings.database) as db:
        if meta(db,'source_run_id')!=status['source_run_id']:raise ValueError('Normalized feature source run mismatch')
        db.execute('DELETE FROM appearance_normalized')
        for entry in normalized['entries']:
            evidence={'signature':signature,'features_sha256':status['features_sha256'],
                'original_crop_sha256':entry['original_crop_sha256'],'gain_rgb':entry['normalization']['gain_rgb']}
            import json
            db.execute('INSERT INTO appearance_normalized VALUES(?,?,?,?,?,?)',(entry['source_track_id'],entry['source_frame_index'],
                vectors[entry['track_id']].astype('<f4').tobytes(),json.dumps(entry['color']),
                np.asarray(entry['head_signature'],dtype='<f4').tobytes(),json.dumps(evidence)))
        set_meta(db,'normalized_appearance',status)
    print('[appearance-normalize] complete '+str({k:status[k] for k in ('samples','wall_seconds','device')}),flush=True)
    return status
