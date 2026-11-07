"""Build a searchable application index from validated FULL and CPU track records."""
import gzip
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from ..probe import sha256
from ..sync.api import TimeAlignment
from ..detection.common import atomic_json,read_json,object_hash
from ..detection.evidence import load_evidence
from ..detection.store import validate_run
from .config import Settings
from .db import connect,set_meta
from .episodes import lifecycle


def track_key(source_id):return hashlib.sha256(source_id.encode()).hexdigest()[:20]


def prepare_database(settings: Settings,detection_run: Path,track_dir: Path):
    evidence=load_evidence(settings.evidence)
    manifest,chunks=validate_run(detection_run,evidence,collect_records=False)
    tracks=read_json(track_dir/'manifest.json')
    if tracks['source_run_id']!=manifest['run_id']:raise ValueError('Track/detection source run mismatch')
    if sha256(detection_run/'run_manifest.json')!=tracks['source_manifest_sha256']:raise ValueError('Detection manifest changed')
    for entry in tracks['files']:
        if sha256(track_dir/entry['path'])!=entry['sha256']:raise ValueError('Track hash mismatch')
    alignment=TimeAlignment.load(settings.alignment)
    settings.artifacts.mkdir(parents=True,exist_ok=True)
    temporary=settings.database.with_name('shadowtrack.build.sqlite3')
    if temporary.exists():
        with connect(temporary,readonly=True) as stale:
            from .db import meta
            if meta(stale,'source_run_id') not in (None,manifest['run_id']) or meta(stale,'prepared_at') is not None:
                raise ValueError('Build file belongs to another completed run')
        temporary.unlink()
    started=time.perf_counter()
    with connect(temporary) as db:
        set_meta(db,'source_run_id',manifest['run_id']);set_meta(db,'source_manifest_sha256',sha256(detection_run/'run_manifest.json'))
        set_meta(db,'detection_run',os.path.relpath(detection_run,settings.root));set_meta(db,'track_dir',os.path.relpath(track_dir,settings.root))
        set_meta(db,'episode_timeout_seconds',settings.episode_timeout);set_meta(db,'global_association_status','GLOBAL_ASSOCIATION_PENDING')
        set_meta(db,'inference','PRECOMPUTED_YOLO_DETECTION_V2');set_meta(db,'tracker',tracks['tracker_config'])
        set_meta(db,'detector',read_json(detection_run/'benchmark.json'))
        for cid,camera in evidence['cameras'].items():
            print(f'[MVP] index camera={cid} start',flush=True)
            pts0=camera['frames'][0][1];pts1=camera['frames'][-1][1]
            a=next(c for c in alignment.document['cameras'] if c['camera_id']==cid)
            db.execute('INSERT INTO camera VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(cid,camera['source_name'],camera['source_sha256'],camera['width'],camera['height'],camera['nominal_fps'],camera['frame_count'],pts0,pts1,pts1-pts0,a['synchronization_status'],json.dumps({'supported_regions':a['supported_regions'],'unsupported_regions':a['unsupported_regions']})))
            entry=next(c for c in tracks['files'] if c['camera_id']==cid)
            summaries={};frame_rows=[];observation_rows=[]
            with gzip.open(track_dir/entry['path'],'rt',encoding='utf-8') as stream:
                for n,line in enumerate(stream):
                    row=json.loads(line);i=row['source_frame_index'];raw=row['raw_media_pts'];sid=row['timestamp_segment_id']
                    if [i,raw,sid]!=camera['frames'][i]:raise ValueError('Track frame identity differs from audit')
                    global_time,uncertainty,status=alignment.map_time(cid,raw)
                    frame_rows.append((cid,i,raw,raw-pts0,sid,global_time,json.dumps(uncertainty),row['input_empty']))
                    for t in row['local_tracks']:
                        key=track_key(t['local_track_id'])
                        if key not in summaries:summaries[key]={'source_id':t['local_track_id'],'segment':sid,'start_frame':i,'start':raw-pts0,'count':0,'confidence':0.}
                        s=summaries[key];s.update(end_frame=i,end=raw-pts0);s['count']+=1;s['confidence']+=t['confidence']
                        observation_rows.append((key,cid,i,raw-pts0,raw,json.dumps(t['bbox_xyxy']),t['confidence'],t['detection_index']))
                    if len(frame_rows)>=1024:
                        db.executemany('INSERT INTO frame VALUES(?,?,?,?,?,?,?,?)',frame_rows);frame_rows=[]
                        db.executemany('INSERT INTO observation VALUES(?,?,?,?,?,?,?,?)',observation_rows);observation_rows=[]
            if n+1!=camera['frame_count']:raise ValueError('FULL track rows incomplete')
            db.executemany('INSERT INTO frame VALUES(?,?,?,?,?,?,?,?)',frame_rows)
            db.executemany('INSERT INTO observation VALUES(?,?,?,?,?,?,?,?)',observation_rows)
            for number,(key,s) in enumerate(sorted(summaries.items(),key=lambda p:p[1]['start']),1):
                display=f'C{cid}-T{number:03d}'
                db.execute('INSERT INTO local_track VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(key,s['source_id'],cid,s['segment'],display,s['start_frame'],s['end_frame'],s['start'],s['end'],s['count'],s['confidence']/s['count'],None))
                segment=next(sg for sg in camera['segments'] if sg['id']==s['segment'])
                endpoint=segment['raw_end']-pts0
                state=lifecycle(s['end'],endpoint,settings.episode_timeout)
                episode='local-'+key
                db.execute('INSERT INTO episode VALUES(?,?,?,?,?,?,?,?,?,?)',(episode,'LOCAL',0,state.status,s['start'],s['end'],state.terminated_at,state.reason,1.,json.dumps([cid])))
                db.execute('INSERT INTO episode_member VALUES(?,?)',(episode,key))
                for kind,when in [('track_started',s['start']),('last_observed',s['end'])]:
                    db.execute('INSERT INTO event(camera_id,track_id,kind,media_time,detail) VALUES(?,?,?,?,?)',(cid,key,kind,when,json.dumps({'display_id':display,'scope':'LOCAL'})))
                if state.terminated_at is not None:
                    db.execute('INSERT INTO event(camera_id,track_id,kind,media_time,detail) VALUES(?,?,?,?,?)',(cid,key,'episode_terminated',state.terminated_at,json.dumps({'reason':state.reason,'timeout_seconds':settings.episode_timeout,'scope':'LOCAL'})))
            detections=[]
            for h,chunk,_ in chunks:
                if chunk['camera_id']!=cid or h!=tracks['selected_config_hash']:continue
                from ..detection.store import chunk_name,validate_chunk
                from ..detection.plan import model_for_config
                _,records=validate_chunk(detection_run/chunk_name(h,chunk),camera,model_for_config(manifest['descriptor'],h)['weights_sha256'],h,chunk)
                for r in records:
                    for j,d in enumerate(r['detections']):detections.append((cid,r['source_frame_index'],j,json.dumps(d['bbox_xyxy']),d['confidence'],d['scene_filter']['decision'],json.dumps(d['scene_filter']['reasons'])))
                if len(detections)>2000:db.executemany('INSERT INTO detection VALUES(?,?,?,?,?,?,?)',detections);detections=[]
            db.executemany('INSERT INTO detection VALUES(?,?,?,?,?,?,?)',detections)
            db.commit()
            print(f'[MVP] index camera={cid} frames={n+1} local_tracks={len(summaries)}',flush=True)
        set_meta(db,'prepared_at',time.time());set_meta(db,'prepare_wall_seconds',time.perf_counter()-started)
        set_meta(db,'schema_version',1)
        if settings.database.exists():
            with connect(settings.database,readonly=True) as previous:
                media=[tuple(r) for r in previous.execute('SELECT * FROM media_segment')]
            db.executemany('INSERT OR REPLACE INTO media_segment VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',media)
        db.commit()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    os.replace(temporary,settings.database)
    result={'status':'COMPLETE','source_run_id':manifest['run_id'],'frames':sum(c['frame_count'] for c in evidence['cameras'].values()),'global_association':'GLOBAL_ASSOCIATION_PENDING','episode_timeout_seconds':settings.episode_timeout}
    atomic_json(settings.artifacts/'prepare_manifest.json',result)
    return result
