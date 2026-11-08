"""FastAPI serves indexed real observations; no inference runs on HTTP requests."""
import csv
import gzip
import hashlib
import io
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Literal

import av
from fastapi import FastAPI,HTTPException,Query,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,JSONResponse,Response,StreamingResponse
from pydantic import BaseModel,Field,model_validator

from ..detection.common import read_json,object_hash
from ..sync.api import TimeAlignment
from .config import Settings
from .db import connect,meta,rows,set_meta
from .episodes import lifecycle
from .render import color_for

log=logging.getLogger('shadowtrack.api')
CAMERA_NAMES={'1':'Игровая зона','2':'Вход и лестница','3':'Лаунж','4':'Коридор','5':'Дальний коридор'}


class CalibrationInput(BaseModel):
    camera_id: str=Field(pattern='^[1-5]$')
    segment_id: str=Field(min_length=3,max_length=40)
    raw_start: float=Field(ge=0,le=86400,allow_inf_nan=False)
    raw_end: float=Field(ge=0,le=86400,allow_inf_nan=False)
    offset_seconds: float=Field(ge=-86400,le=86400,allow_inf_nan=False)
    drift_ppm: float=Field(default=0,ge=-10000,le=10000,allow_inf_nan=False)
    uncertainty_seconds: float=Field(ge=.001,le=120,allow_inf_nan=False)
    checked: bool=False
    evidence: str=Field(default='',max_length=2000)

    @model_validator(mode='after')
    def valid_interval(self):
        if self.raw_end<=self.raw_start:raise ValueError('raw_end must exceed raw_start')
        if self.checked and len(self.evidence.strip())<8:raise ValueError('Checked calibration needs an explicit shared event/evidence description')
        return self


def create_app(settings=None):
    settings=settings or Settings.environment()
    from .snapshot import bootstrap_native_database
    bootstrap_native_database(settings)
    app=FastAPI(title='ShadowTrack',version='1.0.0',description='Precomputed Detection v2 → CPU ByteTrack → evidence-bound episodes. No online YOLO.')
    origins=os.environ.get('SHADOWTRACK_CORS','http://localhost:3000,http://127.0.0.1:3000').split(',')
    app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST','DELETE'],allow_headers=['*'])
    app.state.settings=settings;app.state.worker=threading.Lock()
    alignment=TimeAlignment.load(settings.alignment) if settings.alignment.is_file() else None

    def db():
        if not settings.database.exists():raise HTTPException(503,'Run shadowtrack mvp prepare before starting playback')
        return connect(settings.database,readonly=True)

    @app.middleware('http')
    async def timing(request:Request,call_next):
        started=time.monotonic()
        response=await call_next(request)
        log.info(json.dumps({'method':request.method,'path':request.url.path,'status':response.status_code,'wall_ms':round((time.monotonic()-started)*1000,2)}))
        return response

    @app.get('/api/health')
    def health():
        if not settings.database.exists():return JSONResponse({'status':'NOT_PREPARED','gpu_required':False},503)
        with db() as con:count=con.execute('SELECT COUNT(*) FROM camera').fetchone()[0]
        return JSONResponse({'status':'READY' if count==5 else 'PREPARING','cameras':count,'gpu_required':False},200 if count==5 else 503)

    @app.get('/api/cameras')
    def cameras():
        with db() as con:
            result=rows(con,'SELECT * FROM camera ORDER BY id')
            for c in result:
                c['name']=CAMERA_NAMES.get(c['id'],'Камера '+c['id']);c['sync']=json.loads(c.pop('sync_json'))
                c['media']=rows(con,'SELECT id,kind,start_time,end_time,frames,width,height,segment_id,sha256 FROM media_segment WHERE camera_id=? ORDER BY kind,start_time',(c['id'],))
                for item in c['media']:item['url']='/api/media/'+item['id']+'?v='+item['sha256'][:16]
                c['source_url']=next((v['url'] for v in c['media'] if v['kind']=='source'),None)
                c['poster_url']=f"/api/cameras/{c['id']}/poster"
                c['local_track_count']=con.execute('SELECT COUNT(*) FROM local_track WHERE camera_id=?',(c['id'],)).fetchone()[0]
                c['calibrations']=rows(con,'SELECT * FROM calibration WHERE camera_id=?',(c['id'],))
            return result

    @app.get('/api/stats')
    def stats():
        with db() as con:
            counts={name:con.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0] for name in ('camera','frame','detection','local_track','observation')}
            counts['global_episodes']=con.execute('SELECT COUNT(*) FROM mtmc_episode').fetchone()[0]
            counts['reviewed_episodes']=con.execute('SELECT COUNT(*) FROM mtmc_episode WHERE status="CONFIRMED"').fetchone()[0]
            counts['derived_tracklets']=con.execute('SELECT COUNT(*) FROM mtmc_tracklet').fetchone()[0]
            counts['provisional_episodes']=con.execute('SELECT COUNT(*) FROM mtmc_episode WHERE status="PROVISIONAL"').fetchone()[0]
            counts['local_episodes']=con.execute('SELECT COUNT(*) FROM episode WHERE scope="LOCAL"').fetchone()[0]
            from .accounting import accounting,counts as episode_counts
            counts['episode_counts']=episode_counts(accounting(con))
            return {**counts,'source_run_id':meta(con,'source_run_id'),'global_association_status':meta(con,'global_association_status','GLOBAL_ASSOCIATION_PENDING'),
                'episode_timeout_seconds':meta(con,'episode_timeout_seconds',120),'scene_counts':rows(con,'SELECT decision,COUNT(*) AS count FROM detection GROUP BY decision'),
                'empty_frames':con.execute('SELECT COUNT(*) FROM frame WHERE input_empty=1').fetchone()[0],
                'by_camera':rows(con,'SELECT camera_id,COUNT(*) AS tracks,SUM(observations) AS observations FROM local_track GROUP BY camera_id'),
                'episode_states':rows(con,'SELECT scope,status,COUNT(*) AS count FROM episode GROUP BY scope,status'),
                'detector':meta(con,'detector',{}),'tracker':meta(con,'tracker',{}),
                'embeddings':meta(con,'embedding_status',{'status':'PENDING','tracks':0}),'associations':meta(con,'association_summary',{}),'mtmc':meta(con,'mtmc_summary',{}),
                'identity_count':None,'identity_count_reason':'Local fragments and episode counts are not unique physical people',
                'sync_status':'PARTIAL','inference':'PRECOMPUTED','data_status':'FULL_VALIDATED'}

    @app.get('/api/timeline')
    def timeline():
        with db() as con:
            segments=rows(con,'SELECT camera_id,segment_id,MIN(media_time) AS start_time,MAX(media_time) AS end_time,MIN(raw_pts) AS raw_start,MAX(raw_pts) AS raw_end,COUNT(*) AS frames FROM frame GROUP BY camera_id,segment_id')
            activity=rows(con,'SELECT camera_id,CAST(media_time/30 AS INTEGER)*30 AS time,COUNT(*) AS observations,COUNT(DISTINCT track_id) AS local_tracks FROM observation GROUP BY camera_id,CAST(media_time/30 AS INTEGER) ORDER BY time')
            return {'time_basis':'relative_to_each_source_first_decoded_pts','physical_synchronization':'PARTIAL','segments':segments,'activity':activity,
                'duration':con.execute('SELECT MAX(duration) FROM camera').fetchone()[0],'reference_support':alignment.document['reference_support'] if alignment else []}

    @app.get('/api/overlays/{camera_id}')
    def overlays(camera_id:str,start:float=Query(ge=0,le=86400),end:float=Query(ge=0,le=86400)):
        if not math.isfinite(start+end) or not 0 < end-start <= 8:raise HTTPException(422,'Request a positive interval no longer than 8 seconds')
        with db() as con:
            frames=rows(con,'SELECT * FROM frame WHERE camera_id=? AND media_time>=? AND media_time<? ORDER BY frame_index',(camera_id,max(0,start-.12),end))
            if not frames:return {'camera_id':camera_id,'frames':[]}
            lo,hi=frames[0]['frame_index'],frames[-1]['frame_index']
            observed=rows(con,'SELECT o.*,t.display_id,t.global_episode_id,t.source_id FROM observation o JOIN local_track t ON t.id=o.track_id WHERE o.camera_id=? AND o.frame_index BETWEEN ? AND ? ORDER BY o.frame_index',(camera_id,lo,hi))
            detected=rows(con,'SELECT * FROM detection WHERE camera_id=? AND frame_index BETWEEN ? AND ? ORDER BY frame_index,detection_index',(camera_id,lo,hi))
            by_frame={f['frame_index']:{**f,'detections':[],'tracks':[]} for f in frames}
            for r in detected:
                r['bbox_xyxy']=json.loads(r.pop('bbox'));r['reasons']=json.loads(r['reasons']);by_frame[r['frame_index']]['detections'].append(r)
            for r in observed:
                r['bbox_xyxy']=json.loads(r.pop('bbox'));r['color']='#'+''.join(f'{c:02x}' for c in color_for(r['source_id']))
                from .mtmc import identity
                r['source_track_id']=r['track_id'];r.update(identity(con,r['track_id'],r['frame_index']));r['track_id']=r['derived_tracklet_id']
                by_frame[r['frame_index']]['tracks'].append(r)
            return {'camera_id':camera_id,'time_basis':'media_relative_raw_pts','frames':list(by_frame.values())}

    @app.get('/api/tracks')
    def tracks(camera_id:str|None=None,q:str=Query(default='',max_length=100),start:float|None=Query(default=None,ge=0,le=86400),end:float|None=Query(default=None,ge=0,le=86400),limit:int=Query(default=40,ge=1,le=100),offset:int=Query(default=0,ge=0)):
        where=['1=1'];args=[]
        for condition,value in [('t.camera_id=?',camera_id),('t.end_time>=?',start),('t.start_time<=?',end)]:
            if value is not None:where.append(condition);args.append(value)
        if q:where.append('(t.display_id LIKE ? OR t.id LIKE ? OR e.label LIKE ? OR g.episode_id LIKE ?)');args.extend(['%'+q+'%']*4)
        clause=' AND '.join(where)
        with db() as con:
            base=' FROM association_track t LEFT JOIN local_global g ON g.track_id=t.id LEFT JOIN mtmc_episode e ON e.id=g.episode_id WHERE '
            total=con.execute('SELECT COUNT(*)'+base+clause,args).fetchone()[0]
            found=rows(con,'SELECT t.*,e.lifecycle AS status,e.termination_reason,e.terminated_at'+base+clause+' ORDER BY t.observations DESC,t.id LIMIT ? OFFSET ?',(*args,limit,offset))
            from .mtmc import identity
            for r in found:
                r['thumbnail_url']=f"/api/tracks/{r['id']}/crop";r['scope']='LOCAL';r['duration']=r['end_time']-r['start_time'];r.update(identity(con,r['id']))
            return {'total':total,'items':found,'count_kind':'local_track_fragments_not_people'}

    @app.get('/api/tracks/{track_id}')
    def track_detail(track_id:str):
        with db() as con:
            from .tracklets import source_track
            item=source_track(con,track_id)
            if not item:raise HTTPException(404,'Unknown track')
            parent=item['parent_id'];bounds=(parent,item['start_frame'],item['end_frame'])
            stride=max(1,item['observations']//150)
            item['observations_sample']=rows(con,'SELECT frame_index,media_time,raw_pts,confidence,bbox FROM observation WHERE track_id=? AND frame_index BETWEEN ? AND ? AND frame_index%?=0 ORDER BY frame_index LIMIT 200',(*bounds,stride))
            item['episodes']=rows(con,'SELECT e.id,"GLOBAL" AS scope,e.lifecycle AS status,e.termination_reason,e.terminated_at FROM mtmc_episode e JOIN local_global g ON g.episode_id=e.id WHERE g.track_id=?',(track_id,))
            from .accounting import accounting,lifecycle_for
            ledger=accounting(con)
            for episode in item['episodes']:
                state=lifecycle_for(ledger,episode['id'])
                if state:episode.update(status=state['state'],termination_reason=state['reason'],terminated_at=state['terminated_at'],right_censored=state['right_censored'],lifecycle_clock='source_recordings')
            item['events']=rows(con,'SELECT * FROM event WHERE track_id=? AND media_time BETWEEN ? AND ? ORDER BY media_time',(parent,item['start_time'],item['end_time']))
            item['associations']=rows(con,'SELECT * FROM mtmc_candidate WHERE track_a=? OR track_b=? ORDER BY similarity DESC LIMIT 30',(track_id,track_id))
            from .mtmc import identity
            item.update(identity(con,track_id))
            item['global_members']=rows(con,'SELECT t.id,t.parent_id,t.display_id,t.camera_id,t.start_time,t.end_time FROM association_track t JOIN local_global g ON g.track_id=t.id WHERE g.episode_id=? ORDER BY start_time',(item['global_episode_id'],)) if item['global_episode_id'] else []
            for member in item['global_members']:
                estimate=con.execute('SELECT a.*,c.first_pts FROM association_track t JOIN alignment_estimate a ON a.segment_id=t.segment_id JOIN camera c ON c.id=t.camera_id WHERE t.id=?',(member['id'],)).fetchone()
                member['estimated_start']=member['start_time']+estimate['first_pts']+estimate['offset']-json.loads(estimate['evidence'])['reference_first_pts'] if estimate else None
            item['thumbnail_url']=f'/api/tracks/{track_id}/crop';item['association_run_status']=meta(con,'global_association_status','GLOBAL_ASSOCIATION_PENDING')
            item['duration']=item['end_time']-item['start_time'];return item

    @app.get('/api/events')
    def events(camera_id:str|None=None,limit:int=Query(default=40,ge=1,le=200)):
        with db() as con:return rows(con,'SELECT e.*,t.display_id FROM event e LEFT JOIN local_track t ON t.id=e.track_id '+('WHERE e.camera_id=? ' if camera_id else '')+'ORDER BY e.media_time DESC LIMIT ?',((camera_id,limit) if camera_id else (limit,)))

    @app.get('/api/episodes')
    def episodes(scope:Literal['LOCAL','GLOBAL']='GLOBAL',limit:int=Query(default=50,ge=1,le=200)):
        with db() as con:
            if scope=='GLOBAL':return {'status':meta(con,'global_association_status'),'items':rows(con,'SELECT * FROM mtmc_episode ORDER BY start_time LIMIT ?',(limit,)),'scope':'GLOBAL','timeout_seconds':120}
            return {'status':meta(con,'global_association_status'),'items':rows(con,'SELECT * FROM episode WHERE scope=? ORDER BY start_time LIMIT ?',(scope,limit)),
                    'timeout_seconds':meta(con,'episode_timeout_seconds',120),'scope':scope}

    def snapshot(camera_id,media_time,track_id=None):
        with db() as con:
            camera=con.execute('SELECT * FROM camera WHERE id=?',(camera_id,)).fetchone()
            if camera is None:raise HTTPException(404,'Unknown camera')
            if track_id:
                from .tracklets import source_track
                track=source_track(con,track_id)
                if not track:raise HTTPException(404,'Unknown track')
                observation=con.execute('SELECT * FROM observation WHERE track_id=? AND frame_index BETWEEN ? AND ? ORDER BY confidence DESC,frame_index LIMIT 1',(track['parent_id'],track['start_frame'],track['end_frame'])).fetchone()
                if observation is None:raise HTTPException(404,'No observations')
                raw=observation['raw_pts'];index=observation['frame_index'];bbox=json.loads(observation['bbox'])
            else:
                frame=con.execute('SELECT * FROM frame WHERE camera_id=? ORDER BY ABS(media_time-?) LIMIT 1',(camera_id,media_time)).fetchone()
                raw=frame['raw_pts'];index=frame['frame_index'];bbox=None
        folder=settings.artifacts/'thumbnails';folder.mkdir(parents=True,exist_ok=True)
        target=folder/f'{camera_id}_{index}_{track_id or "poster"}.jpg'
        if not target.exists():
            with av.open(str(settings.raw/camera['source_name'])) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=2
                container.seek(round(raw/float(stream.time_base)),stream=stream,backward=True)
                selected=None
                for frame in container.decode(stream):
                    value=float(frame.pts*frame.time_base)
                    if abs(value-raw)<.000001:selected=frame;break
                    if value>raw+.001:break
                if selected is None:raise HTTPException(409,'Exact audited frame unavailable at seek target')
                image=selected.to_image()
                if bbox:image=image.crop(tuple(round(v) for v in bbox))
                image.thumbnail((640,480) if bbox is None else (240,360))
                buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=86)
                from ..detection.common import atomic_bytes
                atomic_bytes(target,buffer.getvalue())
        return FileResponse(target,media_type='image/jpeg',headers={'Cache-Control':'private,max-age=86400'})

    @app.get('/api/cameras/{camera_id}/poster')
    def poster(camera_id:str,time:float=Query(default=0,ge=0,le=3600)):return snapshot(camera_id,time)

    @app.get('/api/tracks/{track_id}/crop')
    def crop(track_id:str):
        from .tracklets import source_track
        with db() as con:r=source_track(con,track_id)
        if not r:raise HTTPException(404,'Unknown track')
        return snapshot(r['camera_id'],0,track_id)

    @app.get('/api/media/{media_id}')
    def media(media_id:str,download:bool=False):
        with db() as con:r=con.execute('SELECT * FROM media_segment WHERE id=?',(media_id,)).fetchone()
        if not r:raise HTTPException(404,'Unknown media')
        target=(settings.media/r['path']).resolve()
        if not target.is_relative_to(settings.media.resolve()) or not target.is_file():raise HTTPException(404,'Media not yet available')
        return FileResponse(target,media_type='video/mp4',filename=target.name if download else None,headers={'Cache-Control':'private,max-age=3600'})

    @app.get('/api/exports')
    def exports():
        with db() as con:
            found=rows(con,'SELECT * FROM media_segment ORDER BY kind,camera_id,start_time')
            revision=meta(con,'mtmc_summary',{}).get('revision')
            found=[r for r in found if r['kind']!='global_review' or json.loads(r['metadata']).get('revision')==revision]
            for r in found:
                if r['kind']=='global_review':
                    evidence=json.loads(r['metadata'])
                    r.update({k:evidence.get(k) for k in ('global_episode_id','global_label','identity_status','revision')})
            for r in found:r['url']='/api/media/'+r['id']+'?download=true';r.pop('metadata',None)
            return {'media':found,'tracks_csv':'/api/exports/tracks.csv','tracks_json':'/api/exports/tracks.json','report':'/api/report'}

    @app.get('/api/exports/tracks.json')
    def json_tracks():
        manifest_path=settings.artifacts/'local_tracks/manifest.json'
        if not manifest_path.exists():raise HTTPException(404,'Machine track records unavailable')
        manifest=read_json(manifest_path)
        from .mtmc import identity
        from .prepare import track_key
        with db() as con:
            identities={}
            for r in con.execute('SELECT * FROM mtmc_tracklet ORDER BY parent_id,start_frame'):
                identities.setdefault(r['parent_id'],[]).append((r['start_frame'],r['end_frame'],identity(con,r['id'])))
            revision=meta(con,'mtmc_summary',{}).get('revision')
        def content():
            header={'schema_version':2,'scope':'LOCAL','source_run_id':manifest['source_run_id'],
                'source_manifest_sha256':manifest['source_manifest_sha256'],'global_association_revision':revision,'includes_empty_frames':True}
            yield (json.dumps(header,ensure_ascii=False)[:-1]+',"frames":[').encode('utf-8')
            first=True;buffer=bytearray()
            for entry in manifest['files']:
                with gzip.open(settings.artifacts/'local_tracks'/entry['path'],'rb') as stream:
                    for line in stream:
                        if not first:buffer.extend(b',')
                        row=json.loads(line)
                        for t in row['local_tracks']:
                            choices=identities.get(track_key(t['local_track_id']),[])
                            match=next((entry[2] for entry in choices if entry[0]<=row['source_frame_index']<=entry[1]),{})
                            t.update(match)
                        buffer.extend(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode('utf-8'));first=False
                        if len(buffer)>=65536:yield bytes(buffer);buffer.clear()
            if buffer:yield bytes(buffer)
            yield b']}'
        return StreamingResponse(content(),media_type='application/json',headers={'Content-Disposition':'attachment; filename="shadowtrack_all_frame_tracks.json"'})

    @app.get('/api/exports/tracks.csv')
    def csv_tracks():
        from .mtmc import identity
        with db() as con:
            found=rows(con,'SELECT * FROM association_track ORDER BY camera_id,start_time')
            for r in found:r.update(identity(con,r['id']))
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(found[0]) if found else ['id']);writer.writeheader();writer.writerows(found)
        return Response('\ufeff'+buffer.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="shadowtrack_local_tracks.csv"'})

    @app.get('/api/exports/records/{camera_id}')
    def frame_track_records(camera_id:str):
        if camera_id not in {'1','2','3','4','5'}:raise HTTPException(404,'Unknown camera')
        path=settings.artifacts/'local_tracks'/f'camera_{camera_id}_tracks.jsonl.gz'
        if not path.exists():raise HTTPException(404,'Machine track records unavailable')
        return FileResponse(path,media_type='application/gzip',filename=path.name)

    @app.get('/api/calibration')
    def calibration():
        with db() as con:return {'items':rows(con,'SELECT * FROM calibration ORDER BY camera_id,raw_start'),'reference_support':alignment.document['reference_support'] if alignment else [],'status':'PARTIAL','policy':'Unchecked manual offsets never become synchronized timestamps; checked values remain manual evidence with uncertainty'}

    @app.post('/api/calibration')
    def save_calibration(value:CalibrationInput):
        with connect(settings.database) as con:
            extent=con.execute('SELECT MIN(raw_pts),MAX(raw_pts) FROM frame WHERE camera_id=? AND segment_id=?',(value.camera_id,value.segment_id)).fetchone()
            if extent[0] is None or value.raw_start<extent[0]-.000001 or value.raw_end>extent[1]+.000001:raise HTTPException(422,'Calibration must stay inside the named audited timestamp segment')
            key=object_hash({k:value.model_dump()[k] for k in ('camera_id','segment_id','raw_start','raw_end')})[:20]
            conflicts=con.execute('SELECT id FROM calibration WHERE camera_id=? AND checked=1 AND id<>? AND raw_start<? AND raw_end>?',(value.camera_id,key,value.raw_end,value.raw_start)).fetchall()
            if value.checked and conflicts:raise HTTPException(422,'Overlapping checked intervals are ambiguous; remove or edit the existing interval first')
            previous=con.execute('SELECT checked FROM calibration WHERE id=?',(key,)).fetchone()
            if value.checked or (previous and previous['checked']):restore_window(con,value.camera_id,value.raw_start,value.raw_end)
            con.execute('INSERT OR REPLACE INTO calibration VALUES(?,?,?,?,?,?,?,?,?,?,?)',(key,value.camera_id,value.segment_id,value.raw_start,value.raw_end,value.offset_seconds,value.drift_ppm,value.uncertainty_seconds,value.checked,value.evidence,str(time.time())))
            # Mutate the derived index only; original Time Alignment artifacts stay immutable.
            if value.checked:
                origin=con.execute('SELECT first_pts FROM camera WHERE id=?',(value.camera_id,)).fetchone()[0]
                uncertainty=json.dumps({'radius_seconds':value.uncertainty_seconds,'source':'manual_checked','calibration_id':key})
                con.execute('UPDATE frame SET global_time=(raw_pts-?)*?+?,global_uncertainty=? WHERE camera_id=? AND segment_id=? AND raw_pts BETWEEN ? AND ?',
                    (origin,1+value.drift_ppm*1e-6,value.offset_seconds,uncertainty,value.camera_id,value.segment_id,value.raw_start,value.raw_end))
            con.execute('DELETE FROM association');con.execute('DELETE FROM episode_member WHERE episode_id IN (SELECT id FROM episode WHERE scope="GLOBAL")')
            con.execute('DELETE FROM episode WHERE scope="GLOBAL"');con.execute('UPDATE local_track SET global_episode_id=NULL')
            set_meta(con,'global_association_status','GLOBAL_ASSOCIATION_PENDING');set_meta(con,'calibration_revision',time.time())
            invalidate_mtmc(con)
        return {'id':key,'status':'MANUAL_CHECKED' if value.checked else 'UNCHECKED','associations_invalidated':True}

    def restore_window(con,camera_id,raw_start,raw_end):
        changes=[]
        for f in con.execute('SELECT frame_index,raw_pts FROM frame WHERE camera_id=? AND raw_pts BETWEEN ? AND ?',(camera_id,raw_start,raw_end)):
            t,u,status=alignment.map_time(camera_id,f['raw_pts']) if alignment else (None,None,'unavailable')
            changes.append((t,json.dumps(u),camera_id,f['frame_index']))
        con.executemany('UPDATE frame SET global_time=?,global_uncertainty=? WHERE camera_id=? AND frame_index=?',changes)

    def invalidate_mtmc(con):
        con.execute('DELETE FROM local_global');con.execute('DELETE FROM mtmc_episode');con.execute('DELETE FROM mtmc_candidate')
        set_meta(con,'mtmc_summary',{'status':'STALE_RECOMPUTE_REQUIRED'})

    @app.delete('/api/calibration/{calibration_id}')
    def remove_calibration(calibration_id:str):
        with connect(settings.database) as con:
            item=con.execute('SELECT * FROM calibration WHERE id=?',(calibration_id,)).fetchone()
            if not item:raise HTTPException(404,'Unknown calibration')
            if item['checked']:restore_window(con,item['camera_id'],item['raw_start'],item['raw_end'])
            con.execute('DELETE FROM calibration WHERE id=?',(calibration_id,))
            con.execute('DELETE FROM association');con.execute('DELETE FROM episode_member WHERE episode_id IN (SELECT id FROM episode WHERE scope="GLOBAL")')
            con.execute('DELETE FROM episode WHERE scope="GLOBAL"');con.execute('UPDATE local_track SET global_episode_id=NULL')
            set_meta(con,'global_association_status','GLOBAL_ASSOCIATION_PENDING')
            invalidate_mtmc(con)
        return {'status':'REMOVED','original_evidence_restored':True,'associations_invalidated':True}

    @app.get('/api/sync/at')
    def sync_at(camera_id:str,media_time:float=Query(ge=0,le=86400)):
        with db() as con:
            row=con.execute('SELECT raw_pts,global_time,global_uncertainty,segment_id,media_time FROM frame WHERE camera_id=? AND media_time BETWEEN ? AND ? ORDER BY ABS(media_time-?) LIMIT 1',(camera_id,max(0,media_time-.06),media_time+.06,media_time)).fetchone()
        if not row:return {'status':'unavailable','reason':'source timestamp gap'}
        return {**dict(row),'uncertainty':json.loads(row['global_uncertainty']),'status':'SUPPORTED_WITH_UNCERTAINTY' if row['global_time'] is not None else 'UNCALIBRATED'}

    @app.get('/api/sync/playback')
    def sync_playback(global_time:float=Query(ge=-86400,le=86400)):
        result={}
        with db() as con:
            for c in con.execute('SELECT id FROM camera'):
                r=con.execute('SELECT media_time,raw_pts,global_time,global_uncertainty,segment_id FROM frame WHERE camera_id=? AND global_time BETWEEN ? AND ? ORDER BY ABS(global_time-?) LIMIT 1',(c['id'],global_time-.07,global_time+.07,global_time)).fetchone()
                result[c['id']]={'available':r is not None,**(dict(r) if r else {}),'uncertainty':json.loads(r['global_uncertainty']) if r else None}
        return {'global_time':global_time,'cameras':result,'time_basis':'supported_conditional_global_coordinate_not_UTC'}

    @app.get('/api/jobs')
    def jobs():
        with db() as con:found=rows(con,'SELECT * FROM job ORDER BY updated_at DESC')
        render=settings.artifacts/'render_progress.json'
        return {'items':found,'render':read_json(render) if render.exists() else None}

    @app.post('/api/associations/recompute')
    def recompute():
        from .mtmc import run_mtmc
        from .mtmc_api import run_exclusive
        return run_exclusive(app,lambda:run_mtmc(settings))

    @app.get('/api/report')
    def report():
        return {'statistics':stats(),'synchronization':calibration(),'limitations':['Estimated clocks and PROVISIONAL identity links are experimental; only explicit reviewer decisions become CONFIRMED','Camera/segment scoped local tracks and episode counts are not unique people','Global termination after 120 seconds requires supported camera coverage; estimated coverage cannot prove absence','Raw PTS gaps are preserved; no floor coordinates without separate checked correspondences'],
            'source_records':'immutable Detection schema v2','web_gpu_required':False}

    from .mtmc_api import register
    register(app,settings)
    return app
