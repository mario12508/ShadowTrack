"""MTMC, review, Gantt and manual floor geometry endpoints."""
import json
import csv
import io
from typing import Literal

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse,Response
from pydantic import BaseModel, Field, model_validator

from ..detection.common import atomic_json
from .db import connect, meta, rows
from .floor import floor_state, save_mapping
from .mtmc import gantt, load_config, run_mtmc, save_review


class Review(BaseModel):
    decision: Literal['CONFIRM','REJECT','RESET']
    reviewer: str=Field(min_length=2,max_length=100)
    evidence: str=Field(min_length=8,max_length=2000)


class Correspondence(BaseModel):
    image: tuple[float,float]
    floor: tuple[float,float]


class Geometry(BaseModel):
    camera_id: str=Field(pattern='^[1-5]$')
    points: list[Correspondence]=Field(min_length=4,max_length=30)
    checked: bool=False
    evidence: str=Field(default='',max_length=2000)


class Offset(BaseModel):
    camera_id: str=Field(pattern='^[1-5]$')
    segment_id: str=Field(min_length=3,max_length=40)
    offset: float=Field(ge=-120,le=120,allow_inf_nan=False)
    uncertainty: float=Field(ge=.001,le=120,allow_inf_nan=False)
    evidence: str=Field(min_length=8,max_length=2000)


def run_exclusive(app,fn):
    if not app.state.worker.acquire(blocking=False):raise HTTPException(409,'Another association task is running')
    try:return fn()
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    finally:app.state.worker.release()


def register(app,settings):
    def db():return connect(settings.database,readonly=True)

    def compute(fn):
        return run_exclusive(app,fn)

    @app.get('/api/mtmc/progress')
    def progress():
        from ..detection.common import read_json
        path=settings.artifacts/'mtmc/telemetry/progress.json'
        return read_json(path) if path.exists() else {'status':'IDLE','phase':'idle','elapsed_s':0}

    @app.get('/api/episode-counts')
    def episode_counts(time:float|None=Query(default=None,ge=0,le=86400,allow_inf_nan=False)):
        from .accounting import accounting,counts
        with db() as con:return counts(accounting(con),time)

    @app.get('/api/exports/episodes.json')
    def episode_json():
        from .accounting import accounting
        with db() as con:return accounting(con)

    @app.get('/api/exports/episodes.csv')
    def episode_csv():
        from .accounting import accounting
        with db() as con:document=accounting(con)
        output=io.StringIO(newline='');writer=csv.writer(output)
        writer.writerow(['episode_id','label','start_time','last_observed_time','camera_ids','fragment_count','identity_status','lifecycle','terminated_at','right_censored','maximum_observed_absence_seconds','clock','revision'])
        for r in document['episodes']:
            state=r['final_lifecycle'];writer.writerow([r['episode_id'],r['label'],r['start_time'],r['last_observed_time'],';'.join(r['camera_ids']),r['fragment_count'],r['identity_status'],state['state'],state['terminated_at'],state['right_censored'],r['maximum_observed_absence_seconds'],document['clock'],document['revision']])
        return Response(output.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="shadowtrack_episodes.csv"'})

    @app.get('/api/mtmc')
    def summary():
        with db() as con:return {'summary':meta(con,'mtmc_summary',{}),'config':meta(con,'mtmc_config',{}),'reviews':rows(con,'SELECT * FROM mtmc_review ORDER BY id DESC'),'estimated_offsets':rows(con,'SELECT * FROM alignment_estimate ORDER BY camera_id,raw_start')}

    @app.post('/api/mtmc/recompute')
    def recompute():return compute(lambda:run_mtmc(settings))

    @app.get('/api/global-tracks')
    def global_tracks(camera_id:str|None=None,q:str=Query(default='',max_length=100),start:float|None=Query(default=None,ge=0),end:float|None=Query(default=None,ge=0),limit:int=Query(default=24,ge=1,le=100),offset:int=Query(default=0,ge=0)):
        from .catalog import global_catalog
        with db() as con:return global_catalog(con,camera_id,q,start,end,limit,offset)

    @app.get('/api/mtmc/candidates')
    def candidates(status:Literal['ALL','PROVISIONAL','CONFIRMED','UNRESOLVED','REJECTED']='ALL',camera_id:str|None=None,limit:int=Query(default=40,ge=1,le=200),offset:int=Query(default=0,ge=0)):
        where=[];args=[]
        if status!='ALL':where.append('p.status=?');args.append(status)
        if camera_id:where.append('(a.camera_id=? OR b.camera_id=?)');args.extend([camera_id,camera_id])
        clause=' WHERE '+' AND '.join(where) if where else ''
        base=' FROM mtmc_candidate p JOIN association_track a ON a.id=p.track_a JOIN association_track b ON b.id=p.track_b'
        with db() as con:
            found=rows(con,'SELECT p.*,a.display_id AS label_a,b.display_id AS label_b,a.camera_id AS camera_a,b.camera_id AS camera_b,a.start_time AS time_a,b.start_time AS time_b'+base+clause+' ORDER BY CASE p.status WHEN "PROVISIONAL" THEN 0 WHEN "CONFIRMED" THEN 1 ELSE 2 END,p.similarity DESC,p.id LIMIT ? OFFSET ?',(*args,limit,offset))
            for r in found:
                r['reasons']=json.loads(r['reasons']);r['evidence']=json.loads(r['evidence']);r['thumbnail_a']='/api/tracks/'+r['track_a']+'/crop';r['thumbnail_b']='/api/tracks/'+r['track_b']+'/crop'
                r['episode_a']=con.execute('SELECT episode_id FROM local_global WHERE track_id=?',(r['track_a'],)).fetchone()[0]
                r['episode_b']=con.execute('SELECT episode_id FROM local_global WHERE track_id=?',(r['track_b'],)).fetchone()[0]
            return {'items':found,'total':con.execute('SELECT COUNT(*)'+base+clause,args).fetchone()[0],'confidence_kind':'cosine heuristic, not calibrated identity probability'}

    @app.post('/api/mtmc/candidates/{candidate_id}/review')
    def review(candidate_id:str,value:Review):return compute(lambda:save_review(settings,candidate_id,**value.model_dump()))

    @app.get('/api/gantt')
    def get_gantt():
        with db() as con:return gantt(con)

    @app.get('/api/global-episodes/{episode_id}')
    def episode(episode_id:str):
        with db() as con:
            e=con.execute('SELECT * FROM mtmc_episode WHERE id=?',(episode_id,)).fetchone()
            if e is None:raise HTTPException(404,'Unknown global episode')
            members=rows(con,'SELECT t.* FROM association_track t JOIN local_global g ON g.track_id=t.id WHERE g.episode_id=? ORDER BY t.start_time',(episode_id,))
            ids={t['id'] for t in members}
            return {**dict(e),'members':members,'decisions':[p for p in rows(con,'SELECT * FROM mtmc_candidate') if p['track_a'] in ids and p['track_b'] in ids]}

    @app.get('/api/mtmc/export')
    def export():
        with db() as con:return {'schema_version':2,'source_run_id':meta(con,'source_run_id'),'summary':meta(con,'mtmc_summary',{}),'mapping':rows(con,'SELECT g.*,e.label,t.parent_id AS source_track_id,t.start_frame,t.end_frame FROM local_global g JOIN mtmc_episode e ON e.id=g.episode_id JOIN mtmc_tracklet t ON t.id=g.track_id'),'episodes':rows(con,'SELECT * FROM mtmc_episode'),'reviews':rows(con,'SELECT * FROM mtmc_review'),'clock_estimates':rows(con,'SELECT * FROM alignment_estimate')}

    @app.post('/api/mtmc/offsets')
    def offset(value:Offset):
        with db() as con:
            if con.execute('SELECT 1 FROM frame WHERE camera_id=? AND segment_id=? LIMIT 1',(value.camera_id,value.segment_id)).fetchone() is None:raise HTTPException(422,'Unknown camera/segment')
        def work():
            from ..detection.common import read_json
            path=settings.artifacts/'mtmc/manual_offsets.json'
            previous=read_json(path) if path.exists() else []
            values=previous
            values=[v for v in values if (v['camera_id'],v['segment_id'])!=(value.camera_id,value.segment_id)]+[value.model_dump()]
            atomic_json(path,values)
            try:return run_mtmc(settings)
            except BaseException:
                atomic_json(path,previous);raise
        return compute(work)

    @app.get('/api/floor')
    def floor(time:float=Query(ge=-120,le=86400,allow_inf_nan=False)):
        with db() as con:return floor_state(con,time)

    @app.get('/api/floor/mappings')
    def mappings():
        with db() as con:
            values=rows(con,'SELECT * FROM floor_mapping ORDER BY camera_id')
            for v in values:v['points']=json.loads(v['points']);v['matrix']=json.loads(v['matrix'])
            return {'items':values,'default':'UNCALIBRATED','coordinates':'normalized design image, not metres'}

    @app.post('/api/floor/mappings')
    def geometry(value:Geometry):
        try:
            with connect(settings.database) as con:
                result=save_mapping(con,value.camera_id,[p.model_dump() for p in value.points],value.checked,value.evidence)
                saved=rows(con,'SELECT * FROM floor_mapping')
            atomic_json(settings.artifacts/'mtmc/floor_mappings.json',{'schema_version':1,'mappings':saved})
            return result
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc

    @app.delete('/api/floor/mappings/{camera_id}')
    def delete_geometry(camera_id:str):
        with connect(settings.database) as con:
            con.execute('DELETE FROM floor_mapping WHERE camera_id=?',(camera_id,));saved=rows(con,'SELECT * FROM floor_mapping')
        atomic_json(settings.artifacts/'mtmc/floor_mappings.json',{'schema_version':1,'mappings':saved})
        return {'status':'REMOVED'}
