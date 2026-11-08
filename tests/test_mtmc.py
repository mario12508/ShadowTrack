"""Controlled identity hypotheses do not measure accuracy on real office video."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shadowtrack.mvp.api import create_app
from shadowtrack.mvp.association import global_coverage
from shadowtrack.mvp.config import Settings
from shadowtrack.mvp.db import connect, meta, rows, set_meta
from shadowtrack.mvp.floor import fit_mapping, floor_state, project_footpoint, save_mapping
from shadowtrack.mvp.mtmc import gantt, identity, load_config, pair_key, run_mtmc, save_review

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def world(tmp_path):
    (tmp_path/'configs').mkdir();shutil.copyfile(ROOT/'configs/mtmc.json',tmp_path/'configs/mtmc.json')
    s=Settings(tmp_path,tmp_path/'app',tmp_path/'raw',tmp_path/'evidence',tmp_path/'no_alignment.json')
    with connect(s.database) as db:
        set_meta(db,'source_run_id','controlled');set_meta(db,'episode_timeout_seconds',120)
        for cid in '12345':
            fps=20 if cid=='4' else 25
            db.execute('INSERT INTO camera VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(cid,'synthetic.mkv','a'*64,1000,800,fps,8001,0,400,400,'UNCALIBRATED','{}'))
            db.executemany('INSERT INTO frame VALUES(?,?,?,?,?,?,?,?)',[(cid,i,i*.05,i*.05,cid+':s0',i*.05,'{"radius_seconds":0.0}',1) for i in range(8001)])
    def add(key,camera,start,end,vector=None,gaps=()):
        times=[i*.05 for i in range(round(start*20),round(end*20)+1) if not any(a<i*.05<b for a,b in gaps)]
        with connect(s.database) as db:
            db.execute('INSERT INTO local_track VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(key,'source/'+key,camera,camera+':s0','C'+camera+'-'+key,round(start*20),round(end*20),start,end,len(times),.9,None))
            db.executemany('INSERT INTO observation VALUES(?,?,?,?,?,?,?,?)',[(key,camera,round(t*20),t,t,'[100,100,200,700]',.9,0) for t in times])
            if vector is not None:
                v=np.asarray(vector,dtype='float32');v/=np.linalg.norm(v)
                db.execute('INSERT INTO embedding VALUES(?,?,?,?,?,?)',(key,v.astype('<f4').tobytes(),len(v),4,'model',json.dumps({'sample_coherence':.98,'samples':[{'source_frame_index':round(t*20)} for t in times[:4]]})))
        return key
    return s,add


def test_4_3_5_global_episode_short_gap_and_deterministic_replay(world):
    s,add=world
    add('a','4',10,12,[1,0,0]);add('b','3',14.5,17,[1,0,0]);add('c','5',20,23,[1,0,0])
    result=run_mtmc(s);assert result['episodes_by_status']=={'PROVISIONAL':1}
    with connect(s.database,readonly=True) as db:
        mapping=rows(db,'SELECT * FROM local_global ORDER BY track_id');assert len({r['episode_id'] for r in mapping})==1
        assert all(identity(db,k)['global_label']=='ID 001' for k in 'abc')
        assert db.execute('SELECT lifecycle FROM mtmc_episode').fetchone()[0]=='terminated'
        assert db.execute('SELECT termination_reason FROM mtmc_episode').fetchone()[0]=='estimated_clock_coverage_timeout_120s'
    again=run_mtmc(s)
    assert again['revision']==result['revision']
    with connect(s.database,readonly=True) as db:assert mapping==rows(db,'SELECT * FROM local_global ORDER BY track_id')


def test_two_similar_simultaneous_people_do_not_merge(world):
    s,add=world
    add('a','4',10,15,[1,0,0]);add('b','4',10,15,[1,.001,0]);add('c','3',16,18,[1,0,0])
    run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        assert identity(db,'a')['global_episode_id']!=identity(db,'b')['global_episode_id']
        assert identity(db,'a')['global_status']=='SINGLE_CAMERA'
        assert any('competing_simultaneous_lookalikes' in r['reasons'] for r in db.execute('SELECT reasons FROM mtmc_candidate'))


def test_transitive_component_cannot_merge_simultaneous_same_camera(world):
    s,add=world
    add('a','4',10,18,[1,0]);add('b','3',20,22,[1,0]);add('c','4',11,17,[1,.1])
    c=load_config(s);c['ambiguity_margin']=0
    run_mtmc(s,c)
    with connect(s.database,readonly=True) as db:
        m={r['track_id']:r['episode_id'] for r in db.execute('SELECT * FROM local_global')}
        assert m['a']!=m['c']


def test_confirmed_coverage_timeout_and_new_episode_after_120(world):
    s,add=world
    add('a','4',10,12,[1,0]);add('b','3',14,16,[1,0]);add('c','4',150,152,[1,0])
    c=load_config(s);c['mode']='supported_alignment';run_mtmc(s,c)
    with connect(s.database,readonly=True) as db:
        assert global_coverage(db,16,136)
        m={r['track_id']:r['episode_id'] for r in db.execute('SELECT * FROM local_global')}
        assert m['a']==m['b'] and m['c']!=m['a']
        e=db.execute('SELECT * FROM mtmc_episode WHERE id=?',(m['a'],)).fetchone()
        assert e['lifecycle']=='terminated' and e['terminated_at']==136
    with connect(s.database) as db:
        db.execute('UPDATE frame SET global_time=NULL WHERE camera_id=? AND raw_pts BETWEEN 50 AND 51',('2',))
        assert not global_coverage(db,16,136)


def test_human_reviews_are_explicit_replayable_and_cannot_link_is_transitive(world):
    s,add=world
    for key,cam,a,b in [('a','4',10,12),('b','3',14,16),('c','5',20,22)]:add(key,cam,a,b,[1,0])
    run_mtmc(s)
    key=pair_key('a','b');save_review(s,key,'CONFIRM','test reviewer','Controlled synthetic evidence, not a real identity assertion')
    save_review(s,pair_key('b','c'),'CONFIRM','test reviewer','Controlled synthetic chain evidence')
    with connect(s.database,readonly=True) as db:
        assert identity(db,'a')['global_status']=='CONFIRMED'
        assert meta(db,'mtmc_summary')['episodes_by_status']=={'CONFIRMED':1}
    save_review(s,pair_key('a','c'),'REJECT','test reviewer','Controlled negative observation overrides automatic chain')
    with connect(s.database,readonly=True) as db:
        m={r['track_id']:r['episode_id'] for r in db.execute('SELECT * FROM local_global')};assert m['a']!=m['c']
        assert db.execute('SELECT COUNT(*) FROM mtmc_review').fetchone()[0]==3
    saved=json.loads((s.artifacts/'mtmc/human_reviews.json').read_text());assert saved['human_reviewed']


def test_no_embeddings_remains_unresolved_and_raw_pts_immutable(world):
    s,add=world;add('a','4',10,12);add('b','3',15,18)
    with connect(s.database,readonly=True) as db:before=rows(db,'SELECT * FROM frame ORDER BY camera_id,frame_index')
    run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        assert before==rows(db,'SELECT * FROM frame ORDER BY camera_id,frame_index')
        assert identity(db,'a')['global_status']=='SINGLE_CAMERA'


def test_frame_identity_respects_derived_slices_without_rewriting_bytetrack(world):
    s,add=world;add('contaminated','5',10,18,[1,0])
    with connect(s.database) as db:
        for i,t in enumerate(range(10,19)):
            vector=np.asarray([1,0] if i<4 else [0,1],dtype='<f4')
            color=[1,0,1,0] if i<4 else [0,1,0,1]
            ev={'source_frame_index':t*20,'raw_media_pts':t,'source_track_id':'contaminated'}
            db.execute('INSERT INTO appearance_sample VALUES(?,?,?,?,?,?,?)',('contaminated',t*20,t,vector.tobytes(),.95,json.dumps(color),json.dumps(ev)))
        db.execute('UPDATE observation SET bbox=? WHERE track_id=? AND frame_index>=280',('[700,100,800,700]','contaminated'))
        original=[tuple(r) for r in db.execute('SELECT * FROM observation')]
    run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        assert original==[tuple(r) for r in db.execute('SELECT * FROM observation')]
        assert identity(db,'contaminated',200)['global_episode_id']!=identity(db,'contaminated',360)['global_episode_id']
    api=TestClient(create_app(s))
    before=api.get('/api/overlays/5?start=10&end=11').json();after=api.get('/api/overlays/5?start=17&end=18').json()
    a=next(f['tracks'][0] for f in before['frames'] if f['tracks']);b=next(f['tracks'][0] for f in after['frames'] if f['tracks'])
    assert a['source_track_id']==b['source_track_id']=='contaminated'
    assert a['track_id']!=b['track_id'] and a['global_episode_id']!=b['global_episode_id']
    assert api.get('/api/tracks/'+b['track_id']).json()['parent_id']=='contaminated'


def test_gantt_keeps_gaps_parallel_cameras_and_segment_boundaries(world):
    s,add=world
    add('a','4',10,16,[1,0],gaps=[(12,14)]);add('b','3',11,15,[1,0]);run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        data=gantt(db);assert len(data['rows'])==1
        a=next(r for r in data['rows'][0]['segments'] if r['track_id']=='a');assert a['intervals']==[[10,12],[14,16]]
        assert len(data['rows'][0]['segments'])==2


def test_floor_without_calibration_no_person_coordinates(world):
    s,add=world;add('a','4',10,16,[1,0]);run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        state=floor_state(db,11)
        assert not state['markers'] and state['unlocated'][0]['position'] is None
        assert state['unlocated'][0]['localization_status']=='UNCALIBRATED'


def test_manual_floor_geometry_checked_only_no_extrapolation():
    points=[{'image':a,'floor':b} for a,b in [([.1,.1],[.2,.2]),([.9,.1],[.8,.2]),([.9,.9],[.8,.8]),([.1,.9],[.2,.8])]]
    matrix,residual=fit_mapping(points);assert residual<1e-5
    mapping={'points':json.dumps(points),'matrix':json.dumps(matrix),'status':'UNCHECKED'}
    assert project_footpoint(mapping,[450,100,550,600],1000,800)==(None,'UNCALIBRATED')
    mapping['status']='MANUAL_APPROXIMATE'
    point,status=project_footpoint(mapping,[450,100,550,600],1000,800);assert status=='APPROXIMATE_PLANAR' and np.allclose(point,[.5,.6875])
    assert project_footpoint(mapping,[450,100,550,800],1000,800)[1]=='FEET_CLIPPED'
    assert project_footpoint(mapping,[450,0,550,10],1000,800)[1]=='OUTSIDE_CALIBRATED_REGION'
    with pytest.raises(ValueError):fit_mapping([{'image':[i/10,i/10],'floor':[i/10,i/10]} for i in range(4)])


def test_estimated_offsets_do_not_change_supported_clock_and_invalid_offset_rolls_back(world):
    s,add=world;add('a','4',10,15,[1,0]);run_mtmc(s);client=TestClient(create_app(s))
    assert client.get('/api/sync/at?camera_id=4&media_time=10').json()['global_time']==10
    result=client.post('/api/mtmc/offsets',json={'camera_id':'4','segment_id':'4:s0','offset':2,'uncertainty':3,'evidence':'Controlled independent manual hypothesis'})
    assert result.status_code==200
    assert client.get('/api/sync/at?camera_id=4&media_time=10').json()['global_time']==10
    assert client.get('/api/floor?time=12').json()['unlocated'][0]['raw_media_pts']==12
    assert client.post('/api/mtmc/offsets',json={'camera_id':'4','segment_id':'unknown','offset':2,'uncertainty':3,'evidence':'Invalid test segment'}).status_code==422


@pytest.mark.integration
def test_actual_full_mtmc_mapping_overlays_gantt_floor_api_read_only():
    s=Settings.environment()
    if not (s.artifacts/'mtmc/summary.json').exists():pytest.skip('Actual FULL MTMC not prepared')
    client=TestClient(create_app(s));summary=client.get('/api/mtmc').json()['summary']
    assert summary['local_tracks']==503 and summary['genuine_embedding_tracks']==390
    export=client.get('/api/mtmc/export').json();assert len(export['mapping'])==summary['derived_tracklets']
    assert len({m['source_track_id'] for m in export['mapping']})==503
    rows_=client.get('/api/gantt').json()['rows'];assert sum(len(e['segments']) for e in rows_)==summary['derived_tracklets']
    candidate=next(e for e in rows_ if e['status']=='PROVISIONAL' and {'3','4','5'}<={t['camera_id'] for t in e['segments']})
    labels=[]
    for segment in candidate['segments']:
        t=client.get('/api/tracks/'+segment['track_id']).json();labels.append(t['global_label']);assert t['global_status']=='PROVISIONAL'
        overlay=client.get(f"/api/overlays/{t['camera_id']}?start={t['start_time']}&end={t['start_time']+1}").json()
        assert any(r['global_episode_id']==candidate['id'] for f in overlay['frames'] for r in f['tracks'])
    assert len(set(labels))==1
    floor=client.get('/api/floor?time=1030').json()
    assert floor['markers'] and all(m['localization_status']=='SCHEMATIC_PROJECTED' for m in floor['markers'])
    assert all(m['floor_zone']['radius_normalized']>0 for m in floor['markers'])
    assert client.get('/api/exports/tracks.csv').status_code==200
    assert client.get('/api/mtmc/candidates?status=PROVISIONAL&limit=2').json()['total']>0


def test_global_catalog_one_card_preserves_separate_local_intervals(world):
    from shadowtrack.mvp.catalog import global_catalog
    s,add=world;add('a','3',10,12,[1,0]);add('b','3',14,18,[1,0]);run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        catalog=global_catalog(db)
        assert catalog['total']==1 and catalog['items'][0]['fragment_count']==2
        item=catalog['items'][0]
        assert item['start_time']==10 and item['end_time']==18
        assert [i['start_time'] for i in item['intervals']]==[10,14]
        assert global_catalog(db,camera_id='4')['total']==0
        assert global_catalog(db,q='1')['total']==1
        assert global_catalog(db,q='ID 2')['total']==0
    client=TestClient(create_app(s))
    assert client.get('/api/global-tracks').json()['total']==1
    assert client.get('/api/tracks').json()['total']==2  # Technical local API retained.


def test_display_numbers_follow_first_observation_and_replay_stably(world):
    s,add=world;add('z_early','5',1,3,[1,0]);add('a_late','3',20,22,[0,1])
    run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        assert identity(db,'z_early')['global_label']=='ID 001'
        assert identity(db,'a_late')['global_label']=='ID 002'
        internal=identity(db,'z_early')['global_episode_id']
    run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        assert identity(db,'z_early')['global_label']=='ID 001'
        assert identity(db,'z_early')['global_episode_id']==internal


@pytest.mark.integration
def test_real_continuous_turn_handoff_and_duplicate_cards_regression():
    from shadowtrack.mvp.catalog import global_catalog
    s=Settings.environment()
    if not (s.artifacts/'mtmc/summary.json').exists():pytest.skip('Actual FULL MTMC not prepared')
    with connect(s.database,readonly=True) as db:
        get=lambda name:rows(db,'SELECT t.id,g.episode_id,e.label FROM association_track t JOIN local_global g ON g.track_id=t.id JOIN mtmc_episode e ON e.id=g.episode_id WHERE t.display_id=?',(name,))
        a,b,c=get('C4-T001'),get('C5-T006'),get('C5-T007')
        assert len(a)==1  # Turning around does not cut a continuously tracked person.
        assert a[0]['episode_id']==b[0]['episode_id']==c[0]['episode_id']
        assert get('C3-T011')[0]['episode_id']==get('C3-T013')[0]['episode_id']
        label=get('C3-T011')[0]['label']
        catalog=global_catalog(db,q=label)
        assert sum(item['global_label']==label for item in catalog['items'])==1
        labels=sorted(int(r[0].split()[-1]) for r in db.execute('SELECT label FROM mtmc_episode'))
        assert labels==list(range(1,len(labels)+1))


@pytest.mark.integration
def test_reported_partial_woman_cross_view_and_low_quality_head_regressions():
    from shadowtrack.mvp.catalog import global_catalog
    s=Settings.environment()
    if not (s.artifacts/'mtmc/summary.json').exists():pytest.skip('Actual FULL MTMC not prepared')
    with connect(s.database,readonly=True) as db:
        def episode(label):
            result=rows(db,'SELECT g.episode_id,e.label FROM association_track t JOIN local_global g ON g.track_id=t.id JOIN mtmc_episode e ON e.id=g.episode_id WHERE t.display_id=?',(label,))
            assert len(result)==1
            return result[0]
        # Regression assertions are independent of the engine; no source IDs or
        # control labels are read by any candidate generation/partition function.
        woman=[episode(k) for k in ('C3-T001','C3-T002','C5-T001','C5-T002','C5-T003')]
        assert len({r['episode_id'] for r in woman})==1 and woman[0]['label']=='ID 002'
        catalog=global_catalog(db,q=woman[0]['label'])
        assert catalog['total']==1 and catalog['items'][0]['fragment_count']>=5
        assert episode('C3-T003')['episode_id']==episode('C3-T005')['episode_id']
        assert episode('C5-T206')['episode_id']!=episode('C5-T204')['episode_id']
