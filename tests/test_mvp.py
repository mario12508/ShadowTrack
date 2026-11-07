"""Contract tests plus a read-only smoke against the actual imported FULL run."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shadowtrack.mvp.api import create_app
from shadowtrack.mvp.association import compare_candidate
from shadowtrack.mvp.config import Settings
from shadowtrack.mvp.db import connect,set_meta
from shadowtrack.mvp.episodes import lifecycle,may_continue

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('timeout',[60.,90.,120.])
def test_episode_timeout_and_no_revival(timeout):
    assert lifecycle(10,10,timeout).status=='active'
    assert lifecycle(10,10+timeout-.01,timeout).status=='lost'
    assert lifecycle(10,10+timeout,timeout).status=='terminated'
    assert lifecycle(10,10+timeout,timeout).terminated_at==10+timeout
    assert not may_continue(10,10+timeout,timeout)
    assert may_continue(10,10+timeout-.01,timeout)


def test_missing_global_coverage_does_not_prove_absence():
    state=lifecycle(10,300,120,coverage_complete=False)
    assert state.status=='lost' and state.terminated_at is None and state.reason=='coverage_unknown'
    with pytest.raises(ValueError):lifecycle(20,10)


def test_reid_similarity_alone_never_calibrates_cameras():
    a={'camera_id':'1','global_start':None,'global_end':None}
    b={'camera_id':'3','global_start':1,'global_end':4}
    result=compare_candidate(a,b,.99,4,4)
    assert not result['eligible'] and 'unsupported_time' in result['reasons']


def test_association_rejects_same_camera_conflict_and_terminated_episode():
    a={'camera_id':'1','global_start':0.,'global_end':4.}
    b={'camera_id':'1','global_start':3.,'global_end':6.}
    assert 'simultaneous_same_camera_conflict' in compare_candidate(a,b,.99,4,4)['reasons']
    b={'camera_id':'3','global_start':124.,'global_end':130.}
    assert 'episode_already_terminated' in compare_candidate(a,b,.99,4,4)['reasons']
    b={'camera_id':'3','global_start':3.,'global_end':6.}
    assert 'insufficient_independent_crops' in compare_candidate(a,b,.99,1,4)['reasons']


@pytest.fixture
def small_app(tmp_path):
    settings=Settings(tmp_path,tmp_path/'artifacts',tmp_path/'raw',tmp_path/'evidence',tmp_path/'missing_alignment.json')
    with connect(settings.database) as db:
        db.execute('INSERT INTO camera VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',('1','test.mkv','a'*64,160,120,25,3,5.,5.08,.08,'UNCALIBRATED','{}'))
        db.executemany('INSERT INTO frame VALUES(?,?,?,?,?,?,?,?)',[('1',i,5.+i*.04,i*.04,'1:s0',None,'null',1) for i in range(3)])
        set_meta(db,'source_run_id','test');set_meta(db,'episode_timeout_seconds',120)
    return settings,TestClient(create_app(settings))


def test_api_empty_frames_and_bounded_intervals(small_app):
    settings,client=small_app
    r=client.get('/api/overlays/1?start=0&end=0.1')
    assert r.status_code==200
    assert len(r.json()['frames'])==3 and all(not f['detections'] and not f['tracks'] for f in r.json()['frames'])
    assert client.get('/api/overlays/1?start=0&end=100').status_code==422
    assert client.get('/api/overlays/1?start=2&end=1').status_code==422
    assert client.get('/api/stats').json()['empty_frames']==3
    assert client.get('/api/sync/playback?global_time=0').json()['cameras']['1']['available'] is False


def test_unchecked_calibration_not_applied_and_checked_requires_evidence(small_app):
    settings,client=small_app
    value={'camera_id':'1','segment_id':'1:s0','raw_start':5.,'raw_end':5.08,'offset_seconds':1.,'uncertainty_seconds':.2,'checked':False,'evidence':''}
    assert client.post('/api/calibration',json=value).status_code==200
    with connect(settings.database,readonly=True) as db:assert db.execute('SELECT COUNT(*) FROM frame WHERE global_time IS NOT NULL').fetchone()[0]==0
    assert client.post('/api/calibration',json={**value,'checked':True}).status_code==422
    assert client.post('/api/calibration',json={**value,'raw_end':50.}).status_code==422
    assert client.post('/api/calibration',json={**value,'checked':True,'evidence':'Shared event checked on both source PTS'}).status_code==200
    mapped=client.get('/api/sync/at?camera_id=1&media_time=0.04').json()
    assert abs(mapped['global_time']-1.04)<1e-8 and mapped['uncertainty']['source']=='manual_checked'
    conflict={**value,'raw_start':5.02,'raw_end':5.06,'checked':True,'evidence':'Another checked shared event interval'}
    assert client.post('/api/calibration',json=conflict).status_code==422
    assert client.post('/api/calibration',json={**conflict,'checked':False}).status_code==200
    assert abs(client.get('/api/sync/at?camera_id=1&media_time=0.04').json()['global_time']-1.04)<1e-8
    # Unchecking restores the original unsupported clock; it must not leave stale globals.
    assert client.post('/api/calibration',json=value).status_code==200
    assert client.get('/api/sync/at?camera_id=1&media_time=0.04').json()['global_time'] is None


def test_media_http_ranges_and_path_guard(small_app):
    settings,client=small_app;settings.media.mkdir()
    path=settings.media/'test.mp4';path.write_bytes(b'0123456789'*50)
    with connect(settings.database) as db:
        db.execute('INSERT INTO media_segment VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',('file','1','1:s0','source','test.mp4','b'*64,0,.08,0,3,3,160,120,'{}'))
        db.execute('INSERT INTO media_segment VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',('escape','1','1:s0','source','../shadowtrack.sqlite3','b'*64,0,.08,0,3,3,160,120,'{}'))
    r=client.get('/api/media/file',headers={'Range':'bytes=10-19'})
    assert r.status_code==206 and r.content==b'0123456789'
    assert r.headers['content-range']=='bytes 10-19/500'
    assert client.get('/api/media/file',headers={'Range':'bytes=900-1000'}).status_code==416
    assert client.get('/api/media/escape').status_code==404


def test_missing_preparation_has_explicit_health_status(tmp_path):
    s=Settings(tmp_path,tmp_path/'app',tmp_path/'raw',tmp_path/'evidence',tmp_path/'missing.json')
    assert TestClient(create_app(s)).get('/api/health').status_code==503


@pytest.mark.integration
def test_actual_full_index_api_video_range_smoke():
    s=Settings.environment()
    if not (s.artifacts/'prepare_manifest.json').exists():pytest.skip('Real FULL has not been imported on this checkout')
    client=TestClient(create_app(s));stats=client.get('/api/stats').json()
    assert stats['source_run_id']=='3f764abf202cfb544e6f6a67'
    assert stats['frame']==144110 and stats['detection']==136020 and stats['camera']==5 and stats['empty_frames']==84904
    cameras=client.get('/api/cameras').json()
    assert len(cameras)==5 and all(c['source_url'] for c in cameras)
    for camera in cameras:
        r=client.get(camera['source_url'],headers={'Range':'bytes=0-1023'})
        assert r.status_code==206 and b'ftyp' in r.content[:40]
    overlay=client.get('/api/overlays/3?start=129&end=131').json()
    assert any(f['tracks'] for f in overlay['frames'])
    assert all(f['camera_id']=='3' and f['raw_pts']==f['media_time']+next(c['first_pts'] for c in cameras if c['id']=='3') for f in overlay['frames'])
    tracks=client.get('/api/tracks?camera_id=3&limit=2').json()
    assert tracks['total']>0 and len(tracks['items'])==2
    assert client.get('/api/tracks/'+tracks['items'][0]['id']).status_code==200
