"""Counts, censoring, replay and exports must not invent future observations."""
import csv
import io

import pytest
from fastapi.testclient import TestClient

from test_mtmc import world
from shadowtrack.mvp.accounting import accounting,counts,lifecycle_for
from shadowtrack.mvp.api import create_app
from shadowtrack.mvp.db import connect,meta
from shadowtrack.mvp.mtmc import run_mtmc


def test_two_camera_episode_is_counted_once_and_future_episodes_are_not_started(world):
    s,add=world;add('a','4',10,12,[1,0]);add('b','3',14,16,[1,0]);add('c','5',200,202,[0,1])
    run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        document=accounting(db);first=counts(document,11);before=counts(document,0);later=counts(document,150)
        assert first['total_episodes']==2 and first['started_episodes']==1 and first['active']==1
        assert before['started_episodes']==before['active']==before['terminated']==0
        assert later['started_episodes']==1 and later['terminated']==1 and later['multi_camera_episodes']==1
        assert counts(document)['single_camera_episodes']==1 and document['identity_count'] is None


def test_replay_is_lost_between_real_observations_and_timeout_has_exact_boundary(world):
    s,add=world;add('a','4',10,12,[1,0]);add('b','3',20,22,[1,0]);run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        document=accounting(db);key=document['episodes'][0]['episode_id']
        assert lifecycle_for(document,key,15)['state']=='lost'
        assert lifecycle_for(document,key,21)['state']=='active'
        assert lifecycle_for(document,key,141.99)['state']=='lost'
        ended=lifecycle_for(document,key,142)
        assert ended['state']=='terminated' and ended['terminated_at']==142
        assert not ended['right_censored']
        assert document['episodes'][0]['maximum_observed_absence_seconds']==8


def test_recording_cutoff_does_not_fabricate_a_terminated_episode(world):
    s,add=world;add('a','4',380,382,[1,0]);run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        document=accounting(db);state=document['episodes'][0]['final_lifecycle']
        assert state['state']=='lost' and state['right_censored'] and state['terminated_at'] is None
        assert state['reason']=='recording_ended_before_timeout'
        assert counts(document)['right_censored']==1 and counts(document)['terminated']==0


def test_missing_source_coverage_blocks_timeout_even_when_late_camera_data_exists(world):
    s,add=world;add('a','4',10,12,[1,0])
    with connect(s.database) as db:db.execute("DELETE FROM frame WHERE camera_id='2' AND media_time BETWEEN 50 AND 51")
    run_mtmc(s)
    with connect(s.database,readonly=True) as db:
        state=accounting(db)['episodes'][0]['final_lifecycle']
        assert state['state']=='lost' and state['right_censored'] and state['reason']=='coverage_gap'


def test_episode_exports_have_one_record_per_id_and_match_stats_catalog(world):
    s,add=world;add('a','4',10,12,[1,0]);add('b','3',14,16,[1,0]);add('c','5',380,382,[0,1]);run_mtmc(s)
    client=TestClient(create_app(s));document=client.get('/api/exports/episodes.json').json()
    table=list(csv.DictReader(io.StringIO(client.get('/api/exports/episodes.csv').text)))
    totals=client.get('/api/episode-counts').json();stats=client.get('/api/stats').json()
    assert len(table)==len(document['episodes'])==totals['total_episodes']==stats['global_episodes']==2
    assert len({r['episode_id'] for r in table})==2
    assert totals['multi_camera_episodes']==1 and totals['single_camera_episodes']==1
    assert totals['active']+totals['lost']+totals['terminated']==totals['started_episodes']
    assert stats['episode_counts']==totals
    catalog=client.get('/api/global-tracks').json()['items']
    assert sum(t['status']=='terminated' for t in catalog)==totals['terminated']
    assert sum(t['right_censored'] for t in catalog)==totals['right_censored']
    assert client.get('/api/episode-counts?time=-1').status_code==422


@pytest.mark.parametrize('path',['/api/mtmc/recompute','/api/associations/recompute'])
def test_both_recompute_routes_share_exclusive_job_lock(world,path):
    s,add=world;app=create_app(s);client=TestClient(app);app.state.worker.acquire()
    try:assert client.post(path).status_code==409
    finally:app.state.worker.release()


def test_candidate_links_expose_the_actual_global_episode_on_each_side(world):
    s,add=world;add('a','4',10,12,[1,0]);add('b','4',10,12,[1,0]);run_mtmc(s)
    client=TestClient(create_app(s));items=client.get('/api/mtmc/candidates?status=REJECTED').json()['items']
    assert items and items[0]['episode_a']!=items[0]['episode_b']


@pytest.mark.integration
def test_actual_full_episode_accounting_and_continuity_are_self_consistent():
    from shadowtrack.mvp.config import Settings
    s=Settings.environment()
    if not (s.artifacts/'mtmc/episode_accounting.json').exists():pytest.skip('Actual FULL accounting not prepared')
    with connect(s.database,readonly=True) as db:
        document=accounting(db);totals=counts(document)
        assert len(document['episodes'])==db.execute('SELECT COUNT(*) FROM mtmc_episode').fetchone()[0]
        assert len({r['episode_id'] for r in document['episodes']})==totals['total_episodes']
        assert all(r['maximum_observed_absence_seconds']<120 for r in document['episodes'])
        assert document['revision']==meta(db,'mtmc_summary')['revision']
