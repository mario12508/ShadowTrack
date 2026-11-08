"""Meaningful failure cases: contaminated identities, duplicates and global conflicts."""
import json
from dataclasses import replace

import numpy as np
import pytest

from shadowtrack.mvp.assignment import global_partition
from shadowtrack.mvp.config import Settings
from shadowtrack.mvp.db import connect,set_meta,meta
from shadowtrack.mvp.snapshot import publish_snapshot,bootstrap_native_database
from shadowtrack.mvp.tracklets import partial_duplicate,split_samples


def series(vectors,colors):
    result=[]
    for i,(v,color) in enumerate(zip(vectors,colors)):
        v=np.asarray(v,dtype='float32');v/=np.linalg.norm(v)
        result.append({'feature':v,'quality':.95,'histogram':np.asarray(color,dtype=float),'raw_pts':float(i),'frame_index':i*25})
    return result


def test_identity_switch_requires_sustained_appearance_and_clothing_change():
    ss=series([[1,0]]*4+[[0,1]]*4,[[1,0,1,0]]*4+[[0,1,0,1]]*4)
    obs=[{'frame_index':i,'bbox':'[10,10,30,80]' if i<90 else '[110,10,130,80]'} for i in range(176)]
    changes=split_samples(ss,obs)
    assert len(changes)==1 and changes[0]['uncertainty_frame_interval']==[75,100]
    assert changes[0]['frame_index']==90
    continuous=[{'frame_index':i,'bbox':'[10,10,30,80]'} for i in range(176)]
    assert split_samples(ss,continuous)==[]  # Turning in place is not an ID switch.
    # A severe pose change with unchanged clothing does not itself split a person.
    stable=series([[1,0]]*4+[[0,1]]*4,[[1,0,1,0]]*8)
    assert split_samples(stable,obs)==[]


def fake_track(box,camera='5'):
    return {'camera_id':camera,'observed_rows':[{'frame_index':i,'bbox':json.dumps(box)} for i in range(25)]}


def test_partial_duplicate_needs_nested_different_size_boxes_and_same_head():
    full=fake_track([100,100,200,500]);upper=fake_track([100,100,200,300])
    assert partial_duplicate(full,upper,.95)['shared_frames']==25
    assert partial_duplicate(full,fake_track([210,100,310,500]),.95) is None
    assert partial_duplicate(full,fake_track([100,100,200,500]),.95) is None
    # Torso and full-body embeddings differ even for identical pixel heads.
    assert partial_duplicate(full,upper,.45)['appearance_similarity_diagnostic']==.45
    assert partial_duplicate(full,fake_track([100,250,200,500]),.95) is None


def test_global_optimizer_selects_joint_solution_instead_of_strongest_greedy_edge():
    tracks=[{'id':k,'camera_id':cam,'aligned_start':0,'aligned_end':5,'time_uncertainty':0} for k,cam in [('a','4'),('b','3'),('c','3'),('d','4')]]
    candidates=[]
    for a,b,score in [('a','b',.91),('a','c',.89),('b','d',.89)]:
        candidates.append({'track_a':a,'track_b':b,'score':score,'similarity':score,'status':'ELIGIBLE','reasons':[],'evidence':{'human_review':None}})
    config={'appearance_threshold':.82,'component_minimum_similarity':.72,'ambiguity_margin':.001,'clock_tolerance_seconds':3,
            'transitions':[{'cameras':['3','4'],'seconds':[0,45],'overlap_possible':True}]}
    groups,proof=global_partition(tracks,candidates,np.full((4,4),.9),config)
    assert {frozenset(tracks[i]['id'] for i in group) for group in groups}=={frozenset('ac'),frozenset('bd')}
    assert proof['mip_gap']==0 and proof['cuts']>0


def test_native_database_snapshot_is_closed_hash_bound_and_separate_from_host(tmp_path):
    s=Settings(tmp_path,tmp_path/'outputs',tmp_path/'raw',tmp_path/'evidence',tmp_path/'sync.json')
    with connect(s.database) as db:set_meta(db,'source_run_id','source');set_meta(db,'counter',1)
    published=publish_snapshot(s)
    assert published['closed_snapshot']
    native=replace(s,database_override=tmp_path/'native/state.sqlite3')
    bootstrap_native_database(native)
    with connect(native.database) as db:set_meta(db,'counter',2)
    bootstrap_native_database(native)  # An unchanged source never overwrites API state.
    with connect(native.database,readonly=True) as db:assert meta(db,'counter')==2
    with connect(s.database,readonly=True) as db:assert meta(db,'counter')==1
    manifest=s.artifacts/'server/snapshot.json';value=json.loads(manifest.read_text());value['sha256']='bad';manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='hash mismatch'):bootstrap_native_database(native)
