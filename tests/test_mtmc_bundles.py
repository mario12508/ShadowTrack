"""Pose, partial visibility, competition and real motion evidence regressions."""
import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from shadowtrack.detection.common import atomic_json,object_hash,read_json
from shadowtrack.mvp.assignment import global_partition
from shadowtrack.mvp.association_features import clothing_similarity,prototype_bank,gallery_similarity
from shadowtrack.mvp.local_bundles import consolidate
from shadowtrack.mvp.normalized_appearance import balanced,scene_gain,head_signature,validate_cache
from shadowtrack.mvp.spatial import opposing_corridor_motion,local_reentry
from shadowtrack.probe import sha256


def track(key,box,start=0,end=3,vector=(1.,0.),quality=.9):
    v=np.asarray(vector);v=v/np.linalg.norm(v)
    obs=[{'frame_index':round(t*25),'media_time':float(t),'bbox':json.dumps(box),'confidence':.9} for t in np.arange(start,end+.001,.04)]
    samples=[{'frame_index':obs[k]['frame_index'],'raw_pts':obs[k]['media_time'],'quality':quality,'feature':v,
              'histogram':np.ones(4)/2,'head_feature':np.array([1.,0.]),'provenance':{'frame':obs[k]['frame_index']}} for k in (0,len(obs)//2,-1)]
    return {'id':key,'camera_id':'5','observed_rows':obs,'start_time':start,'end_time':end,'observations':len(obs),
            'vector':v,'samples':3,'coherence':.95,'appearance_samples':samples,'color_histogram':np.ones(4)/2}


def test_nested_partial_geometry_pools_full_body_features_without_modifying_boxes():
    full=track('full',[100,100,200,500]);upper=track('torso',[100,100,200,300],vector=(0.,1.))
    saved=deepcopy([full['observed_rows'],upper['observed_rows']])
    proof=consolidate([full,upper])
    assert proof['geometric_links']==1 and full['local_bundle_id']==upper['local_bundle_id']
    assert np.allclose(upper['vector'],[1,0]) and upper['bundle_evidence']['rich_sources']==['full']
    assert [full['observed_rows'],upper['observed_rows']]==saved
    assert json.loads(upper['geometry_rows'][0]['bbox'])==[100,100,200,500]
    from shadowtrack.mvp.mtmc import candidate
    for t in (full,upper):
        t.update(aligned_start=0,aligned_end=8,time_uncertainty=3,time_status='SOURCE',endpoint_boxes=[[0,0,1,1]]*2,crop_evidence=[])
    c={'component_minimum_similarity':.72,'appearance_threshold':.82,'minimum_sample_coherence':.65,'clock_tolerance_seconds':3,
       'episode_timeout_seconds':120,'mode':'estimated_alignment','transition_status':'HYPOTHESIS','topology_source':'synthetic','transitions':[]}
    assert candidate(full,upper,1.,c)['status']=='ELIGIBLE'  # Duplicates are allowed beyond the clock tolerance too.


def test_partial_duplicate_cannot_bridge_two_co_visible_people():
    full=track('a',[100,100,200,500]);other=track('b',[110,100,210,500]);upper=track('c',[108,100,198,250])
    consolidate([full,other,upper])
    assert full['local_bundle_id']!=other['local_bundle_id']


def test_occluded_continuation_needs_crop_quality_and_unique_head_trajectory():
    short=track('short',[100,100,150,240],end=.64);later=track('full',[110,110,170,500],start=2,end=5)
    consolidate([short,later]);assert short['local_bundle_id']==later['local_bundle_id']
    weak=track('head_only',[100,100,150,140],end=.64,quality=.16);later=track('full',[110,110,170,500],start=2,end=5)
    consolidate([weak,later]);assert weak['local_bundle_id']!=later['local_bundle_id']
    short=track('short',[100,100,150,240],end=.64)
    alternatives=[track('a',[100,110,160,500],start=2,end=5),track('b',[100,110,160,500],start=2,end=5)]
    consolidate([short,*alternatives]);assert len({t['local_bundle_id'] for t in [short,*alternatives]})==3


def test_neutral_scene_balance_removes_cast_and_rejects_unusable_reference():
    image=Image.new('RGB',(40,40),(180,150,100));gain,rgb=scene_gain(image,[0,0,1,1])
    pixel=np.asarray(balanced(image,gain))[0,0]
    assert np.ptp(pixel)<=1 and np.allclose(rgb,[180,150,100])
    with pytest.raises(ValueError,match='insufficient'):scene_gain(Image.new('RGB',(40,40),'black'),[0,0,1,1])
    assert head_signature(image).shape==(128,) and np.isfinite(head_signature(image)).all()


def test_clothing_tolerates_adjacent_exposure_levels_without_equating_blue_and_black():
    a=np.zeros((2,16));b=a.copy();c=a.copy();a[:,12]=1;b[:,13]=1;c[:,7]=1
    assert clothing_similarity(a.ravel(),b.ravel())>clothing_similarity(a.ravel(),c.ravel())
    assert clothing_similarity(a.ravel(),a.ravel())==pytest.approx(1)


def test_gallery_retains_two_supported_poses_but_not_a_single_outlier():
    samples=[{'feature':np.asarray(v),'quality':.9,'raw_pts':t} for t,v in enumerate([(1.,0.),(1.,0.),(0.,1.),(0.,1.)])]
    bank=prototype_bank(samples,np.array([1.,0.]))
    assert gallery_similarity({'appearance_bank':bank},{'appearance_bank':np.array([[0.,1.]])})==1
    assert len(prototype_bank(samples[:3],np.array([1.,0.])))==2  # fallback + supported first view


def test_global_component_charges_pose_disagreement_but_preserves_physical_exclusion():
    def solve(negative,co_visible=False):
        ts=[{'id':k,'camera_id':cam,'aligned_start':start,'aligned_end':start+3,'time_uncertainty':0}
            for k,cam,start in [('a','1',0),('b','2',0),('c','1' if co_visible else '3',0 if co_visible else 4)]]
        edges=[{'track_a':a,'track_b':b,'score':.9,'similarity':.9,'status':'ELIGIBLE','reasons':[],'evidence':{}}
               for a,b in [('a','b'),('b','c')]]
        scores=np.ones((3,3))*.9;scores[0,2]=scores[2,0]=negative
        config={'appearance_threshold':.82,'component_minimum_similarity':.72,'ambiguity_margin':0,'clock_tolerance_seconds':3,
                'transitions':[{'cameras':[a,b],'seconds':[0,30],'overlap_possible':True} for a,b in [('1','2'),('2','3'),('1','3')]]}
        return global_partition(ts,edges,scores,config)
    groups,proof=solve(.66);assert len(groups)==1 and proof['component_appearance_cost']==pytest.approx(.06)
    assert len(solve(.3)[0])==2  # Negative whole-component evidence outweighs one weak bridge.
    assert len(solve(.95,True)[0])==2  # Even perfect appearance cannot override co-visibility.


def test_global_transitive_topology_spends_estimated_clock_budget_only_once():
    ts=[{'id':k,'camera_id':cam,'aligned_start':0,'aligned_end':5,'time_uncertainty':3} for k,cam in [('a','1'),('b','3'),('c','4')]]
    edges=[{'track_a':a,'track_b':b,'score':.9,'status':'ELIGIBLE','reasons':[],'evidence':{}} for a,b in [('a','b'),('b','c')]]
    config={'appearance_threshold':.82,'component_minimum_similarity':.72,'ambiguity_margin':0,'clock_tolerance_seconds':3,
            'mode':'estimated_alignment','transitions':[{'cameras':[a,b],'seconds':[0,30],'overlap_possible':True} for a,b in [('1','3'),('3','4')]]}
    groups,_=global_partition(ts,edges,np.ones((3,3)),config)
    assert len(groups)==2  # 1 and 4 cannot overlap for 5 s via a transitive chain either.


def test_nested_boxes_do_not_cast_multiple_independent_cross_camera_votes():
    ts=[{'id':k,'camera_id':cam,'aligned_start':0,'aligned_end':5,'time_uncertainty':0,
         'local_bundle_id':'A' if k in ('a','b') else k,'local_bundle_size':2 if k in ('a','b') else 1}
        for k,cam in [('a','4'),('b','4'),('c','4'),('d','3')]]
    edges=[{'track_a':a,'track_b':b,'score':score,'status':'ELIGIBLE','reasons':[],
            'evidence':{'partial_duplicate':True} if (a,b)==('a','b') else {}}
           for a,b,score in [('a','b',.98),('a','d',.9),('b','d',.9),('c','d',.94)]]
    config={'appearance_threshold':.82,'component_minimum_similarity':.72,'ambiguity_margin':0,'clock_tolerance_seconds':3,
            'transitions':[{'cameras':['3','4'],'seconds':[0,30],'overlap_possible':True}]}
    saved=deepcopy(edges);groups,_=global_partition(ts,edges,np.ones((4,4)),config)
    assert {frozenset(ts[k]['id'] for k in g) for g in groups}=={frozenset('ab'),frozenset('cd')}
    # An occluded bundle member with no eligible edge must not dilute the
    # evidence supplied by its visible member.
    fewer=[p for p in saved if (p['track_a'],p['track_b'])!=('b','d')]
    next(p for p in fewer if p['track_a']=='c')['score']=.89
    groups,_=global_partition(ts,fewer,np.ones((4,4)),config)
    assert {frozenset(ts[k]['id'] for k in g) for g in groups}=={frozenset('abd'),frozenset('c')}


def corridor(camera,lag=0,opposite=True):
    q=[[.4,.1],[.6,.1],[.6,.9],[.4,.9]]
    if camera=='5':q=q[2:]+q[:2] if opposite else q
    projection={'kind':'APPROXIMATE_FLOOR_PLANE','floor_points':q,'image_points':[[0,0],[1,0],[1,1],[0,1]]}
    result={'camera_id':camera,'width':1000,'height':1000,'spatial_projection':projection,'observed_rows':[]}
    for i,t in enumerate(np.arange(0,5,.05)):
        depth=3+.6*t if camera=='3' else 7-.6*t;h=1000/depth
        result['observed_rows'].append({'media_time':float(t+lag),'frame_index':i,'confidence':.9,'bbox':json.dumps([500-h/7,800-h,500+h/7,800])})
    return result


def test_opposing_motion_uses_static_orientation_and_genuine_observed_support():
    a,b=corridor('3'),corridor('5',lag=.8);proof=opposing_corridor_motion(a,b)
    assert proof and proof['inverse_height_correlation']<-.99 and proof['paired_samples']>=16
    assert opposing_corridor_motion(a,corridor('5',opposite=False)) is None
    stationary=corridor('5')
    for o in stationary['observed_rows']:o['bbox']='[450,500,550,800]'
    assert opposing_corridor_motion(a,stationary) is None
    a,b=corridor('3'),corridor('5')
    a['observed_rows']=a['observed_rows'][:35];b['observed_rows']=b['observed_rows'][60:]
    assert opposing_corridor_motion(a,b,0) is None  # No interpolation through a detection gap.


def test_same_camera_reentry_rejects_remote_or_late_portals():
    a={'camera_id':'3','start_time':0,'end_time':5,'zones':{'status':'APPROXIMATE_IMAGE_PORTALS','exit_point':[.4,.5]}}
    b={'camera_id':'3','start_time':18,'end_time':25,'zones':{'status':'APPROXIMATE_IMAGE_PORTALS','entry_point':[.41,.5]}}
    assert local_reentry(a,b)['gap_seconds']==13
    b['zones']['entry_point']=[.7,.5];assert local_reentry(a,b) is None
    b['start_time']=40;assert local_reentry(a,b) is None


def cache_files(path):
    (path/'crops').mkdir();(path/'encoded').mkdir()
    cm={'normalization_signature':'signature','source_run_id':'source','entries':[{'track_id':'crop1'}]}
    atomic_json(path/'crops/crop_manifest.json',cm)
    np.savez(path/'encoded/embeddings.npz',track_ids=np.array(['crop1']),features=np.array([[1.,0.]]))
    em={'weights_sha256':'model','crop_manifest_sha256':sha256(path/'crops/crop_manifest.json'),
        'features_sha256':sha256(path/'encoded/embeddings.npz')}
    atomic_json(path/'encoded/embedding_manifest.json',em)
    status={'signature':'signature','model_hash':'model','source_run_id':'source','samples':1,
            'crop_manifest_sha256':em['crop_manifest_sha256'],'features_sha256':em['features_sha256']}
    atomic_json(path/'manifest.json',status)


@pytest.mark.parametrize('broken',['source','features','manifest','coverage','missing'])
def test_normalized_cache_validates_before_import_and_rejects_interruption(tmp_path,broken):
    cache_files(tmp_path)
    status,cm,vectors=validate_cache(tmp_path,'signature','model');assert np.allclose(vectors['crop1'],[1,0])
    if broken=='features':(tmp_path/'encoded/embeddings.npz').write_bytes(b'broken')
    elif broken=='manifest':(tmp_path/'crops/crop_manifest.json').write_text('{}')
    elif broken=='missing':(tmp_path/'manifest.json').unlink()
    else:
        status['source_run_id']='wrong' if broken=='source' else 'source'
        if broken=='coverage':status['samples']=2
        atomic_json(tmp_path/'manifest.json',status)
    with pytest.raises((ValueError,FileNotFoundError)):validate_cache(tmp_path,'signature','model')


def test_complete_encoding_resumes_after_interrupted_final_commit_without_inference(tmp_path,monkeypatch):
    from shadowtrack.mvp import normalized_appearance as module
    from shadowtrack.mvp.config import Settings
    from shadowtrack.mvp.db import connect,set_meta,meta
    settings=Settings(tmp_path,tmp_path/'app',tmp_path/'raw',tmp_path/'evidence',tmp_path/'sync.json')
    source=settings.artifacts/'mtmc/appearance/crops';source.mkdir(parents=True)
    atomic_json(source/'crop_manifest.json',{'source_run_id':'source','entries':[]})
    atomic_json(tmp_path/'configs/reid_model_lock.json',{'weights_sha256':'model','weights_filename':'never.pt'})
    signature=object_hash({'input':sha256(source/'crop_manifest.json'),'rois':module.NEUTRAL_FLOOR_ROIS,'version':1,'model':'model'})
    out=settings.artifacts/'mtmc/normalized_appearance'/signature[:16];out.mkdir(parents=True)
    cache_files(out);(out/'manifest.json').unlink()  # Inference completed; DB/marker commit was interrupted.
    cm=read_json(out/'crops/crop_manifest.json');cm['normalization_signature']=signature;cm['scene_references']={}
    cm['entries']=[{'track_id':'crop1','source_track_id':'parent','source_frame_index':1,'original_crop_sha256':'original',
                    'normalization':{'gain_rgb':[1,1,1]},'color':[1,0,1,0],'head_signature':[1,0]}]
    atomic_json(out/'crops/crop_manifest.json',cm)
    em=read_json(out/'encoded/embedding_manifest.json');em['crop_manifest_sha256']=sha256(out/'crops/crop_manifest.json');em['wall_seconds']=2.
    atomic_json(out/'encoded/embedding_manifest.json',em)
    with connect(settings.database) as db:set_meta(db,'source_run_id','source')
    def forbidden(*args,**kwargs):raise AssertionError('Resume must not decode sources or rerun OSNet')
    monkeypatch.setattr(module,'encode_crop_package',forbidden);monkeypatch.setattr(module,'scene_gain',forbidden)
    result=module.prepare(settings)
    assert result['resumed_completed_encoding'] and result['wall_seconds'] is None
    with connect(settings.database,readonly=True) as db:
        assert db.execute('SELECT COUNT(*) FROM appearance_normalized').fetchone()[0]==1
        assert meta(db,'normalized_appearance')['features_sha256']==result['features_sha256']
