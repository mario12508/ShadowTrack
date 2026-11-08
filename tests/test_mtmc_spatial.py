"""Shared-view geometry must add evidence, not override identity conflicts."""
import json

import numpy as np

from shadowtrack.mvp.spatial import shared_motion,minimap_position
from shadowtrack.mvp.mtmc import candidate


def moving_track(camera,lag=0,reverse=False,shift=0):
    flip=camera=='2'
    quad=[[1,0],[0,0],[0,1],[1,1]] if flip else [[0,0],[1,0],[1,1],[0,1]]
    projection={'image_points':quad,'shared_points':[[0,0],[2,0],[2,1],[0,1]],
                'shared_plane':'synthetic_table','body_fraction':.55}
    observations=[]
    for i,t in enumerate(np.arange(0,3,.1)):
        x=.2+.1*(2.9-t if reverse else t)+shift
        if flip:x=1-x
        observations.append({'frame_index':i,'media_time':float(t+lag),'confidence':.9,
                             'bbox':json.dumps([x*1000-30,412,x*1000+30,572])})
    return {'id':camera,'camera_id':camera,'width':1000,'height':1000,'spatial_projection':projection,
            'observed_rows':observations,'aligned_start':lag,'aligned_end':lag+3,'time_uncertainty':3,
            'vector':np.array([1.,0.]),'samples':4,'coherence':.95,'time_status':'SOURCE_CLOCK',
            'endpoint_boxes':[[0,0,1,1]]*2,'color_histogram':np.ones(2),'crop_evidence':[]}


def test_shared_motion_requires_static_geometry_position_direction_and_duration():
    a,b=moving_track('1'),moving_track('2',lag=.8)
    proof=shared_motion(a,b)
    assert proof['paired_samples']>=16 and proof['median_distance_table_widths']<.001
    assert abs(proof['pair_lag_seconds']-.8)<.01
    assert shared_motion(a,moving_track('2',reverse=True)) is None
    assert shared_motion(a,moving_track('2',shift=.45)) is None
    b['observed_rows']=b['observed_rows'][:10]
    assert shared_motion(a,b) is None
    b=moving_track('2');b['spatial_projection']=None
    assert shared_motion(a,b) is None


def test_geometry_supports_back_front_match_but_not_weak_appearance_or_different_clothes():
    a,b=moving_track('1'),moving_track('2',lag=.8)
    c={'component_minimum_similarity':.72,'appearance_threshold':.82,'minimum_sample_coherence':.65,'clock_tolerance_seconds':3,
       'episode_timeout_seconds':120,'mode':'estimated_alignment','transition_status':'HYPOTHESIS',
       'topology_source':'synthetic scene','transitions':[{'cameras':['1','2'],'seconds':[0,30],'overlap_possible':True}]}
    assert candidate(a,b,.76,c)['status']=='ELIGIBLE'
    assert candidate(a,b,.76,c)['evidence']['shared_view_motion']
    assert candidate(a,b,.71,c)['status']=='UNRESOLVED'
    b['color_histogram']=np.zeros(2)
    assert candidate(a,b,.76,c)['status']=='UNRESOLVED'


def test_minimap_moves_with_observations_and_rejects_far_extrapolation():
    layout={'floorplan_sha256':'synthetic','walkable_regions':[[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]],
            'cameras':{'1':{'projection':{'image_points':[[0,0],[1,0],[1,1],[0,1]],
                'floor_points':[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'body_fraction':1,
                'uncertainty_radius':.04,'landmarks':'Synthetic floor corners','kind':'APPROXIMATE_FLOOR_PLANE'}}}}
    p,e=minimap_position(layout,'1',[100,100,300,500],1000,1000)
    q,_=minimap_position(layout,'1',[300,100,500,500],1000,1000)
    assert q[0]>p[0] and e['radius_normalized']>=.04
    assert minimap_position(layout,'1',[2000,100,2400,500],1000,1000)==(None,None)


def test_map_handoff_uses_junction_motion_and_rejects_opposite_or_unreachable_entry():
    from copy import deepcopy
    from shadowtrack.mvp.spatial import map_handoff
    layout={'walkable_regions':[[[.4,.1],[.5,.1],[.5,.9],[.4,.9]],[[.1,.4],[.5,.4],[.5,.5],[.1,.5]]]}
    a={'camera_id':'4','start_time':0,'spatial_projection':{'floor_points':[[.45,.42],[.45,.48],[.15,.48],[.15,.42]]},
       'floor_motion':{'exit':{'point':[.43,.45],'direction':[.1,0],'time':10}}}
    b={'camera_id':'5','start_time':11,'spatial_projection':{'floor_points':[[.43,.85],[.47,.85],[.47,.45],[.43,.45]]},
       'floor_motion':{'entry':{'point':[.45,.47],'direction':[0,.1],'time':11}}}
    proof=map_handoff(a,b,layout)
    assert proof['status']=='MAP_EXIT_ENTRY_HANDOFF' and min(proof['direction_agreements'])>.99
    opposite=deepcopy(b);opposite['floor_motion']['entry']['direction']=[0,-.1]
    assert map_handoff(a,opposite,layout) is None
    remote=deepcopy(b);remote['floor_motion']['entry']['point']=[.45,.85]
    assert map_handoff(a,remote,layout) is None


def test_walkable_route_cannot_cross_a_wall_or_disconnected_area():
    from shadowtrack.mvp.spatial import walkable_distance
    layout={'walkable_regions':[[[0,0],[.3,0],[.3,.3],[0,.3]],[[.7,.7],[1,.7],[1,1],[.7,1]]]}
    assert walkable_distance([.1,.1],[.2,.2],layout)>0
    assert walkable_distance([.1,.1],[.8,.8],layout) is None
