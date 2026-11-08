"""Qualitative portals and motion from actual image trajectories, not homographies."""
import json

import numpy as np


def endpoint_zones(track):
    obs=track['observed_rows'];width,height=track['width'],track['height']
    points=np.asarray([[(b[0]+b[2])/2/width,b[3]/height] for b in (json.loads(o['bbox']) for o in obs)])
    def label(point):
        x,y=point;cid=track['camera_id']
        if not track.get('scene_ports_supported',False):return 'uncalibrated_image_sector'
        if cid=='4':
            if x<.30 and y>.52 or y>.86:return 'main_hall_portal'
            if .25<x<.60 and y<.62:return 'rear_room_or_occlusion'
        if cid=='5':
            if y<.38:return 'north_corridor'
            if x<.30 and y>.5 or y>.86:return 'south_corridor'
            if x>.7:return 'room_door_or_occlusion'
        if cid=='3':
            if x<.48 and y<.73:return 'corridor_door'
            if x<.5:return 'foyer'
        if cid in ('1','2'):return 'foyer_or_stair_zone'
        return 'interior_or_occlusion'
    n=min(6,len(points));first=np.median(points[:n],axis=0);last=np.median(points[-n:],axis=0)
    return {'entry':label(first),'exit':label(last),'entry_point':first.tolist(),'exit_point':last.tolist(),
            'entry_direction':(points[min(len(points)-1,max(1,n))]-first).tolist(),
            'exit_direction':(last-points[max(0,len(points)-n-1)]).tolist(),
            'source':'qualitative scene review + normalized saved bbox footpoints; camera plan adjacency',
            'status':'APPROXIMATE_IMAGE_PORTALS','unknown_means':'occlusion or missing detection is possible, not a demonstrated physical exit'}


def transition_evidence(a,b,gap):
    if a['aligned_start']>b['aligned_start']:a,b=b,a
    za,zb=a.get('zones'),b.get('zones')
    result={'from_zone':za['exit'] if za else None,'to_zone':zb['entry'] if zb else None,'status':'UNAVAILABLE','score_adjustment':0.}
    if not za or not zb:return result
    if a['camera_id']==b['camera_id']:
        distance=float(np.linalg.norm(np.asarray(za['exit_point'])-zb['entry_point']))
        result.update(status='IMAGE_MOTION_CONTINUITY',footpoint_distance=distance,
                      score_adjustment=.04 if 0<=gap<=3 and distance<.22 else 0.)
        return result
    routes={('4','5'):('main_hall_portal','north_corridor'),('5','4'):('north_corridor','main_hall_portal'),
            ('3','4'):('corridor_door','main_hall_portal'),('4','3'):('main_hall_portal','corridor_door'),
            ('3','5'):('corridor_door','north_corridor'),('5','3'):('north_corridor','corridor_door')}
    expected=routes.get((a['camera_id'],b['camera_id']))
    if expected:
        matched=int(result['from_zone']==expected[0])+int(result['to_zone']==expected[1])
        result.update(status='QUALITATIVE_PORTAL_PRIOR',expected=list(expected),matched_portals=matched,score_adjustment=.025*matched)
    return result


def approximate_floor_zone(layout,camera,point):
    """Return a broad reviewed floor sector, never a metric point estimate."""
    camera_layout=layout.get('cameras',{}).get(camera,{})
    regions=camera_layout.get('regions',[])
    x,y=point
    for region in regions:
        a,b,c,d=region['image_bounds']
        if a<=x<=c and b<=y<=d:
            left,top,right,bottom=region['floor_bounds']
            return [(left+right)/2,(top+bottom)/2],{
                'status':'ZONE_ESTIMATE','floor_bounds':region['floor_bounds'],'zone':region['name'],
                'source':layout['source'],'accuracy':'Whole shaded region; centre is a display anchor, not an exact person coordinate',
                'floorplan_sha256':layout['floorplan_sha256']}
    return None,None
