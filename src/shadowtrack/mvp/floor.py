"""Manual image/floor correspondences; no invented person coordinates."""
import json
import time

import cv2
import numpy as np

from .db import rows,meta
from .mtmc import identity


def fit_mapping(points):
    if not 4<=len(points)<=30:raise ValueError('Use 4–30 independent correspondences')
    array=np.asarray([[*p['image'],*p['floor']] for p in points],dtype=float)
    if array.shape!=(len(points),4) or not np.isfinite(array).all() or np.any((array<0)|(array>1)):
        raise ValueError('Correspondences use finite normalized [0,1] image/floor coordinates')
    a,b=array[:,:2],array[:,2:]
    if len(np.unique(a,axis=0))!=len(a) or len(np.unique(b,axis=0))!=len(b):raise ValueError('Duplicate correspondence')
    for values in (a,b):
        if cv2.contourArea(cv2.convexHull(values.astype('float32'))) < .005:raise ValueError('Correspondences are collinear or cover too little area')
    h,mask=cv2.findHomography(a,b,cv2.RANSAC,.025)
    if h is None or mask is None or not mask.all() or not np.isfinite(h).all():raise ValueError('Inconsistent correspondences; review all points')
    projected=cv2.perspectiveTransform(a.reshape(-1,1,2),h).reshape(-1,2)
    residual=float(np.max(np.linalg.norm(projected-b,axis=1)))
    if residual>.025:raise ValueError('Reprojection residual exceeds 2.5% of floor image')
    return h.tolist(),residual


def save_mapping(db,camera_id,points,checked,evidence):
    if db.execute('SELECT 1 FROM camera WHERE id=?',(camera_id,)).fetchone() is None:raise ValueError('Unknown camera')
    if checked and len(evidence.strip())<8:raise ValueError('Checked geometry requires a source description')
    matrix,residual=fit_mapping(points)
    status='MANUAL_APPROXIMATE' if checked else 'UNCHECKED'
    db.execute('INSERT OR REPLACE INTO floor_mapping VALUES(?,?,?,?,?,?,?)',(camera_id,json.dumps(points),json.dumps(matrix),residual,status,evidence,str(time.time())))
    return {'camera_id':camera_id,'status':status,'maximum_reprojection_error_normalized':residual,
            'accuracy':'Approximate planar model. Lens distortion, foot occlusion and room scale remain uncalibrated.'}


def project_footpoint(mapping,bbox,width,height):
    if not mapping or mapping['status']!='MANUAL_APPROXIMATE':return None,'UNCALIBRATED'
    x1,y1,x2,y2=bbox;point=np.asarray([[(x1+x2)/2/width,y2/height]],dtype='float32')
    if y2>=height*.995:return None,'FEET_CLIPPED'
    points=json.loads(mapping['points']);h=np.asarray(json.loads(mapping['matrix']))
    hull=cv2.convexHull(np.asarray([p['image'] for p in points],dtype='float32'))
    if cv2.pointPolygonTest(hull,tuple(float(v) for v in point[0]),False)<0:return None,'OUTSIDE_CALIBRATED_REGION'
    q=cv2.perspectiveTransform(point.reshape(1,1,2),h)[0,0]
    if not np.isfinite(q).all() or np.any((q<0)|(q>1)):return None,'OUTSIDE_FLOORPLAN'
    return [float(v) for v in q],'APPROXIMATE_PLANAR'


def floor_state(db,playhead):
    markers=[];unlocated=[];frames=[]
    layout=meta(db,'scene_layout',{})
    tolerance=meta(db,'mtmc_config',{}).get('clock_tolerance_seconds',3.)
    for camera in rows(db,'SELECT * FROM camera ORDER BY id'):
        cid=camera['id']
        frame=db.execute('SELECT * FROM frame WHERE camera_id=? AND media_time BETWEEN ? AND ? ORDER BY ABS(media_time-?) LIMIT 1',(cid,playhead-.10,playhead+.10,playhead)).fetchone()
        if not frame:
            frames.append({'camera_id':cid,'available':False,'reason':'no_source_frame'});continue
        frames.append({'camera_id':cid,'available':True,'media_time':frame['media_time'],'raw_media_pts':frame['raw_pts'],'source_frame_index':frame['frame_index'],
                       'time_status':'ORGANIZER_SOURCE_CLOCK','uncertainty':tolerance,'segment_id':frame['segment_id']})
        mapping=db.execute('SELECT * FROM floor_mapping WHERE camera_id=?',(cid,)).fetchone()
        for obs in rows(db,'SELECT o.*,t.display_id FROM observation o JOIN local_track t ON t.id=o.track_id WHERE o.camera_id=? AND o.frame_index=?',(cid,frame['frame_index'])):
            if obs['confidence']<.35:continue
            point,status=project_footpoint(mapping,json.loads(obs['bbox']),camera['width'],camera['height'])
            zone=None;bbox=json.loads(obs['bbox'])
            if point is None and layout.get('cameras',{}).get(cid,{}).get('source_sha256')==camera['source_sha256']:
                from .spatial import minimap_position
                point,zone=minimap_position(layout,cid,bbox,camera['width'],camera['height'])
                if point is not None:status='SCHEMATIC_PROJECTED'
            ident=identity(db,obs['track_id'],obs['frame_index']);item={**ident,'track_id':ident['derived_tracklet_id'],'source_track_id':obs['track_id'],'camera_id':cid,'local_label':obs['display_id'],'position':point,'localization_status':status,
                'confidence':obs['confidence'],'raw_media_pts':obs['raw_pts'],'source_frame_index':obs['frame_index'],'media_time':obs['media_time'],
                'footpoint_image':[(json.loads(obs['bbox'])[0]+json.loads(obs['bbox'])[2])/2,json.loads(obs['bbox'])[3]],
                'location_uncertainty':zone or 'Planar approximation; lens distortion and foot occlusion unmodelled','floor_zone':zone,'time_uncertainty':tolerance}
            (markers if point else unlocated).append(item)
    # One dot per global ID. Pick the least uncertain observed projection;
    # averaging approximate sources can put a person inside a wall.
    grouped={}
    for m in markers:
        key=m['global_episode_id'] or m['track_id'];previous=grouped.get(key)
        if previous is None:
            grouped[key]=m;m['camera_sources']=[m['camera_id']]
        else:
            sources=previous['camera_sources']
            if m['camera_id'] not in sources:sources.append(m['camera_id'])
            old_radius=(previous.get('floor_zone') or {}).get('radius_normalized',.02)
            radius=(m.get('floor_zone') or {}).get('radius_normalized',.02)
            if radius<old_radius-.01:
                m['camera_sources']=sources;grouped[key]=m
    return {'playhead':playhead,'time_basis':'source_recordings','stream':'ARCHIVE_REPLAY','markers':list(grouped.values()),'unlocated':unlocated,'cameras':frames,
            'policy':'One moving dot per global ID; static-landmark schematic projection, approximate position. Video playback uses its native source clock.'}
