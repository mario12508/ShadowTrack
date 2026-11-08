"""Static-landmark projection and conservative shared-view motion evidence.

Schematic floor coordinates are for the minimap. Shared-view matching uses a
separate plane defined by the same four stationary table corners in both views;
it never uses global IDs or user-labelled person correspondences.
"""
import json
from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=32)
def _matrix(source, target):
    return cv2.getPerspectiveTransform(np.asarray(source, dtype='float32'), np.asarray(target, dtype='float32'))


def project(points, projection, shared=False):
    source=tuple(map(tuple,projection['image_points']))
    target=tuple(map(tuple,projection['shared_points' if shared else 'floor_points']))
    array=np.asarray(points,dtype='float32').reshape(-1,1,2)
    return cv2.perspectiveTransform(array,_matrix(source,target)).reshape(-1,2)


def body_points(boxes,width,height,fraction):
    boxes=np.asarray(boxes,dtype=float)
    return np.c_[(boxes[:,0]+boxes[:,2])/2/width,(boxes[:,1]+fraction*(boxes[:,3]-boxes[:,1]))/height]


def minimap_position(layout,camera,bbox,width,height):
    """A moving approximate point with explicit error, never surveyed position."""
    item=layout.get('cameras',{}).get(camera,{})
    projection=item.get('projection')
    if not projection:return None,None
    if camera=='3' and bbox[3]/height>.65 and item.get('secondary_projection'):
        projection=item['secondary_projection']
    p=body_points([bbox],width,height,projection['body_fraction'])
    q=project(p,projection)[0]
    if not np.isfinite(q).all() or np.any((q<-.1)|(q>1.1)):return None,None
    # Reject distant extrapolation. Small schematic errors are snapped to the
    # nearest reviewed walkable polygon, never averaged through an office wall.
    best=None
    for polygon in layout.get('walkable_regions',[]):
        poly=np.asarray(polygon,dtype='float32')
        if cv2.pointPolygonTest(poly,tuple(map(float,q)),False)>=0:
            best=(0.,q);break
        for a,b in zip(poly,np.roll(poly,-1,axis=0)):
            point=a+np.clip(np.dot(q-a,b-a)/max(np.dot(b-a,b-a),1e-9),0,1)*(b-a)
            distance=float(np.linalg.norm(q-point))
            if best is None or distance<best[0]:best=(distance,point)
    if best is None or best[0]>projection.get('maximum_extrapolation',.08):return None,None
    return best[1].astype(float).tolist(),{'status':'SCHEMATIC_PROJECTED','kind':projection['kind'],
        'radius_normalized':projection['uncertainty_radius']+best[0],
        'source':projection['landmarks'],'floorplan_sha256':layout['floorplan_sha256'],
        'accuracy':'Approximate minimap point from reviewed static landmarks; lens distortion, body height and schematic scale remain uncertain'}


def shared_motion(a,b,tolerance=3.):
    """Cross-view torso trajectories near a known common tabletop plane.

    The fitted lag is pairwise association tolerance, never a playback offset.
    No free translation/rotation is fitted to people. Appearance, uniqueness and
    whole-component conflicts are checked separately by the association engine.
    """
    pa,pb=a.get('spatial_projection'),b.get('spatial_projection')
    if not pa or not pb or a['camera_id']==b['camera_id'] or not pa.get('shared_plane') or pa.get('shared_plane')!=pb.get('shared_plane'):
        return None
    series=[]
    for track,projection in ((a,pa),(b,pb)):
        observations=[o for o in track.get('geometry_rows',track.get('observed_rows',[])) if o['confidence']>=.45]
        if len(observations)<16:return None
        times=np.asarray([o['media_time'] for o in observations])
        boxes=[json.loads(o['bbox']) for o in observations]
        points=project(body_points(boxes,track['width'],track['height'],projection['body_fraction']),projection,True)
        valid=np.isfinite(points).all(axis=1)&(np.abs(points).max(axis=1)<5)
        if valid.sum()<16:return None
        series.append((times[valid],points[valid]))
    (ta,xa),(tb,xb)=series
    minimum_span=max(1.2,min(ta[-1]-ta[0],tb[-1]-tb[0])*.65)
    best=None
    for lag in np.arange(-tolerance,tolerance+.001,.2):
        lo,hi=max(ta[0],tb[0]-lag),min(ta[-1],tb[-1]-lag)
        if hi-lo<minimum_span:continue
        grid=np.arange(lo,hi,.1)
        u=np.stack([np.interp(grid,ta,xa[:,i]) for i in (0,1)],axis=1)
        v=np.stack([np.interp(grid+lag,tb,xb[:,i]) for i in (0,1)],axis=1)
        distances=np.linalg.norm(u-v,axis=1)
        du=np.median(u[-3:],axis=0)-np.median(u[:3],axis=0)
        dv=np.median(v[-3:],axis=0)-np.median(v[:3],axis=0)
        lengths=[float(np.linalg.norm(p)) for p in (du,dv)]
        if min(lengths)<.25:continue
        direction=float(du@dv/max(lengths[0]*lengths[1],1e-9))
        median,p90=float(np.median(distances)),float(np.percentile(distances,90))
        ratio=lengths[0]/lengths[1]
        if median>.22 or p90>.38 or direction<.85 or not .45<ratio<2.2:continue
        score=median+.25*p90
        evidence={'status':'SHARED_VIEW_MOTION','plane':pa['shared_plane'],'median_distance_table_widths':median,
            'p90_distance_table_widths':p90,'direction_cosine':direction,'displacement_ratio':ratio,
            'paired_samples':len(grid),'support_seconds':float(hi-lo),'pair_lag_seconds':float(round(lag,3)),
            'source':'Static corresponding table corners + saved bbox torso trajectories; no free person-to-person spatial transform',
            'clock_effect':'None: lag is matching tolerance only, never applied to video or raw PTS'}
        if best is None or score<best[0]:best=(score,evidence)
    return best[1] if best else None


def motion_endpoints(track,layout):
    """Robust entry/exit motion from observed frames, not interpolated gaps."""
    projection=track.get('spatial_projection')
    if not projection or projection.get('kind')!='APPROXIMATE_FLOOR_PLANE':return None
    observations=[o for o in track.get('geometry_rows',track['observed_rows']) if o['confidence']>=.45]
    if len(observations)<8:return None
    result={}
    for name,window in [('entry',[o for o in observations if o['media_time']<=observations[0]['media_time']+2]),
                        ('exit',[o for o in observations if o['media_time']>=observations[-1]['media_time']-2])]:
        # Bound CPU cost independently of source FPS. Only genuine samples enter.
        window=[window[i] for i in np.unique(np.linspace(0,len(window)-1,min(12,len(window))).astype(int))]
        points=[];times=[]
        for o in window:
            box=json.loads(o['bbox'])
            if box[3]>track['height']*.99:continue
            point,evidence=minimap_position(layout,track['camera_id'],box,track['width'],track['height'])
            if point is not None:points.append(point);times.append(o['media_time'])
        if len(points)<4 or times[-1]-times[0]<.4:return None
        p=np.asarray(points);first=np.median(p[:3],axis=0);last=np.median(p[-3:],axis=0)
        result[name]={'point':(first if name=='entry' else last).tolist(),
                      'direction':(last-first).tolist(),'time':times[0] if name=='entry' else times[-1],
                      'observed_seconds':times[-1]-times[0],'samples':len(p)}
    return result


def _inside(point,polygons):
    return any(cv2.pointPolygonTest(p,tuple(map(float,point)),False)>=-1e-7 for p in polygons)


def _visible(a,b,polygons):
    return all(_inside(point,polygons) for point in np.linspace(a,b,25))


@lru_cache(maxsize=8)
def _walk_graph(serialized):
    """Visibility graph of the reviewed walkable areas, not identity routes."""
    polygons=[np.asarray(p,dtype='float32') for p in json.loads(serialized)]
    nodes=[p.mean(axis=0) for p in polygons];junctions=[]
    for i,a in enumerate(polygons):
        for b in polygons[i+1:]:
            area,intersection=cv2.intersectConvexConvex(a,b)
            if area>1e-7:
                point=intersection.reshape(-1,2).mean(axis=0);nodes.append(point);junctions.append(point)
    return polygons,nodes,junctions


def walkable_distance(start,end,layout):
    """Shortest visible polyline through connected walkable polygons."""
    import heapq
    polygons,nodes,_=_walk_graph(json.dumps(layout.get('walkable_regions',[]),sort_keys=True))
    if not polygons or not _inside(start,polygons) or not _inside(end,polygons):return None
    if _visible(start,end,polygons):return float(np.linalg.norm(np.asarray(end)-start))
    points=[np.asarray(start),np.asarray(end),*nodes];dist={0:0.};queue=[(0.,0)]
    while queue:
        cost,i=heapq.heappop(queue)
        if cost>dist[i]:continue
        if i==1:return cost
        for j,p in enumerate(points):
            if j==i or not _visible(points[i],p,polygons):continue
            value=cost+float(np.linalg.norm(p-points[i]))
            if value<dist.get(j,float('inf')):dist[j]=value;heapq.heappush(queue,(value,j))
    return None


def map_handoff(a,b,layout):
    """Independent spatial support for a short exit/entry handoff.

    Camera rays and shared junctions derive from the static floor geometry.
    Similar clothing/OSNet, ambiguity and component conflicts remain mandatory.
    """
    if a['camera_id']==b['camera_id'] or not a.get('floor_motion') or not b.get('floor_motion'):return None
    if a['start_time']>b['start_time']:a,b=b,a
    exit_,entry=a['floor_motion']['exit'],b['floor_motion']['entry']
    gap=entry['time']-exit_['time']
    if not -3<=gap<=12:return None
    distance=walkable_distance(exit_['point'],entry['point'],layout)
    if distance is None or distance>.08+.025*max(0,gap):return None
    _,_,junctions=_walk_graph(json.dumps(layout.get('walkable_regions',[]),sort_keys=True))
    if not junctions:return None
    ends=[]
    for t in (a,b):
        p=np.asarray(t['spatial_projection']['floor_points']);ends.append(np.stack([p[:2].mean(axis=0),p[2:].mean(axis=0)]))
    junction=min(junctions,key=lambda j:sum(float(np.min(np.linalg.norm(e-j,axis=1))) for e in ends))
    agreements=[]
    for index,(t,endpoint,field) in enumerate(zip((a,b),(exit_,entry),ends)):
        toward_camera=field[0]-field[1]
        near=np.linalg.norm(field[0]-junction)<np.linalg.norm(field[1]-junction)
        expected=toward_camera*(1 if near else -1)*(1 if index==0 else -1)
        observed=np.asarray(endpoint['direction']);length=float(np.linalg.norm(observed))
        if length<.012:return None
        agreement=float(observed@expected/max(length*np.linalg.norm(expected),1e-9))
        if agreement<.55:return None
        agreements.append(agreement)
    return {'status':'MAP_EXIT_ENTRY_HANDOFF','from_camera':a['camera_id'],'to_camera':b['camera_id'],
        'gap_seconds':gap,'walkable_path_length_normalized':distance,'junction':junction.astype(float).tolist(),
        'direction_agreements':agreements,'exit':exit_,'entry':entry,
        'source':'Observed endpoint motion + camera field orientation + walkable visibility graph; no person IDs',
        'accuracy':'Schematic spatial gate, not measured metres or speed; never applied to video playback'}


def local_reentry(a,b):
    """Same-camera return through the same occlusion/exit, without clock fitting."""
    if a['camera_id']!=b['camera_id']:return None
    if a['start_time']>b['start_time']:a,b=b,a
    gap=b['start_time']-a['end_time']
    if not 0<=gap<=30:return None
    za,zb=a.get('zones'),b.get('zones')
    if not za or not zb or za['status']!='APPROXIMATE_IMAGE_PORTALS':return None
    distance=float(np.linalg.norm(np.asarray(za['exit_point'])-zb['entry_point']))
    if distance>.055:return None
    return {'status':'SAME_VIEW_REENTRY','gap_seconds':gap,'image_endpoint_distance':distance,
        'source':'Same camera, observed exit/entry image positions; no manual identity assertion'}


def opposing_corridor_motion(a,b,tolerance=3.):
    """Inverse full-body heights in opposing views obey a constant-depth sum.

    Fit only lag/focal-scale nuisance terms. This independent motion cue is not
    a metric position fix and never changes video playback.
    """
    if a['camera_id']==b['camera_id']:return None
    projections=[t.get('spatial_projection') for t in (a,b)]
    if any(not p or p.get('kind')!='APPROXIMATE_FLOOR_PLANE' for p in projections):return None
    axes=[]
    for p in projections:
        q=np.asarray(p['floor_points']);v=q[:2].mean(axis=0)-q[2:].mean(axis=0);axes.append(v/max(np.linalg.norm(v),1e-9))
    if float(axes[0]@axes[1])>-.9:return None
    series=[]
    for t,p in zip((a,b),projections):
        if '_opposing_depth_series' in t:
            cached=t['_opposing_depth_series']
            if cached is None:return None
            series.append(cached);continue
        hull=cv2.convexHull(np.asarray(p['image_points'],dtype='float32'));times=[];heights=[]
        for o in t.get('geometry_rows',t['observed_rows']):
            if o['confidence']<.45:continue
            x1,y1,x2,y2=json.loads(o['bbox']);h=y2-y1
            if h<70 or y2>t['height']*.97 or h/max(x2-x1,1)<2.1:continue
            point=((x1+x2)/2/t['width'],y2/t['height'])
            if cv2.pointPolygonTest(hull,point,True)<-.035:continue
            times.append(o['media_time']);heights.append(t['height']/h)
        if len(times)<16:t['_opposing_depth_series']=None;return None
        values=np.asarray(heights);smooth=np.asarray([np.median(values[max(0,i-2):i+3]) for i in range(len(values))])
        t['_opposing_depth_series']=(np.asarray(times),smooth)
        series.append(t['_opposing_depth_series'])
    (ta,xa),(tb,xb)=series;best=None
    for lag in np.arange(-tolerance,tolerance+.001,.2):
        start,end=max(ta[0],tb[0]-lag),min(ta[-1],tb[-1]-lag)
        if end-start<1.6:continue
        grid=np.linspace(start,end,min(80,max(16,round((end-start)*10))))
        def nearest(times,values):
            indexes=np.searchsorted(times,values)
            return np.minimum(abs(times[np.clip(indexes,0,len(times)-1)]-values),abs(times[np.clip(indexes-1,0,len(times)-1)]-values))
        valid=(nearest(ta,grid)<.11)&(nearest(tb,grid+lag)<.11)
        if valid.sum()<16:continue
        grid=grid[valid];x=np.interp(grid,ta,xa);y=np.interp(grid+lag,tb,xb)
        if min(np.ptp(x)/max(np.median(x),1e-9),np.ptp(y)/max(np.median(y),1e-9))<.10:continue
        corr=float(np.corrcoef(x,y)[0,1])
        if not np.isfinite(corr) or corr>-.86:continue
        scale=float(-np.cov(x,y,ddof=0)[0,1]/max(np.var(y),1e-9))
        if not .2<=scale<=5:continue
        residual=x+scale*y;relative=float(np.percentile(abs(residual-np.median(residual)),90)/max(np.ptp(x),1e-9))
        if relative>.25:continue
        score=-corr-relative*.3
        evidence={'status':'OPPOSING_CORRIDOR_DEPTH_MOTION','paired_samples':len(grid),'support_seconds':float(grid[-1]-grid[0]),
            'inverse_height_correlation':corr,'positive_focal_scale':scale,'relative_p90_residual':relative,'pair_lag_seconds':float(round(lag,3)),
            'source':'Source-bound opposing camera fields + observed full-body size trajectories; partial boxes excluded',
            'clock_effect':'Association tolerance only; does not change playback or raw PTS'}
        if best is None or score>best[0]:best=(score,evidence)
    return best[1] if best else None
