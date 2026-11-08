"""Geometric local identity bundles before cross-camera appearance association.

Actual detections/observations remain immutable. Partial boxes contribute their
source evidence but do not force a torso descriptor to match a full-body one.
"""
import json
from itertools import combinations

import numpy as np

from .tracklets import partial_duplicate,weighted_descriptor
from .association_features import prototype_bank


def _head(track,side):
    rows=[r for r in track['observed_rows'] if r['confidence']>=.35]
    if len(rows)<6:return None
    subset=rows[:5] if side=='entry' else rows[-5:]
    boxes=np.asarray([json.loads(r['bbox']) for r in subset]);point=np.median(np.c_[(boxes[:,0]+boxes[:,2])/2,boxes[:,1]],axis=0)
    return point,float(np.median(boxes[:,2]-boxes[:,0]))


def _head_feature(track):
    samples=[s for s in track.get('appearance_samples',[]) if s.get('head_feature') is not None and s['quality']>=.15]
    if not samples:return None
    v=np.average(np.stack([s['head_feature'] for s in samples]),axis=0,weights=[s['quality'] for s in samples])
    return v/max(float(np.linalg.norm(v)),1e-9)


def consolidate(tracks):
    roots=list(range(len(tracks)));links=[];duplicates={};partial=set()
    def root(k):
        while roots[k]!=k:roots[k]=roots[roots[k]];k=roots[k]
        return k
    def can_union(i,j):
        left=[k for k in range(len(tracks)) if root(k)==root(i)];right=[k for k in range(len(tracks)) if root(k)==root(j)]
        for a in left:
            for b in right:
                x,y=tracks[a],tracks[b]
                if min(x['end_time'],y['end_time'])>max(x['start_time'],y['start_time'])+.001 and tuple(sorted((a,b))) not in duplicates:return False
        return True
    by_camera={}
    for i,t in enumerate(tracks):by_camera.setdefault(t['camera_id'],[]).append(i)
    for indices in by_camera.values():
        for i,j in combinations(indices,2):
            a,b=tracks[i],tracks[j]
            if min(a['end_time'],b['end_time'])<=max(a['start_time'],b['start_time']):continue
            sim=float(a['vector']@b['vector']) if a['vector'] is not None and b['vector'] is not None else -1.
            ev=partial_duplicate(a,b,sim)
            if not ev:continue
            duplicates[i,j]=ev
            areas=[]
            for t in (a,b):
                boxes=np.asarray([json.loads(o['bbox']) for o in t['observed_rows']]);areas.append(float(np.median(np.prod(boxes[:,2:]-boxes[:,:2],axis=1))))
            partial.add(i if areas[0]<areas[1] else j)
        for (i,j),ev in list(duplicates.items()):
            if root(i)!=root(j) and can_union(i,j):roots[root(j)]=root(i);links.append((i,j,ev))
    # Attach short occluded fragments only with same-view head appearance AND
    # a tight head-position continuation. An ambiguous alternative stays split.
    proposals=[]
    heads={i:_head_feature(t) for i,t in enumerate(tracks)}
    for indices in by_camera.values():
        for i in indices:
            a=tracks[i]
            if a['end_time']-a['start_time']>1.2 or len(a['observed_rows'])<6:continue
            # A barely visible head can share gradients and background with a
            # different passer-by. Require two usable genuine crops before a
            # short fragment is allowed to lend identity to a larger bundle.
            if sum(s['quality']>=.25 for s in a.get('appearance_samples',[]))<2:continue
            endpoint=_head(a,'exit')
            if endpoint is None or heads[i] is None:continue
            alternatives={}
            for j in indices:
                b=tracks[j];gap=b['start_time']-a['end_time']
                if root(i)==root(j) or not 0<=gap<=3 or heads[j] is None:continue
                entry=_head(b,'entry')
                if entry is None:continue
                radius=max(endpoint[1],entry[1])*2+5
                distance=float(np.linalg.norm(entry[0]-endpoint[0]));head=float(heads[i]@heads[j])
                if distance>radius or head<.70:continue
                sim=float(a['vector']@b['vector']) if a['vector'] is not None and b['vector'] is not None else -1.
                if sim<.45:continue
                score=.5*head+.3*(1-distance/radius)+.2*sim
                ev={'status':'OCCLUDED_LOCAL_CONTINUATION','gap_seconds':gap,'head_pixels':distance,'head_gate_pixels':radius,
                    'head_signature_similarity':head,'appearance_similarity':sim,'source':'Same-camera observed head position and image patch; no person labels'}
                key=root(j)
                if key not in alternatives or score>alternatives[key][0]:alternatives[key]=(score,j,ev)
            choices=sorted(alternatives.values(),key=lambda v:v[0],reverse=True)
            if choices and (len(choices)==1 or choices[0][0]-choices[1][0]>=.04):proposals.append((choices[0][0],i,*choices[0][1:]))
    for score,i,j,ev in sorted(proposals,reverse=True):
        if root(i)!=root(j) and can_union(i,j):roots[root(j)]=root(i);partial.add(i);links.append((i,j,ev))
    groups={}
    for i in range(len(tracks)):groups.setdefault(root(i),[]).append(i)
    for group in groups.values():
        rich=[i for i in group if i not in partial] or group
        samples={}
        for i in rich:
            for sample in tracks[i].get('appearance_samples',[]):
                key=sample['frame_index']
                if key not in samples or sample['quality']>samples[key]['quality']:samples[key]=sample
        pool=sorted(samples.values(),key=lambda s:s['raw_pts']);fallback=tracks[rich[0]]['vector']
        vector,coherence,independent,hist=weighted_descriptor(pool)
        if vector is None:vector=fallback
        bank=prototype_bank(pool,vector)
        bundle_links=[{'from_track':tracks[i]['id'],'to_track':tracks[j]['id'],**ev} for i,j,ev in links if i in group and j in group]
        geometry={}
        for i in sorted(rich,key=lambda k:tracks[k]['observations']):
            for row in tracks[i]['observed_rows']:geometry[row['frame_index']]=row
        for i in group:
            t=tracks[i];t['appearance_bank']=bank;t['local_bundle_id']=min(tracks[k]['id'] for k in group);t['local_bundle_size']=len(group)
            if len(group)>1:
                t['vector']=vector;t['coherence']=coherence;t['samples']=len(independent);t['color_histogram']=hist
                t['geometry_rows']=[geometry.get(o['frame_index'],o) for o in t['observed_rows']]
                t['bundle_evidence']={'members':[tracks[k]['id'] for k in group],'rich_sources':[tracks[k]['id'] for k in rich],
                    'links':bundle_links,'feature_samples':[s['provenance'] for s in pool]}
    return {'bundles':len(groups),'geometric_links':len(links),'partial_nodes':len(partial)}
