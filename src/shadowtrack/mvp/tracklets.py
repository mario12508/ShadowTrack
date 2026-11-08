"""Derived temporal slices and robust descriptors; immutable ByteTrack stays intact."""
import hashlib
import json

import numpy as np

from .db import rows


def unit(vector):
    return vector/max(float(np.linalg.norm(vector)),1e-12)


def weighted_descriptor(samples):
    if not samples:return None,0.,[],None
    good=[s for s in samples if s['quality']>=.30] or samples
    vectors=np.stack([s['feature'] for s in good]);weights=np.asarray([s['quality'] for s in good])
    medoid=int(np.argmax(np.median(vectors@vectors.T,axis=1)))
    mask=vectors@vectors[medoid]>=.60
    vectors=vectors[mask];weights=weights[mask];used=[s for s,keep in zip(good,mask) if keep]
    descriptor=unit(np.average(vectors,axis=0,weights=weights));coherence=float(np.quantile(vectors@descriptor,.1))
    hist=np.average(np.asarray([s['histogram'] for s in used]),axis=0,weights=weights)
    independent=[]
    for s in used:
        if not independent or s['raw_pts']-independent[-1]['raw_pts']>=.4:independent.append(s)
    return descriptor,coherence,independent,hist


def split_samples(samples,observations):
    """Detect a sustained change in both identity features and clothing.

    A view/pose change alone is not enough. Boundaries are estimates and retain
    the sampled uncertainty interval; the crop labels are never supplied by a
    reviewer or by the control-set manifest.
    """
    good=[s for s in samples if s['quality']>=.4]
    if len(good)<4:return []
    cuts=[]
    def visit(group):
        if len(group)<4:return
        best=None
        for k in range(2,len(group)-1):
            left,right=group[:k],group[k:]
            if min(left[-1]['raw_pts']-left[0]['raw_pts'],right[-1]['raw_pts']-right[0]['raw_pts'])<1.:continue
            a,ca,ia,ha=weighted_descriptor(left);b,cb,ib,hb=weighted_descriptor(right)
            if min(len(ia),len(ib))<2:continue
            cosine=float(a@b);color=float(np.minimum(ha,hb).sum()/2)
            if cosine>=.72 or color>=.65 or min(ca,cb)<.80:continue
            before,after=group[k-1],group[k]
            window=[o for o in observations if before['frame_index']<=o['frame_index']<=after['frame_index']]
            jumps=[]
            for previous,current in zip(window,window[1:]):
                x=np.asarray(json.loads(previous['bbox']));y=np.asarray(json.loads(current['bbox']))
                scale=max(1.,np.linalg.norm(x[2:]-x[:2]))
                jump=float(np.linalg.norm((x[:2]+x[2:]-y[:2]-y[2:])/2)/scale)
                jumps.append((jump,current['frame_index']))
            if not jumps:continue
            peak=max(jumps);baseline=float(np.median([j[0] for j in jumps]))
            # Smooth motion supports continuity even if pose and illumination
            # change both descriptors. Only a corroborated swap may split it.
            if peak[0]<.08 or peak[0]<max(.03,baseline*4):continue
            gain=(1-cosine)*min(len(left),len(right))/len(group)
            if best is None or gain>best[0]:best=(gain,k,cosine,color,peak,baseline)
        if best is None:return
        _,k,cosine,color,peak,baseline=best;before,after=group[k-1],group[k]
        cut=peak[1]
        cuts.append({'frame_index':cut,'uncertainty_frame_interval':[before['frame_index'],after['frame_index']],
                     'raw_pts_interval':[before['raw_pts'],after['raw_pts']],'cosine_across_change':cosine,'clothing_intersection':color,
                     'motion_jump_body_diagonals':peak[0],'motion_baseline':baseline,
                     'method':'sustained appearance/clothing change AND abrupt bbox trajectory discontinuity'})
        visit(group[:k]);visit(group[k:])
    visit(good)
    return sorted(cuts,key=lambda c:c['frame_index'])


def temporal_slices(db,parent,observations):
    samples=rows(db,'SELECT a.*,n.vector AS normalized_vector,n.color AS normalized_color,n.head_signature,n.evidence AS normalized_evidence FROM appearance_sample a LEFT JOIN appearance_normalized n ON n.parent_id=a.parent_id AND n.frame_index=a.frame_index WHERE a.parent_id=? ORDER BY a.frame_index',(parent['id'],))
    for s in samples:
        s['feature']=np.frombuffer(s['normalized_vector'] if s['normalized_vector'] is not None else s['vector'],dtype='<f4');s['histogram']=np.asarray(json.loads(s['normalized_color'] or s['color']));s['provenance']=json.loads(s['evidence'])
        s['head_feature']=np.frombuffer(s['head_signature'],dtype='<f4') if s['head_signature'] is not None else None
        if s['normalized_evidence']:s['provenance']['scene_balance']=json.loads(s['normalized_evidence'])
    changes=split_samples(samples,observations)
    stops=[parent['start_frame']]+[c['frame_index'] for c in changes]+[parent['end_frame']+1]
    result=[]
    for i,(start,stop) in enumerate(zip(stops,stops[1:])):
        obs=[o for o in observations if start<=o['frame_index']<stop]
        if not obs:continue
        subset=[s for s in samples if start<=s['frame_index']<stop]
        vector,coherence,independent,hist=weighted_descriptor(subset)
        key=parent['id'] if i==0 else hashlib.sha256(f"{parent['id']}/{obs[0]['frame_index']}".encode()).hexdigest()[:20]
        evidence={'parent_id':parent['id'],'part':i+1,'parts':len(stops)-1,'changes':changes,'sample_frames':[s['frame_index'] for s in subset],
                  'independent_sample_frames':[s['frame_index'] for s in independent],'appearance_source':'genuine cached OSNet temporal samples' if subset else 'original track descriptor fallback'}
        db.execute('INSERT INTO mtmc_tracklet VALUES(?,?,?,?,?,?,?,?)',(key,parent['id'],obs[0]['frame_index'],obs[-1]['frame_index'],obs[0]['media_time'],obs[-1]['media_time'],len(obs),json.dumps(evidence)))
        result.append({'id':key,'parent_id':parent['id'],'start_frame':obs[0]['frame_index'],'end_frame':obs[-1]['frame_index'],'start_time':obs[0]['media_time'],'end_time':obs[-1]['media_time'],
                       'observations':len(obs),'temporal_vector':vector,'temporal_coherence':coherence,'temporal_samples':independent,'appearance_samples':subset,'color_histogram':hist,
                       'observed_rows':obs,'slice_evidence':evidence})
    return result


def source_track(db,track_id):
    row=db.execute('SELECT * FROM association_track WHERE id=?',(track_id,)).fetchone()
    if row:return dict(row)
    row=db.execute('SELECT * FROM local_track WHERE id=?',(track_id,)).fetchone()
    return {**dict(row),'parent_id':row['id']} if row else None


def frame_tracklet(db,parent_id,frame_index):
    row=db.execute('SELECT id FROM mtmc_tracklet WHERE parent_id=? AND start_frame<=? AND end_frame>=?',(parent_id,frame_index,frame_index)).fetchone()
    return row[0] if row else parent_id


def partial_duplicate(a,b,similarity):
    """A nested upper-body/full-body detection of one moving image subject.

    Equal-sized overlapping boxes are never treated as duplicates. Require many
    common frames, a shared head trajectory and sustained near-containment.
    """
    if a['camera_id']!=b['camera_id']:return None
    aa={o['frame_index']:o for o in a.get('observed_rows',[])};bb={o['frame_index']:o for o in b.get('observed_rows',[])}
    common=sorted(aa.keys()&bb.keys())
    if len(common)<12:return None
    values=[]
    for frame in common:
        x=np.asarray(json.loads(aa[frame]['bbox']));y=np.asarray(json.loads(bb[frame]['bbox']))
        areas=[float(np.prod(x[2:]-x[:2])),float(np.prod(y[2:]-y[:2]))]
        intersection=max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
        values.append((intersection/max(1,min(areas)),abs(x[1]-y[1])/max(1,min(x[3]-x[1],y[3]-y[1])),
                       abs((x[0]+x[2]-y[0]-y[2])/2)/max(1,x[2]-x[0],y[2]-y[0]),max(areas)/max(1,min(areas))))
    v=np.asarray(values);qualifying=float(np.mean((v[:,0]>=.90)&(v[:,1]<.08)))
    if qualifying<.85 or np.median(v[:,2])>.10 or np.median(v[:,3])<1.3:return None
    return {'status':'PARTIAL_DUPLICATE','shared_frames':len(common),'nested_head_agreement_fraction':qualifying,
            'median_containment':float(np.median(v[:,0])),'median_area_ratio':float(np.median(v[:,3])),
            'appearance_similarity_diagnostic':float(similarity),'source':'saved detection bbox trajectories; sustained nested boxes and aligned heads, independent of crop coverage'}
