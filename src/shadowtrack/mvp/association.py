"""Conservative track-level association, with explicit unresolved candidates."""
import hashlib
import json
from collections import defaultdict

import numpy as np

from .db import connect,rows,set_meta,meta
from .episodes import lifecycle,may_continue


def compare_candidate(a,b,similarity,samples_a,samples_b,threshold=.88,timeout=120.):
    reasons=[]
    if samples_a<3 or samples_b<3:reasons.append('insufficient_independent_crops')
    if similarity<threshold:reasons.append('appearance_below_threshold')
    if a['global_start'] is None or b['global_start'] is None:reasons.append('unsupported_time')
    elif a['camera_id']==b['camera_id'] and max(a['global_start'],b['global_start'])<=min(a['global_end'],b['global_end']):reasons.append('simultaneous_same_camera_conflict')
    else:
        gap=max(a['global_start'],b['global_start'])-min(a['global_end'],b['global_end'])
        if gap>=timeout:reasons.append('episode_already_terminated')
        if gap>30:reasons.append('unsupported_transition_duration')
        if a['camera_id']!=b['camera_id'] and frozenset((a['camera_id'],b['camera_id'])) not in {frozenset(('1','2')),frozenset(('1','3')),frozenset(('2','3')),frozenset(('3','4')),frozenset(('4','5'))}:
            reasons.append('no_supported_floorplan_adjacency')
    return {'eligible':not reasons,'reasons':reasons,'similarity':float(similarity),'confidence_kind':'heuristic_support_not_identity_probability'}


def associate(settings):
    with connect(settings.database) as db:
        db.execute('DELETE FROM association');db.execute('DELETE FROM episode_member WHERE episode_id IN (SELECT id FROM episode WHERE scope="GLOBAL")')
        db.execute('DELETE FROM episode WHERE scope="GLOBAL"');db.execute('UPDATE local_track SET global_episode_id=NULL')
        evidence=rows(db,'SELECT * FROM embedding WHERE samples>=3 ORDER BY track_id')
        if not evidence:
            set_meta(db,'global_association_status','GLOBAL_ASSOCIATION_PENDING')
            return {'status':'GLOBAL_ASSOCIATION_PENDING','reason':'No genuine multi-crop embeddings available','global_episodes':0}
        models={r['model_hash'] for r in evidence}
        if len(models)!=1:raise ValueError('Embeddings from different model versions cannot be compared')
        tracks=[];vectors=[];samples=[]
        for e in evidence:
            t=dict(db.execute('SELECT * FROM local_track WHERE id=?',(e['track_id'],)).fetchone())
            support=db.execute('SELECT MIN(f.global_time),MAX(f.global_time),COUNT(f.global_time),COUNT(*) FROM observation o JOIN frame f ON f.camera_id=o.camera_id AND f.frame_index=o.frame_index WHERE o.track_id=?',(t['id'],)).fetchone()
            # A handful of aligned observations cannot calibrate an entire longer track.
            t['global_start']=support[0] if support[2]==support[3] else None;t['global_end']=support[1] if support[2]==support[3] else None
            tracks.append(t);vectors.append(np.frombuffer(e['vector'],dtype='<f4'));samples.append(e['samples'])
        matrix=np.vstack(vectors);scores=np.clip(matrix@matrix.T,-1,1);np.fill_diagonal(scores,-2)
        parent=list(range(len(tracks)))
        def find(i):
            while parent[i]!=i:i=parent[i]
            return i
        supported=0;unresolved=0
        for i,a in enumerate(tracks):
            if len(tracks)<2:break
            candidates=np.argsort(scores[i])[::-1];j=int(candidates[0]);b=tracks[j]
            if i>=j or float(scores[i,j])<.70:continue
            result=compare_candidate(a,b,float(scores[i,j]),samples[i],samples[j])
            reverse=np.argsort(scores[j])[::-1]
            margin=float(scores[i,j]-scores[i,int(candidates[1])]) if len(tracks)>2 else 1.
            reverse_margin=float(scores[j,i]-scores[j,int(reverse[1])]) if len(tracks)>2 else 1.
            if int(reverse[0])!=i or min(margin,reverse_margin)<.06:result['reasons'].append('ambiguous_appearance_neighbour')
            # Transitive components must also be free of same-camera overlap conflicts.
            if not result['reasons']:
                cluster=[k for k in range(len(tracks)) if find(k) in {find(i),find(j)}]
                for x in cluster:
                    for y in cluster:
                        if x>=y:continue
                        if tracks[x]['camera_id']==tracks[y]['camera_id'] and max(tracks[x]['global_start'],tracks[y]['global_start'])<=min(tracks[x]['global_end'],tracks[y]['global_end']):
                            result['reasons'].append('component_same_camera_conflict')
            status='SUPPORTED_CONDITIONAL' if not result['reasons'] else 'UNRESOLVED'
            key=hashlib.sha256((a['id']+b['id']).encode()).hexdigest()[:20]
            db.execute('INSERT INTO association VALUES(?,?,?,?,?,?,?,?)',(key,a['id'],b['id'],float(scores[i,j]),float(scores[i,j]),status,json.dumps(result['reasons']),json.dumps({'samples':[samples[i],samples[j]],'model_hash':next(iter(models)),'mutual_margin':[margin,reverse_margin],'time_support_complete':a['global_start'] is not None and b['global_start'] is not None,'topology_source':'qualitative floorplan adjacency; not metric geometry'})))
            if status=='SUPPORTED_CONDITIONAL':parent[find(j)]=find(i);supported+=1
            else:unresolved+=1
        groups=defaultdict(list)
        for i in range(len(tracks)):groups[find(i)].append(i)
        episodes=0
        for group in groups.values():
            if len(group)<2:continue
            ids=sorted(tracks[i]['id'] for i in group)
            key='G-'+hashlib.sha256(('|'.join(ids)+str(meta(db,'calibration_revision',0))).encode()).hexdigest()[:10].upper()
            start=min(tracks[i]['global_start'] for i in group);end=max(tracks[i]['global_end'] for i in group)
            cameras=sorted({tracks[i]['camera_id'] for i in group})
            # Global absence needs recorded, supported coverage of every source camera.
            timeout=meta(db,'episode_timeout_seconds',120)
            coverage=global_coverage(db,end,end+timeout)
            state=lifecycle(end,end+timeout if coverage else end+.3,timeout,coverage_complete=coverage)
            db.execute('INSERT INTO episode VALUES(?,?,?,?,?,?,?,?,?,?)',(key,'GLOBAL',1,state.status,start,end,state.terminated_at,state.reason,min(float(scores[i,j]) for i in group for j in group if i!=j),json.dumps(cameras)))
            for track_id in ids:
                db.execute('INSERT INTO episode_member VALUES(?,?)',(key,track_id));db.execute('UPDATE local_track SET global_episode_id=? WHERE id=?',(key,track_id))
            episodes+=1
        status='PARTIAL' if episodes else 'GLOBAL_ASSOCIATION_PENDING'
        set_meta(db,'global_association_status',status)
        reasons=defaultdict(int)
        for row in db.execute('SELECT reasons FROM association WHERE status="UNRESOLVED"'):
            for reason in json.loads(row['reasons']):reasons[reason]+=1
        set_meta(db,'association_summary',{'supported_links':supported,'unresolved_candidates':unresolved,'global_episodes':episodes,'unresolved_by_reason':dict(reasons)})
        return {'status':status,'global_episodes':episodes,'supported_links':supported,'unresolved_candidates':unresolved,'policy':'multi-crop appearance + mutual margin + complete supported time + topology + component conflicts; no identity labels'}


def global_coverage(db,start,end):
    cameras=[r[0] for r in db.execute('SELECT id FROM camera')]
    if len(cameras)!=5:return False
    for camera in cameras:
        intervals=db.execute('SELECT MIN(global_time),MAX(global_time) FROM frame WHERE camera_id=? AND global_time IS NOT NULL GROUP BY segment_id',(camera,))
        if not any(a<=start and b>=end for a,b in intervals):return False
    return True
