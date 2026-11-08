"""Reproducible experimental MTMC over immutable observations and genuine OSNet.

Estimated clocks and machine links are PROVISIONAL. A maximum-weight path cover
competes for observations globally, then whole components undergo conflict checks.
No identity probability or validated clock calibration is inferred from appearance.
"""
import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from ..detection.common import atomic_json, object_hash, read_json
from ..detection.progress import observed
from .db import connect, meta, rows, set_meta
from .episodes import lifecycle
from .render import color_for


def pair_key(a,b):
    return hashlib.sha256('|'.join(sorted((a,b))).encode()).hexdigest()[:24]


def load_config(settings):
    c=read_json(settings.root/'configs/mtmc.json')
    layout=settings.root/'configs/scene_layout.json'
    if layout.exists():c['scene_layout']=read_json(layout)
    manual=settings.artifacts/'mtmc/manual_offsets.json'
    if manual.exists():c['manual_offsets']=read_json(manual)
    if c['mode'] not in ('estimated_alignment','supported_alignment'):raise ValueError('Unknown MTMC clock mode')
    if not 0<=c['clock_tolerance_seconds']<=30:raise ValueError('Invalid clock tolerance')
    if c['episode_timeout_seconds']!=120:raise ValueError('Global episode timeout must be 120 seconds')
    for key in ('appearance_threshold','ambiguity_margin','minimum_sample_coherence','component_minimum_similarity'):
        if not 0<=c[key]<=1:raise ValueError('Invalid '+key)
    for m in c['manual_offsets']:
        if not all(math.isfinite(m[k]) for k in ('offset','uncertainty')) or m['uncertainty']<.001 or len(m.get('evidence',''))<8:
            raise ValueError('Manual estimates require finite offsets, uncertainty and evidence')
    return c


def load_tracks(db):
    from .tracklets import temporal_slices
    parents=rows(db,'SELECT t.*,c.first_pts,c.width,c.height,c.source_sha256,e.vector,e.samples,e.evidence AS embedding_evidence,e.model_hash FROM local_track t JOIN camera c ON c.id=t.camera_id LEFT JOIN embedding e ON e.track_id=t.id ORDER BY t.start_time,t.id')
    models={t['model_hash'] for t in parents if t['model_hash']}
    if len(models)>1:raise ValueError('Mixed OSNet models cannot be associated')
    db.execute('DELETE FROM mtmc_tracklet');ts=[]
    for parent in parents:
        original=np.frombuffer(parent['vector'],dtype='<f4') if parent['vector'] else None
        ev=json.loads(parent['embedding_evidence']) if parent['embedding_evidence'] else {}
        observed=rows(db,'SELECT * FROM observation WHERE track_id=? ORDER BY frame_index',(parent['id'],))
        for part in temporal_slices(db,parent,observed):
            t={**parent,**part};t['vector']=part['temporal_vector'] if part['temporal_vector'] is not None else original
            if part['temporal_vector'] is not None:
                t['samples']=len(part['temporal_samples']);t['coherence']=part['temporal_coherence']
                t['crop_evidence']=[s['provenance'] for s in part['temporal_samples']]
                t['model_hash']=meta(db,'temporal_appearance',{}).get('model_hash',parent['model_hash'])
            else:t['coherence']=ev.get('sample_coherence',0);t['crop_evidence']=ev.get('samples',[])
            t['raw_start']=t['start_time']+t['first_pts'];t['raw_end']=t['end_time']+t['first_pts']
            params=(parent['id'],t['start_frame'],t['end_frame'])
            support=db.execute('SELECT MIN(f.global_time),MAX(f.global_time),COUNT(f.global_time),COUNT(*) FROM observation o JOIN frame f ON f.camera_id=o.camera_id AND f.frame_index=o.frame_index WHERE o.track_id=? AND o.frame_index BETWEEN ? AND ?',params).fetchone()
            t['supported_start']=support[0] if support[2]==support[3] and support[3] else None
            t['supported_end']=support[1] if t['supported_start'] is not None else None
            radii=[]
            if t['supported_start'] is not None:
                for r in db.execute('SELECT f.global_uncertainty FROM observation o JOIN frame f ON f.camera_id=o.camera_id AND f.frame_index=o.frame_index WHERE o.track_id=? AND o.frame_index BETWEEN ? AND ?',params):
                    u=json.loads(r[0]) if r[0] else None
                    radius=u.get('radius_seconds',u.get('radius')) if isinstance(u,dict) else None
                    if radius is None and isinstance(u,dict) and u.get('upper') is not None and u.get('lower') is not None:radius=(u['upper']-u['lower'])/2
                    if radius is None or not math.isfinite(radius) or radius<0:radii=[];t['supported_start']=t['supported_end']=None;break
                    radii.append(radius)
            t['supported_uncertainty']=max(radii) if radii else None
            boxes=[json.loads(o['bbox']) for o in part['observed_rows']]
            t['endpoint_boxes']=[boxes[0],boxes[-1]]
            t['movement_pixels']=float(max((b[0]+b[2])/2 for b in boxes)-min((b[0]+b[2])/2 for b in boxes)+max(b[3] for b in boxes)-min(b[3] for b in boxes))
            from .zones import endpoint_zones
            layout=meta(db,'scene_layout',{})
            t['scene_ports_supported']=layout.get('cameras',{}).get(t['camera_id'],{}).get('source_sha256')==t['source_sha256']
            t['zones']=endpoint_zones(t)
            t['spatial_projection']=layout.get('cameras',{}).get(t['camera_id'],{}).get('projection') if t['scene_ports_supported'] else None
            from .spatial import motion_endpoints
            t['floor_motion']=motion_endpoints(t,layout) if t['scene_ports_supported'] else None
            ts.append(t)
    from .local_bundles import consolidate
    from .spatial import motion_endpoints
    summary=consolidate(ts);set_meta(db,'local_bundle_summary',summary)
    for t in ts:t['floor_motion']=motion_endpoints(t,meta(db,'scene_layout',{})) if t['scene_ports_supported'] else None
    return sorted(ts,key=lambda t:(t['start_time'],t['id']))


def estimate_alignment(db,tracks,c):
    """Appearance-weighted occupancy profile, deliberately weak, not a sync proof.

    Record competing offsets and full support plateau. Without narrow independent
    support retain a declared zero-offset raw-PTS prior; never subtract PTS jumps.
    """
    ref=db.execute('SELECT first_pts FROM camera WHERE id=?',(c['reference_camera'],)).fetchone()[0]
    segments=rows(db,'SELECT camera_id,segment_id,MIN(raw_pts) raw_start,MAX(raw_pts) raw_end FROM frame GROUP BY camera_id,segment_id')
    estimates={};reference=[t for t in tracks if t['camera_id']==c['reference_camera'] and t['vector'] is not None]
    for seg in segments:
        cid,sid=seg['camera_id'],seg['segment_id'];group=[t for t in tracks if t['segment_id']==sid and t['vector'] is not None]
        evidence=[];profile=[]
        if cid!=c['reference_camera']:
            for a in group:
                for b in reference:
                    sim=float(a['vector']@b['vector'])
                    if sim<.78 or abs((a['raw_start']+a['raw_end']-b['raw_start']-b['raw_end'])/2)>60:continue
                    evidence.append({'track_a':a['id'],'track_b':b['id'],'similarity':sim,
                        'midpoint_offset':(b['raw_start']+b['raw_end']-a['raw_start']-a['raw_end'])/2,
                        'raw_intervals':[[a['raw_start'],a['raw_end']],[b['raw_start'],b['raw_end']]]})
            grid=np.arange(-c['offset_search_seconds'],c['offset_search_seconds']+.001,c['offset_step_seconds'])
            for offset in grid:
                votes=0.
                for e in evidence:
                    (a0,a1),(b0,b1)=e['raw_intervals'];gap=max(a0+offset,b0)-min(a1+offset,b1)
                    votes+=(e['similarity']-.75)*max(0,1-max(0,gap)/20)
                profile.append({'offset':float(offset),'score':float(votes)})
        best=max((p['score'] for p in profile),default=0.)
        plateau=[p['offset'] for p in profile if best>0 and p['score']>=best*.95]
        # The same appearance evidence cannot independently validate its own clock.
        # Keep best fit diagnostic unless an explicit manual hypothesis is supplied.
        offset=0.;uncertainty=c['clock_tolerance_seconds'];status='RAW_PTS_PRIOR_UNCALIBRATED'
        manual=next((m for m in c['manual_offsets'] if m['camera_id']==cid and m['segment_id']==sid),None)
        if manual:offset=manual['offset'];uncertainty=manual['uncertainty'];status='MANUAL_ESTIMATE'
        ev={'reference_camera':c['reference_camera'],'reference_first_pts':ref,'formula':'estimated_time = raw_media_pts - reference_first_pts + offset',
            'independent_clock_evidence':False,'circularity_warning':'Appearance-derived offsets and identity matches are not independent validation',
            'source':'declared raw-PTS prior' if manual is None else manual['evidence'],'appearance_pairs':evidence,
            'profile':profile,'best_offset':max(profile,key=lambda p:p['score'])['offset'] if best else None,
            'near_optimal_interval':[min(plateau),max(plateau)] if plateau else None,
            'reason':'Travel time, overlap and clock error are not identifiable from these interval pairs alone; diagnostic fits are not applied',
            'uncertainty_kind':'configured_tolerance_not_measured_error_bound',
            'pts_jumps':'preserved; no unproved correction','clock_tolerance_seconds':c['clock_tolerance_seconds']}
        db.execute('INSERT OR REPLACE INTO alignment_estimate VALUES(?,?,?,?,?,?,?,?)',(cid,sid,seg['raw_start'],seg['raw_end'],offset,uncertainty,status,json.dumps(ev)))
        estimates[sid]={**seg,'offset':offset,'uncertainty':uncertainty,'status':status,'evidence':ev,'reference_first_pts':ref}
    for cid in {e['camera_id'] for e in estimates.values()}:
        previous=None
        for e in sorted((e for e in estimates.values() if e['camera_id']==cid),key=lambda e:e['raw_start']):
            if previous is not None and e['raw_start']+e['offset']<=previous:raise ValueError('Estimated offsets would reverse or overlap timestamp segments')
            previous=e['raw_end']+e['offset']
    return estimates


def topology(c,a,b):
    if a==b:return {'seconds':[0,120],'overlap_possible':False,'path':[a]}
    graph=defaultdict(list)
    for edge in c['transitions']:
        x,y=edge['cameras'];graph[x].append((y,edge));graph[y].append((x,edge))
    queue=[(a,[a],0.,False)]
    while queue:
        node,path,maximum,overlap=queue.pop(0)
        for nxt,e in graph[node]:
            if nxt in path:continue
            if nxt==b:return {'seconds':[e['seconds'][0] if len(path)==1 else 0,maximum+e['seconds'][1]],'overlap_possible':e['overlap_possible'] if len(path)==1 else False,'path':path+[nxt]}
            queue.append((nxt,path+[nxt],maximum+e['seconds'][1],overlap))
    return None


def candidate(a,b,sim,c,review=None):
    reasons=[];hard=[];path=topology(c,a['camera_id'],b['camera_id']);gap=max(a['aligned_start'],b['aligned_start'])-min(a['aligned_end'],b['aligned_end'])
    uncertainty=c['clock_tolerance_seconds'] if c['mode']=='estimated_alignment' else a['time_uncertainty']+b['time_uncertainty']
    if a['vector'] is None or b['vector'] is None:reasons.append('missing_embedding')
    elif min(a['samples'],b['samples'])<3 or min(a['coherence'],b['coherence'])<c['minimum_sample_coherence']:reasons.append('unstable_multicrop_appearance')
    if sim<c['appearance_threshold']:reasons.append('appearance_below_threshold')
    if a['camera_id']==b['camera_id'] and gap<-.001:hard.append('simultaneous_same_camera_conflict')
    from .tracklets import partial_duplicate
    duplicate=partial_duplicate(a,b,sim)
    if duplicate:hard=[r for r in hard if r!='simultaneous_same_camera_conflict']
    if gap>=c['episode_timeout_seconds']:hard.append('episode_gap_exceeds_timeout')
    if not path:hard.append('no_topology_path')
    elif gap>path['seconds'][1]+uncertainty:hard.append('transition_too_long')
    elif gap < -uncertainty and not path['overlap_possible'] and not duplicate:hard.append('incompatible_camera_overlap')
    if c['mode']=='supported_alignment' and (a['supported_start'] is None or b['supported_start'] is None):hard.append('unsupported_time')
    if review and review['decision']=='REJECT':hard.append('human_rejected')
    if review and review['decision']=='CONFIRM':reasons=[]
    from .zones import transition_evidence
    zones=transition_evidence(a,b,gap)
    local_continuity=(a['camera_id']==b['camera_id'] and 0<=gap<=3 and zones.get('footpoint_distance',1)<.20 and sim>=.74)
    if local_continuity:reasons=[r for r in reasons if r not in ('appearance_below_threshold','unstable_multicrop_appearance')]
    hist_a,hist_b=a.get('color_histogram'),b.get('color_histogram')
    from .association_features import clothing_similarity,gallery_similarity
    color=clothing_similarity(hist_a,hist_b)
    gallery=gallery_similarity(a,b)
    if color is not None and color<.22:reasons.append('strong_clothing_disagreement')
    from .spatial import shared_motion
    spatial=shared_motion(a,b,c['clock_tolerance_seconds']) if sim>=.74 and gap<=c['clock_tolerance_seconds'] else None
    if spatial and color is not None and color>=.35:
        reasons=[r for r in reasons if r!='appearance_below_threshold']
    else:spatial=None
    from .spatial import map_handoff
    handoff=map_handoff(a,b,c.get('scene_layout',{})) if sim>=c['component_minimum_similarity'] and color is not None and color>=.60 and gap<=12 else None
    if handoff:reasons=[r for r in reasons if r!='appearance_below_threshold']
    from .spatial import local_reentry,opposing_corridor_motion
    reentry=local_reentry(a,b) if gallery>=.75 and color is not None and color>=.55 else None
    corridor=opposing_corridor_motion(a,b,c['clock_tolerance_seconds']) if gallery>=c['component_minimum_similarity'] and color is not None and color>=.5 and path and path['overlap_possible'] else None
    bundle=bool(a.get('local_bundle_id') and a.get('local_bundle_id')==b.get('local_bundle_id'))
    if reentry or corridor or bundle:reasons=[r for r in reasons if r not in ('appearance_below_threshold','strong_clothing_disagreement')]
    short_supported=(reentry or corridor or bundle) and any(len(short.get('observed_rows',[]))>=6 and short['coherence']>=.8 and reference['samples']>=3 and reference['coherence']>=c['minimum_sample_coherence'] and short['end_time']-short['start_time']<=2.5 for short,reference in ((a,b),(b,a)))
    if short_supported or bundle:reasons=[r for r in reasons if r!='unstable_multicrop_appearance']
    status='REJECTED' if hard else ('UNRESOLVED' if reasons else 'ELIGIBLE')
    # Bounded bonus for close transitions, independent of clock tolerance.
    score=sim+.025*max(0,1-max(0,gap)/max(1,path['seconds'][1] if path else 1))+zones['score_adjustment']
    if reentry or corridor:score=max(score,c['appearance_threshold']+.025+max(0,gallery-c['component_minimum_similarity'])*.1)
    if bundle:score=max(score,.98)
    if handoff:score=max(score,c['appearance_threshold']+.02+.01*min(handoff['direction_agreements']))
    if spatial:score=max(score,c['appearance_threshold']+.02+(.22-spatial['median_distance_table_widths'])*.10)
    return {'id':pair_key(a['id'],b['id']),'track_a':a['id'],'track_b':b['id'],'similarity':sim,'score':score,'status':status,'reasons':hard+reasons,
        'evidence':{'time_basis':c['mode'],'time_status':[a['time_status'],b['time_status']],'clock_uncertainty_seconds':uncertainty,
            'estimated_intervals':[[a['aligned_start'],a['aligned_end']],[b['aligned_start'],b['aligned_end']]],'gap_seconds':gap,
            'uncertainty_kind':'propagated_support' if c['mode']=='supported_alignment' else 'configured_tolerance_not_measured_error_bound',
            'transition':path,'transition_status':c['transition_status'],'topology_source':c['topology_source'],
            'endpoint_image_boxes':[a['endpoint_boxes'],b['endpoint_boxes']],'endpoint_zone_status':'QUALITATIVE_PORTALS_AND_MOTION',
            'zone_transition':zones,'shared_view_motion':spatial,'map_handoff':handoff,'local_reentry':reentry,'opposing_corridor_motion':corridor,'same_camera_bundle':bundle,'appearance_gallery_similarity':gallery,'similarity_kind':'scene-balanced descriptor; supported gallery for component compatibility','clothing_intersection':color,'short_occlusion_continuity':local_continuity,'partial_duplicate':duplicate,
            'sample_coherence':[a['coherence'],b['coherence']],'samples':[a['samples'],b['samples']],
            'crop_evidence':[a['crop_evidence'],b['crop_evidence']],'confidence_kind':'heuristic_not_identity_probability',
            'human_review':review}}


@observed('mtmc')
def run_mtmc(settings,config=None,progress=None):
    c=config or load_config(settings);started=time.monotonic();out=settings.artifacts/'mtmc';out.mkdir(exist_ok=True)
    progress.attach(out/'telemetry')
    progress.update('load_tracks_start',phase='load_tracks')
    # SQLite backup is consistent even while the read-only web app is serving.
    backup=out/'before_mtmc.sqlite3'
    if not backup.exists():
        with sqlite3.connect(settings.database) as src,sqlite3.connect(backup) as dst:src.backup(dst)
    with connect(settings.database) as db:
        set_meta(db,'scene_layout',c.get('scene_layout',{}))
        tracks=load_tracks(db);estimates=estimate_alignment(db,tracks,c)
        for t in tracks:
            e=estimates[t['segment_id']]
            supported=c['mode']=='supported_alignment' and t['supported_start'] is not None
            t['aligned_start']=t['supported_start'] if supported else t['raw_start']-e['reference_first_pts']+e['offset']
            t['aligned_end']=t['supported_end'] if supported else t['raw_end']-e['reference_first_pts']+e['offset']
            t['time_uncertainty']=t['supported_uncertainty'] if supported else e['uncertainty'];t['time_status']='SUPPORTED_CONDITIONAL' if supported else e['status']
        reviews={}
        for r in rows(db,'SELECT * FROM mtmc_review ORDER BY id'):reviews[r['candidate_id']]=r
        n=len(tracks);scores=np.full((n,n),-1.);candidates=[]
        progress.update('pair_scoring_start',phase='pair_scoring',total_pairs=n*(n-1)//2)
        for pair_index,(i,j) in enumerate(combinations(range(n),2)):
            if pair_index%16384==0:progress.update('pair_progress',processed_pairs=pair_index)
            a,b=tracks[i],tracks[j]
            sim=float(a['vector']@b['vector']) if a['vector'] is not None and b['vector'] is not None else -1.
            from .association_features import gallery_similarity
            gallery=gallery_similarity(a,b)
            scores[i,j]=scores[j,i]=gallery
            # Save all plausible appearance edges, even if time/topology rejects
            # them; no N^2 export of wholly uninformative low-similarity pairs.
            if max(sim,gallery)<.70 and pair_key(a['id'],b['id']) not in reviews:continue
            candidates.append(candidate(a,b,sim,c,reviews.get(pair_key(a['id'],b['id']))))
        from .assignment import global_partition
        progress.update('global_assignment_start',phase='global_assignment')
        groups,solver=global_partition(tracks,candidates,scores,c)
        progress.update('write_results_start',phase='write_results')
        evidence_signature=object_hash({t['id']:{'source':t['source_id'],'model':t['model_hash'],'crops':t['crop_evidence'],'local_bundle':t.get('bundle_evidence')} for t in tracks})
        algorithm_hash=object_hash({name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('mtmc.py','assignment.py','tracklets.py','zones.py','spatial.py','local_bundles.py','association_features.py','accounting.py')})
        revision=object_hash({'algorithm_sha256':algorithm_hash,'config':c,'source':meta(db,'source_run_id'),'evidence':evidence_signature,'reviews':reviews,'estimates':estimates})[:24]
        old={r['track_id']:r['episode_id'] for r in db.execute('SELECT * FROM local_global')};registry=rows(db,'SELECT * FROM mtmc_registry ORDER BY number');used=set()
        db.execute('DELETE FROM mtmc_episode');db.execute('DELETE FROM local_global');db.execute('DELETE FROM mtmc_candidate');db.execute('UPDATE local_track SET global_episode_id=NULL')
        number=max((r['number'] for r in registry),default=0);mapping=[]
        for group in sorted(groups,key=lambda g:min(tracks[i]['id'] for i in g)):
            ids=sorted(tracks[i]['id'] for i in group);previous=[r for r in registry if r['id'] not in used and r['seed_track'] in ids]
            if previous:r=previous[0];key=r['id'];num=r['number']
            else:
                number+=1;num=number;key=f'G{num:04d}';db.execute('INSERT INTO mtmc_registry VALUES(?,?,?,?)',(key,num,ids[0],revision))
            used.add(key);links=[p for p in candidates if p['track_a'] in ids and p['track_b'] in ids and p['status'] in ('CONFIRMED','PROVISIONAL')]
            # Confirmed components require a connected graph of explicit human
            # decisions; an unreviewed transitive edge keeps the group provisional.
            roots={tid:tid for tid in ids}
            def root(k):
                while roots[k]!=k:k=roots[k]
                return k
            for p in links:
                if p['status']=='CONFIRMED':roots[root(p['track_b'])]=root(p['track_a'])
            state='UNRESOLVED' if len(group)==1 else ('CONFIRMED' if len({root(k) for k in ids})==1 else 'PROVISIONAL')
            start=min(tracks[i]['aligned_start'] for i in group);end=max(tracks[i]['aligned_end'] for i in group)
            confidence=min((p['similarity'] for p in links),default=0.)
            from .association import global_coverage
            covered=global_coverage(db,end,end+120) if c['mode']=='supported_alignment' else estimated_coverage(db,estimates,end,end+120)
            life=lifecycle(end,end+120 if covered else end+.3,120,coverage_complete=covered)
            termination_reason='estimated_clock_coverage_timeout_120s' if covered and c['mode']=='estimated_alignment' else life.reason
            db.execute('INSERT INTO mtmc_episode VALUES(?,?,?,?,?,?,?,?,?,?,?)',(key,f'ID {num:03d}',state,life.status,start,end,life.terminated_at,termination_reason,confidence,c['mode'],revision))
            for tid in ids:
                db.execute('INSERT INTO local_global VALUES(?,?,?,?,?)',(tid,key,state,confidence,revision))
                db.execute('UPDATE local_track SET global_episode_id=? WHERE id=?',(key,tid))
                source=next(t for t in tracks if t['id']==tid)
                mapping.append({'local_track_id':tid,'source_track_id':source['parent_id'],'source_frame_range':[source['start_frame'],source['end_frame']],'global_episode_id':key,'internal_episode_id':key,'status':state,'confidence':confidence})
        ordered=rows(db,'SELECT e.id,MIN(t.start_time) AS first_time,MIN(t.camera_id) AS first_camera FROM mtmc_episode e JOIN local_global g ON g.episode_id=e.id JOIN association_track t ON t.id=g.track_id GROUP BY e.id ORDER BY first_time,first_camera,e.id')
        for display_number,episode in enumerate(ordered,1):
            db.execute('UPDATE mtmc_episode SET label=? WHERE id=?',(f'ID {display_number:03d}',episode['id']))
        labels=dict(db.execute('SELECT id,label FROM mtmc_episode'))
        for record in mapping:record['global_label']=labels[record['global_episode_id']]
        for p in candidates:
            db.execute('INSERT INTO mtmc_candidate VALUES(?,?,?,?,?,?,?,?,?)',(p['id'],p['track_a'],p['track_b'],p['similarity'],p['score'],p['status'],json.dumps(p['reasons']),json.dumps(p['evidence']),revision))
        counts=dict(Counter(r[0] for r in db.execute('SELECT status FROM mtmc_episode')));decisions=dict(Counter(p['status'] for p in candidates))
        summary={'status':'AUTOMATIC_ESTIMATED_TRACKING','revision':revision,'source_run_id':meta(db,'source_run_id'),
            'local_tracks':db.execute('SELECT COUNT(*) FROM local_track').fetchone()[0],'derived_tracklets':n,
            'normalized_appearance':meta(db,'normalized_appearance',{}),'local_bundles':meta(db,'local_bundle_summary',{}),
            'genuine_embedding_tracks':db.execute('SELECT COUNT(*) FROM embedding').fetchone()[0],'temporal_appearance':meta(db,'temporal_appearance',{}),
            'episodes_by_status':counts,'decisions':decisions,'global_solver':solver,
            'time_mode':c['mode'],'clock_tolerance_seconds':c['clock_tolerance_seconds'],'identity_count':None,'algorithm_sha256':algorithm_hash,
            'policy':'Automatic MILP global assignment, temporal identity slices, image portals and motion; reviews optional; raw PTS and TimeAlignment immutable',
            'wall_seconds':time.monotonic()-started}
        db.execute('INSERT OR REPLACE INTO mtmc_run VALUES(?,?,?,?)',(revision,json.dumps(c),json.dumps(summary),str(time.time())))
        set_meta(db,'mtmc_summary',summary);set_meta(db,'mtmc_config',c);set_meta(db,'global_association_status',summary['status'])
        from .accounting import build_accounting,counts
        progress.update('episode_accounting_start',phase='episode_accounting')
        ledger=build_accounting(db,revision,c['episode_timeout_seconds'])
        set_meta(db,'episode_accounting',ledger)
        atomic_json(out/'episode_accounting.json',ledger)
        atomic_json(out/'episode_counts.json',counts(ledger))
        atomic_json(out/'alignment_estimates.json',{'mode':c['mode'],'estimates':list(estimates.values()),'historical_alignment_modified':False})
        atomic_json(out/'local_to_global.json',{'schema_version':2,'revision':revision,'mappings':mapping,'identity_count':None})
        atomic_json(out/'candidates.json',{'schema_version':1,'revision':revision,'candidates':candidates})
        atomic_json(out/'summary.json',summary)
    print('[MTMC] '+json.dumps(summary),flush=True);return summary


def identity(db,track_id,frame_index=None):
    if frame_index is not None:
        from .tracklets import frame_tracklet
        track_id=frame_tracklet(db,track_id,frame_index)
    r=db.execute('SELECT g.episode_id,g.status,g.confidence,e.label FROM local_global g JOIN mtmc_episode e ON e.id=g.episode_id WHERE g.track_id=?',(track_id,)).fetchone()
    if not r:return {'global_episode_id':None,'global_status':'UNRESOLVED','global_label':None,'global_color':None,'global_confidence':0.,'derived_tracklet_id':track_id}
    return {'global_episode_id':r['episode_id'],'global_status':'SINGLE_CAMERA' if r['status']=='UNRESOLVED' else r['status'],'global_label':r['label'],'derived_tracklet_id':track_id,
            'global_confidence':r['confidence'],'global_color':'#'+''.join(f'{v:02x}' for v in color_for(r['episode_id']))}


def estimated_coverage(db,estimates,start,end):
    """Coverage in the declared estimated clock; not proven physical absence."""
    cameras=[r[0] for r in db.execute('SELECT id FROM camera')]
    if len(cameras)!=5:return False
    for camera in cameras:
        intervals=sorted((e['raw_start']-e['reference_first_pts']+e['offset'],e['raw_end']-e['reference_first_pts']+e['offset']) for e in estimates.values() if e['camera_id']==camera)
        cursor=start
        for a,b in intervals:
            if b<cursor:continue
            if a>cursor+.2:break
            cursor=max(cursor,b)
            if cursor>=end:break
        if cursor<end:return False
    return True


def save_review(settings,candidate_id,decision,reviewer,evidence):
    if decision not in ('CONFIRM','REJECT','RESET') or len(reviewer.strip())<2 or len(evidence.strip())<8:raise ValueError('Review requires a reviewer, decision and actual evidence')
    with connect(settings.database) as db:
        row=db.execute('SELECT * FROM mtmc_candidate WHERE id=?',(candidate_id,)).fetchone()
        if not row:raise ValueError('Unknown candidate')
        # A confirmed UI decision is a human assertion, never an automatic label.
        db.execute('INSERT INTO mtmc_review(candidate_id,track_a,track_b,decision,reviewer,evidence,created_at,source_run_id) VALUES(?,?,?,?,?,?,?,?)',
            (candidate_id,row['track_a'],row['track_b'],decision,reviewer.strip(),evidence.strip(),str(time.time()),meta(db,'source_run_id')))
        reviews=rows(db,'SELECT * FROM mtmc_review ORDER BY id')
    atomic_json(settings.artifacts/'mtmc/human_reviews.json',{'schema_version':1,'human_reviewed':True,'reviews':reviews})
    return run_mtmc(settings)


def observation_intervals(db,track,offset=0.,ref=0.,max_gap=.2):
    """Actual observed runs, split at missing observations and segment boundaries."""
    times=[r[0]-ref+offset for r in db.execute('SELECT raw_pts FROM observation WHERE track_id=? AND frame_index BETWEEN ? AND ? ORDER BY frame_index',(track.get('parent_id',track['id']),track['start_frame'],track['end_frame']))]
    intervals=[]
    for value in times:
        if not intervals or value-intervals[-1][1]>max_gap:intervals.append([value,value])
        else:intervals[-1][1]=value
    return intervals


def gantt(db):
    result=[]
    estimates={r['segment_id']:dict(r) for r in db.execute('SELECT * FROM alignment_estimate')}
    for e in rows(db,'SELECT * FROM mtmc_episode ORDER BY start_time,id'):
        segments=[]
        for t in rows(db,'SELECT t.* FROM association_track t JOIN local_global g ON g.track_id=t.id WHERE g.episode_id=? ORDER BY t.start_time',(e['id'],)):
            est=estimates.get(t['segment_id']);ev=json.loads(est['evidence']) if est else {};ref=ev.get('reference_first_pts',0)
            intervals=observation_intervals(db,t,est['offset'] if est else 0,ref)
            if e['time_basis']=='supported_alignment':
                actual=rows(db,'SELECT f.global_time FROM observation o JOIN frame f ON f.camera_id=o.camera_id AND f.frame_index=o.frame_index WHERE o.track_id=? AND o.frame_index BETWEEN ? AND ? ORDER BY o.frame_index',(t['parent_id'],t['start_frame'],t['end_frame']))
                intervals=[]
                if all(p['global_time'] is not None for p in actual):
                    for p in actual:
                        value=p['global_time']
                        if not intervals or value-intervals[-1][1]>.2:intervals.append([value,value])
                        else:intervals[-1][1]=value
            segments.append({'track_id':t['id'],'source_track_id':t['parent_id'],'source_frame_range':[t['start_frame'],t['end_frame']],'local_label':t['display_id'],'camera_id':t['camera_id'],'segment_id':t['segment_id'],
                'intervals':intervals,'raw_start':t['start_time']+db.execute('SELECT first_pts FROM camera WHERE id=?',(t['camera_id'],)).fetchone()[0],
                'media_start':t['start_time'],'uncertainty':est['uncertainty'] if est else None,'time_status':est['status'] if est else 'UNAVAILABLE'})
        result.append({**e,'segments':segments,'display_label':e['label']})
    return {'rows':result,'time_basis':'estimated_alignment','physical_synchronization':'PARTIAL','gaps':'absence of observations, not proven departure','duration':max((e['end_time'] for e in result),default=1200)}
