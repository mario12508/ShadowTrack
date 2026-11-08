"""Episode counting and replay lifecycle over genuine source observations.

These counts describe the current automatic partition, not physical people or
measured identity accuracy. Replay time is explicit and never changes media PTS.
"""
from collections import Counter,defaultdict

from .db import meta,rows


def append_interval(intervals,time,segment=None):
    if intervals and time-intervals[-1][1]<=.2 and (segment is None or intervals[-1][2]==segment):
        intervals[-1][1]=time
    else:intervals.append([time,time,segment])


def covers(coverage,start,end):
    if len(coverage)!=5:return False
    for intervals in coverage.values():
        cursor=start
        for a,b,*_ in intervals:
            if b<cursor:continue
            if a>cursor+.2:break
            cursor=max(cursor,b)
            if cursor>=end:break
        if cursor<end:return False
    return True


def replay_state(record,now,coverage,recording_end,timeout):
    seen=None
    for a,b,*_ in record['observed_intervals']:
        if a>now:break
        seen=min(b,now)
    if seen is None:return {'state':'not_started','last_seen':None,'terminated_at':None,'reason':'not_started','right_censored':False}
    deadline=seen+timeout;complete=covers(coverage,seen,deadline)
    if now-seen<=.2:state,reason='active','observed'
    elif now>=deadline and complete:state,reason='terminated','source_recording_coverage_timeout'
    else:state,reason='lost',('within_inactivity_timeout' if now<deadline else 'coverage_unknown')
    censored=state!='terminated' and now>=recording_end-.001
    if censored:reason='recording_ended_before_timeout' if deadline>recording_end else 'coverage_gap'
    return {'state':state,'last_seen':seen,'terminated_at':deadline if state=='terminated' else None,
            'reason':reason,'right_censored':censored,'coverage_support':'source_recordings_prior_not_independent_clock_calibration'}


def build_accounting(db,revision,timeout=120.):
    coverage=defaultdict(list)
    for row in db.execute('SELECT camera_id,media_time,segment_id FROM frame ORDER BY camera_id,media_time,frame_index'):
        append_interval(coverage[row[0]],row[1],row[2])
    recording_end=max((x[1] for values in coverage.values() for x in values),default=0.)
    members=defaultdict(list)
    for row in rows(db,'SELECT g.episode_id,t.id,t.parent_id,t.camera_id,t.start_time,t.end_time,t.observations FROM local_global g JOIN association_track t ON t.id=g.track_id ORDER BY t.start_time,t.id'):
        members[row['episode_id']].append(row)
    intervals=defaultdict(list)
    for gid,time in db.execute('SELECT g.episode_id,o.media_time FROM observation o JOIN mtmc_tracklet t ON t.parent_id=o.track_id AND o.frame_index BETWEEN t.start_frame AND t.end_frame JOIN local_global g ON g.track_id=t.id ORDER BY g.episode_id,o.media_time'):
        append_interval(intervals[gid],time)
    records=[]
    for episode in rows(db,'SELECT * FROM mtmc_episode ORDER BY start_time,id'):
        parts=members[episode['id']];observed=intervals[episode['id']]
        if not parts or not observed:continue
        record={'episode_id':episode['id'],'label':episode['label'],'identity_status':episode['status'],
                'start_time':observed[0][0],'last_observed_time':observed[-1][1],
                'aligned_start_time':episode['start_time'],'aligned_end_time':episode['end_time'],'alignment_basis':episode['time_basis'],
                'camera_ids':sorted({p['camera_id'] for p in parts}),'fragment_ids':[p['id'] for p in parts],
                'source_track_ids':sorted({p['parent_id'] for p in parts}),'fragment_count':len(parts),
                'observations':sum(p['observations'] for p in parts),'observed_intervals':observed,
                'maximum_observed_absence_seconds':max((b[0]-a[1] for a,b in zip(observed,observed[1:])),default=0.)}
        record['final_lifecycle']=replay_state(record,recording_end,coverage,recording_end,timeout)
        records.append(record)
    records.sort(key=lambda r:(r['start_time'],r['label']))
    return {'schema_version':1,'revision':revision,'clock':'source_recordings','clock_is_calibrated':False,
            'source_run_id':meta(db,'source_run_id'),'timeout_seconds':timeout,'recording_end':recording_end,
            'camera_coverage':dict(coverage),'episodes':records,'identity_count':None,
            'definition':'Each global track episode counted once, including single-camera episodes; multi-camera episodes are a subset',
            'validation':'automatic_estimated_partition_not_ground_truth'}


def accounting(db):
    revision=meta(db,'mtmc_summary',{}).get('revision');cached=meta(db,'episode_accounting',{})
    if cached.get('revision')==revision and cached.get('schema_version')==1:return cached
    return build_accounting(db,revision)


def counts(document,now=None):
    now=document['recording_end'] if now is None else now;states=Counter();started=[];censored=0
    for episode in document['episodes']:
        state=replay_state(episode,now,document['camera_coverage'],document['recording_end'],document['timeout_seconds'])
        if state['state']=='not_started':continue
        started.append(episode);states[state['state']]+=1;censored+=state['right_censored']
    return {'revision':document['revision'],'clock':document['clock'],'time':now,'total_episodes':len(document['episodes']),
            'started_episodes':len(started),'multi_camera_episodes':sum(len(r['camera_ids'])>1 for r in started),
            'single_camera_episodes':sum(len(r['camera_ids'])==1 for r in started),
            'active':states['active'],'lost':states['lost'],'terminated':states['terminated'],'right_censored':censored,
            'timeout_seconds':document['timeout_seconds'],'identity_count':None,'validation':document['validation']}


def lifecycle_for(document,episode_id,now=None):
    record=next((r for r in document['episodes'] if r['episode_id']==episode_id),None)
    if record is None:return None
    return replay_state(record,document['recording_end'] if now is None else now,document['camera_coverage'],document['recording_end'],document['timeout_seconds'])
