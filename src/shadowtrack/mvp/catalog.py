"""One archive card per global episode; local fragments remain in its details."""
from collections import defaultdict

from .db import rows


def global_catalog(db,camera_id=None,q='',start=None,end=None,limit=24,offset=0):
    from .accounting import accounting,lifecycle_for
    ledger=accounting(db)
    members=defaultdict(list)
    for item in rows(db,'SELECT t.id,t.parent_id,t.display_id,t.camera_id,t.start_time,t.end_time,t.observations,t.mean_confidence,g.episode_id FROM association_track t JOIN local_global g ON g.track_id=t.id ORDER BY t.start_time,t.id'):
        members[item['episode_id']].append(item)
    found=[];query=q.strip().casefold();numeric=query.removeprefix('id').strip()
    number=int(numeric) if numeric.isdigit() else None
    for episode in rows(db,'SELECT * FROM mtmc_episode'):
        parts=members[episode['id']]
        if not parts or camera_id and not any(p['camera_id']==camera_id for p in parts):continue
        first=min(p['start_time'] for p in parts);last=max(p['end_time'] for p in parts)
        if start is not None and last<start or end is not None and first>end:continue
        if number is not None and int(episode['label'].split()[-1])!=number:continue
        if number is None and query and query not in ' '.join([episode['label'],episode['id'],*(p['display_id'] for p in parts)]).casefold():continue
        representative=max(parts,key=lambda p:(p['observations'],p['mean_confidence'],p['id']))
        replay=lifecycle_for(ledger,episode['id'])
        found.append({'id':representative['id'],'camera_id':representative['camera_id'],
            'representative_start_time':representative['start_time'],'display_id':episode['label'],
            'global_episode_id':episode['id'],'global_label':episode['label'],'scope':'GLOBAL',
            'status':replay['state'] if replay else episode['lifecycle'],'termination_reason':replay['reason'] if replay else episode['termination_reason'],
            'right_censored':replay['right_censored'] if replay else None,'lifecycle_clock':'source_recordings',
            'start_time':first,'end_time':last,'duration':last-first,
            'observations':sum(p['observations'] for p in parts),'fragment_count':len(parts),
            'camera_ids':sorted({p['camera_id'] for p in parts}),
            'intervals':[{'camera_id':p['camera_id'],'track_id':p['id'],'start_time':p['start_time'],'end_time':p['end_time']} for p in parts],
            'thumbnail_url':f"/api/tracks/{representative['id']}/crop"})
    found.sort(key=lambda e:(e['start_time'],int(e['global_label'].split()[-1])))
    total=len(found);found=found[offset:offset+limit]
    from .mtmc import identity
    for item in found:item.update(identity(db,item['id']))
    return {'total':total,'items':found,'count_kind':'global_track_episodes','identity_count':None}
