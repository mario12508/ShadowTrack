"""Read-only visual-control evaluation. The association engine never reads labels."""
from collections import Counter
from pathlib import Path

from shadowtrack.detection.common import atomic_json,read_json
from shadowtrack.mvp.config import Settings
from shadowtrack.mvp.db import connect,meta
from shadowtrack.mvp.tracklets import frame_tracklet


def evaluate(db,cases,labels,derived):
    totals=Counter();results=[]
    for case in cases:
        key=case['control_id'];label=labels[key];ids=[]
        for sequence in case['sequences']:
            values=[]
            for sample in sequence['samples']:
                source=sequence['track_id'];track=frame_tracklet(db,source,sample['source_frame_index']) if derived else source
                row=db.execute('SELECT episode_id FROM local_global WHERE track_id=?',(track,)).fetchone()
                values.append(row[0] if row else None)
            ids.append(values)
        decision=case.get('status');direct_status=[]
        if derived:
            aa=[r[0] for r in db.execute('SELECT id FROM mtmc_tracklet WHERE parent_id=?',(case['track_a'],))]
            bb=[r[0] for r in db.execute('SELECT id FROM mtmc_tracklet WHERE parent_id=?',(case['track_b'],))]
        else:aa=[case['track_a']];bb=[case['track_b']]
        for a in aa:
            for b in bb:
                row=db.execute('SELECT status FROM mtmc_candidate WHERE (track_a=? AND track_b=?) OR (track_a=? AND track_b=?)',(a,b,b,a)).fetchone()
                if row:direct_status.append(row[0])
        matched=sum(a is not None and a==b for a in ids[0] for b in ids[1]);count=len(ids[0])*len(ids[1])
        if label['kind'] in ('same_person_continuation','different_person'):
            positive=label['kind']=='same_person_continuation'
            totals['tp' if positive else 'fp']+=matched
            totals['fn' if positive else 'tn']+=count-matched
            totals['decisive_cases']+=1
            totals['positive_cases' if positive else 'negative_cases']+=1
            totals['fully_consistent_positive_cases']+=int(positive and matched==count)
            totals['partially_connected_positive_cases']+=int(positive and matched>0)
        elif label['kind']=='same_person_later_not_direct':
            totals['later_visit_cases']+=1
            totals['inadmissible_direct_links']+=int(any(s in ('PROVISIONAL','CONFIRMED') for s in direct_status))
        else:totals['mixed_tracklet_cases']+=1
        results.append({'control_id':key,'source_tracks':[case.get('la'),case.get('lb')],'label':label,'observed_global_ids':ids,
                        'matching_review_frame_pairs':matched,'review_frame_pairs':count,'direct_decisions':direct_status})
    tp,fp,fn=totals['tp'],totals['fp'],totals['fn']
    return {'counts':dict(totals),'precision_on_reviewed_frame_pairs':tp/(tp+fp) if tp+fp else None,
            'recall_on_reviewed_frame_pairs':tp/(tp+fn) if tp+fn else None,'cases':results}


def main():
    s=Settings.environment();folder=s.artifacts/'mtmc/control';manifest=read_json(folder/'evaluation_manifest.json');labels=read_json(folder/'visual_labels.json')
    with connect(folder/'baseline.sqlite3',readonly=True) as before:baseline=evaluate(before,manifest['cases'],labels['labels'],False)
    with connect(s.database,readonly=True) as after:
        current=evaluate(after,manifest['cases'],labels['labels'],True);revision=meta(after,'mtmc_summary')['revision']
        split_sources=[dict(r) for r in after.execute('SELECT parent_id,COUNT(*) AS parts FROM mtmc_tracklet GROUP BY parent_id HAVING COUNT(*)>1')]
    report={'schema_version':1,'revision':revision,'selection':manifest['selection'],'review_method':labels['method'],
            'limitations':['Curated visual control, not random population sampling','No IDF1/HOTA or claim of true unique-person count','Reused visual control; not a blind held-out benchmark','Five later-visit pairs evaluate direct-link rejection, not physical identity or complete episode separation','Mixed source tracklets excluded from pair precision/recall and reported separately'],
            'baseline':baseline,'current':current,'derived_split_sources':split_sources,'labels_are_algorithm_input':False}
    atomic_json(folder/'quality.json',report)
    print({'baseline':{k:v for k,v in baseline.items() if k!='cases'},'current':{k:v for k,v in current.items() if k!='cases'}})


if __name__=='__main__':main()
