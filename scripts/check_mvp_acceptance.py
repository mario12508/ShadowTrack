"""Read-only requirement audit; internal consistency is not identity ground truth."""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import xml.etree.ElementTree as ET

from shadowtrack.detection.common import atomic_json,read_json
from shadowtrack.mvp.accounting import accounting,counts
from shadowtrack.mvp.config import Settings
from shadowtrack.mvp.db import connect,meta,rows
from shadowtrack.mvp.tracklets import partial_duplicate


def main():
    settings=Settings.environment();out=settings.artifacts/'mtmc'
    with connect(settings.database,readonly=True) as db:
        summary=meta(db,'mtmc_summary');ledger=accounting(db);totals=counts(ledger)
        groups=defaultdict(list)
        for t in rows(db,'SELECT t.*,g.episode_id FROM association_track t JOIN local_global g ON g.track_id=t.id'):
            groups[t['episode_id']].append(t)
        conflicts=[];nested=0;cache={}
        for gid,members in groups.items():
            for a,b in combinations(members,2):
                if a['camera_id']!=b['camera_id'] or min(a['end_time'],b['end_time'])<=max(a['start_time'],b['start_time'])+.001:continue
                for t in (a,b):
                    if t['id'] not in cache:
                        cache[t['id']]={**t,'observed_rows':rows(db,'SELECT frame_index,bbox FROM observation WHERE track_id=? AND frame_index BETWEEN ? AND ? ORDER BY frame_index',(t['parent_id'],t['start_frame'],t['end_frame']))}
                proof=partial_duplicate(cache[a['id']],cache[b['id']],-1)
                if proof:nested+=1
                else:conflicts.append({'episode_id':gid,'track_a':a['id'],'track_b':b['id'],'reason':'overlapping_different_same_camera_fragments'})
        identity_checks={'status':'SELF_CONSISTENCY_PASSED' if not conflicts else 'FAILED','same_camera_conflicts':conflicts,
            'geometrically_supported_nested_pairs':nested,'maximum_observed_absence_seconds':max((r['maximum_observed_absence_seconds'] for r in ledger['episodes']),default=0),
            'episodes_with_absence_at_least_timeout':[r['episode_id'] for r in ledger['episodes'] if r['maximum_observed_absence_seconds']>=ledger['timeout_seconds']],
            'labels_used_as_algorithm_input':False,'identity_accuracy':None,'meaning':'Physical exclusions and timeout continuity only; not IDF1/HOTA or proof of identity'}
        assert not conflicts and not identity_checks['episodes_with_absence_at_least_timeout']
        assert totals['total_episodes']==len(groups)==db.execute('SELECT COUNT(*) FROM mtmc_episode').fetchone()[0]
        cameras=rows(db,'SELECT id,fps,frame_count FROM camera ORDER BY id')
        sizes={name:db.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] for name in ('frame','detection','local_track','observation')}
        assert len(cameras)==5 and sum(r['frame_count'] for r in cameras)==sizes['frame']
    browser=read_json(out/'playwright-results.json')['stats'];quality=read_json(out/'control/quality.json')
    suites=ET.parse(out/'tests.xml').getroot().findall('testsuite')
    cpu=sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped'))
    assert browser['expected']>=20 and browser['unexpected']==browser['flaky']==browser['skipped']==0
    assert quality['revision']==summary['revision']==ledger['revision']
    requirements=[
        {'requirement':'process_five_videos','status':'VERIFIED','evidence':{'cameras':cameras,'frames':sizes['frame']}},
        {'requirement':'detect_people','status':'IMPLEMENTED','evidence':{'raw_detections':sizes['detection'],'all_frame_export':True,'detector_accuracy':None}},
        {'requirement':'show_bounding_boxes','status':'VERIFIED','evidence':'Real browser canvas and saved local-ID MP4 tests'},
        {'requirement':'single_camera_tracking','status':'IMPLEMENTED','evidence':{'local_tracklets':sizes['local_track'],'observations':sizes['observation']}},
        {'requirement':'web_video','status':'VERIFIED','evidence':{'browser_tests':browser['expected'],'map_people_modes_separate':True,'media_error_recovery_tested':True}},
        {'requirement':'cross_camera_identity','status':'PARTIAL','evidence':{'multi_camera_episodes':totals['multi_camera_episodes'],'control_counts':quality['current']['counts'],'full_video_identity_accuracy':None}},
        {'requirement':'absence_timeout_60_120s','status':'IMPLEMENTED_CONDITIONAL','evidence':{'timeout_seconds':totals['timeout_seconds'],'terminated':totals['terminated'],'right_censored':totals['right_censored'],'clock':totals['clock']}},
        {'requirement':'unique_track_episode_count','status':'IMPLEMENTED_ESTIMATED','evidence':totals},
        {'requirement':'validate_real_identity_accuracy','status':'PARTIAL','evidence':{'curated_visual_control':quality['current']['counts'],'population_accuracy':None,'not_blind_held_out':True}},
    ]
    report={'acceptance':'IMPLEMENTED_ESTIMATED_MVP_WITH_PARTIAL_IDENTITY_VALIDATION','revision':summary['revision'],
            'requirements':requirements,'episode_counts':totals,'component_consistency':identity_checks,'pytest_passed':cpu,'browser_passed':browser['expected'],
            'native_recompute':read_json(out/'recompute_verification.json'),
            'regression_reported_by_user':'historic_video_status; current confirmed frontend defect was identical map/people content',
            'limitations':['All counts depend on automatic association','Unique track episodes are not unique physical people','Source replay clock follows organizer prior, not independent all-camera calibration','Censored records are not proven completed departures'],
            'hour_plan_minutes':[{'from':0,'to':10,'work':'Rank long unresolved fragments and inspect the highest-impact scene/visibility failures'},
                {'from':10,'to':30,'work':'Add quality-aware portal handoff and multi-view continuation using existing crops, with physical cannot-links'},
                {'from':30,'to':45,'work':'Compare a frozen baseline on positives, co-visible lookalikes, occlusions and 120-second boundary cases'},
                {'from':45,'to':55,'work':'Run CPU and browser checks, reconcile episode exports and all-frame source linkage'},
                {'from':55,'to':60,'work':'Freeze code/data hashes, acceptance report and repeatable demo'}]}
    atomic_json(out/'acceptance_progress.json',report)
    print(json.dumps({'acceptance':report['acceptance'],'counts':totals,'consistency':identity_checks,'pytest':cpu,'browser':browser['expected']},ensure_ascii=True))


if __name__=='__main__':main()
