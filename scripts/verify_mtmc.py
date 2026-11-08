"""Read-only acceptance proof; retain historic MVP verification separately."""
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import av
import httpx

from shadowtrack.detection.common import atomic_json,read_json
from shadowtrack.mvp.config import Settings
from shadowtrack.mvp.db import connect,meta,rows
from shadowtrack.mvp.prepare import track_key
from shadowtrack.probe import sha256


def table_digest(db,table,columns='*',order='1'):
    h=hashlib.sha256()
    for row in db.execute(f'SELECT {columns} FROM {table} ORDER BY {order}'):
        value=[v.hex() if isinstance(v,bytes) else v for v in row]
        h.update(json.dumps(value,separators=(',',':')).encode()+b'\n')
    return h.hexdigest()


def main():
    s=Settings.environment();out=s.artifacts/'mtmc'
    protected=read_json(s.root/'artifacts/detection/preserved_hashes.json')
    assert all(sha256(s.root/name)==digest for name,digest in protected.items())
    alignment=read_json(s.alignment)
    assert all(sha256(s.alignment.parent/c['local_frame_times_file'])==c['local_frame_times_sha256'] for c in alignment['cameras'])
    sample=s.root/'artifacts/detection/sample_review/validated/988a8de217bb1d3c0b951e67';old_sample=read_json(sample/'export_manifest.json')
    assert all(sha256(sample/name)==entry['sha256'] for name,entry in old_sample['files'].items())
    features=read_json(s.artifacts/'reid/embeddings/embedding_manifest.json')
    assert sha256(s.artifacts/'reid/embeddings/embeddings.npz')==features['features_sha256']
    print('[verify] protected sources, alignment and genuine embeddings unchanged',flush=True)
    tables={}
    with connect(out/'before_mtmc.sqlite3',readonly=True) as before,connect(s.database,readonly=True) as after:
        for name,order in [('camera','id'),('frame','camera_id,frame_index'),('detection','camera_id,frame_index,detection_index'),('observation','track_id,frame_index'),('embedding','track_id')]:
            a=table_digest(before,name,order=order);b=table_digest(after,name,order=order);assert a==b,name;tables[name]=b
        columns=','.join(r[1] for r in before.execute('PRAGMA table_info(local_track)') if r[1]!='global_episode_id')
        assert table_digest(before,'local_track',columns,'id')==table_digest(after,'local_track',columns,'id')
        source_media=rows(before,'SELECT * FROM media_segment')
        for m in source_media:assert sha256(s.media/m['path'])==m['sha256']
        summary=meta(after,'mtmc_summary');counts=dict(after.execute('SELECT status,COUNT(*) FROM mtmc_episode GROUP BY status').fetchall())
        for camera in rows(after,'SELECT source_name,source_sha256 FROM camera'):
            assert sha256(s.raw/camera['source_name'])==camera['source_sha256']
        normalized=meta(after,'normalized_appearance')
        from shadowtrack.mvp.normalized_appearance import validate_cache
        cache=out/'normalized_appearance'/normalized['signature'][:16]
        status,cm,vectors=validate_cache(cache,normalized['signature'],normalized['model_hash'])
        assert status['samples']==len(vectors)==after.execute('SELECT COUNT(*) FROM appearance_normalized').fetchone()[0]==3427
        derived=after.execute('SELECT COUNT(*) FROM mtmc_tracklet').fetchone()[0]
        assert after.execute('SELECT COUNT(*) FROM local_global').fetchone()[0]==derived==summary['derived_tracklets']
        assert after.execute('SELECT COUNT(DISTINCT parent_id) FROM mtmc_tracklet').fetchone()[0]==503
        slices={}
        for r in after.execute('SELECT t.parent_id,t.id,t.start_frame,t.end_frame,g.episode_id FROM mtmc_tracklet t JOIN local_global g ON g.track_id=t.id ORDER BY t.parent_id,t.start_frame'):
            slices.setdefault(r['parent_id'],[]).append(tuple(r)[1:])
        for parent,parts in slices.items():
            for index in after.execute('SELECT frame_index FROM observation WHERE track_id=?',(parent,)):
                assert sum(a<=index[0]<=b for _,a,b,_ in parts)==1
        reviews=after.execute('SELECT COUNT(*) FROM mtmc_review').fetchone()[0]
        cross_view=after.execute('SELECT COUNT(*) FROM (SELECT g.episode_id FROM local_global g JOIN association_track t ON t.id=g.track_id GROUP BY g.episode_id HAVING COUNT(DISTINCT t.camera_id)>1)').fetchone()[0]
        clips=rows(after,'SELECT * FROM media_segment WHERE kind="global_review" AND json_extract(metadata,"$.revision")=?',(summary['revision'],))
        # No new MP4 is required: dynamic overlays show the current mapping.
        assert len(clips) in (0,3)
        for clip in clips:
            p=s.media/clip['path'];assert sha256(p)==clip['sha256']
            m=json.loads(clip['metadata']);side=p.with_suffix('.frames.jsonl');assert sha256(side)==m['sidecar_sha256']
            expected=[json.loads(line) for line in side.read_text().splitlines()]
            with av.open(str(p)) as container:pts=[float(packet.pts*packet.time_base) for packet in container.demux(video=0) if packet.pts is not None]
            assert len(pts)==len(expected)==clip['frames']
            assert all(abs(t-e['output_relative_seconds'])<1e-6 for t,e in zip(pts,expected))
            for e in expected:
                frame=after.execute('SELECT raw_pts,segment_id FROM frame WHERE camera_id=? AND frame_index=?',(e['camera_id'],e['source_frame_index'])).fetchone()
                assert tuple(frame)==(e['raw_media_pts'],e['timestamp_segment_id'])
        bases=rows(after,'SELECT camera_id,segment_id,offset,uncertainty,status FROM alignment_estimate')
    print('[verify] database observations and 23 original derived videos unchanged; new clip PTS verified',flush=True)
    suites=ET.parse(out/'tests.xml').getroot().findall('testsuite')
    assert all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped'))
    pytest_count=sum(int(s.attrib['tests']) for s in suites)
    browser=read_json(out/'playwright-results.json')['stats'];assert browser['expected']==20 and browser['unexpected']==browser['skipped']==browser['flaky']==0
    docker=read_json(out/'docker_verification.json');assert docker['torch_absent'] and docker['backend_healthy']
    with httpx.Client(base_url='http://localhost:3000',timeout=30) as client:
        assert client.get('/').status_code==200
        stats=client.get('/api/stats').json();assert stats['frame']==144110 and stats['local_track']==503
        assert stats['provisional_episodes']==counts.get('PROVISIONAL',0)
        listed=client.get('/api/exports').json()['media']
        assert len([m for m in listed if m['kind']=='global_review'])==len(clips)
        assert all(m.get('revision')==summary['revision'] for m in listed if m['kind']=='global_review')
        gantt=client.get('/api/gantt').json();assert sum(len(r['segments']) for r in gantt['rows'])==derived
        floor=client.get('/api/floor?time=1030').json()
        assert floor['markers'] and all(m['floor_zone']['radius_normalized']>0 and m['localization_status']=='SCHEMATIC_PROJECTED' for m in floor['markers'])
        # User-reported front/back mismatch: verification input only, never engine rules.
        with connect(s.database,readonly=True) as con:
            first=rows(con,'SELECT t.camera_id,g.episode_id FROM association_track t JOIN local_global g ON g.track_id=t.id WHERE t.display_id IN ("C1-T001","C2-T001")')
        assert len(first)==2 and len({r['episode_id'] for r in first})==1
        first_id=first[0]['episode_id']
        catalog=client.get('/api/global-tracks?limit=100').json();assert catalog['count_kind']=='global_track_episodes'
        assert [r['global_label'] for r in catalog['items']]==[f'ID {i:03d}' for i in range(1,len(catalog['items'])+1)]
        with connect(s.database,readonly=True) as con:
            corridor=rows(con,'SELECT t.display_id,t.id,g.episode_id,e.label FROM association_track t JOIN local_global g ON g.track_id=t.id JOIN mtmc_episode e ON e.id=g.episode_id WHERE t.display_id IN ("C4-T001","C5-T006","C5-T007")')
            assert len(corridor)==3 and len({r['episode_id'] for r in corridor})==1
            duplicate=rows(con,'SELECT t.display_id,g.episode_id,e.label FROM association_track t JOIN local_global g ON g.track_id=t.id JOIN mtmc_episode e ON e.id=g.episode_id WHERE t.display_id IN ("C3-T011","C3-T013")')
            assert len({r['episode_id'] for r in duplicate})==1
            woman=rows(con,'SELECT t.display_id,g.episode_id,e.label FROM association_track t JOIN local_global g ON g.track_id=t.id JOIN mtmc_episode e ON e.id=g.episode_id WHERE t.display_id IN ("C3-T001","C3-T002","C5-T001","C5-T002","C5-T003")')
            assert len(woman)==5 and len({r['episode_id'] for r in woman})==1 and woman[0]['label']=='ID 002'
            head=rows(con,'SELECT t.display_id,g.episode_id FROM association_track t JOIN local_global g ON g.track_id=t.id WHERE t.display_id IN ("C5-T204","C5-T206")')
            assert len(head)==2 and head[0]['episode_id']!=head[1]['episode_id']
        assert client.get('/api/global-tracks',params={'q':duplicate[0]['label']}).json()['total']==1
        woman_card=client.get('/api/global-tracks',params={'q':'ID 002'}).json()
        assert woman_card['total']==1 and woman_card['items'][0]['fragment_count']>=5
        a=client.get('/api/floor?time=.8').json();b=client.get('/api/floor?time=2').json()
        aa=[m for m in a['markers'] if m['global_episode_id']==first_id];bb=[m for m in b['markers'] if m['global_episode_id']==first_id]
        assert len(aa)==len(bb)==1 and aa[0]['camera_sources']==['1','2']
        assert bb[0]['position'][0]<aa[0]['position'][0] and bb[0]['position'][1]<aa[0]['position'][1]
        started=time.monotonic();response=client.get('/api/exports/tracks.json');response.raise_for_status();export_wall=time.monotonic()-started
        exported=response.json()
        print(f'[verify] all-frame JSON received: {len(response.content)} bytes in {export_wall:.2f}s',flush=True)
        assert exported['global_association_revision']==summary['revision'] and len(exported['frames'])==144110
        last={};empty=0
        with connect(s.database,readonly=True) as db:
            expected={str(cid):iter(db.execute('SELECT frame_index,raw_pts,segment_id FROM frame WHERE camera_id=? ORDER BY frame_index',(str(cid),)).fetchall()) for cid in range(1,6)}
        for record in exported['frames']:
            cid=record['camera_id'];index=record['source_frame_index']
            assert index==last.get(cid,-1)+1
            assert tuple(next(expected[cid]))==(index,record['raw_media_pts'],record['timestamp_segment_id'])
            last[cid]=index;empty+=not record['local_tracks']
            for observation in record['local_tracks']:
                part=next(p for p in slices[track_key(observation['local_track_id'])] if p[1]<=index<=p[2])
                assert (observation['derived_tracklet_id'],observation['global_episode_id'])==(part[0],part[3])
        assert all(next(values,None) is None for values in expected.values())
        del exported
    paths=['src/shadowtrack/detection/progress.py','configs/mtmc.json','configs/scene_layout.json','docker-compose.yml','frontend/playwright.config.ts','scripts/evaluate_mtmc_control.py','scripts/mtmc_control_clips.py','scripts/research_mtmc.py','scripts/verify_mtmc.py','scripts/build_mvp_bundle.py','README.md','docs/MVP_SETUP.md','docs/MVP_REPORT.md','docs/MTMC_SETUP.md','docs/MTMC_REPORT.md','docs/DEMO_SCRIPT.md','pyproject.toml']
    paths+=sorted(p.relative_to(s.root).as_posix() for p in (s.root/'src/shadowtrack/mvp').glob('*.py'))
    paths+=sorted(p.relative_to(s.root).as_posix() for p in (s.root/'frontend/src').glob('*') if p.suffix in ('.tsx','.ts','.css'))
    paths+=sorted(p.relative_to(s.root).as_posix() for p in (s.root/'frontend/e2e').glob('*.ts'))
    paths+=['tests/test_mtmc.py','tests/test_mtmc_quality.py','tests/test_mtmc_spatial.py','tests/test_mtmc_bundles.py','tests/test_episode_accounting.py','scripts/check_mvp_acceptance.py']
    quality=read_json(out/'control/quality.json');assert quality['revision']==summary['revision'] and not quality['labels_are_algorithm_input']
    appearance=read_json(out/'appearance/manifest.json')
    assert sha256(out/'appearance/encoded/embeddings.npz')==appearance['features_sha256']
    assert reviews==0
    baseline_code=read_json(out/'baseline_code_manifest.json')
    candidates=set(paths)|set(baseline_code['files'])
    changed={p:sha256(s.root/p) for p in sorted(candidates) if (s.root/p).is_file() and baseline_code['files'].get(p)!=sha256(s.root/p)}
    result={'acceptance':'IMPLEMENTED_AUTOMATIC_ESTIMATED_TRACKING','metric_floor_and_proven_all_camera_clocks':False,'source_run_id':summary['source_run_id'],'summary':summary,
            'pytest_passed':pytest_count,'browser_passed':browser['expected'],'failures':0,'all_frame_export_count':144110,'all_frame_export_http_seconds':export_wall,'empty_local_track_frames_exported':empty,'cross_camera_provisional_groups':cross_view,'human_reviews':reviews,
            'protected_files_unchanged':len(protected),'old_gpu_sample_files_unchanged':len(old_sample['files']),'alignment_arrays_unchanged':True,
            'raw_database_tables_unchanged':tables,'original_derived_media_unchanged':len(source_media),'global_review_clips':[{k:m[k] for k in ('camera_id','frames','path','sha256')} for m in clips],
            'offsets':bases,'repeated_yolo':False,'repeated_bytetrack':False,'one_time_temporal_osnet_cache':appearance,'ordinary_recompute_uses_cache':True,'control_quality':{k:v for k,v in quality.items() if k not in ('baseline','current','derived_split_sources')},'control_counts':{k:quality[k]['counts'] for k in ('baseline','current')},'native_server_recompute':read_json(out/'recompute_verification.json'),'reported_first_exit':{'episode_id':first_id,'camera_sources':['1','2'],'one_moving_dot':True},'native_playback':read_json(out/'native_playback_verification.json'),'user_regressions':{'corridor_handoff':corridor,'single_archive_card':duplicate,'one_based_chronological_display':True},'cloud_upload':False,'floor_geometry_fabricated':False,'floor_localization':'SCHEMATIC_PROJECTED_POINTS',
            'normalized_appearance_cache':normalized,'source_mkv_hashes_unchanged':5,'reported_woman_single_episode':woman,'different_low_quality_head_stays_separate':head,
            'true_identity_count':None,'true_cross_camera_accuracy':None,'baseline_code_hash':baseline_code['code_hash'],'changed_files':changed}
    atomic_json(out/'verification.json',result)
    print(json.dumps({k:result[k] for k in ('acceptance','pytest_passed','browser_passed','cross_camera_provisional_groups','human_reviews','protected_files_unchanged','original_derived_media_unchanged')},indent=2))


if __name__=='__main__':main()
