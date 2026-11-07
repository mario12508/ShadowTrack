"""Read derived MP4 packet PTS and counts; do not re-probe original MKV."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from shadowtrack.probe import sha256
from shadowtrack.detection.common import atomic_json,read_json,object_hash
from shadowtrack.detection.evidence import load_evidence
from shadowtrack.mvp.config import Settings


def main():
    settings=Settings.environment();evidence=load_evidence(settings.evidence)
    ffprobe=os.environ.get('SHADOWTRACK_FFPROBE') or shutil.which('ffprobe') or str(settings.root/'.tools/ffprobe.exe')
    sources=[read_json(p) for p in (settings.media/'source').glob('camera_*_source.json')]
    local=read_json(settings.media/'local/videos.json')
    assert local['status']=='COMPLETE' and local['frames']==144110
    result=[]
    for kind,items in [('source',sources),('local',local['videos'])]:
        for item in items:
            camera=evidence['cameras'][item['camera_id']]
            folder=settings.media/kind;path=folder/item['path']
            assert sha256(path)==item['sha256']
            raw=subprocess.run([ffprobe,'-v','error','-select_streams','v:0','-show_packets','-show_entries',
                'packet=pts_time:stream=codec_name,width,height,time_base,start_time,duration,nb_frames','-of','json',str(path)],capture_output=True,text=True,check=True,timeout=120)
            data=json.loads(raw.stdout);pts=sorted(float(p['pts_time']) for p in data['packets'])
            if kind=='source':expected=[r[1]-camera['frames'][0][1] for r in camera['frames']]
            else:
                sidecar=path.with_suffix('.frames.jsonl')
                assert sha256(sidecar)==item['sidecar_sha256']
                rows=[json.loads(line) for line in sidecar.read_text(encoding='utf-8').splitlines()]
                expected=[r['output_relative_seconds'] for r in rows]
                assert all([r['source_frame_index'],r['raw_media_pts'],r['timestamp_segment_id']]==camera['frames'][r['source_frame_index']] for r in rows)
                assert [r['source_frame_index'] for r in rows]==list(range(item['frame_start'],item['frame_stop']))
            assert len(pts)==len(expected)==item['frames'],f'{path.name}: packet/frame count mismatch'
            error=max((abs(a-b) for a,b in zip(pts,expected)),default=0.)
            assert error<=.0000011,f'{path.name}: output/source PTS mismatch {error}'
            assert data['streams'][0]['codec_name']=='h264'
            result.append({'camera_id':item['camera_id'],'kind':kind,'path':item['path'],'frames':len(pts),'max_pts_error_seconds':error,
                'source_indices_unchanged':True,'stream':data['streams'][0],'sha256':item['sha256']})
            print('[MVP verify]',kind,item['camera_id'],item['path'],len(pts),'PTS PASS',flush=True)
    for cid,camera in evidence['cameras'].items():
        clips=sorted((v for v in local['videos'] if v['camera_id']==cid),key=lambda v:v['frame_start'])
        indices=[i for c in clips for i in range(c['frame_start'],c['frame_stop'])]
        assert indices==list(range(camera['frame_count'])),'Missing/duplicate frames between rendered segments'
    atomic_json(settings.artifacts/'media_verification.json',{'status':'PASS','source_cameras':5,'annotated_cameras':5,'annotated_segments':len(local['videos']),
        'source_frames':144110,'annotated_frames':local['frames'],'all_output_packets_match_audited_frame_pts':True,
        'original_mkv_probe_repeated':False,'global_ids_claimed':False,'files':result})
    print('PASS: five full cameras, all 144110 frames, source and annotated PTS preserved.')


if __name__=='__main__':main()
