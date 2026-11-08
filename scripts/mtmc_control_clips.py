"""Extract private temporal evidence for an independent 20-pair control review.

No identity labels are input to the association algorithm. Existing source MP4s
are decoded, never changed. Samples retain exact audited frame/PTS provenance.
"""
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw

from shadowtrack.detection.common import atomic_json,read_json
from shadowtrack.mvp.config import Settings
from shadowtrack.mvp.db import connect,rows
from shadowtrack.probe import sha256


def main():
    s=Settings.environment();out=s.artifacts/'mtmc/control';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Control selection is frozen; reuse its manifest instead of resampling after algorithm changes')
    with connect(s.database,readonly=True) as db:
        all_pairs=rows(db,"SELECT p.*,a.camera_id ca,b.camera_id cb,a.display_id la,b.display_id lb FROM mtmc_candidate p JOIN local_track a ON a.id=p.track_a JOIN local_track b ON b.id=p.track_b WHERE a.camera_id<>b.camera_id AND a.camera_id IN ('3','4','5') AND b.camera_id IN ('3','4','5') ORDER BY p.similarity DESC")
        # Systematic selection before review: strong accepted, strongest eligible
        # rejects, and temporal-conflict negatives. Not a random test-population.
        selected=[];used=set()
        buckets=[('accepted',lambda p:p['status']=='PROVISIONAL',10),('contested',lambda p:p['status']=='UNRESOLVED' and abs(json.loads(p['evidence'])['gap_seconds'])<50,5),('time_conflict',lambda p:p['status']=='REJECTED' and p['similarity']>.85,5)]
        for kind,rule,count in buckets:
            options=[p for p in all_pairs if rule(p) and p['id'] not in used]
            # Prefer distinct endpoint tracks; repeated appearance is still
            # deliberately included in the time-conflict bucket as a hard case.
            seen=set()
            for p in options:
                if kind!='time_conflict' and (p['track_a'] in seen or p['track_b'] in seen):continue
                selected.append({**p,'sampling_bucket':kind});used.add(p['id']);seen.update((p['track_a'],p['track_b']))
                if len([p for p in selected if p['sampling_bucket']==kind])==count:break
        if len(selected)!=20:raise ValueError(f'Need 20 systematic candidates, got {len(selected)}')
        for i,p in enumerate(selected):
            p['control_id']=f'C{i+1:02d}';p['split']='development' if i%2==0 else 'holdout'
            sheet=Image.new('RGB',(1440,740),'#11170f');draw=ImageDraw.Draw(sheet)
            draw.text((12,8),f"{p['control_id']} {p['la']} <-> {p['lb']} original={p['status']} cos={p['similarity']:.4f} UNREVIEWED",fill='white')
            p['sequences']=[]
            for row,key in enumerate((p['track_a'],p['track_b'])):
                t=dict(db.execute('SELECT t.*,c.first_pts,c.source_sha256,c.width,c.height FROM local_track t JOIN camera c ON c.id=t.camera_id WHERE t.id=?',(key,)).fetchone())
                obs=rows(db,'SELECT * FROM observation WHERE track_id=? ORDER BY frame_index',(key,))
                # Six frames across the local fragment (bounded excerpt), plus
                # beginning/end context. Full crop sequences are linked by ID.
                chosen=[]
                for q in (0,.2,.4,.6,.8,1):
                    item=obs[round((len(obs)-1)*q)]
                    if item['frame_index'] not in {o['frame_index'] for o in chosen}:chosen.append(item)
                media=db.execute('SELECT path FROM media_segment WHERE kind="source" AND camera_id=?',(t['camera_id'],)).fetchone()[0]
                sequence=[]
                with av.open(str(s.media/media)) as container:
                    stream=container.streams.video[0];stream.codec_context.thread_count=1
                    for col,o in enumerate(chosen):
                        container.seek(round(o['media_time']/float(stream.time_base)),stream=stream,backward=True)
                        found=None
                        for frame in container.decode(stream):
                            pts=float(frame.pts*frame.time_base)
                            if abs(pts-o['media_time'])<1e-5:found=frame;break
                            if pts>o['media_time']+.001:break
                        if found is None:raise ValueError('Control frame not at exact audited PTS')
                        image=found.to_image();sx=image.width/t['width'];box=[v*sx for v in json.loads(o['bbox'])]
                        crop=image.crop(tuple(round(v) for v in box));crop.thumbnail((155,175))
                        full=image.resize((234,176));paint=ImageDraw.Draw(full);b=[v*234/t['width'] for v in json.loads(o['bbox'])];paint.rectangle(b,outline='#d4ff38',width=2)
                        x=col*240;y=40+row*345;sheet.paste(full,(x,y));sheet.paste(crop,(x,y+184))
                        draw.text((x+3,y+166),f"{t['display_id']} raw={o['raw_pts']:.3f}",fill='white')
                        filename=f"{p['control_id']}_{t['camera_id']}_{o['frame_index']}.jpg";image.thumbnail((960,720));image.save(out/filename,quality=88)
                        sequence.append({'path':filename,'sha256':sha256(out/filename),'camera_id':t['camera_id'],'source_frame_index':o['frame_index'],'raw_media_pts':o['raw_pts'],'source_video_sha256':t['source_sha256'],'bbox_xyxy':json.loads(o['bbox'])})
                p['sequences'].append({'track_id':key,'camera_id':t['camera_id'],'media_start':t['start_time'],'media_end':t['end_time'],'samples':sequence})
            sheet.save(out/f"{p['control_id']}.jpg",quality=91)
            print(f"[control] {p['control_id']} {p['la']} {p['lb']} {p['status']} extracted",flush=True)
    atomic_json(out/'manifest.json',{'schema_version':1,'purpose':'independent pairwise visual control, not algorithm input','selection':'10 accepted, 5 contested, 5 temporal-conflict hard negatives; stratified nonrandom','cases':selected,'reviewed':False})


if __name__=='__main__':main()
