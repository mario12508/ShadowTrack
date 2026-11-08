"""Short private candidate previews with revision-bound global overlays; no YOLO."""
import json
from pathlib import Path

import av

from ..detection.common import atomic_json
from .db import connect, rows
from .mtmc import identity
from .render import RenderVideo, paint, register_media


def review_highlight(settings,episode_id,seconds=6):
    if not 1<=seconds<=15:raise ValueError('Review preview is limited to 1–15 seconds per camera')
    with connect(settings.database,readonly=True) as db:
        episode=db.execute('SELECT * FROM mtmc_episode WHERE id=?',(episode_id,)).fetchone()
        if not episode or episode['status']=='UNRESOLVED':raise ValueError('Select an actual linked episode')
        episode=dict(episode)
        members=rows(db,'SELECT t.*,c.width,c.height,c.fps,c.first_pts,c.source_name FROM association_track t JOIN local_global g ON g.track_id=t.id JOIN camera c ON c.id=t.camera_id WHERE g.episode_id=? ORDER BY t.start_time+c.first_pts',(episode_id,))
        labels={t['id']:t['display_id'] for t in members};identities={t['id']:identity(db,t['id']) for t in members}
        selected=[];seen=set()
        for t in members:
            if t['camera_id'] not in seen:selected.append(t);seen.add(t['camera_id'])
        results=[];output=settings.media/'global_review'/episode['revision']/episode_id;output.mkdir(parents=True,exist_ok=True)
        for t in selected:
            cid=t['camera_id'];start=max(t['start_time'],t['end_time']-seconds) if cid=='4' else t['start_time'];end=min(t['end_time'],start+seconds)
            frames=rows(db,'SELECT * FROM frame WHERE camera_id=? AND segment_id=? AND media_time BETWEEN ? AND ? ORDER BY frame_index',(cid,t['segment_id'],start,end))
            if not frames:continue
            path=output/f"camera_{cid}_{frames[0]['frame_index']:08d}.mp4"
            camera={'camera_id':cid,'width':t['width'],'height':t['height'],'nominal_fps':t['fps'],'frames':[[0,t['first_pts']]]}
            writer=RenderVideo(path,camera,960);count=0
            try:
                with av.open(str(settings.raw/t['source_name'])) as container:
                    stream=container.streams.video[0];stream.codec_context.thread_count=2
                    container.seek(round(frames[0]['raw_pts']/float(stream.time_base)),stream=stream,backward=True)
                    for decoded in container.decode(stream):
                        raw=float(decoded.pts*decoded.time_base)
                        if raw<frames[0]['raw_pts']-1e-6:continue
                        if count==len(frames):break
                        f=frames[count]
                        if abs(raw-f['raw_pts'])>1e-6:raise ValueError('Preview decoder differs from saved frame PTS')
                        observations=rows(db,'SELECT o.*,t.source_id FROM observation o JOIN local_track t ON t.id=o.track_id WHERE o.camera_id=? AND o.frame_index=?',(cid,f['frame_index']))
                        row={'camera_id':cid,'source_frame_index':f['frame_index'],'raw_media_pts':f['raw_pts'],'timestamp_segment_id':f['segment_id'],
                             'local_tracks':[{'local_track_id':o['source_id'],'bbox_xyxy':json.loads(o['bbox']),'confidence':o['confidence']} for o in observations]}
                        rgb=decoded.reformat(width=writer.width,height=writer.height,format='rgb24').to_ndarray()
                        writer.add(paint(rgb,row,labels,writer.width/t['width'],identities=identities),row);count+=1
                if count!=len(frames):raise ValueError('Incomplete preview source coverage')
                item=writer.close();writer=None
                item.update(kind='global_review',identity_status=episode['status'],global_episode_id=episode_id,global_label=episode['label'],revision=episode['revision'],
                            selected_source_track=t['id'],description='Separate candidate clip, not a confirmed continuous cross-camera passage')
                subdir=path.parent.relative_to(settings.media);register_media(settings,item,subdir);results.append(item)
                print(f'[MTMC review] {episode_id} camera={cid} frames={count} {episode["status"]}',flush=True)
            finally:
                if writer is not None:writer.container.close();writer.side.close()
        manifest={'schema_version':1,'episode':episode,'clips':results,'human_reviewed':False,'identity_accuracy_claim':None,'source_pts_unchanged':True}
        atomic_json(output/'review_manifest.json',manifest);return manifest
