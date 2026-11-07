"""Short presentation clips decoded from verified local-ID MP4, without inference."""
import json
from pathlib import Path

import av

from ..detection.common import atomic_json,read_json
from ..detection.evidence import load_evidence
from ..probe import sha256
from .render import RenderVideo,register_media


def make_highlights(settings):
    evidence=load_evidence(settings.evidence);local=read_json(settings.media/'local/videos.json')
    output=settings.media/'highlights';output.mkdir(parents=True,exist_ok=True);result=[]
    for cid,start,seconds,name in [('1',0.,6.,'camera_1_walk'),('3',129.,16.,'camera_3_doorway'),('4',670.,16.,'camera_4_corridor')]:
        camera=evidence['cameras'][cid];first_pts=camera['frames'][0][1]
        selected=[i for i,pts,sid in camera['frames'] if start<=pts-first_pts<start+seconds]
        first,stop=selected[0],selected[-1]+1
        parent=next(v for v in local['videos'] if v['camera_id']==cid and v['frame_start']<=first<v['frame_stop'])
        stop=min(stop,parent['frame_stop'])  # Never cross a timestamp segment.
        path=settings.media/'local'/parent['path']
        if sha256(path)!=parent['sha256']:raise ValueError('Highlight source video checksum mismatch')
        side=path.with_suffix('.frames.jsonl')
        if sha256(side)!=parent['sidecar_sha256']:raise ValueError('Highlight source sidecar checksum mismatch')
        lookup={round(r['output_relative_seconds']*1e6):r for r in (json.loads(line) for line in side.read_text(encoding='utf-8').splitlines())}
        writer=RenderVideo(output/(name+'.mp4'),camera,parent['width'])
        try:
            with av.open(str(path)) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=2
                relative=camera['frames'][first][1]-camera['frames'][parent['frame_start']][1]
                container.seek(round(relative/float(stream.time_base)),stream=stream,backward=True)
                for frame in container.decode(stream):
                    key=round(float(frame.pts*frame.time_base)*1e6)
                    if key not in lookup:raise ValueError('Rendered MP4 frame has no source sidecar mapping')
                    row=lookup[key];i=row['source_frame_index']
                    if i<first:continue
                    if i>=stop:break
                    writer.add(frame.to_ndarray(format='rgb24'),row)
            if writer.frames!=stop-first:raise ValueError('Highlight frame count mismatch')
            item=writer.close();writer=None
        finally:
            if writer is not None:writer.container.close();writer.side.close()
        item.update(kind='highlight',parent_sha256=parent['sha256'],inference_repeated=False)
        atomic_json(output/(name+'.json'),item);register_media(settings,item,'highlights');result.append(item)
        print(f'[MVP] highlight {name} frames={item["frames"]}',flush=True)
    atomic_json(output/'highlights.json',{'status':'COMPLETE','videos':result,'time_basis':'each source relative raw PTS; scenes are not claimed simultaneous'})
    return result
