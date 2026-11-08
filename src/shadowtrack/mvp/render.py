"""CPU render, two bounded workers. No detector or embedding model is imported."""
import colorsys
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections import defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from ..probe import sha256
from ..detection.common import atomic_json,canonical,read_json
from ..detection.decoder import checked_frames
from ..detection.evidence import load_evidence
from .config import Settings
from .db import connect,rows
from .prepare import track_key


def color_for(identity):
    hue=int(hashlib.sha256(identity.encode()).hexdigest()[:6],16)/0xffffff
    return tuple(round(v*255) for v in colorsys.hsv_to_rgb(hue,.45,.94))


def paint(rgb,row,labels,scale,trails=None,identities=None):
    image=np.ascontiguousarray(rgb.copy());h,w=image.shape[:2]
    if trails is not None:
        alive={t['local_track_id'] for t in row['local_tracks']}
        for key in list(trails):
            if key not in alive:trails.pop(key,None)
    for t in row['local_tracks']:
        key=t['local_track_id'];identity=(identities or {}).get(track_key(key),{})
        color=color_for(identity.get('global_episode_id') or key)
        x1,y1,x2,y2=[round(v*scale) for v in t['bbox_xyxy']]
        cv2.rectangle(image,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)
        if trails is not None:
            trail=trails.setdefault(key,deque(maxlen=18));trail.append(((x1+x2)//2,y2))
            for j,point in enumerate(trail):cv2.circle(image,point,1+int(j>len(trail)*.7),color,-1,cv2.LINE_AA)
        identity_label=identity.get('global_label')
        if identity_label and identity.get('global_status')=='PROVISIONAL':identity_label+=' ?'
        label=f"{identity_label or labels.get(track_key(key),key.split('/')[-1])}  {t['confidence']:.2f}"
        size=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.43,1)[0]
        y=max(size[1]+10,y1-5);right=min(w-1,x1+size[0]+12)
        cv2.rectangle(image,(max(0,x1),y-size[1]-8),(right,y+4),(20,24,20),-1)
        cv2.putText(image,label,(max(0,x1)+5,y),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
    cv2.rectangle(image,(0,0),(w,34),(17,21,15),-1)
    status='CONFIRMED' if identities and all(v.get('global_status')=='CONFIRMED' for v in identities.values()) else 'PROVISIONAL'
    kind='GLOBAL REVIEW / '+status if identities is not None else 'LOCAL TRACKING'
    cv2.putText(image,f"SHADOWTRACK / CAM {row['camera_id']} / {kind}",(12,23),cv2.FONT_HERSHEY_SIMPLEX,.45,(211,255,56),1,cv2.LINE_AA)
    pts=f"raw PTS {row['raw_media_pts']:.3f}s"
    cv2.putText(image,pts,(max(12,w-190),23),cv2.FONT_HERSHEY_SIMPLEX,.43,(225,230,221),1,cv2.LINE_AA)
    return image


class RenderVideo:
    def __init__(self,path,camera,width=960):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.pending=self.path.with_suffix('.pending.mp4')
        self.container=av.open(str(self.pending),'w')
        self.stream=self.container.add_stream('libx264',rate=Fraction(str(camera['nominal_fps'])))
        self.width=width;self.height=round(camera['height']*width/camera['width']/2)*2
        self.stream.width,self.stream.height=width,self.height;self.stream.pix_fmt='yuv420p'
        self.base=Fraction(1,1_000_000);self.stream.time_base=self.stream.codec_context.time_base=self.base
        self.stream.codec_context.thread_count=2
        self.stream.options={'preset':'ultrafast','crf':'23','bf':'0'}
        self.frames=0;self.first_pts=None;self.last_index=None;self.sid=None
        self.sidecar=self.path.with_suffix('.frames.jsonl');self.side_pending=self.sidecar.with_suffix('.pending')
        self.side=self.side_pending.open('wb');self.camera=camera

    def add(self,image,row):
        pts=Fraction(str(row['raw_media_pts']))
        if self.last_index is not None and (row['source_frame_index']!=self.last_index+1 or row['timestamp_segment_id']!=self.sid):raise ValueError('Cannot bridge missing frames or timestamp segments')
        if self.first_pts is None:self.first_pts=pts
        frame=av.VideoFrame.from_ndarray(image,format='rgb24');frame.pts=round((pts-self.first_pts)/self.base);frame.time_base=self.base
        for packet in self.stream.encode(frame):self.container.mux(packet)
        self.side.write(canonical({'output_frame_index':self.frames,'source_frame_index':row['source_frame_index'],'raw_media_pts':row['raw_media_pts'],
            'timestamp_segment_id':row['timestamp_segment_id'],'camera_id':row['camera_id'],'output_relative_seconds':float(pts-self.first_pts)})+b'\n')
        self.frames+=1;self.last_index=row['source_frame_index'];self.sid=row['timestamp_segment_id'];self.last_pts=pts

    def close(self):
        try:
            for packet in self.stream.encode():self.container.mux(packet)
        finally:
            self.container.close();self.side.close()
        os.replace(self.pending,self.path);os.replace(self.side_pending,self.sidecar)
        return {'camera_id':self.camera['camera_id'],'segment_id':self.sid,'kind':'local','path':self.path.name,'sha256':sha256(self.path),
            'frames':self.frames,'width':self.width,'height':self.height,'frame_start':self.last_index-self.frames+1,'frame_stop':self.last_index+1,
            'start_time':float(self.first_pts)-self.camera['frames'][0][1],'end_time':float(self.last_pts)-self.camera['frames'][0][1],
            'sidecar_sha256':sha256(self.sidecar),'time_policy':'original source PTS rebased inside this segment; no synchronized global time'}


def register_media(settings,item,subdir):
    relative=str(Path(subdir)/item['path']).replace('\\','/')
    key=hashlib.sha256(relative.encode()).hexdigest()[:20]
    with connect(settings.database) as db:
        db.execute('INSERT OR REPLACE INTO media_segment VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(key,item['camera_id'],item.get('segment_id'),item['kind'],relative,item['sha256'],
            item['start_time'],item['end_time'],item['frame_start'],item['frame_stop'],item['frames'],item['width'],item['height'],json.dumps(item)))
    return key


def browser_sources(settings):
    """H264 originals remux exactly; the HEVC source is encoded to browser H264."""
    evidence=load_evidence(settings.evidence)
    ffmpeg=os.environ.get('SHADOWTRACK_FFMPEG') or shutil.which('ffmpeg') or str(settings.root/'.tools/ffmpeg.exe')
    output=settings.media/'source';output.mkdir(parents=True,exist_ok=True)
    result=[]
    for cid,camera in evidence['cameras'].items():
        target=output/f'camera_{cid}_source.mp4';side=target.with_suffix('.json')
        if target.exists() and side.exists():
            item=read_json(side)
            if sha256(target)==item['sha256'] and (cid!='4' or item.get('encoder_time_base')=='1/1000000'):
                register_media(settings,item,'source');result.append(item);continue
        pending=target.with_suffix('.pending.mp4')
        cmd=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-copyts','-itsoffset',str(-camera['frames'][0][1]),'-i',str(settings.raw/camera['source_name']),'-map','0:v:0','-an']
        if cid=='4':cmd+=['-vf','scale=1280:-2','-c:v','libx264','-preset','ultrafast','-crf','23','-threads','2','-bf','0','-fps_mode','passthrough','-enc_time_base','1:1000000']
        else:cmd+=['-c:v','copy']
        cmd+=['-movflags','+faststart','-video_track_timescale','1000000','-y',str(pending)]
        print(f'[MVP] browser source camera={cid} start',flush=True)
        subprocess.run(cmd,check=True,capture_output=True,timeout=1800)
        with av.open(str(pending)) as container:
            first=next(container.decode(video=0))
            if abs(float(first.pts*first.time_base))>.003:raise ValueError(f'Browser source PTS zero mismatch camera {cid}')
        os.replace(pending,target)
        width=1280 if cid=='4' else camera['width'];height=round(camera['height']*width/camera['width']/2)*2
        item={'camera_id':cid,'segment_id':None,'kind':'source','path':target.name,'sha256':sha256(target),
            'start_time':0.,'end_time':camera['frames'][-1][1]-camera['frames'][0][1],'frame_start':0,'frame_stop':camera['frame_count'],
            'frames':camera['frame_count'],'width':width,'height':height,'encoder_time_base':'1/1000000' if cid=='4' else 'stream_copy',
            'time_policy':'raw source PTS minus first decoded PTS; discontinuities retained; overlays served separately'}
        atomic_json(side,item);register_media(settings,item,'source');result.append(item)
        print(f'[MVP] browser source camera={cid} ready',flush=True)
    return result


def render_all(settings: Settings,width=960,workers=2):
    cv2.setNumThreads(1)
    evidence=load_evidence(settings.evidence)
    track_dir=settings.artifacts/'local_tracks';manifest=read_json(track_dir/'manifest.json')
    with connect(settings.database,readonly=True) as db:labels={r['id']:r['display_id'] for r in db.execute('SELECT id,display_id FROM local_track')}
    output=settings.media/'local';output.mkdir(parents=True,exist_ok=True)
    progress={};lock=threading.Lock();all_results=[];started=time.monotonic()
    def camera_render(entry):
        cid=entry['camera_id'];camera=evidence['cameras'][cid];camera_side=output/f'camera_{cid}_manifest.json'
        if camera_side.exists():
            saved=read_json(camera_side)
            if saved.get('track_sha256')==entry['sha256'] and all((output/v['path']).exists() and sha256(output/v['path'])==v['sha256'] for v in saved['videos']):
                for item in saved['videos']:register_media(settings,item,'local')
                print(f'[MVP] render camera={cid} reused',flush=True);return saved['videos']
        writer=None;results=[];trails={}
        try:
            with gzip.open(track_dir/entry['path'],'rt',encoding='utf-8') as stream:
                for i,frame in checked_frames(settings.raw/camera['source_name'],camera,settings.artifacts/f'render_camera_{cid}_diagnostic.json'):
                    line=stream.readline()
                    if not line:raise ValueError('Missing track row')
                    row=json.loads(line)
                    if [row['source_frame_index'],row['raw_media_pts'],row['timestamp_segment_id']]!=camera['frames'][i]:raise ValueError('Render frame/PTS mismatch')
                    if writer is not None and writer.sid!=row['timestamp_segment_id']:
                        item=writer.close();results.append(item);register_media(settings,item,'local');writer=None;trails.clear()
                    if writer is None:writer=RenderVideo(output/f"camera_{cid}_local_{i:08d}.mp4",camera,width)
                    rgb=frame.reformat(width=writer.width,height=writer.height,format='rgb24').to_ndarray()
                    writer.add(paint(rgb,row,labels,width/camera['width'],trails),row)
                    if i%250==0:
                        with lock:
                            progress[cid]={'frames':i+1,'total':camera['frame_count'],'status':'RUNNING'}
                            atomic_json(settings.artifacts/'render_progress.json',{'cameras':progress,'wall_seconds':time.monotonic()-started})
                        if i%2000==0:print(f'[MVP] render camera={cid} frame={i+1}/{camera["frame_count"]}',flush=True)
                if stream.readline():raise ValueError('Extra track row')
            if writer is not None:
                item=writer.close();results.append(item);register_media(settings,item,'local');writer=None
            if sum(v['frames'] for v in results)!=camera['frame_count']:raise ValueError('Rendered frame coverage incomplete')
            atomic_json(camera_side,{'status':'COMPLETE','source_sha256':camera['source_sha256'],'track_sha256':entry['sha256'],'videos':results})
            with lock:
                progress[cid]={'frames':camera['frame_count'],'total':camera['frame_count'],'status':'COMPLETE'}
                atomic_json(settings.artifacts/'render_progress.json',{'cameras':progress,'wall_seconds':time.monotonic()-started})
            print(f'[MVP] render camera={cid} COMPLETE',flush=True)
            return results
        finally:
            if writer is not None:
                writer.container.close();writer.side.close()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for results in pool.map(camera_render,manifest['files']):all_results.extend(results)
    result={'status':'COMPLETE','frames':sum(v['frames'] for v in all_results),'videos':all_results,'global_ids':False,'wall_seconds':time.monotonic()-started}
    atomic_json(output/'videos.json',result)
    return result
