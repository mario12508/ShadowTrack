"""Derived annotated MP4. Original PTS live in records and a per-video sidecar."""

from fractions import Fraction
import gzip
import os
from pathlib import Path

import numpy as np

from ..probe import sha256
from .common import DetectionError,atomic_json,canonical,read_json
from .decoder import checked_frames
from .progress import observed


def annotate(rgb, row, show_all=False):
    import cv2
    image = np.ascontiguousarray(rgb.copy())
    local = "local_tracks" in row
    boxes = row["local_tracks"] if local else [d for d in row["detections"] if show_all or d.get("scene_filter",{}).get("decision","accepted") == "accepted"]
    for d in boxes:
        x1,y1,x2,y2 = [int(v) for v in d["bbox_xyxy"]]
        label = f"person {d['confidence']:.2f}"
        if local:
            label = f"LOCAL {d['local_track_id'].split('/')[-1]} {d['confidence']:.2f}"
        decision = d.get("scene_filter",{}).get("decision","accepted")
        color = (40,230,90) if decision == "accepted" else (240,70,70) if decision == "rejected" else (255,175,30)
        if show_all and not local:
            label += " "+decision
        cv2.rectangle(image,(x1,y1),(x2,y2),color,max(2,image.shape[1]//600))
        cv2.putText(image,label,(x1,max(24,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,220,30),2,cv2.LINE_AA)
    header = f"CAM {row['camera_id']} | frame {row['source_frame_index']} | raw PTS {row['raw_media_pts']:.3f} | {row['timestamp_segment_id']}"
    cv2.rectangle(image,(0,0),(min(image.shape[1]-1,1600),42),(0,0,0),-1)
    cv2.putText(image,header,(8,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
    return image


class AnnotatedVideo:
    def __init__(self, path, camera):
        import av
        self.av,self.path,self.camera = av,Path(path),camera
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.temp = self.path.with_name(self.path.stem+".pending.mp4")
        self.sidecar = self.path.with_suffix(".frames.jsonl")
        self.side_temp = self.sidecar.with_suffix(".jsonl.pending")
        self.container = av.open(str(self.temp),mode="w",format="mp4")
        self.stream = self.container.add_stream("libx264",rate=Fraction(str(camera["nominal_fps"])))
        self.stream.width = camera["width"]+(camera["width"]%2)
        self.stream.height = camera["height"]+(camera["height"]%2)
        self.stream.pix_fmt = "yuv420p"
        self.time_base = Fraction(1,1_000_000)
        self.stream.time_base = self.stream.codec_context.time_base = self.time_base
        self.stream.options = {"preset":"veryfast","crf":"23","bf":"0"}
        self.output = self.side_temp.open("wb")
        self.count,self.first,self.last = 0,None,None
        self.local_ids = set()

    def add(self, rgb, row):
        if row["camera_id"] != self.camera["camera_id"] or rgb.shape[:2] != (self.camera["height"],self.camera["width"]):
            raise DetectionError("Annotated video source dimensions/camera mismatch")
        pts = Fraction(str(row["raw_media_pts"]))
        if self.last and (row["source_frame_index"] != self.last["source_frame_index"]+1
                         or row["timestamp_segment_id"] != self.last["timestamp_segment_id"]
                         or pts <= Fraction(str(self.last["raw_media_pts"]))):
            raise DetectionError("Video cannot concatenate segments, gaps or non-increasing PTS")
        if self.first is None:
            self.first = pts
        painted = annotate(rgb,row)
        if painted.shape[:2] != (self.stream.height,self.stream.width):
            padded = np.zeros((self.stream.height,self.stream.width,3),dtype=np.uint8)
            padded[:painted.shape[0],:painted.shape[1]] = painted
            painted = padded
        frame = self.av.VideoFrame.from_ndarray(painted,format="rgb24")
        frame.pts,frame.time_base = round((pts-self.first)/self.time_base),self.time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        ids = [d["local_track_id"] for d in row.get("local_tracks",[])]
        self.local_ids.update(ids)
        self.output.write(canonical({"output_frame_index":self.count,"camera_id":row["camera_id"],
            "source_frame_index":row["source_frame_index"],"raw_media_pts":row["raw_media_pts"],
            "timestamp_segment_id":row["timestamp_segment_id"],"output_relative_seconds":float(pts-self.first),
            "local_track_ids":ids,"global_time_status":"NOT_REQUESTED"})+b"\n")
        self.last = row
        self.count += 1

    def close(self, commit=True):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
            self.output.flush()
            os.fsync(self.output.fileno())
            self.output.close()
        if commit and self.count:
            os.replace(self.temp,self.path)
            os.replace(self.side_temp,self.sidecar)
            return {"camera_id":self.camera["camera_id"],"path":self.path.name,"sha256":sha256(self.path),
                    "sidecar":self.sidecar.name,"sidecar_sha256":sha256(self.sidecar),"frames":self.count,
                    "local_id_count":len(self.local_ids),"time_policy":"source raw PTS rebased for display; not synchronized global time"}
        self.temp.unlink(missing_ok=True)
        self.side_temp.unlink(missing_ok=True)
        return None


@observed("annotate_local_tracks")
def render_local_tracks(track_dir, evidence, raw_dir, output_dir, progress=None):
    """CPU-only replay; reads immutable detections indirectly through saved track rows."""
    track_dir,output_dir = Path(track_dir),Path(output_dir).resolve()
    for protected in (Path(raw_dir).resolve(),evidence["directory"].resolve(),track_dir.resolve()):
        if output_dir.is_relative_to(protected):
            raise DetectionError("Derived videos must be outside source/evidence/track directories")
    track_manifest = read_json(track_dir/"manifest.json")
    results = []
    for entry in track_manifest["files"]:
        camera = evidence["cameras"][entry["camera_id"]]
        source = track_dir/entry["path"]
        if sha256(source) != entry["sha256"]:
            raise DetectionError("Track file SHA256 mismatch")
        progress.update("annotate_camera",camera_id=camera["camera_id"])
        writer = None
        try:
            with gzip.open(source,"rt",encoding="utf-8") as stream:
                rows = iter(stream)
                row = __import__("json").loads(next(rows)) if entry["processed_frames"] else None
                consumed = 0
                for i,frame in checked_frames(Path(raw_dir)/camera["source_name"],camera,
                        output_dir/f"camera_{camera['camera_id']}_diagnostic.json",progress=progress):
                    if row is None or i != row["source_frame_index"]:
                        continue
                    if row["raw_media_pts"] != camera["frames"][i][1] or row["timestamp_segment_id"] != camera["frames"][i][2]:
                        raise DetectionError("Track row differs from source evidence")
                    boundary = writer is not None and (i != writer.last["source_frame_index"]+1 or row["timestamp_segment_id"] != writer.last["timestamp_segment_id"]
                                                       or row["raw_media_pts"] <= writer.last["raw_media_pts"])
                    if boundary:
                        results.append(writer.close());writer = None
                    if writer is None:
                        writer = AnnotatedVideo(output_dir/f"camera_{camera['camera_id']}_local_{i:08d}.mp4",camera)
                    writer.add(frame.to_ndarray(format="rgb24"),row)
                    consumed += 1
                    progress.update(processed_frames=consumed)
                    line = next(rows,None)
                    row = __import__("json").loads(line) if line is not None else None
                if row is not None or consumed != entry["processed_frames"]:
                    raise DetectionError("Track/video frame coverage mismatch")
            if writer is not None:
                results.append(writer.close());writer = None
        finally:
            if writer is not None:
                writer.close(commit=False)
    manifest = {"schema_version":1,"status":"COMPLETE","source_run_id":track_manifest["source_run_id"],
                "global_association":False,"videos":results}
    atomic_json(output_dir/"videos.json",manifest)
    return manifest
