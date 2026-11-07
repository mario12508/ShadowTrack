"""Sensitive media are written only to an explicitly private output directory.

Never return images to a notebook cell, logger or public export by default.
"""

from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from .common import atomic_json,read_json
from .schema import decode_rle

CHECKS = ["televisions_and_reflections","table_occlusion","distant_people","doors","feet_visibility","20_vs_25_fps"]


class PrivateQuality:
    def __init__(self, output: Path, camera_id: str, config_hash: str, enabled: bool):
        self.output,self.enabled = output,enabled
        self.camera_id,self.config_hash = camera_id,config_hash
        self.stills,self.clip,self.seen = [],[],set()
        if enabled and (output/"quality_index.json").exists():
            self.stills = read_json(output/"quality_index.json")["stills"]
            self.seen = {r["source_frame_index"] for r in self.stills}

    def add(self, rgb, record, window):
        if not self.enabled:
            return
        i = record["source_frame_index"]
        start,stop = window["first_frame"],window["stop_frame"]
        midpoint = (start+stop)//2
        still = i in (start,midpoint)
        clip = start <= i < start+12
        if not still and not clip:
            return
        image = np.asarray(rgb).copy()
        for d in record["detections"]:
            mask = decode_rle(d["mask_rle"]).astype(bool)
            image[mask] = (image[mask]*.6+np.array([20,210,120])*.4).astype(np.uint8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for d in record["detections"]:
            draw.rectangle(d["bbox_xyxy"],outline=(255,170,30),width=3)
        image = ImageOps.contain(image,(720,540))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0,0,image.width,22),fill="black")
        draw.text((4,4),f"cam {self.camera_id} frame {i} raw PTS {record['raw_media_pts']} persons {len(record['detections'])}",fill="white")
        self.output.mkdir(parents=True,exist_ok=True)
        if still and len(self.stills)<6 and i not in self.seen:
            name = f"frame_{i:08d}.jpg"
            image.save(self.output/name,quality=85)
            self.stills.append({"path":name,"source_frame_index":i,"raw_media_pts":record["raw_media_pts"],"checked":False})
            self.seen.add(i)
            self.save_index()
        if clip and len(self.clip)<36:
            self.clip.append(image.copy())

    def finish(self):
        if not self.enabled:
            return
        self.output.mkdir(parents=True,exist_ok=True)
        if self.clip:
            # Contact-style preview at a deliberately labelled playback rate, never synchronized video.
            self.clip[0].save(self.output/"annotated_preview.gif",save_all=True,append_images=self.clip[1:],duration=100,loop=0)
        if self.stills:
            sheet = Image.new("RGB",(720*2,580*((len(self.stills)+1)//2)),"#15202a")
            draw = ImageDraw.Draw(sheet)
            for j,entry in enumerate(self.stills):
                with Image.open(self.output/entry["path"]) as im:
                    sheet.paste(im,(j%2*720,j//2*580))
                draw.text((j%2*720+5,j//2*580+545),f"frame {entry['source_frame_index']} | unchecked",fill="white")
            sheet.save(self.output/"gallery.jpg",quality=88)
        self.save_index()

    def save_index(self):
        atomic_json(self.output/"quality_index.json",{"private_only":True,"camera_id":self.camera_id,"config_hash":self.config_hash,
                     "stills":self.stills,"checks":{k:{"checked":False,"notes":""} for k in CHECKS},
                     "preview_playback":"10 fps contact preview; not source timing; never publish"})


class DetectionQuality:
    """Three numeric-linked JPEG examples per camera/config; no GIF or masks."""
    def __init__(self, output, window):
        self.output,self.window = Path(output),window
        self.targets = {window["first_frame"],(window["first_frame"]+window["stop_frame"]-1)//2,window["stop_frame"]-1}
        self.entries = read_json(self.output/"index.json")["examples"] if (self.output/"index.json").exists() else []

    def add(self, rgb, record):
        if record["source_frame_index"] not in self.targets:
            return
        from .video import annotate
        image = Image.fromarray(annotate(rgb,record,show_all=True))
        image = ImageOps.contain(image,(1280,960))
        self.output.mkdir(parents=True,exist_ok=True)
        name = f"frame_{record['source_frame_index']:08d}.jpg"
        image.save(self.output/name,quality=90)
        entry = {"path":name,"source_frame_index":record["source_frame_index"],"raw_media_pts":record["raw_media_pts"],
            "timestamp_segment_id":record["timestamp_segment_id"],"raw_count":len(record["detections"]),
            "scene_counts":{k:sum(d["scene_filter"]["decision"]==k for d in record["detections"]) for k in ("accepted","rejected","uncertain")}}
        self.entries = [e for e in self.entries if e["source_frame_index"] != record["source_frame_index"]]+[entry]
        atomic_json(self.output/"index.json",{"private_only":True,"legend":"green accepted; orange uncertain; red rejected; all raw boxes retained",
            "examples":sorted(self.entries,key=lambda e:e["source_frame_index"]),"checked":False})
