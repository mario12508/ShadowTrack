"""Versioned scene decisions annotate raw boxes; never delete source detections."""

import copy
import math

from .common import DetectionError,object_hash


def iou(a,b):
    overlap = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/area if area > 0 else 0.


def translate_crop(detections, crop, width, height):
    x,y,right,bottom = crop
    if not 0 <= x < right <= width or not 0 <= y < bottom <= height:
        raise DetectionError("Crop outside source resolution")
    output = []
    for raw in detections:
        d = copy.deepcopy(raw)
        a,b,c,e = d["bbox_xyxy"]
        if not 0 <= a < c <= right-x or not 0 <= b < e <= bottom-y:
            raise DetectionError("Crop detector coordinates outside crop")
        d.update(bbox_xyxy=[a+x,b+y,c+x,e+y],inference_source="far_crop",crop_xyxy=list(crop))
        output.append(d)
    return output


class SceneFilter:
    def __init__(self, config):
        if config.get("schema_version") != 1 or not config.get("version"):
            raise DetectionError("Invalid scene filter version")
        self.config,self.hash = config,object_hash(config)

    def camera_config(self, camera):
        result = self.config.get("cameras",{}).get(camera["camera_id"],{})
        if result.get("source_sha256",camera["source_sha256"]) != camera["source_sha256"]:
            raise DetectionError("Scene rules refer to a different source video")
        if result.get("resolution",[camera["width"],camera["height"]]) != [camera["width"],camera["height"]]:
            raise DetectionError("Scene rules resolution differs from source")
        return result

    def crop(self, camera):
        box = self.camera_config(camera).get("far_crop")
        if box is not None and (len(box) != 4 or any(type(v) is not int for v in box)
                or not 0 <= box[0]<box[2]<=camera["width"] or not 0<=box[1]<box[3]<=camera["height"]):
            raise DetectionError("Invalid far crop")
        return box

    def apply(self, camera, detections):
        rules = self.camera_config(camera).get("rules",[])
        output = []
        for index,raw in enumerate(detections):
            d = copy.deepcopy(raw)
            x1,y1,x2,y2 = d["bbox_xyxy"]
            if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or x2<=x1 or y2<=y1:
                raise DetectionError("Invalid raw bbox for scene filtering")
            decision = "accepted" if d["confidence"] >= self.config.get("review_confidence",.25) else "uncertain"
            reasons = ["outside_reviewed_exclusions" if decision == "accepted" else "low_confidence_preserved"]
            for rule in rules:
                a,b,c,e = rule["bbox_xyxy"]
                fraction = max(0,min(x2,c)-max(x1,a))*max(0,min(y2,e)-max(y1,b))/((x2-x1)*(y2-y1))
                if fraction >= rule.get("contained_fraction",.95):
                    if rule["decision"] not in ("rejected","uncertain"):
                        raise DetectionError("Scene rule must reject or mark uncertainty")
                    if decision != "rejected":
                        decision = rule["decision"]
                    reasons.append(rule["reason"])
            d.setdefault("inference_source","full_frame")
            d["scene_filter"] = {"decision":decision,"reasons":reasons,"rule_version":self.config["version"],"config_hash":self.hash}
            output.append(d)
        # Reconcile full-frame/crop duplicates but keep every raw candidate and original index.
        kept = []
        for index in sorted(range(len(output)),key=lambda i:(-output[i]["confidence"],i)):
            d = output[index]
            if d["scene_filter"]["decision"] == "rejected":
                continue
            duplicate = next((j for j in kept if iou(d["bbox_xyxy"],output[j]["bbox_xyxy"]) > self.config.get("duplicate_iou",.65)),None)
            if duplicate is not None:
                d["scene_filter"].update(decision="rejected",duplicate_of=duplicate)
                d["scene_filter"]["reasons"].append("overlapping_raw_candidate")
            else:
                kept.append(index)
        return output
