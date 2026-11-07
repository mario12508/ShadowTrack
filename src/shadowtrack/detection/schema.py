"""Version 1: one processed-frame envelope, including an explicit empty mark.

Each detection repeats the frame identity; consumers need not infer timestamps.
Masks use uncompressed COCO RLE, column-major, zero run first, original H x W.
"""

import math
import re

import numpy as np

from .common import DetectionError

IDENTITY = ("camera_id", "source_frame_index", "raw_media_pts", "timestamp_segment_id", "model_hash", "config_hash")


def encode_rle(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2 or not np.all((mask == 0) | (mask == 1)):
        raise DetectionError("Mask must be two-dimensional binary data")
    flat = mask.ravel(order="F")
    edges = np.flatnonzero(flat[1:] != flat[:-1])+1
    counts = np.diff(np.r_[0, edges, len(flat)]).tolist()
    if len(flat) and flat[0]:
        counts.insert(0, 0)
    return {"encoding": "coco_rle_uncompressed", "size": list(mask.shape), "counts": counts}


def validate_rle(rle, height, width):
    if not isinstance(rle, dict) or rle.get("encoding") != "coco_rle_uncompressed" or rle.get("size") != [height, width]:
        raise DetectionError("Missing or wrong-size segmentation mask")
    counts = rle.get("counts")
    if not isinstance(counts, list) or not counts or len(counts) > height*width+1:
        raise DetectionError("Invalid RLE counts")
    if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != height*width or sum(counts[1::2]) <= 0:
        raise DetectionError("Invalid or empty instance mask RLE")


def decode_rle(rle):
    h,w = rle["size"]
    validate_rle(rle,h,w)
    out = np.zeros(h*w, dtype=np.uint8)
    offset = 0
    for i,count in enumerate(rle["counts"]):
        if i % 2:
            out[offset:offset+count] = 1
        offset += count
    return out.reshape((h,w), order="F")


def frame_record(camera, index, detections, model_hash, config_hash):
    source = camera["frames"][index]
    identity = {"camera_id": camera["camera_id"], "source_frame_index": index,
                "raw_media_pts": source[1], "timestamp_segment_id": source[2],
                "model_hash": model_hash, "config_hash": config_hash}
    return {"schema_version": 1, **identity, "processed": True,
            "frame_status": "detections" if detections else "empty",
            "detections": [{**d, **identity} for d in detections]}


def detection_record(camera, index, detections, model_hash, config_hash, scene_hash):
    record = frame_record(camera,index,detections,model_hash,config_hash)
    record.update(schema_version=2,task="detect",scene_config_hash=scene_hash)
    return record


def validate_frame(record, camera, model_hash, config_hash):
    version = record.get("schema_version")
    if version not in (1,2) or record.get("processed") is not True:
        raise DetectionError("Unsupported schema or missing processed-frame marker")
    if version == 2 and (record.get("task") != "detect" or not re.fullmatch("[0-9a-f]{64}",record.get("scene_config_hash", ""))):
        raise DetectionError("Invalid v2 task/scene configuration identity")
    i = record.get("source_frame_index")
    if type(i) is not int or not 0 <= i < camera["frame_count"]:
        raise DetectionError("Invalid source_frame_index")
    expected = camera["frames"][i]
    if (record.get("camera_id") != camera["camera_id"] or record.get("raw_media_pts") != expected[1]
            or record.get("timestamp_segment_id") != expected[2]):
        raise DetectionError("Frame identity/PTS/segment disagrees with audited source")
    if record.get("model_hash") != model_hash or record.get("config_hash") != config_hash:
        raise DetectionError("Model or config hash mismatch")
    for digest in (model_hash, config_hash):
        if not isinstance(digest,str) or not re.fullmatch("[0-9a-f]{64}",digest):
            raise DetectionError("Invalid SHA256 identity")
    detections = record.get("detections")
    if not isinstance(detections,list) or record.get("frame_status") != ("detections" if detections else "empty"):
        raise DetectionError("Empty frame must be explicitly processed")
    for detection in detections:
        if any(detection.get(k) != record[k] for k in IDENTITY):
            raise DetectionError("Detection identity disagrees with frame envelope")
        box,confidence = detection.get("bbox_xyxy"),detection.get("confidence")
        if not isinstance(box,list) or len(box) != 4 or any(type(v) not in (float,int) or not math.isfinite(v) for v in box):
            raise DetectionError("Invalid bounding box")
        x1,y1,x2,y2 = box
        if not 0 <= x1 < x2 <= camera["width"] or not 0 <= y1 < y2 <= camera["height"]:
            raise DetectionError("Bounding box outside original image")
        if type(confidence) not in (float,int) or not math.isfinite(confidence) or not 0 <= confidence <= 1 or detection.get("class_id") != 0:
            raise DetectionError("Invalid person confidence/class")
        if version == 1:
            validate_rle(detection.get("mask_rle"),camera["height"],camera["width"])
        else:
            if detection.get("mask_rle") is not None:
                raise DetectionError("Detection v2 does not carry segmentation masks")
            decision = detection.get("scene_filter",{})
            if (decision.get("decision") not in ("accepted","rejected","uncertain")
                    or not isinstance(decision.get("reasons"),list) or not decision["reasons"]
                    or not all(isinstance(r,str) and r for r in decision["reasons"])
                    or not isinstance(decision.get("rule_version"),str) or not decision["rule_version"]
                    or decision.get("config_hash") != record["scene_config_hash"]):
                raise DetectionError("Missing or invalid scene filter evidence")
