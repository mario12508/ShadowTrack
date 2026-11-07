"""CPU ByteTrack replay over validated imported detections; no model inference."""

import gzip
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
import tomllib

import numpy as np

from ..probe import sha256
from .common import DetectionError,atomic_bytes,atomic_json,canonical,object_hash,read_json
from .evidence import load_evidence,verify_local_sources
from .store import validate_run,validate_chunk,chunk_name
from .progress import observed
from .plan import model_for_config


def _write_stream(path, rows):
    fd,name = tempfile.mkstemp(prefix=".pending-",dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd,"wb") as raw:
            with gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as compressed:
                for row in rows:
                    compressed.write(canonical(row)+b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        if path.exists() and sha256(path) != sha256(temporary):
            raise DetectionError("Existing track replay differs; select a new output directory")
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def replay(records, camera, tracker_config):
    if importlib.metadata.version("supervision") != "0.25.1":
        raise DetectionError("Local ByteTrack requires supervision==0.25.1")
    import supervision as sv
    tracker,previous,segment,namespace = None,None,None,None
    for record in records:
        i,sid = record["source_frame_index"],record["timestamp_segment_id"]
        if tracker is None or sid != segment or i != previous+1:
            tracker = sv.ByteTrack(frame_rate=int(camera["nominal_fps"]),**tracker_config)
            tracker.reset()
            namespace = f"camera_{camera['camera_id']}/{sid}/start_{i}"
        raw = record["detections"]
        allowed = [j for j,d in enumerate(raw) if d.get("scene_filter",{}).get("decision") != "rejected"]
        detections = [raw[j] for j in allowed]
        if detections:
            observed = sv.Detections(xyxy=np.array([d["bbox_xyxy"] for d in detections],dtype=np.float32),
                                    confidence=np.array([d["confidence"] for d in detections],dtype=np.float32),
                                    class_id=np.zeros(len(detections),dtype=int),
                                    data={"detection_index":np.array(allowed,dtype=int)})
        else:
            observed = sv.Detections(xyxy=np.empty((0,4),dtype=np.float32),confidence=np.empty(0,dtype=np.float32),class_id=np.empty(0,dtype=int))
        tracked = tracker.update_with_detections(observed)
        tracks = []
        assigned = set()
        for j,tid in enumerate(tracked.tracker_id):
            source_index = int(tracked.data["detection_index"][j])
            assigned.add(source_index)
            tracks.append({"local_track_id":f"{namespace}/{int(tid)}","detection_index":source_index,
                           "bbox_xyxy":raw[source_index]["bbox_xyxy"],"confidence":raw[source_index]["confidence"],
                           "mask_ref":{"source_frame_index":i,"detection_index":source_index} if raw[source_index].get("mask_rle") else None,"observed":True})
        yield {"schema_version":record["schema_version"],"camera_id":camera["camera_id"],"source_frame_index":i,
               "raw_media_pts":record["raw_media_pts"],"timestamp_segment_id":sid,"processed":True,
               "input_empty":not raw,"model_hash":record["model_hash"],"config_hash":record["config_hash"],
               "local_tracks":tracks,"untracked_detection_indices":sorted(set(allowed)-assigned),
               "rejected_detection_indices":sorted(set(range(len(raw)))-set(allowed)),
               "global_time_status":"NOT_REQUESTED","global_association":False}
        previous,segment = i,sid


@observed("local_track")
def local_track(config_path: Path, progress=None):
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    root = config_path.resolve().parent
    paths = {k:(root/v).resolve() for k,v in config["paths"].items()}
    with progress.stage("tracking_evidence",announce=True):
        evidence = load_evidence(paths["evidence_dir"])
    verify_local_sources(evidence,paths["raw_dir"],progress=progress)
    receipt = read_json(paths["detection_run"]/"IMPORT_RECEIPT.json")
    if receipt.get("validated") is not True or receipt.get("manifest_sha256") != sha256(paths["detection_run"]/"run_manifest.json") or receipt.get("source_evidence_hash") != evidence["manifest_hash"]:
        raise DetectionError("Tracking requires unchanged, validated detection import")
    with progress.stage("tracking_validate_chunks",announce=True):
        manifest,chunks = validate_run(paths["detection_run"],evidence)
    choices = [(h,cfg) for h,cfg in manifest["descriptor"]["configurations"].items() if cfg["name"] == config["tracking"]["configuration_name"]]
    if len(choices) != 1:
        raise DetectionError("Select exactly one imported inference configuration")
    selected,_ = choices[0]
    output = paths["output_dir"]
    for protected in (paths["detection_run"],paths["evidence_dir"],paths["raw_dir"]):
        if output.is_relative_to(protected):
            raise DetectionError("Track output must be separate from read-only inputs")
    tracker_config = config["bytetrack"]
    summary = {"schema_version":manifest["schema_version"],"package_kind":"shadowtrack_local_tracks","tracker":"supervision.ByteTrack",
               "supervision_version":"0.25.1","deterministic_replay_scope":"same inputs, config and pinned CPU implementation",
               "source_run_id":manifest["run_id"],"source_manifest_sha256":sha256(paths["detection_run"]/"run_manifest.json"),
               "selected_config_hash":selected,"tracker_config":tracker_config,"tracker_config_hash":object_hash(tracker_config),
               "global_association":False,"mask_policy":"none for detection v2; references original masks for segmentation v1",
               "fps_usage":"nominal camera FPS scales lost-track buffer only; never generates source timestamps","files":[]}
    output.mkdir(parents=True,exist_ok=True)
    for cid,camera in evidence["cameras"].items():
        progress.update("tracking_camera_start",phase="bytetrack_replay",camera_id=cid,processed_frames=0)
        items = sorted([(chunk,records) for h,chunk,records in chunks if h == selected and chunk["camera_id"] == cid],key=lambda pair:pair[0]["start"])
        def source_records():
            for chunk,_ in items:
                folder = paths["detection_run"]/chunk_name(selected,chunk)
                _,records = validate_chunk(folder,camera,model_for_config(manifest["descriptor"],selected)["weights_sha256"],selected,chunk)
                yield from records
        stats = {"processed_frames":0,"track_ids":set()}
        def result_rows():
            for row in replay(source_records(),camera,tracker_config):
                stats["processed_frames"] += 1
                progress.update(processed_frames=stats["processed_frames"],source_frame_index=row["source_frame_index"],raw_media_pts=row["raw_media_pts"])
                for t in row["local_tracks"]:
                    if t["mask_ref"] is not None:
                        t["mask_ref"].update(source_run_id=manifest["run_id"],config_hash=selected,camera_id=cid)
                    stats["track_ids"].add(t["local_track_id"])
                yield row
        path = output/f"camera_{cid}_tracks.jsonl.gz"
        _write_stream(path,result_rows())
        progress.update("tracking_camera_saved",processed_frames=stats["processed_frames"],message=str(path))
        summary["files"].append({"camera_id":cid,"path":path.name,"sha256":sha256(path),"processed_frames":stats["processed_frames"],
                                  "local_track_count":len(stats["track_ids"])})
    atomic_json(output/"manifest.json",summary)
    if config.get("video",{}).get("enabled"):
        from .video import render_local_tracks
        video_output = (root/config["video"]["output_dir"]).resolve()
        try:
            render_local_tracks(output,evidence,paths["raw_dir"],video_output,progress=progress)
            summary["derived_video_status"] = "COMPLETE"
        except Exception as exc:
            summary["derived_video_status"] = "FAILED"
            summary["derived_video_error_type"] = type(exc).__name__
            progress.update("video_failed_tracks_preserved",error_type=type(exc).__name__)
        atomic_json(output/"manifest.json",summary)
    return summary
