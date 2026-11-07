"""Analyze saved video evidence. No invocation of a full video probe."""

from __future__ import annotations

import itertools
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..probe import AuditError, sha256
from .api import TimeAlignment
from .audio import analyze_audio
from .fit import fit_intervals
from .local import build_local, save_local


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_valid(item, root: Path):
    source = item.get("source", {})
    path = root/source.get("manifest", "__missing__")
    return path.is_file() and source.get("sha256") == sha256(path)


def build_models(timelines: dict, events: list[dict], root: Path, settings: dict):
    groups, considered = defaultdict(list), []
    for event in events:
        reason = None
        if event.get("checked") is not True:
            reason = "unchecked"
        elif not event.get("reviewer") or not event.get("description") or not _source_valid(event,root):
            reason = "missing review provenance or changed source manifest"
        observations = {}
        if reason is None:
            for cid, bounds in event["observations"].items():
                if cid not in timelines or len(bounds) != 2 or bounds[1] < bounds[0]:
                    reason = "invalid camera/interval"
                    break
                a,b = (timelines[cid].at(t) for t in bounds)
                if a is None or b is None or a["segment_id"] != b["segment_id"] or a.get("interval") is None or b.get("interval") is None:
                    reason = "anchor crosses timestamp break or unavailable local time"
                    break
                observations[cid] = {"node": a["segment_id"], "interval": [a["interval"][0],b["interval"][1]], "raw_interval": bounds}
        considered.append({**event,"accepted_for_fit":reason is None,"exclusion_reason":reason})
        if reason is not None:
            continue
        for a,b in itertools.combinations(sorted(observations),2):
            oa,ob = observations[a],observations[b]
            groups[(oa["node"],ob["node"])].append({"id":event["id"],"x":oa["interval"],"y":ob["interval"],
                                                    "raw_x":oa["raw_interval"],"raw_y":ob["raw_interval"],
                                                    "source":event["source"],"kind":event["kind"]})
    models = []
    for (a,b), anchors in sorted(groups.items()):
        fit = fit_intervals(anchors, **settings)
        fit["raw_pts_model"] = fit_intervals([dict(anchor,x=anchor["raw_x"],y=anchor["raw_y"]) for anchor in anchors],**settings)
        models.append({"id":f"{a}->{b}","node_a":a,"node_b":b,"anchors":anchors,**fit})
    return models,considered


def analyze(stats_path: Path, evidence_path: Path, output: Path,
            reference_camera="3", audio=True, root: Path | None = None) -> dict:
    stats_path, evidence_path, output = stats_path.resolve(),evidence_path.resolve(),output.resolve()
    root = root.resolve() if root else Path.cwd().resolve()
    stats, evidence = _read(stats_path),_read(evidence_path)
    if output.is_relative_to(stats_path.parent) or output.is_relative_to(root/"data"):
        raise AuditError("Sync outputs must be separate from source data and the completed audit")
    output.mkdir(parents=True,exist_ok=True)
    cameras = []
    # Explicit IDs in evidence; never assume sorted file index is a camera identity.
    for cid, filename in evidence["camera_files"].items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+",cid):
            raise AuditError("Camera IDs must contain only letters, digits, _ or -")
        matches = [c for c in stats["cameras"] if Path(c["file"]).name == filename]
        if len(matches) != 1:
            raise AuditError(f"No unique audited source for camera {cid}")
        camera = dict(matches[0],camera_id=cid)
        if camera["sha256"] != evidence["input_sha256"].get(filename) or sha256(Path(camera["file"])) != camera["sha256"]:
            raise AuditError(f"Input hash changed for camera {cid}; use evidence for the actual source")
        cameras.append(camera)
    if reference_camera not in evidence["camera_files"]:
        raise AuditError("Unknown reference camera")
    timelines, infos = {},[]
    for camera in cameras:
        cid = camera["camera_id"]
        saved = stats_path.parent/camera["key"]/"timestamps/frames.json"
        if evidence.get("frame_pts_sha256",{}).get(cid) != sha256(saved):
            raise AuditError(f"Saved video frame evidence changed for camera {cid}")
        frames = _read(saved)["frames"]
        corrections = []
        for c in evidence.get("local_corrections",[]):
            if c["camera_id"] == cid and c.get("checked") is True:
                if not _source_valid(c,root):
                    raise AuditError(f"Correction evidence changed: {c['id']}")
                corrections.append(c)
        timeline = build_local(cid,[f.get("best_effort_timestamp_time") for f in frames],camera["nominal_fps"],corrections)
        timelines[cid] = timeline
        path = output/f"camera_{cid}_frame_times.npz"
        save_local(timeline,path)
        infos.append({"camera_id":cid,"file":camera["file"],"sha256":camera["sha256"],
                      "nominal_fps":camera["nominal_fps"],"frame_count":len(frames),
                      "raw_pts_evidence":str(saved),"raw_pts_evidence_sha256":sha256(saved),
                      "local_frame_times_file":path.name,"local_frame_times_sha256":sha256(path),
                      "local_segments":timeline.segments,"anomalies":timeline.anomalies,
                      "local_time_status":"ESTIMATE_ONLY_RATE_UNVERIFIED",
                      "synchronization_status":"UNCALIBRATED"})
    models,events = build_models(timelines,evidence.get("events",[]),root,evidence.get("regression",{}))
    reference_node = timelines[reference_camera].segments[0]["id"]
    supports = [m["x_support"] if m["node_a"] == reference_node else m["y_support"]
                for m in models if m["status"] == "SUPPORTED" and reference_node in (m["node_a"],m["node_b"])]
    audio_result = analyze_audio(cameras,output/"audio") if audio else {"status":"NOT_REQUESTED","automatically_applied_anchors":[]}
    doc = {"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),
           "status":"UNCALIBRATED","acceptance_status":"UNCALIBRATED",
           "global_time_basis":f"arbitrary relative seconds from first decoded frame of camera {reference_camera}, first uninterrupted epoch; NOT UTC",
           "reference_camera":reference_camera,"reference_node":reference_node,"reference_support":supports,
           "reference_coordinate_resolution_seconds":.001,
           "cameras":infos,"pair_models":models,"events":events,"audio":audio_result,
           "source_stats":str(stats_path),"source_stats_sha256":sha256(stats_path),
           "source_evidence":str(evidence_path),"source_evidence_sha256":sha256(evidence_path),
           "full_video_probe_repeated":False,"raw_media_modified":False,
           "uncertainty_policy":"interval parameter envelopes + local correction bounds; not statistical confidence; no extrapolation",
           "decisions":["Unresolved gaps retain raw duration and have no interior mapping.",
                        "Confirmed clock jumps are local estimates only, never automatic inter-camera anchors.",
                        "Models are fitted separately for each pair of uninterrupted timestamp segments.",
                        "Unchecked events and uninformative audio cannot calibrate cameras.",
                        "A single initial offset is not extended beyond checked anchor support."]}
    api = TimeAlignment(doc,timelines)
    supported_count = 0
    for info in infos:
        timeline = timelines[info["camera_id"]]
        domains = []
        for segment in timeline.segments:
            # Map only model-supported candidate intervals; report failed paths explicitly.
            candidates = []
            for model in models:
                if model["status"] != "SUPPORTED":
                    continue
                if model["node_a"] == segment["id"]:
                    candidates.append(model["x_support"])
                if model["node_b"] == segment["id"]:
                    candidates.append(model["y_support"])
            for low,high in candidates:
                midpoint = (low+high)/2
                raw = timeline.inverse(midpoint)
                mapped = api.map_time(info["camera_id"],raw) if raw is not None else (None,None,"unavailable")
                if mapped[0] is not None:
                    domains.append({"local_interval_candidate":[low,high],
                                    "raw_interval_candidate":[timeline.inverse(low),timeline.inverse(high)],
                                    "status":"SUPPORTED_CONDITIONAL",
                                    "note":"Query API checks exact path support and uncertainty; boundary values may be unavailable."})
        info["supported_regions"] = domains
        unsupported = []
        for segment in timeline.segments:
            cursor,end = segment["raw_start"],segment["raw_end"]
            for a,b in sorted(d["raw_interval_candidate"] for d in domains):
                if b < cursor or a > end:
                    continue
                if a > cursor+1e-8:
                    unsupported.append({"raw_interval":[cursor,a],"open_end":True,"reason":"no calibrated global support"})
                cursor = max(cursor,b)
            if cursor < end-1e-8 or not domains:
                unsupported.append({"raw_interval":[cursor,end],"open_start":bool(domains),"reason":"no calibrated global support"})
        info["unsupported_regions"] = unsupported
        info["unavailable_raw_gaps"] = [{"raw_interval":[a["before_pts"],a["after_pts"]],"reason":a["classification"]}
                                        for a in timeline.anomalies if "before_pts" in a]
        if domains:
            info["synchronization_status"] = "PARTIAL"
            supported_count += 1
    invalid_cameras = []
    for info in infos:
        order = sorted(info["supported_regions"],key=lambda d:d["raw_interval_candidate"][0])
        previous_high = None
        previous_raw_high = None
        for domain in order:
            low,high = domain["raw_interval_candidate"]
            first,last = api.map_time(info["camera_id"],low),api.map_time(info["camera_id"],high)
            if first[0] is not None and last[0] is not None:
                if previous_high is not None and low > previous_raw_high and first[0] <= previous_high-1e-6:
                    invalid_cameras.append(info["camera_id"])
                previous_high = last[0]
                previous_raw_high = high
    doc["invalid_cameras"] = sorted(set(invalid_cameras))
    for info in infos:
        if info["camera_id"] in invalid_cameras:
            info["synchronization_status"] = "INCONSISTENT"
            info["supported_regions"] = []
            info["unsupported_regions"] = [{"raw_interval":[s["raw_start"],s["raw_end"]],"reason":"inconsistent calibration"} for s in info["local_segments"]]
            supported_count -= 1
    if supported_count < 2:
        doc["reference_support"] = []
        supported_count = 0
        for info in infos:
            if info["synchronization_status"] == "PARTIAL":
                info["synchronization_status"] = "UNCALIBRATED"
                info["supported_regions"] = []
                info["unsupported_regions"] = [{"raw_interval":[s["raw_start"],s["raw_end"]],"reason":"no reliable connected camera pair"} for s in info["local_segments"]]
    if supported_count:
        doc["status"] = doc["acceptance_status"] = "PARTIAL"
    doc["linked_camera_count"] = supported_count
    doc["fully_calibrated"] = (supported_count == len(infos) and not invalid_cameras and
                               all(not c["unsupported_regions"] for c in infos))
    if doc["fully_calibrated"]:
        doc["status"] = doc["acceptance_status"] = "CALIBRATED_CONDITIONAL"
    (output/"time_alignment.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return doc
