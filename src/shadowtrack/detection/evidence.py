"""Portable audit contract: only source identity, frame index, original PTS and segments."""

import gzip
import hashlib
import json
from pathlib import Path

from ..probe import sha256
from .common import DetectionError, atomic_bytes, atomic_json, canonical, code_fingerprint, object_hash, read_json, safe_path
from .progress import observed


@observed("prepare_evidence")
def prepare_evidence(root: Path, output: Path, progress=None):
    root,output = root.resolve(),output.resolve()
    if output.is_relative_to(root/"data") or output.is_relative_to(root/"artifacts/inspect") or output.is_relative_to(root/"artifacts/sync"):
        raise DetectionError("Portable evidence must be outside original data and audit/alignment artifacts")
    if (output/"source_manifest.json").exists():
        raise DetectionError("Refusing to overwrite existing portable source evidence")
    stats_path,alignment_path = root/"artifacts/inspect/stats.json",root/"artifacts/sync/time_alignment.json"
    stats,alignment = read_json(stats_path),read_json(alignment_path)
    manifest = {"schema_version":1,"package_kind":"shadowtrack_source_evidence",
                "audit_stats_sha256":sha256(stats_path),"time_alignment_sha256":sha256(alignment_path),
                "producer_code_hash":code_fingerprint(root),"cameras":{}}
    for c in alignment["cameras"]:
        progress.update("prepare_camera",camera_id=c["camera_id"],phase="prepare_evidence")
        camera = next(v for v in stats["cameras"] if Path(v["file"]).name == Path(c["file"]).name)
        if progress.hash_file(Path(camera["file"]),"source_sha256") != camera["sha256"]:
            raise DetectionError("Original video hash changed")
        path = stats_path.parent/camera["key"]/"timestamps/frames.json"
        if sha256(path) != c["raw_pts_evidence_sha256"]:
            raise DetectionError("Saved video frame evidence changed")
        frames = read_json(path)["frames"]
        segments = c["local_segments"]
        labels = [None]*len(frames)
        for s in segments:
            for i in range(s["frame_start"],s["frame_end"]+1):
                if labels[i] is not None:
                    raise DetectionError("Overlapping timestamp segments")
                labels[i] = s["id"]
        if any(label is None for label in labels):
            raise DetectionError("Cannot export frames with ambiguous/missing timestamp segment")
        records = [[i,float(f["best_effort_timestamp_time"]),labels[i]] for i,f in enumerate(frames)]
        cid = c["camera_id"]
        filename = f"camera_{cid}_frames.json"
        atomic_bytes(output/filename,canonical(records))
        manifest["cameras"][cid] = {"camera_id":cid,"source_name":Path(camera["file"]).name,
            "source_sha256":camera["sha256"],"frame_count":len(frames),"width":camera["width"],"height":camera["height"],
            "nominal_fps":camera["nominal_fps"],"frames_file":filename,"frames_sha256":sha256(output/filename),
            "frames_content_sha256":sha256(output/filename),
            "audited_frames_sha256":sha256(path),"segments":segments}
    atomic_json(output/"source_manifest.json",manifest)
    return load_evidence(output)


def load_evidence(directory: Path):
    directory = directory.resolve()
    manifest = read_json(directory/"source_manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("package_kind") != "shadowtrack_source_evidence":
        raise DetectionError("Unsupported source evidence schema")
    cameras = {}
    for cid,info in manifest["cameras"].items():
        path = safe_path(directory,info["frames_file"])
        if path.is_file():
            if sha256(path) != info["frames_sha256"]:
                raise DetectionError("Portable frame evidence SHA256 mismatch")
            payload = path.read_bytes()
            if path.suffix == ".gz":
                payload = gzip.decompress(payload)
        else:
            unpacked = path.with_suffix("") if path.suffix == ".gz" else path
            if not unpacked.is_file():
                raise DetectionError(f"Camera {cid}: missing evidence {path.name} (also checked {unpacked.name}). "
                                     "Attach the complete input_evidence folder from the current code package.")
            if not info.get("frames_content_sha256"):
                raise DetectionError("Legacy evidence was decompressed but has no content SHA256. "
                                     "Use the plain JSON input_evidence bundled with the current code package.")
            payload = unpacked.read_bytes()
        if info.get("frames_content_sha256") and hashlib.sha256(payload).hexdigest() != info["frames_content_sha256"]:
            raise DetectionError("Portable frame evidence content SHA256 mismatch")
        frames = json.loads(payload)
        if len(frames) != info["frame_count"] or any(v[0] != i for i,v in enumerate(frames)):
            raise DetectionError("Shuffled/missing source frame indices")
        if any(not isinstance(v[1],(int,float)) or v[2] is None for v in frames):
            raise DetectionError("Missing source timestamps/segments")
        for s in info["segments"]:
            if any(frames[i][2] != s["id"] for i in range(s["frame_start"],s["frame_end"]+1)):
                raise DetectionError("Source timestamp segment mismatch")
        cameras[cid] = {**info,"frames":frames}
    return {"manifest":manifest,"manifest_hash":sha256(directory/"source_manifest.json"),"cameras":cameras,"directory":directory}


def verify_local_sources(evidence, raw_dir: Path, progress=None):
    for camera in evidence["cameras"].values():
        path = raw_dir/camera["source_name"]
        digest = (progress.hash_file(path,"source_sha256") if progress is not None else sha256(path)) if path.is_file() else None
        if digest != camera["source_sha256"]:
            raise DetectionError("Local source video missing or SHA256 differs from original audit")


def sample_plan(evidence, seconds_per_camera=30., chunk_frames=128, mode="SAMPLE"):
    if mode not in ("SAMPLE","FULL") or not 1 <= chunk_frames <= 1024:
        raise DetectionError("Invalid mode/chunk size")
    selections,windows,chunks = {},[],[]
    for cid,camera in evidence["cameras"].items():
        selected = []
        if mode == "FULL":
            selected = list(range(camera["frame_count"]))
        else:
            segments = camera["segments"]
            duration = seconds_per_camera/len(segments)
            for s in segments:
                # Same source windows in both configurations; distributed across every segment.
                lo,hi = s["frame_start"],s["frame_end"]
                available = camera["frames"][hi][1]-camera["frames"][lo][1]
                if available < duration:
                    raise DetectionError("Timestamp segment too short for requested SAMPLE window")
                start = camera["frames"][lo][1]+max(0,(available-duration)*.45)
                indices = [i for i in range(lo,hi+1) if start <= camera["frames"][i][1] < start+duration]
                selected.extend(indices)
                windows.append({"camera_id":cid,"segment_id":s["id"],"raw_interval":[start,start+duration],
                                "first_frame":indices[0],"stop_frame":indices[-1]+1})
        selections[cid] = selected
        group = []
        for i in selected:
            if group and (i != group[-1]+1 or camera["frames"][i][2] != camera["frames"][group[-1]][2] or len(group) >= chunk_frames):
                chunks.append({"camera_id":cid,"segment_id":camera["frames"][group[0]][2],"start":group[0],"stop":group[-1]+1})
                group = []
            group.append(i)
        if group:
            chunks.append({"camera_id":cid,"segment_id":camera["frames"][group[0]][2],"start":group[0],"stop":group[-1]+1})
    return {"mode":mode,"sample_seconds_per_camera":seconds_per_camera if mode == "SAMPLE" else None,
            "chunk_frames":chunk_frames,"windows":windows,"chunks":chunks,
            "selected_frame_counts":{cid:len(indices) for cid,indices in selections.items()}}
