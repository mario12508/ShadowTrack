"""Atomic camera/segment/frame-range chunks and strict completion markers."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from ..probe import sha256
from .common import DetectionError, atomic_bytes, atomic_json, canonical, object_hash, read_json, safe_path
from .schema import validate_frame
from .evidence import sample_plan
from .plan import detection_plan,evidence_identity,model_for_config


def chunk_name(config_hash, chunk):
    segment = chunk["segment_id"].replace(":","_")
    return f"chunks/{config_hash}/camera_{chunk['camera_id']}/{segment}/{chunk['start']:08d}_{chunk['stop']:08d}"


def validate_chunk(folder: Path, camera, model_hash, config_hash, expected=None):
    manifest_path = folder/"manifest.json"
    if not manifest_path.is_file():
        raise DetectionError("Interrupted chunk: manifest missing")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") not in (1,2) or manifest.get("source_sha256") != camera["source_sha256"]:
        raise DetectionError("Chunk schema/source hash mismatch")
    if manifest.get("config_hash") != config_hash or manifest.get("model_hash") != model_hash:
        raise DetectionError("Chunk model/config mismatch")
    data = safe_path(folder,manifest.get("records_file",""))
    try:
        if data.is_file():
            if sha256(data) != manifest.get("records_sha256"):
                raise DetectionError("Corrupt/incomplete chunk SHA256")
            payload = gzip.decompress(data.read_bytes())
        elif data.suffix == ".gz" and data.with_suffix("").is_file() and manifest.get("records_content_sha256"):
            payload = data.with_suffix("").read_bytes()
        else:
            raise DetectionError("Corrupt/incomplete chunk SHA256: records missing")
        if manifest.get("records_content_sha256") and hashlib.sha256(payload).hexdigest() != manifest["records_content_sha256"]:
            raise DetectionError("Corrupt chunk content SHA256")
        lines = payload.splitlines()
        records = [json.loads(line) for line in lines]
    except (OSError,ValueError,EOFError) as exc:
        raise DetectionError("Invalid compressed detection records") from exc
    start,stop = manifest.get("start"),manifest.get("stop")
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= camera["frame_count"]:
        raise DetectionError("Invalid chunk frame range")
    if expected and any(manifest.get(k) != expected[k] for k in ("camera_id","segment_id","start","stop")):
        raise DetectionError("Chunk does not belong to the declared selection plan")
    if len(records) != stop-start or manifest.get("processed_frames") != len(records):
        raise DetectionError("Interrupted chunk: processed-frame count mismatch")
    for i,record in enumerate(records,start):
        validate_frame(record,camera,model_hash,config_hash)
        if record["schema_version"] != manifest["schema_version"]:
            raise DetectionError("Mixed record/chunk schema versions")
        if record["source_frame_index"] != i or record["timestamp_segment_id"] != manifest.get("segment_id"):
            raise DetectionError("Shuffled, duplicate, missing or cross-segment chunk frames")
    if manifest.get("detection_count") != sum(len(r["detections"]) for r in records):
        raise DetectionError("Detection count mismatch")
    if manifest.get("empty_frames") != sum(not r["detections"] for r in records):
        raise DetectionError("Empty-frame count mismatch")
    return manifest,records


class RunStore:
    def __init__(self, root: Path, descriptor: dict, evidence, progress=None):
        self.root,self.evidence = root.resolve(),evidence
        self.root.mkdir(parents=True,exist_ok=True)
        self.path = self.root/"run_manifest.json"
        self.descriptor_hash = object_hash(descriptor)
        self.order = {chunk_name(h,c):(h,c["camera_id"],c["start"]) for h in descriptor["configurations"] for c in descriptor["plan"]["chunks"]}
        if self.path.exists():
            self.manifest = read_json(self.path)
            if self.manifest.get("descriptor_hash") != self.descriptor_hash or self.manifest.get("descriptor") != descriptor:
                raise DetectionError("Resume inputs/config/code/model differ from the checkpoint")
            if self.manifest.get("status") == "FAILED_DECODE":
                raise DetectionError("Decode mismatch run cannot be resumed as valid; inspect diagnostic and use a new output")
        else:
            self.manifest = {"schema_version":descriptor.get("records_schema_version",1),"package_kind":"shadowtrack_detections","run_id":self.descriptor_hash[:24],
                             "descriptor_hash":self.descriptor_hash,"descriptor":descriptor,"status":"IN_PROGRESS",
                             "chunks":[],"decode_validation":{},"runtimes":{},"benchmark_status":"NOT_RUN"}
        self.completed = {}
        # Recover fully committed orphan chunks after a crash before root-index update.
        for config_hash in descriptor["configurations"]:
            for chunk in descriptor["plan"]["chunks"]:
                name = chunk_name(config_hash,chunk)
                folder = safe_path(self.root,name)
                if folder.exists():
                    meta,_ = validate_chunk(folder,evidence["cameras"][chunk["camera_id"]],model_for_config(descriptor,config_hash)["weights_sha256"],config_hash,chunk)
                    self.completed[name] = {"path":name,"manifest_sha256":sha256(folder/"manifest.json"),"processed_frames":meta["processed_frames"]}
                    if progress is not None:
                        progress.update(checkpoint_chunks_verified=len(self.completed))
        for entry in self.manifest["chunks"]:
            if self.completed.get(entry["path"]) != entry:
                raise DetectionError("Previously committed chunk is missing or altered")
        self.save()

    def save(self):
        self.manifest["chunks"] = [self.completed[k] for k in sorted(self.completed,key=self.order.__getitem__)]
        atomic_json(self.path,self.manifest)

    def commit(self, config_hash, chunk, records, metrics):
        name = chunk_name(config_hash,chunk)
        target = safe_path(self.root,name)
        if name in self.completed:
            return False
        if len(records) != chunk["stop"]-chunk["start"]:
            raise DetectionError("Cannot commit an incomplete frame range")
        camera = self.evidence["cameras"][chunk["camera_id"]]
        for i,record in enumerate(records,chunk["start"]):
            validate_frame(record,camera,model_for_config(self.manifest["descriptor"],config_hash)["weights_sha256"],config_hash)
            if record["source_frame_index"] != i or record["timestamp_segment_id"] != chunk["segment_id"]:
                raise DetectionError("Cannot commit shuffled or cross-segment frames")
        target.parent.mkdir(parents=True,exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-",dir=target.parent))
        try:
            payload = b"\n".join(canonical(r) for r in records)+b"\n"
            atomic_bytes(staging/"frames.jsonl.gz",gzip.compress(payload,mtime=0))
            meta = {"schema_version":self.manifest["schema_version"],**chunk,"config_hash":config_hash,
                    "model_hash":model_for_config(self.manifest["descriptor"],config_hash)["weights_sha256"],
                    "source_sha256":camera["source_sha256"],"processed_frames":len(records),
                    "empty_frames":sum(not r["detections"] for r in records),
                    "detection_count":sum(len(r["detections"]) for r in records),
                    "records_file":"frames.jsonl.gz","records_sha256":sha256(staging/"frames.jsonl.gz"),
                    "records_content_sha256":hashlib.sha256(payload).hexdigest(),"metrics":metrics}
            atomic_json(staging/"manifest.json",meta)
            validate_chunk(staging,camera,meta["model_hash"],config_hash,chunk)
            os.rename(staging,target)  # Final directory is visible only after both files are durable.
        finally:
            if staging.exists():
                if not staging.resolve().is_relative_to(target.parent.resolve()) or not staging.name.startswith(".staging-"):
                    raise DetectionError("Refusing to clean an unverified staging path")
                shutil.rmtree(staging)
        self.completed[name] = {"path":name,"manifest_sha256":sha256(target/"manifest.json"),"processed_frames":len(records)}
        self.save()
        return True

    def finish(self):
        descriptor = self.manifest["descriptor"]
        required = {chunk_name(h,c) for h in descriptor["configurations"] for c in descriptor["plan"]["chunks"]}
        if set(self.completed) != required:
            raise DetectionError("Run cannot complete: missing chunks")
        for config_hash in descriptor["configurations"]:
            for cid,camera in self.evidence["cameras"].items():
                result = self.manifest["decode_validation"].get(config_hash+":"+cid,{})
                if result.get("status") != "VALIDATED" or result.get("frame_count") != camera["frame_count"]:
                    raise DetectionError("Run cannot complete: whole-source decoding not validated")
        self.manifest["status"] = "COMPLETE"
        self.save()
        atomic_json(self.root/"COMPLETE.json",{"schema_version":1,"manifest_sha256":sha256(self.path),
                    "processed_frames":sum(v["processed_frames"] for v in self.completed.values()),"chunk_count":len(self.completed)})


def validate_run(root: Path, evidence, require_complete=True, collect_records=False):
    manifest = read_json(root/"run_manifest.json")
    if manifest.get("schema_version") not in (1,2) or manifest.get("package_kind") != "shadowtrack_detections":
        raise DetectionError("Unsupported detection package schema_version")
    descriptor = manifest.get("descriptor",{})
    if manifest["schema_version"] != descriptor.get("records_schema_version",1):
        raise DetectionError("Package schema_version differs from record contract")
    if object_hash(descriptor) != manifest.get("descriptor_hash") or manifest.get("run_id") != object_hash(descriptor)[:24]:
        raise DetectionError("Run descriptor hash mismatch")
    same_evidence = (descriptor.get("evidence_identity") == evidence_identity(evidence) if manifest["schema_version"] == 2
                     else descriptor.get("evidence_manifest_hash") == evidence["manifest_hash"])
    if not same_evidence or descriptor.get("sources") != evidence["manifest"]["cameras"]:
        raise DetectionError("Run references different source video/audit/segment evidence")
    for h,config in descriptor["configurations"].items():
        if object_hash(config) != h:
            raise DetectionError("Inference config hash mismatch")
    plan = descriptor.get("plan",{})
    expected_plan = (detection_plan(evidence,plan.get("mode"),plan.get("raw_starts",{}),plan.get("window_seconds",6.),plan.get("chunk_frames"))
                     if manifest["schema_version"] == 2 else
                     sample_plan(evidence,plan.get("sample_seconds_per_camera") or 30.,plan.get("chunk_frames"),plan.get("mode")))
    if plan != expected_plan:
        raise DetectionError("Selection plan does not match canonical source frame coverage")
    expected = {chunk_name(h,c):(h,c) for h in descriptor["configurations"] for c in descriptor["plan"]["chunks"]}
    listed = [entry["path"] for entry in manifest.get("chunks",[])]
    if listed != sorted(listed,key=lambda name:(expected.get(name,("",{}))[0],expected.get(name,("",{}))[1].get("camera_id",""),expected.get(name,("",{}))[1].get("start",-1))):
        raise DetectionError("Shuffled chunk index")
    seen,records_by_chunk,coverage = set(),[],{}
    for entry in manifest.get("chunks",[]):
        name = entry["path"]
        if name in seen or name not in expected:
            raise DetectionError("Duplicate or unexpected chunk")
        seen.add(name)
        folder = safe_path(root,name)
        if not (folder/"manifest.json").is_file() or sha256(folder/"manifest.json") != entry["manifest_sha256"]:
            raise DetectionError("Chunk manifest SHA256 mismatch")
        config_hash,chunk = expected[name]
        meta,records = validate_chunk(folder,evidence["cameras"][chunk["camera_id"]],model_for_config(descriptor,config_hash)["weights_sha256"],config_hash,chunk)
        if meta["schema_version"] != manifest["schema_version"]:
            raise DetectionError("Mixed package/chunk schema versions")
        if manifest["schema_version"] == 2 and any(r.get("scene_config_hash") != object_hash(descriptor.get("scene_filter")) for r in records):
            raise DetectionError("Record scene filtering differs from pinned run rules")
        if entry.get("processed_frames") != meta["processed_frames"]:
            raise DetectionError("Root processed-frame count mismatch")
        records_by_chunk.append((config_hash,chunk,records if collect_records else None))
        coverage.setdefault((config_hash,chunk["camera_id"]),[]).append((chunk["start"],chunk["stop"]))
    # Guard against forged/overlapping/shuffled plan ranges, even if every chunk has a valid hash.
    for ranges in coverage.values():
        ordered = sorted(ranges)
        if any(a[1] > b[0] for a,b in zip(ordered,ordered[1:])):
            raise DetectionError("Duplicate source frames across chunks")
    if require_complete:
        marker = root/"COMPLETE.json"
        if manifest.get("status") != "COMPLETE" or not marker.exists() or seen != set(expected):
            raise DetectionError("Interrupted output/upload: run is not complete")
        complete = read_json(marker)
        if complete.get("manifest_sha256") != sha256(root/"run_manifest.json") or complete.get("chunk_count") != len(seen):
            raise DetectionError("Completion marker mismatch")
        if complete.get("processed_frames") != sum(e["processed_frames"] for e in manifest["chunks"]):
            raise DetectionError("Completion frame count mismatch")
        for h in descriptor["configurations"]:
            for cid,camera in evidence["cameras"].items():
                v = manifest.get("decode_validation",{}).get(h+":"+cid,{})
                if v.get("status") != "VALIDATED" or v.get("frame_count") != camera["frame_count"]:
                    raise DetectionError("Missing complete source decode validation")
    return manifest,records_by_chunk
