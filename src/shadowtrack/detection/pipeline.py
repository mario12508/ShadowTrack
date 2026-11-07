"""One inference implementation shared by Kaggle Notebook and CPU contract tests."""

from __future__ import annotations

import importlib.metadata
import shutil
import time
from pathlib import Path

from .common import DetectionError, atomic_json, canonical, code_fingerprint, object_hash, read_json
from .decoder import checked_frames,pyav_frames
from .evidence import load_evidence,sample_plan
from .quality import CHECKS,PrivateQuality
from .runtime import YoloSegmenter,environment
from .schema import frame_record
from .store import RunStore,chunk_name,validate_run
from .progress import observed


def validate_full_review(review_path, sample_root, evidence, lock):
    from ..probe import sha256
    if review_path is None or sample_root is None:
        raise DetectionError("FULL requires a separately reviewed COMPLETE SAMPLE run")
    sample,_ = validate_run(Path(sample_root),evidence)
    review = read_json(review_path)
    benchmark = Path(sample_root)/"benchmark.json"
    if (sample["descriptor"]["plan"]["mode"] != "SAMPLE" or len(sample["descriptor"]["configurations"]) != 2
            or sample["descriptor"]["backend_kind"] != "ultralytics" or sample["descriptor"]["model"] != lock
            or not benchmark.is_file() or review.get("benchmark_sha256") != sha256(benchmark)
            or review.get("sample_manifest_sha256") != sha256(Path(sample_root)/"run_manifest.json")
            or review.get("approved_full") is not True or not review.get("reviewer")
            or not all(review.get("checks",{}).get(k,{}).get("checked") is True for k in CHECKS)):
        raise DetectionError("FULL review is absent, unchecked, synthetic, stale, or references a different model/sample")
    if read_json(benchmark).get("gpu_benchmark_status") != "MEASURED":
        raise DetectionError("FULL requires a real completed GPU SAMPLE comparison; CPU-only/mixed runs are not a GPU benchmark")
    h = review.get("selected_config_hash")
    if h not in sample["descriptor"]["configurations"]:
        raise DetectionError("FULL review must select one compared configuration")
    return sample["descriptor"]["configurations"][h]


def benchmark(store):
    from .common import read_json,safe_path
    real = store.manifest["descriptor"]["backend_kind"] == "ultralytics"
    used_runtime_ids = {read_json(safe_path(store.root,e["path"])/"manifest.json")["metrics"]["runtime_id"] for e in store.manifest["chunks"]}
    gpu_only = bool(used_runtime_ids) and all(store.manifest["runtimes"][r]["selected_device"].startswith("cuda") for r in used_runtime_ids)
    result = {"schema_version":1,"status":("MEASURED" if store.manifest["chunks"] else "NOT_RUN") if real else "SYNTHETIC_TEST_ONLY",
              "gpu_benchmark_status":"MEASURED" if real and gpu_only else "NOT_RUN_OR_CPU_ONLY",
              "comparison":"same source SAMPLE windows; nano model, imgsz 640 vs 960",
              "throughput_definition":"processed frames / synchronized predict wall seconds; excludes source decoding, mask serialization, warmup and checkpoint I/O",
              "quality_metrics":"manual review required; no ground-truth precision/recall or AP claimed",
              "configurations":{},"runtimes":store.manifest["runtimes"]}
    for h,cfg in store.manifest["descriptor"]["configurations"].items():
        chunks = [read_json(safe_path(store.root,e["path"])/"manifest.json") for e in store.manifest["chunks"]
                  if e["path"].split("/")[1] == h]
        frames = sum(c["processed_frames"] for c in chunks)
        elapsed = sum(c["metrics"]["inference_seconds"] for c in chunks)
        peak = [c["metrics"].get("peak_vram_allocated_bytes") for c in chunks if c["metrics"].get("peak_vram_allocated_bytes") is not None]
        result["configurations"][h] = {"config":cfg,"processed_frames":frames,"detections":sum(c["detection_count"] for c in chunks),
                  "empty_frames":sum(c["empty_frames"] for c in chunks),"inference_seconds":elapsed,
                  "throughput_fps":frames/elapsed if elapsed > 0 else None,"peak_vram_allocated_bytes":max(peak) if peak else None,
                  "runtime_ids":sorted({c["metrics"]["runtime_id"] for c in chunks}),
                  "quality_review":{k:{"checked":False,"notes":""} for k in CHECKS}}
    atomic_json(store.root/"benchmark.json",result)
    return result


@observed("run")
def run(evidence_dir: Path, video_dir: Path, weights: Path, lock_path: Path, output: Path,
        repo_root: Path, mode="SAMPLE", device="auto", save_private_quality=True,
        chunk_frames=128, review_path=None, sample_root=None, max_wall_seconds=10000.,
        predictor_factory=None, decoder=pyav_frames, runtime_override=None, sample_seconds=30., stop_after_chunks=None,
        progress=None):
    """No upload. Full sequential decode is necessary even in SAMPLE to verify EOF.

    Predictor/decoder injection is for synthetic CPU tests, labelled in all outputs.
    Completed chunks may survive interruption, but mismatch invalidates the run.
    """
    with progress.stage("load_evidence",announce=True):
        evidence,lock = load_evidence(Path(evidence_dir)),read_json(lock_path)
    if evidence["manifest"]["producer_code_hash"] != code_fingerprint(repo_root):
        raise DetectionError("Notebook code differs from versioned source evidence; rebuild portable artifacts")
    with progress.stage("environment",announce=True):
        runtime = runtime_override or environment(device)
    progress.update("device_selected",device=runtime["selected_device"],mode=mode)
    if runtime["selected_device"] == "cpu":
        progress.update("cpu_fallback",message="CUDA is not selected; YOLO will run on CPU and can be slow")
    if predictor_factory is None:
        if runtime["packages"].get("av") != lock["pyav_version"]:
            raise DetectionError("PyAV decoder version is not pinned version")
        predictor_factory = lambda cfg:YoloSegmenter(Path(weights),lock,cfg,runtime["selected_device"],progress=progress)
        backend_kind = "ultralytics"
    else:
        backend_kind = "synthetic_test"
    if mode == "FULL" and (backend_kind != "ultralytics" or not runtime["selected_device"].startswith("cuda")):
        raise DetectionError("FULL requires the real pinned backend on an explicitly available GPU plus reviewed SAMPLE")
    configs = lock["configurations"] if mode == "SAMPLE" else [validate_full_review(review_path,sample_root,evidence,lock)]
    plan = sample_plan(evidence,sample_seconds,chunk_frames,mode)
    config_map = {object_hash(cfg):cfg for cfg in configs}
    descriptor = {"evidence_manifest_hash":evidence["manifest_hash"],"sources":evidence["manifest"]["cameras"],
                  "model":lock,"configurations":config_map,"plan":plan,"code_hash":code_fingerprint(repo_root),
                  "backend_kind":backend_kind}
    output = Path(output).resolve()
    if output.is_relative_to(Path(video_dir).resolve()) or output.is_relative_to(Path(evidence_dir).resolve()):
        raise DetectionError("Output must be separate from read-only input mounts")
    progress.attach(output.parent/"telemetry"/output.name)
    with progress.stage("checkpoint_validation",announce=True):
        store = RunStore(output,descriptor,evidence,progress=progress)
    total_decode = sum(c["frame_count"] for c in evidence["cameras"].values())*len(config_map)
    total_inference = sum(plan["selected_frame_counts"].values())*len(config_map)
    processed = sum(c["processed_frames"] for c in store.completed.values())
    progress.update("plan",total_decode_frames=total_decode,total_inference_frames=total_inference,
                    decoded_frames_session=0,processed_frames=processed,inferred_frames_session=0,
                    committed_frames=processed,
                    chunks_done=len(store.completed),chunks_total=len(plan["chunks"])*len(config_map),
                    message="SAMPLE still decodes ALL source frames to EOF for EACH config; only selected windows use YOLO")
    if store.manifest["status"] == "COMPLETE":
        with progress.stage("complete_run_validation",announce=True):
            validate_run(output,evidence)
        progress.update("resume_complete",message="Validated COMPLETE checkpoint; no inference or decoding needed")
        return store.manifest  # Idempotent resume: no decoder or model calls.
    store.manifest["status"] = "IN_PROGRESS"
    runtime_id = object_hash(runtime)
    store.manifest["runtimes"][runtime_id] = runtime
    store.save()
    started,committed = time.perf_counter(),0
    predictor = None
    decoded,inferred,detection_count = 0,0,0
    try:
        for config_hash,cfg in config_map.items():
            progress.update("config_start",config_name=cfg["name"],config_hash=config_hash)
            with progress.stage("model_initialization",announce=True):
                predictor = predictor_factory(cfg)
            for cid,camera in evidence["cameras"].items():
                camera_started = time.perf_counter()
                progress.update("camera_start",camera_id=cid,camera_decoded_frames=0,camera_total_frames=camera["frame_count"],
                                source_frame_index=None,raw_media_pts=None,camera_wall_fps=None,camera_eta_s=None)
                chunks = [c for c in plan["chunks"] if c["camera_id"] == cid]
                windows = [w for w in plan["windows"] if w["camera_id"] == cid]
                quality = PrivateQuality(output/"_private_quality"/config_hash/f"camera_{cid}",cid,config_hash,save_private_quality)
                needed = {i:c for c in chunks if chunk_name(config_hash,c) not in store.completed for i in range(c["start"],c["stop"])}
                records,elapsed,peak_alloc,peak_reserved = [],0.,None,None
                checked = 0
                progress.update("camera_plan",pending_inference_frames=len(needed),message=f"pending YOLO frames={len(needed)}; full decode frames={camera['frame_count']}")
                for i,frame in checked_frames(Path(video_dir)/camera["source_name"],camera,output/"diagnostics"/f"{config_hash}_{cid}.json",decoder,progress=progress):
                    checked += 1
                    decoded += 1
                    wall = time.perf_counter()-camera_started
                    rate = checked/wall if wall > 0 else 0
                    progress.update(decoded_frames_session=decoded,camera_wall_fps=round(rate,2),
                                    camera_eta_s=round((camera["frame_count"]-checked)/rate,1) if rate else None)
                    if time.perf_counter()-started > max_wall_seconds:
                        raise InterruptedError("Configured session budget reached; committed chunks preserved")
                    if i not in needed:
                        continue
                    chunk = needed[i]
                    if i == chunk["start"]:
                        progress.update("inference_chunk_start",message=f"YOLO frames [{chunk['start']}, {chunk['stop']}); segment={chunk['segment_id']}")
                    with progress.stage("rgb_conversion"):
                        rgb = frame.to_ndarray(format="rgb24")
                    with progress.stage("inference_and_masks"):
                        detections,metrics = predictor.predict(rgb)
                    with progress.stage("frame_record"):
                        record = frame_record(camera,i,detections,lock["weights_sha256"],config_hash)
                    inferred += 1
                    detection_count += len(detections)
                    progress.update(inferred_frames_session=inferred,processed_frames=processed+inferred,
                                    detections_session=detection_count,vram_allocated_bytes=metrics.get("peak_vram_allocated_bytes"))
                    records.append(record)
                    elapsed += metrics["inference_seconds"]
                    for name,value in [("alloc",metrics.get("peak_vram_allocated_bytes")),("reserved",metrics.get("peak_vram_reserved_bytes"))]:
                        if value is not None:
                            if name == "alloc": peak_alloc = max(peak_alloc or 0,value)
                            else: peak_reserved = max(peak_reserved or 0,value)
                    for window in windows:
                        if window["first_frame"] <= i < window["stop_frame"]:
                            with progress.stage("private_quality"):
                                quality.add(rgb,record,window)
                    if i == chunk["stop"]-1:
                        # Reserve room for one final data-only snapshot plus the next chunk.
                        with progress.stage("disk_reserve_check"):
                            existing = sum(p.stat().st_size for p in (output/"chunks").rglob("*") if p.is_file()) if (output/"chunks").exists() else 0
                            pending_upper_bound = sum(len(canonical(r))+1 for r in records)
                            if shutil.disk_usage(output).free < existing+pending_upper_bound+512*1024*1024:
                                raise InterruptedError("Low output disk space; export existing checkpoint before continuing")
                        with progress.stage("chunk_commit",announce=True):
                            store.commit(config_hash,chunk,records,{"inference_seconds":elapsed,
                                         "peak_vram_allocated_bytes":peak_alloc,"peak_vram_reserved_bytes":peak_reserved,"runtime_id":runtime_id})
                        progress.update("chunk_committed",chunks_done=len(store.completed),
                                        committed_frames=sum(c["processed_frames"] for c in store.completed.values()),
                                        message="Chunk saved; continuing full decode to next selected window or EOF")
                        records,elapsed,peak_alloc,peak_reserved = [],0.,None,None
                        committed += 1
                        if stop_after_chunks is not None and committed >= stop_after_chunks:
                            raise InterruptedError("Synthetic test interruption")
                with progress.stage("quality_gallery_save",announce=True):
                    quality.finish()
                store.manifest["decode_validation"][config_hash+":"+cid] = {"status":"VALIDATED","frame_count":checked,
                             "source_sha256":camera["source_sha256"],"audited_frames_sha256":camera["audited_frames_sha256"],"runtime_id":runtime_id}
                store.save()
                progress.update("camera_complete",message=f"Full EOF count and PTS verified: {checked} frames")
            with progress.stage("model_close",announce=True):
                predictor.close()
            predictor = None
        with progress.stage("benchmark_and_complete",announce=True):
            result = benchmark(store)
            store.manifest["benchmark_status"] = result["status"]
            store.finish()
        progress.update("run_complete",message="All planned chunks saved and full decode validation complete")
        return store.manifest
    except BaseException as exc:
        progress.update("run_error",error_type=type(exc).__name__,message="Saving interruption state; committed chunks are preserved")
        diagnostic_files = list((output/"diagnostics").glob("*.json")) if (output/"diagnostics").exists() else []
        store.manifest["status"] = "FAILED_DECODE" if diagnostic_files else "INTERRUPTED"
        store.manifest["last_error_type"] = type(exc).__name__
        store.save()
        benchmark(store)
        raise
    finally:
        if predictor is not None:
            predictor.close()
