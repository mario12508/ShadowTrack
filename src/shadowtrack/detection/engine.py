"""Batched detection v2. One source decode per camera; MP4 is a derivative."""

import time
import copy
from pathlib import Path

from ..probe import sha256
from .common import DetectionError,atomic_json,code_fingerprint,object_hash,read_json,safe_path
from .decoder import checked_frames,pyav_frames
from .evidence import load_evidence,verify_local_sources
from .plan import detection_plan,evidence_identity,model_for_config
from .progress import observed
from .runtime import environment,YoloDetector
from .scene_filter import SceneFilter,translate_crop
from .schema import detection_record
from .store import RunStore,chunk_name,validate_chunk,validate_run
from .video import AnnotatedVideo
from .quality import DetectionQuality


def _demo_videos_valid(root, evidence, plan):
    """A complete derivative needs every camera, its MP4 and its original-PTS sidecar."""
    try:
        videos=read_json(root/"demo_videos.json")
        if not isinstance(videos,dict) or not isinstance(videos.get("videos"),list):
            return False
        entries=videos["videos"]
        if any(not isinstance(v,dict) for v in entries):
            return False
        if (videos.get("status") != "COMPLETE" or len(entries) != len(evidence["cameras"])
                or {v["camera_id"] for v in entries} != set(evidence["cameras"])):
            return False
        return all(v["frames"] == plan["selected_frame_counts"][v["camera_id"]]
                   and sha256(safe_path(root/"demo",v["path"])) == v["sha256"]
                   and sha256(safe_path(root/"demo",v["sidecar"])) == v["sidecar_sha256"] for v in entries)
    except (OSError,ValueError,TypeError,KeyError,DetectionError):
        return False


def _discard_video(writer, progress):
    """Failed encoder cleanup must never interrupt machine-record processing."""
    try:
        writer.close(commit=False)
    except Exception as exc:
        progress.update("video_cleanup_failed_records_preserved",error_type=type(exc).__name__)


def full_demo_evidence(path, evidence, model_lock, configuration, backend, scene):
    if path is None:
        raise DetectionError("FULL requires a successful DEMO directory for the selected configuration")
    root = Path(path)
    manifest,_ = validate_run(root,evidence)
    if (manifest["schema_version"] != 2 or manifest["descriptor"]["plan"]["mode"] != "DEMO"
            or manifest["descriptor"]["model"] != model_lock
            or list(manifest["descriptor"]["configurations"].values()) != [configuration]
            or manifest["descriptor"]["backend_kind"] != backend):
        raise DetectionError("FULL DEMO evidence does not match model/configuration/backend")
    if manifest["descriptor"]["scene_filter"] != scene:
        raise DetectionError("FULL scene rules differ from the successful DEMO")
    if not _demo_videos_valid(root,evidence,manifest["descriptor"]["plan"]):
        raise DetectionError("FULL requires complete DEMO videos and sidecars with matching hashes/counts for every camera")


def _benchmark(store):
    manifest = store.manifest
    real = manifest["descriptor"]["backend_kind"] == "ultralytics"
    configs = {}
    for h,cfg in manifest["descriptor"]["configurations"].items():
        chunks = [read_json(store.root/e["path"]/"manifest.json") for e in manifest["chunks"] if e["path"].split('/')[1] == h]
        frames = sum(c["processed_frames"] for c in chunks)
        elapsed = sum(c["metrics"]["inference_seconds"] for c in chunks)
        peaks = [c["metrics"]["peak_vram_allocated_bytes"] for c in chunks if c["metrics"].get("peak_vram_allocated_bytes") is not None]
        configs[h] = {"config":cfg,"processed_frames":frames,"raw_detections":sum(c["detection_count"] for c in chunks),
                      "empty_frames":sum(c["empty_frames"] for c in chunks),"inference_seconds":elapsed,
                      "predict_fps":frames/elapsed if elapsed else None,"peak_vram_allocated_bytes":max(peaks) if peaks else None}
        configs[h]["scene_counts"] = {k:sum(c["metrics"].get("scene_counts",{}).get(k,0) for c in chunks) for k in ("accepted","rejected","uncertain")}
    wall = sum(s["wall_seconds"] for s in manifest.get("sessions",[]))
    frames = sum(c["processed_frames"] for c in configs.values())
    used_runtime_ids = {read_json(store.root/e["path"]/"manifest.json")["metrics"]["runtime_id"] for e in manifest["chunks"]}
    gpu = real and bool(used_runtime_ids) and all(manifest["runtimes"][i]["selected_device"].startswith("cuda") for i in used_runtime_ids)
    result = {"schema_version":2,"status":("MEASURED" if manifest["chunks"] else "NOT_RUN") if real else "SYNTHETIC_TEST_ONLY",
              "gpu_benchmark_status":"MEASURED" if gpu else "NOT_RUN_OR_CPU_ONLY",
              "configurations":configs,"wall_seconds_all_sessions":wall,
              "end_to_end_processed_fps":frames/wall if wall else None,
              "decode_frames_all_sessions":sum(s["decoded_frames"] for s in manifest.get("sessions",[])),
              "throughput_policy":"end-to-end includes decode, inference, scene decisions, checkpoints and MP4 encoding; resume overhead included",
              "memory_policy":"process peak with potentially multiple models loaded; not isolated model memory",
              "runtimes":manifest["runtimes"],"winner":None,"quality":"manual review pending; no AP claimed"}
    atomic_json(store.root/"benchmark.json",result)
    return result


@observed("detection_v2")
def run_detection(evidence_dir, video_dir, weights_dir, lock_path, scene_path, output, repo_root,
                  mode="BENCHMARK", selected_configuration=None, settings=None, batch_size=8,
                  demo_source=None, predictor_factory=None, decoder=pyav_frames, runtime_override=None,
                  stop_after_chunks=None, max_wall_seconds=10000., device="auto",benchmark_configurations=None,progress=None):
    """Always writes every selected raw frame, including empty/rejected-only frames.

    FULL selects all source frames. Predictor injection is explicitly synthetic.
    A failed derived MP4 never makes already valid machine detections disappear.
    """
    started = time.perf_counter()
    if not 1 <= batch_size <= 16:
        raise DetectionError("Detection batch_size must be between 1 and 16")
    with progress.stage("load_v2_inputs",announce=True):
        evidence,lock,scene = load_evidence(Path(evidence_dir)),read_json(lock_path),SceneFilter(read_json(scene_path))
        runtime = copy.deepcopy(runtime_override or environment(device))
    if lock.get("schema_version") != 2 or lock.get("task") != "detect":
        raise DetectionError("Use the v2 detection model lock")
    real = predictor_factory is None
    backend = "ultralytics" if real else "synthetic_test"
    if real and runtime["packages"].get("av") != lock["pyav_version"]:
        raise DetectionError("PyAV version differs from pinned decoder")
    if mode == "FULL" and real and not runtime["selected_device"].startswith("cuda"):
        raise DetectionError("FULL requires an available CUDA device")
    settings = settings or {"benchmark_seconds":6.,"demo_seconds":30.,"chunk_frames":128,"raw_starts":{}}
    configs = lock["configurations"]
    if mode == "BENCHMARK":
        names = benchmark_configurations or ["nano_640","small_640"]
        configs = [c for c in configs if c["name"] in names]
        if len(configs) != len(set(names)) or not configs:
            raise DetectionError("Unknown/empty benchmark configuration selection")
    else:
        configs = [c for c in configs if c["name"] == selected_configuration]
        if len(configs) != 1:
            raise DetectionError("Explicitly select one benchmarked configuration for DEMO/FULL")
    configs = [{**c,"batch_size":batch_size} for c in configs]
    seconds = settings["benchmark_seconds"] if mode == "BENCHMARK" else settings["demo_seconds"]
    plan = detection_plan(evidence,mode,settings["raw_starts"],seconds,settings.get("chunk_frames",128))
    if mode == "FULL":
        full_demo_evidence(demo_source,evidence,lock,configs[0],backend,scene.config)
    config_map = {object_hash(c):c for c in configs}
    descriptor = {"records_schema_version":2,"task":"detect","model":lock,"configurations":config_map,
        "sources":evidence["manifest"]["cameras"],"evidence_manifest_hash":evidence["manifest_hash"],
        "evidence_identity":evidence_identity(evidence),"source_evidence_producer":evidence["manifest"]["producer_code_hash"],
        "code_hash":code_fingerprint(Path(repo_root)),"scene_filter":scene.config,"plan":plan,
        "execution":{"batch_size":batch_size,"decode_policy":"one sequential full validation pass per camera"},"backend_kind":backend}
    output = Path(output).resolve()
    for protected in (Path(video_dir).resolve(),Path(evidence_dir).resolve()):
        if output.is_relative_to(protected):
            raise DetectionError("Output must be outside source inputs")
    progress.attach(output.parent/"telemetry"/output.name)
    store = RunStore(output,descriptor,evidence,progress)
    if store.manifest["status"] == "COMPLETE":
        verify_local_sources(evidence,Path(video_dir),progress)
        validate_run(output,evidence)
        if mode != "DEMO" or _demo_videos_valid(output,evidence,plan):
            progress.update("resume_complete",message="Validated saved records; no inference or decode")
            return store.manifest
    runtime["effective_precision"] = "synthetic" if not real else "FP16" if runtime["selected_device"].startswith("cuda") and all(c.get("half") for c in configs) else "FP32"
    runtime_id = object_hash(runtime)
    store.manifest["runtimes"][runtime_id] = runtime
    store.manifest["status"] = "IN_PROGRESS"
    (output/"COMPLETE.json").unlink(missing_ok=True)
    store.save()
    total_infer = sum(plan["selected_frame_counts"].values())*len(config_map)
    progress.update("plan",device=runtime["selected_device"],total_decode_frames=sum(c["frame_count"] for c in evidence["cameras"].values()),
        total_inference_frames=total_infer,processed_frames=sum(e["processed_frames"] for e in store.completed.values()),
        chunks_done=len(store.completed),chunks_total=len(plan["chunks"])*len(config_map),
        message="One source decode per camera; raw records are primary, annotated MP4 are derived")
    if real and runtime["selected_device"] == "cpu":
        progress.update("cpu_fallback",message="YOLO detection will run on CPU; select cuda:0 for GPU-only execution")
    predictors,video_results,video_errors = {},[],[]
    decoded,inferred,committed = 0,0,0
    active_writer = None
    try:
        for cid,camera in evidence["cameras"].items():
            cam_started = time.perf_counter()
            progress.update("camera_start",camera_id=cid,camera_total_frames=camera["frame_count"],camera_decoded_frames=0)
            chunks = [c for c in plan["chunks"] if c["camera_id"] == cid]
            by_index = {i:c for c in chunks for i in range(c["start"],c["stop"])}
            pending = {h:{i:c for i,c in by_index.items() if chunk_name(h,c) not in store.completed} for h in config_map}
            buffers = {h:[] for h in config_map}
            def fresh_metrics():
                return {"inference_seconds":0.,"peak_vram_allocated_bytes":None,"runtime_id":runtime_id,
                        "scene_counts":{"accepted":0,"rejected":0,"uncertain":0}}
            metrics = {h:fresh_metrics() for h in config_map}
            cached = {}
            batch = []
            window = next((w for w in plan["windows"] if w["camera_id"] == cid),None)
            quality = {h:DetectionQuality(output/"_private_quality"/h/f"camera_{cid}",window) for h in config_map} if window else {}
            if mode == "DEMO":
                try:
                    active_writer = AnnotatedVideo(output/"demo"/f"camera_{cid}_detected.mp4",camera)
                except Exception as exc:
                    video_errors.append({"camera_id":cid,"error_type":type(exc).__name__})
                    progress.update("video_unavailable",error_type=type(exc).__name__)

            def flush():
                nonlocal inferred,committed,active_writer
                if not batch:
                    return
                display = {}
                for h,cfg in config_map.items():
                    jobs = [(i,rgb) for i,rgb in batch if i in pending[h]]
                    if jobs:
                        progress.update(config_name=cfg["name"])
                        if h not in predictors:
                            with progress.stage("model_load_and_warmup",announce=True):
                                predictors[h] = (predictor_factory(cfg) if not real else YoloDetector(
                                    Path(weights_dir)/model_for_config(descriptor,h)["weights_filename"],model_for_config(descriptor,h),cfg,runtime["selected_device"],progress))
                        with progress.stage("detection_batch"):
                            boxes,timing = predictors[h].predict_batch([rgb for i,rgb in jobs])
                            if len(boxes) != len(jobs):
                                raise DetectionError("Batch detector changed source frame count")
                            crop = scene.crop(camera) if cfg.get("crop_camera5") and cid == "5" else None
                            if crop:
                                a,b,c,d = crop
                                extra,crop_timing = predictors[h].predict_batch([rgb[b:d,a:c] for i,rgb in jobs])
                                if len(extra) != len(jobs):
                                    raise DetectionError("Crop batch detector changed source frame count")
                                boxes = [full+translate_crop(part,crop,camera["width"],camera["height"]) for full,part in zip(boxes,extra)]
                                timing = {**timing,"inference_seconds":timing["inference_seconds"]+crop_timing["inference_seconds"]}
                        for (i,rgb),detections in zip(jobs,boxes):
                            with progress.stage("scene_decisions"):
                                filtered = scene.apply(camera,detections)
                                record = detection_record(camera,i,filtered,model_for_config(descriptor,h)["weights_sha256"],h,scene.hash)
                            buffers[h].append(record)
                            if h in quality:
                                with progress.stage("private_detection_examples"):
                                    quality[h].add(rgb,record)
                            for d in filtered:
                                metrics[h]["scene_counts"][d["scene_filter"]["decision"]] += 1
                            display[i] = record
                            inferred += 1
                            metrics[h]["inference_seconds"] += timing["inference_seconds"]/len(jobs)
                            peak = timing.get("peak_vram_allocated_bytes")
                            if peak is not None:
                                metrics[h]["peak_vram_allocated_bytes"] = max(metrics[h]["peak_vram_allocated_bytes"] or 0,peak)
                            chunk = pending[h][i]
                            if i == chunk["stop"]-1:
                                with progress.stage("chunk_commit",announce=True):
                                    store.commit(h,chunk,buffers[h],metrics[h])
                                buffers[h] = []
                                metrics[h] = fresh_metrics()
                                committed += 1
                                progress.update("chunk_committed",chunks_done=len(store.completed),
                                    processed_frames=sum(e["processed_frames"] for e in store.completed.values()))
                                if stop_after_chunks is not None and committed >= stop_after_chunks:
                                    raise InterruptedError("Synthetic test interruption")
                if active_writer is not None:
                    h = next(iter(config_map))
                    try:
                        with progress.stage("demo_mp4_encode"):
                            for i,rgb in batch:
                                record = display.get(i)
                                if record is None:
                                    chunk = by_index[i]
                                    key = chunk_name(h,chunk)
                                    if key not in cached:
                                        cached.clear()
                                        _,saved = validate_chunk(output/key,camera,model_for_config(descriptor,h)["weights_sha256"],h,chunk)
                                        cached[key] = {r["source_frame_index"]:r for r in saved}
                                    record = cached[key][i]
                                active_writer.add(rgb,record)
                    except Exception as exc:
                        video_errors.append({"camera_id":cid,"error_type":type(exc).__name__})
                        _discard_video(active_writer,progress)
                        active_writer = None
                        progress.update("video_failed_records_preserved",error_type=type(exc).__name__)
                batch.clear()

            last_selected = None
            for i,frame in checked_frames(Path(video_dir)/camera["source_name"],camera,
                    output/"diagnostics"/f"camera_{cid}.json",decoder,progress):
                decoded += 1
                wall = time.perf_counter()-cam_started
                rate = (i+1)/wall if wall else 0
                progress.update(decoded_frames_session=decoded,inferred_frames_session=inferred,
                    camera_wall_fps=round(rate,2),camera_eta_s=round((camera["frame_count"]-i-1)/rate,1) if rate else None,
                    end_to_end_processed_fps=round(inferred/(time.perf_counter()-started),2))
                if time.perf_counter()-started > max_wall_seconds:
                    raise InterruptedError("Session budget reached; machine checkpoints preserved")
                if i not in by_index:
                    flush()
                    continue
                if batch and (i != last_selected+1 or camera["frames"][i][2] != camera["frames"][last_selected][2]):
                    flush()
                if active_writer is not None or any(i in pending[h] for h in config_map):
                    with progress.stage("rgb_conversion"):
                        batch.append((i,frame.to_ndarray(format="rgb24")))
                    if len(batch) >= batch_size:
                        flush()
                last_selected = i
            flush()
            decode_proof = {"status":"VALIDATED","frame_count":camera["frame_count"],"source_sha256":camera["source_sha256"],
                "audited_frames_sha256":camera["audited_frames_sha256"],"runtime_id":runtime_id,"shared_decode_pass":cid}
            for h in config_map:
                store.manifest["decode_validation"][h+":"+cid] = decode_proof.copy()
            store.save()
            if active_writer is not None:
                try:
                    with progress.stage("demo_mp4_finalize",announce=True):
                        video_results.append(active_writer.close())
                except Exception as exc:
                    video_errors.append({"camera_id":cid,"error_type":type(exc).__name__})
                    progress.update("video_finalize_failed_records_preserved",error_type=type(exc).__name__)
                    _discard_video(active_writer,progress)
                finally:
                    active_writer = None
            progress.update("camera_complete",message="Source EOF/PTS verified once for all configurations")
        if mode == "DEMO":
            atomic_json(output/"demo_videos.json",{"schema_version":1,"status":"COMPLETE" if not video_errors and len(video_results)==len(evidence["cameras"]) else "FAILED",
                "videos":video_results,"errors":video_errors,"global_time_status":"NOT_REQUESTED","records_are_primary":True})
        store.manifest.setdefault("sessions",[]).append({"wall_seconds":time.perf_counter()-started,"decoded_frames":decoded,"inferred_frames":inferred})
        result = _benchmark(store)
        store.manifest["benchmark_status"] = result["status"]
        store.finish()
        (output/"DETECTION_REPORT.md").write_text(
            f"# Detection v2 {mode}\n\nMachine detections are primary: {total_infer} planned processed-frame records, including empty frames.\n"
            f"\nStatus: {store.manifest['status']}; backend: {backend}; benchmark: {result['status']}.\n"
            "\nRaw PTS and frame indices are unchanged. MP4 uses rebased source PTS for display, no global IDs.\n"
            "\nScene decisions annotate raw boxes; rejected/uncertain candidates remain in records. Quality review is still required.\n",encoding="utf-8")
        return store.manifest
    except BaseException as exc:
        store.manifest["status"] = "FAILED_DECODE" if list((output/"diagnostics").glob("*.json")) else "INTERRUPTED"
        store.manifest["last_error_type"] = type(exc).__name__
        store.manifest.setdefault("sessions",[]).append({"wall_seconds":time.perf_counter()-started,"decoded_frames":decoded,"inferred_frames":inferred})
        store.save()
        _benchmark(store)
        raise
    finally:
        if active_writer is not None:
            _discard_video(active_writer,progress)
        for predictor in predictors.values():
            predictor.close()
