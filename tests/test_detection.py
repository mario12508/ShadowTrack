import copy
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from shadowtrack.probe import sha256
from shadowtrack.detection.common import DetectionError,atomic_bytes,atomic_json,canonical,code_fingerprint,object_hash,read_json
from shadowtrack.detection.decoder import checked_frames
from shadowtrack.detection.evidence import load_evidence,sample_plan
from shadowtrack.detection.pipeline import run
from shadowtrack.detection.runtime import environment
from shadowtrack.detection.schema import encode_rle,decode_rle,frame_record,validate_frame
from shadowtrack.detection.store import RunStore,validate_run,validate_chunk
from shadowtrack.detection.transport import export_run,export_run_directory,import_run,restore_checkpoint,unpack_export
from shadowtrack.detection.tracking import replay,local_track

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT/"configs/detection_model_lock.json"
TRACKER = {"track_activation_threshold":.25,"lost_track_buffer":30,"minimum_matching_threshold":.8,"minimum_consecutive_frames":1}
RUNTIME = {"selected_device":"cpu","torch_cuda_is_available":False,"packages":{},"gpu":None,"nvidia_smi":{"status":"UNAVAILABLE"}}


class FakeFrame:
    width,height = 32,24
    def __init__(self,index): self.index = index
    def to_ndarray(self,format):
        return np.full((self.height,self.width,3),self.index,dtype=np.uint8)


class FakeSegmenter:
    calls = 0
    def predict(self,rgb):
        FakeSegmenter.calls += 1
        i = int(rgb[0,0,0])
        detections = []
        if i % 7:
            mask = np.zeros(rgb.shape[:2],dtype=np.uint8)
            mask[5:18,4:12] = 1
            detections = [{"bbox_xyxy":[4.,5.,12.,18.],"confidence":.9,"class_id":0,"mask_rle":encode_rle(mask)}]
        return detections,{"inference_seconds":.001,"peak_vram_allocated_bytes":None,"peak_vram_reserved_bytes":None}
    def close(self): pass


@pytest.fixture
def sources(tmp_path):
    videos,evidence_dir = tmp_path/"videos",tmp_path/"evidence"
    videos.mkdir()
    evidence_dir.mkdir()
    manifest = {"schema_version":1,"package_kind":"shadowtrack_source_evidence","audit_stats_sha256":"a"*64,
                "time_alignment_sha256":"b"*64,"producer_code_hash":code_fingerprint(ROOT),"cameras":{}}
    for cid,fps in [("1",25),("4",20)]:
        path = videos/f"camera_{cid}.mkv"
        path.write_bytes(b"synthetic saved-audit fixture "+cid.encode())
        frames = [[i,round(i/fps+(10 if i>=10 else 0),3),f"{cid}:s{int(i>=10)}"] for i in range(20)]
        filename = f"frames_{cid}.json.gz"
        atomic_bytes(evidence_dir/filename,gzip.compress(canonical(frames),mtime=0))
        segments = [{"id":f"{cid}:s0","frame_start":0,"frame_end":9,"raw_start":frames[0][1],"raw_end":frames[9][1]},
                    {"id":f"{cid}:s1","frame_start":10,"frame_end":19,"raw_start":frames[10][1],"raw_end":frames[19][1]}]
        manifest["cameras"][cid] = {"camera_id":cid,"source_name":path.name,"source_sha256":sha256(path),"frame_count":20,
             "width":32,"height":24,"nominal_fps":fps,"frames_file":filename,"frames_sha256":sha256(evidence_dir/filename),
             "frames_content_sha256":object_hash(frames),"audited_frames_sha256":object_hash(frames),"segments":segments}
    atomic_json(evidence_dir/"source_manifest.json",manifest)
    evidence = load_evidence(evidence_dir)
    def decoder(path):
        cid = path.stem.split("_")[-1]
        for i,pts,segment in evidence["cameras"][cid]["frames"]:
            yield round(pts*1000),Fraction(1,1000),FakeFrame(i)
    return {"videos":videos,"directory":evidence_dir,"evidence":evidence,"decoder":decoder,"tmp":tmp_path}


def run_synthetic(sources,output=None,**kwargs):
    private_quality = kwargs.pop("save_private_quality",False)
    return run(sources["directory"],sources["videos"],Path("unused-test-weights"),LOCK,output or sources["tmp"]/"run",
               ROOT,save_private_quality=private_quality,predictor_factory=lambda cfg:FakeSegmenter(),
               decoder=sources["decoder"],runtime_override=RUNTIME,sample_seconds=.6,chunk_frames=3,**kwargs)


def test_rle_column_major_roundtrip():
    mask = np.zeros((24,32),dtype=np.uint8)
    mask[0,0] = 1
    mask[5:18,4:12] = 1
    rle = encode_rle(mask)
    assert rle["counts"][0] == 0
    assert np.array_equal(decode_rle(rle),mask)


def test_empty_frame_marker_and_original_pts(sources):
    camera = sources["evidence"]["cameras"]["1"]
    record = frame_record(camera,15,[],"a"*64,"b"*64)
    validate_frame(record,camera,"a"*64,"b"*64)
    assert record["processed"] and record["frame_status"] == "empty"
    assert record["raw_media_pts"] == 10.6  # not frame_index / FPS
    assert record["timestamp_segment_id"] == "1:s1"


def detection_record(sources,index=1):
    camera = sources["evidence"]["cameras"]["1"]
    detections,_ = FakeSegmenter().predict(FakeFrame(index).to_ndarray("rgb24"))
    return camera,frame_record(camera,index,detections,"a"*64,"b"*64)


@pytest.mark.parametrize("mutation",["mask","class","confidence","bbox","pts","processed"])
def test_invalid_detection_schema_rejected(sources,mutation):
    camera,record = detection_record(sources)
    d = record["detections"][0]
    if mutation == "mask": d.pop("mask_rle")
    elif mutation == "class": d["class_id"] = 2
    elif mutation == "confidence": d["confidence"] = float("nan")
    elif mutation == "bbox": d["bbox_xyxy"] = [0,0,100,100]
    elif mutation == "pts": record["raw_media_pts"] = .041
    else: record.pop("processed")
    with pytest.raises(DetectionError):
        validate_frame(record,camera,"a"*64,"b"*64)


def test_sample_distributed_across_timestamp_segments_and_different_fps(sources):
    plan = sample_plan(sources["evidence"],.6,3)
    assert len(plan["windows"]) == 4
    assert plan["selected_frame_counts"]["1"] != plan["selected_frame_counts"]["4"]
    for chunk in plan["chunks"]:
        camera = sources["evidence"]["cameras"][chunk["camera_id"]]
        assert len({r[2] for r in camera["frames"][chunk["start"]:chunk["stop"]]}) == 1


def test_pipeline_complete_and_idempotent_resume_without_inference(sources):
    root = sources["tmp"]/"run"
    manifest = run_synthetic(sources)
    _,chunks = validate_run(root,sources["evidence"],collect_records=True)
    assert manifest["status"] == "COMPLETE"
    assert manifest["benchmark_status"] == "SYNTHETIC_TEST_ONLY"
    assert all(v["frame_count"]==20 for v in manifest["decode_validation"].values())
    assert sum(len(rows) for h,c,rows in chunks) > 0
    before = FakeSegmenter.calls
    run_synthetic(sources)
    assert FakeSegmenter.calls == before
    assert all(rows is None for _,_,rows in validate_run(root,sources["evidence"])[1])


def test_checkpoint_resume_and_incomplete_import_rejected(sources):
    root = sources["tmp"]/"run"
    with pytest.raises(InterruptedError):
        run_synthetic(sources,stop_after_chunks=1)
    manifest = read_json(root/"run_manifest.json")
    assert manifest["status"] == "INTERRUPTED" and len(manifest["chunks"]) == 1
    archive = export_run(root,sources["tmp"]/"checkpoint.zip")
    with pytest.raises(DetectionError,match="Incomplete"):
        import_run(archive,sources["directory"],sources["tmp"]/"imports",LOCK,allow_synthetic=True,raw_dir=sources["videos"])
    restored = restore_checkpoint(archive,sources["tmp"]/"restored")
    prior = read_json(restored/"run_manifest.json")["chunks"][0]
    digest = sha256(restored/prior["path"]/"frames.jsonl.gz")
    run_synthetic(sources,output=restored)
    assert sha256(restored/prior["path"]/"frames.jsonl.gz") == digest
    validate_run(restored,sources["evidence"])


def test_corrupt_chunk_rejected(sources):
    run_synthetic(sources)
    root = sources["tmp"]/"run"
    entry = read_json(root/"run_manifest.json")["chunks"][0]
    (root/entry["path"]/"frames.jsonl.gz").write_bytes(b"truncated")
    with pytest.raises(DetectionError,match="SHA256"):
        validate_run(root,sources["evidence"])


@pytest.mark.parametrize("kind",["shuffle","duplicate","missing_mask"])
def test_chunk_semantics_checked_even_when_file_hash_is_recomputed(sources,kind):
    run_synthetic(sources)
    root = sources["tmp"]/"run"
    manifest = read_json(root/"run_manifest.json")
    entry = next(e for e in manifest["chunks"] if e["processed_frames"]==3)
    folder = root/entry["path"]
    meta = read_json(folder/"manifest.json")
    records = [json.loads(v) for v in gzip.decompress((folder/"frames.jsonl.gz").read_bytes()).splitlines()]
    if kind == "shuffle": records.reverse()
    elif kind == "duplicate": records[1] = copy.deepcopy(records[0])
    else:
        record = next(r for r in records if r["detections"])
        record["detections"][0].pop("mask_rle")
    atomic_bytes(folder/"frames.jsonl.gz",gzip.compress(b"\n".join(canonical(r) for r in records)+b"\n",mtime=0))
    meta["records_sha256"] = sha256(folder/"frames.jsonl.gz")
    meta["records_content_sha256"] = hashlib.sha256(gzip.decompress((folder/"frames.jsonl.gz").read_bytes())).hexdigest()
    atomic_json(folder/"manifest.json",meta)
    camera = sources["evidence"]["cameras"][meta["camera_id"]]
    with pytest.raises(DetectionError):
        validate_chunk(folder,camera,meta["model_hash"],meta["config_hash"])


def test_interrupted_output_without_complete_marker(sources):
    run_synthetic(sources)
    root = sources["tmp"]/"run"
    (root/"COMPLETE.json").unlink()
    with pytest.raises(DetectionError,match="Interrupted"):
        validate_run(root,sources["evidence"])


@pytest.mark.parametrize("kind",["missing","extra","shuffled"])
def test_decoder_count_order_mismatch_creates_data_only_diagnostic(sources,kind):
    camera = sources["evidence"]["cameras"]["1"]
    items = list(sources["decoder"](sources["videos"]/camera["source_name"]))
    if kind == "missing": items.pop()
    elif kind == "extra": items.append(items[-1])
    else: items[2],items[3] = items[3],items[2]
    diagnostic = sources["tmp"]/"diagnostic.json"
    with pytest.raises(DetectionError):
        list(checked_frames(sources["videos"]/camera["source_name"],camera,diagnostic,lambda path:iter(items)))
    report = read_json(diagnostic)
    assert report["status"] == "DECODE_MISMATCH" and report["images_saved"] is False


def test_decode_failure_invalidates_run_even_with_committed_chunks(sources):
    original = sources["decoder"]
    def short(path):
        yield from list(original(path))[:-1]
    sources["decoder"] = short
    with pytest.raises(DetectionError):
        run_synthetic(sources)
    root = sources["tmp"]/"run"
    assert read_json(root/"run_manifest.json")["status"] == "FAILED_DECODE"
    assert not (root/"COMPLETE.json").exists()
    with pytest.raises(DetectionError,match="cannot be resumed"):
        run_synthetic(sources)


def test_wrong_video_hash_blocks_decode_and_import(sources):
    camera = sources["evidence"]["cameras"]["1"]
    run_synthetic(sources)
    archive = export_run(sources["tmp"]/"run",sources["tmp"]/"complete.zip")
    (sources["videos"]/camera["source_name"]).write_bytes(b"wrong video")
    with pytest.raises(DetectionError):
        list(checked_frames(sources["videos"]/camera["source_name"],camera,sources["tmp"]/"bad.json",sources["decoder"]))
    with pytest.raises(DetectionError,match="Local source"):
        import_run(archive,sources["directory"],sources["tmp"]/"imports",LOCK,allow_synthetic=True,raw_dir=sources["videos"])


def test_interrupted_upload_and_path_traversal_rejected(sources):
    run_synthetic(sources)
    archive = export_run(sources["tmp"]/"run",sources["tmp"]/"complete.zip")
    broken = sources["tmp"]/"missing_entry.zip"
    with zipfile.ZipFile(archive) as src,zipfile.ZipFile(broken,"w") as dst:
        for name in src.namelist()[1:]: dst.writestr(name,src.read(name))
    with pytest.raises(DetectionError,match="inventory"):
        unpack_export(broken,sources["tmp"]/"out")
    with zipfile.ZipFile(sources["tmp"]/"escape.zip","w") as dst:
        dst.writestr("../escape.txt","no")
        dst.writestr("export_manifest.json","{}")
    with pytest.raises(DetectionError,match="escapes"):
        unpack_export(sources["tmp"]/"escape.zip",sources["tmp"]/"out2")


def test_import_schema_version_and_synthetic_backend_guard(sources):
    run_synthetic(sources)
    root = sources["tmp"]/"run"
    archive = export_run(root,sources["tmp"]/"complete.zip")
    with pytest.raises(DetectionError,match="Synthetic"):
        import_run(archive,sources["directory"],sources["tmp"]/"imports",LOCK,raw_dir=sources["videos"])
    manifest = read_json(root/"run_manifest.json")
    manifest["schema_version"] = 2
    atomic_json(root/"run_manifest.json",manifest)
    with pytest.raises(DetectionError,match="schema_version"):
        validate_run(root,sources["evidence"])


def test_private_quality_excluded_from_default_export(sources):
    run_synthetic(sources)
    root = sources["tmp"]/"run"
    sensitive = root/"_private_quality"/"private.jpg"
    sensitive.parent.mkdir()
    sensitive.write_bytes(b"synthetic private placeholder")
    archive = export_run(root,sources["tmp"]/"data_only.zip")
    with zipfile.ZipFile(archive) as z:
        assert not any(n.startswith("_private_quality/") for n in z.namelist())


def test_private_synthetic_gallery_and_short_preview_are_not_cell_outputs(sources):
    run_synthetic(sources,save_private_quality=True)
    root = sources["tmp"]/"run"
    galleries = list((root/"_private_quality").glob("**/gallery.jpg"))
    clips = list((root/"_private_quality").glob("**/annotated_preview.gif"))
    assert len(galleries) == len(clips) == 4
    for path in (root/"_private_quality").glob("**/quality_index.json"):
        index = read_json(path)
        assert index["private_only"] is True and index["stills"]
        assert all(not c["checked"] for c in index["checks"].values())
    archive = export_run(root,sources["tmp"]/"data_only.zip")
    with zipfile.ZipFile(archive) as z:
        assert not any(name.lower().endswith((".jpg",".gif",".png")) for name in z.namelist())


def test_cpu_bytetrack_replay_deterministic_and_resets_at_jumps(sources):
    camera = sources["evidence"]["cameras"]["1"]
    rows = []
    for i in range(20):
        detections,_ = FakeSegmenter().predict(FakeFrame(i).to_ndarray("rgb24"))
        rows.append(frame_record(camera,i,detections,"a"*64,"b"*64))
    one,two = list(replay(rows,camera,TRACKER)),list(replay(rows,camera,TRACKER))
    assert canonical(one) == canonical(two)
    assert one[0]["input_empty"] is True
    before = {t["local_track_id"] for r in one[:10] for t in r["local_tracks"]}
    after = {t["local_track_id"] for r in one[10:] for t in r["local_tracks"]}
    assert before and after and before.isdisjoint(after)
    assert [r["raw_media_pts"] for r in one] == [r[1] for r in camera["frames"]]
    noncontiguous = list(replay([rows[1],rows[2],rows[8]],camera,TRACKER))
    assert noncontiguous[0]["local_tracks"][0]["local_track_id"] != noncontiguous[-1]["local_tracks"][0]["local_track_id"]


def test_import_and_local_track_end_to_end_with_frozen_detections(sources):
    run_synthetic(sources)
    archive = export_run(sources["tmp"]/"run",sources["tmp"]/"complete.zip")
    imported = import_run(archive,sources["directory"],sources["tmp"]/"imports",LOCK,allow_synthetic=True,raw_dir=sources["videos"])
    assert import_run(archive,sources["directory"],sources["tmp"]/"imports",LOCK,allow_synthetic=True,raw_dir=sources["videos"]) == imported
    config = sources["tmp"]/"track.toml"
    paths = {"detection_run":str(imported),"evidence_dir":str(sources["directory"]),"raw_dir":str(sources["videos"]),"output_dir":str(sources["tmp"]/"tracks")}
    text = "[paths]\n"+"\n".join(f'{k} = {json.dumps(v.replace(chr(92),"/"))}' for k,v in paths.items())
    text += '\n[tracking]\nconfiguration_name="nano_640"\n[bytetrack]\n'+"\n".join(f"{k}={v}" for k,v in TRACKER.items())
    config.write_text(text,encoding="utf-8")
    calls = FakeSegmenter.calls
    first,second = local_track(config),local_track(config)
    assert canonical(first) == canonical(second)
    assert FakeSegmenter.calls == calls
    assert first["global_association"] is False and len(first["files"]) == 2
    assert "torch" not in sys.modules


def test_full_requires_reviewed_gpu_sample(sources):
    with pytest.raises(DetectionError,match="FULL requires"):
        run_synthetic(sources,mode="FULL")


def test_full_review_binds_sample_benchmark_and_all_quality_checks(sources):
    from shadowtrack.detection.pipeline import validate_full_review
    from shadowtrack.detection.quality import CHECKS
    run_synthetic(sources)
    root = sources["tmp"]/"run"
    manifest = read_json(root/"run_manifest.json")
    # Hypothetical producer metadata ONLY in this synthetic fixture; no GPU measurement.
    manifest["descriptor"]["backend_kind"] = "ultralytics"
    manifest["descriptor_hash"] = object_hash(manifest["descriptor"])
    manifest["run_id"] = manifest["descriptor_hash"][:24]
    atomic_json(root/"run_manifest.json",manifest)
    marker = read_json(root/"COMPLETE.json")
    marker["manifest_sha256"] = sha256(root/"run_manifest.json")
    atomic_json(root/"COMPLETE.json",marker)
    benchmark = read_json(root/"benchmark.json")
    benchmark["gpu_benchmark_status"] = "MEASURED"
    atomic_json(root/"benchmark.json",benchmark)
    selected = next(iter(manifest["descriptor"]["configurations"]))
    review = {"approved_full":True,"reviewer":"synthetic review fixture",
              "sample_manifest_sha256":sha256(root/"run_manifest.json"),"benchmark_sha256":sha256(root/"benchmark.json"),
              "checks":{k:{"checked":True} for k in CHECKS},"selected_config_hash":selected}
    path = sources["tmp"]/"review.json"
    atomic_json(path,review)
    assert validate_full_review(path,root,sources["evidence"],read_json(LOCK)) == manifest["descriptor"]["configurations"][selected]
    for kind in ("approval","check","hash"):
        invalid = copy.deepcopy(review)
        if kind == "approval": invalid["approved_full"] = False
        elif kind == "check": invalid["checks"][CHECKS[0]]["checked"] = False
        else: invalid["benchmark_sha256"] = "0"*64
        atomic_json(path,invalid)
        with pytest.raises(DetectionError,match="FULL review"):
            validate_full_review(path,root,sources["evidence"],read_json(LOCK))


def test_notebook_clean_and_imports_shared_implementation():
    import ast,nbformat
    notebook = nbformat.read(ROOT/"notebooks/kaggle_detection.ipynb",as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert 'MODE = "SAMPLE"' in code and "EXECUTE = False" in code
    assert "from shadowtrack.detection.pipeline import run" in code
    assert "YOLO(" not in code and "imshow(" not in code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not cell.outputs and cell.execution_count is None
            ast.parse(cell.source)


def test_code_artifact_is_versioned_and_data_free(tmp_path):
    from shadowtrack.detection.portable import build_code_artifact,verify_code_tree
    artifact = build_code_artifact(ROOT,tmp_path/"artifact")
    archive = tmp_path/"artifact"/artifact["filename"]
    assert sha256(archive) == artifact["sha256"]
    assert verify_code_tree(tmp_path/"artifact"/artifact["directory"]) == artifact["code_hash"]
    with zipfile.ZipFile(archive) as z:
        assert not any(n.lower().endswith((".mkv",".jpg",".png",".gif",".pt")) for n in z.namelist())
        z.extractall(tmp_path/"code")
    assert verify_code_tree(tmp_path/"code") == artifact["code_hash"]
    with (tmp_path/"code/src/shadowtrack/detection/schema.py").open("a") as stream: stream.write("\n# changed\n")
    with pytest.raises(DetectionError,match="changed"):
        verify_code_tree(tmp_path/"code")


@pytest.mark.parametrize("nested",[False,True])
def test_notebook_bootstrap_uses_unpacked_input_folders(tmp_path,monkeypatch,nested):
    import nbformat
    from shadowtrack.detection.portable import build_code_artifact
    artifact = build_code_artifact(ROOT,tmp_path/"dataset"/"nested")
    code_dir = tmp_path/"dataset" if nested else tmp_path/"dataset"/"nested"/artifact["directory"]
    notebook = nbformat.read(ROOT/"notebooks/kaggle_detection.ipynb",as_version=4)
    bootstrap = next(c.source for c in notebook.cells if c.cell_type == "code" and c.source.startswith("if not EXECUTE:"))
    monkeypatch.setattr(sys,"path",list(sys.path))
    namespace = {"Path":Path,"EXECUTE":True,"MODE":"SAMPLE","CODE_DIR":code_dir}
    previous = {k:v for k,v in sys.modules.items() if k == "shadowtrack" or k.startswith("shadowtrack.")}
    try:
        exec(compile(bootstrap,"notebook_bootstrap","exec"),namespace)
        assert Path(sys.modules["shadowtrack"].__file__).resolve().is_relative_to(namespace["CODE_ROOT"])
    finally:
        for name in tuple(sys.modules):
            if name == "shadowtrack" or name.startswith("shadowtrack."):
                del sys.modules[name]
        sys.modules.update(previous)
    assert namespace["CODE_HASH"] == artifact["code_hash"]
    find = namespace["input_file"]
    with pytest.raises(RuntimeError,match="found 0"):
        find(tmp_path/"missing","source_manifest.json")
    for name in ("a","b"):
        path = tmp_path/"ambiguous"/name/"source_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}",encoding="utf-8")
    with pytest.raises(RuntimeError,match="found 2"):
        find(tmp_path/"ambiguous","source_manifest.json")


def test_folder_checkpoint_resume_and_directory_import(sources):
    root = sources["tmp"]/"run"
    with pytest.raises(InterruptedError):
        run_synthetic(sources,stop_after_chunks=1)
    snapshot = export_run_directory(root,sources["tmp"]/"partial_folder")
    restored = restore_checkpoint(snapshot,sources["tmp"]/"restored_folder")
    prior = read_json(restored/"run_manifest.json")["chunks"][0]
    digest = sha256(restored/prior["path"]/"frames.jsonl.gz")
    run_synthetic(sources,output=restored)
    assert sha256(restored/prior["path"]/"frames.jsonl.gz") == digest
    complete = export_run_directory(restored,sources["tmp"]/"complete_folder")
    assert not (complete/"_private_quality").exists()
    imported = import_run(complete,sources["directory"],sources["tmp"]/"folder_imports",LOCK,
                          allow_synthetic=True,raw_dir=sources["videos"])
    validate_run(imported,sources["evidence"])


@pytest.mark.parametrize("damage",["missing","hash","extra"])
def test_folder_checkpoint_rejects_interrupted_or_corrupt_upload(sources,damage):
    run_synthetic(sources)
    snapshot = export_run_directory(sources["tmp"]/"run",sources["tmp"]/"folder")
    entry = next(iter(read_json(snapshot/"export_manifest.json")["files"]))
    if damage == "missing": (snapshot/entry).unlink()
    elif damage == "hash": (snapshot/entry).write_bytes(b"corrupt")
    else: (snapshot/"unexpected.json").write_text("{}",encoding="utf-8")
    target = sources["tmp"]/"rejected_restore"
    with pytest.raises(DetectionError,match="inventory|SHA256"):
        restore_checkpoint(snapshot,target)
    assert not target.exists()
    assert not list(target.parent.glob(".staging-*"))


def test_interrupted_folder_export_is_not_visible(sources,monkeypatch):
    run_synthetic(sources)
    target = sources["tmp"]/"failed_export"
    def fail(*args,**kwargs): raise OSError("synthetic disk failure")
    monkeypatch.setattr("shadowtrack.detection.transport.shutil.copyfile",fail)
    with pytest.raises(OSError,match="disk failure"):
        export_run_directory(sources["tmp"]/"run",target)
    assert not target.exists()
    assert not list(target.parent.glob(".staging-*"))


@pytest.mark.parametrize("layout",["plain","gzip","auto_unpacked"])
def test_evidence_loads_plain_gzip_and_kaggle_unpacked_payload(sources,layout):
    directory = sources["directory"]
    manifest = read_json(directory/"source_manifest.json")
    for camera in manifest["cameras"].values():
        path = directory/camera["frames_file"]
        if layout != "gzip":
            unpacked = path.with_suffix("")
            unpacked.write_bytes(gzip.decompress(path.read_bytes()))
            path.unlink()
            if layout == "plain":
                camera.update(frames_file=unpacked.name,frames_sha256=sha256(unpacked))
    atomic_json(directory/"source_manifest.json",manifest)
    loaded = load_evidence(directory)
    assert all(loaded["cameras"][cid]["frames"] == old["frames"] for cid,old in sources["evidence"]["cameras"].items())


@pytest.mark.parametrize("damage",["corrupt","legacy_no_hash","missing"])
def test_unpacked_evidence_requires_verified_content(sources,damage):
    directory = sources["directory"]
    manifest = read_json(directory/"source_manifest.json")
    camera = manifest["cameras"]["1"]
    path = directory/camera["frames_file"]
    unpacked = path.with_suffix("")
    unpacked.write_bytes(gzip.decompress(path.read_bytes()))
    path.unlink()
    if damage == "corrupt": unpacked.write_bytes(b"[]")
    elif damage == "legacy_no_hash": camera.pop("frames_content_sha256")
    else: unpacked.unlink()
    atomic_json(directory/"source_manifest.json",manifest)
    with pytest.raises(DetectionError,match="content SHA256|missing evidence"):
        load_evidence(directory)


def test_prepared_bundle_includes_matching_plain_evidence(sources):
    from shadowtrack.detection.portable import build_code_artifact,verify_code_tree
    directory = sources["directory"]
    manifest = read_json(directory/"source_manifest.json")
    for camera in manifest["cameras"].values():
        path = directory/camera["frames_file"]
        plain = path.with_suffix("")
        plain.write_bytes(gzip.decompress(path.read_bytes()))
        camera.update(frames_file=plain.name,frames_sha256=sha256(plain))
    atomic_json(directory/"source_manifest.json",manifest)
    artifact = build_code_artifact(ROOT,sources["tmp"]/"bundle",directory)
    folder = sources["tmp"]/"bundle"/artifact["directory"]
    assert verify_code_tree(folder) == artifact["code_hash"]
    assert artifact["includes_plain_source_evidence"] is True
    bundled = load_evidence(folder/"input_evidence")
    assert bundled["manifest"]["producer_code_hash"] == artifact["code_hash"]
    assert not list((folder/"input_evidence").glob("*.gz"))
    with zipfile.ZipFile(sources["tmp"]/"bundle"/artifact["filename"]) as archive:
        assert "input_evidence/frames_1.json" in archive.namelist()


@pytest.mark.parametrize("complete",[False,True])
def test_kaggle_auto_unpacked_checkpoint_restore_and_import(sources,complete):
    if complete:
        run_synthetic(sources)
    else:
        with pytest.raises(InterruptedError):
            run_synthetic(sources,stop_after_chunks=1)
    snapshot = export_run_directory(sources["tmp"]/"run",sources["tmp"]/"unpacked_snapshot")
    for path in snapshot.rglob("*.gz"):
        path.with_suffix("").write_bytes(gzip.decompress(path.read_bytes()))
        path.unlink()
    restored = restore_checkpoint(snapshot,sources["tmp"]/"unpacked_restored")
    run_synthetic(sources,output=restored)
    validate_run(restored,sources["evidence"])
    if complete:
        imported = import_run(snapshot,sources["directory"],sources["tmp"]/"unpacked_imports",LOCK,
                              allow_synthetic=True,raw_dir=sources["videos"])
        validate_run(imported,sources["evidence"])


def test_auto_unpacked_chunk_cannot_bypass_payload_hash(sources):
    run_synthetic(sources)
    root = sources["tmp"]/"run"
    entry = read_json(root/"run_manifest.json")["chunks"][0]
    folder = root/entry["path"]
    path = folder/"frames.jsonl.gz"
    path.with_suffix("").write_bytes(b"{}\n")
    path.unlink()
    meta = read_json(folder/"manifest.json")
    with pytest.raises(DetectionError,match="content SHA256"):
        validate_chunk(folder,sources["evidence"]["cameras"][meta["camera_id"]],meta["model_hash"],meta["config_hash"])


def test_heartbeat_survives_blocked_call_and_stdout_redirection(tmp_path):
    import contextlib,io,threading
    from shadowtrack.detection.progress import Progress
    heard = threading.Event()
    class Stream(io.StringIO):
        def write(self,value):
            result = super().write(value)
            if " heartbeat " in value:
                heard.set()
            return result
    original,redirected = Stream(),io.StringIO()
    with Progress("blocked_fixture",interval=.01,stream=original) as progress:
        progress.attach(tmp_path/"telemetry")
        with progress.stage("yolo_predict"):
            with contextlib.redirect_stdout(redirected):
                assert heard.wait(3), "Heartbeat must work while the foreground call is blocked"
    assert not progress.thread.is_alive()
    assert "phase=yolo_predict" in original.getvalue() and not redirected.getvalue()
    records = [json.loads(line) for line in (tmp_path/"telemetry/events.jsonl").read_text(encoding="utf-8").splitlines()]
    heartbeat = next(r for r in records if r["event"] == "heartbeat")
    assert heartbeat["phase"] == "yolo_predict" and heartbeat["no_counter_progress_s"] > 0
    final = read_json(tmp_path/"telemetry/progress.json")
    assert final["status"] == "COMPLETE" and final["stage_seconds_inclusive"]["yolo_predict"] > 0


def test_progress_failure_records_stage_and_closes_thread(tmp_path):
    import io
    from shadowtrack.detection.progress import Progress
    progress = Progress("failure",interval=.01,stream=io.StringIO())
    with pytest.raises(RuntimeError,match="fixture failure"):
        with progress:
            progress.attach(tmp_path/"telemetry")
            with progress.stage("chunk_commit",announce=True):
                raise RuntimeError("fixture failure")
    record = read_json(tmp_path/"telemetry/progress.json")
    assert record["status"] == "FAILED" and record["error_type"] == "RuntimeError"
    assert record["failed_phase"] == "chunk_commit"
    assert "chunk_commit" in record["stage_seconds_inclusive"] and not progress.thread.is_alive()


def test_progress_counters_cover_full_decode_and_only_sample_inference(sources):
    import io
    from shadowtrack.detection.progress import Progress
    stream = io.StringIO()
    with Progress("sample",interval=.01,stream=stream) as progress:
        manifest = run_synthetic(sources,progress=progress)
    final = read_json(sources["tmp"]/"telemetry/run/progress.json")
    expected_frames = sum(sample_plan(sources["evidence"],.6,3)["selected_frame_counts"].values())*2
    assert final["decoded_frames_session"] == final["total_decode_frames"] == 80
    assert final["processed_frames"] == final["inferred_frames_session"] == final["total_inference_frames"] == expected_frames
    assert final["committed_frames"] == expected_frames
    assert final["chunks_done"] == final["chunks_total"] == len(manifest["chunks"])
    assert "decode_next_frame" in final["stage_seconds_inclusive"]
    assert "camera_eta_s=" in stream.getvalue() and "source_sha256_start" in stream.getvalue()
    before = [sha256(sources["tmp"]/"run"/entry["path"]/"frames.jsonl.gz") for entry in manifest["chunks"]]
    quiet = run_synthetic(sources,output=sources["tmp"]/"quiet",progress=False)
    assert [sha256(sources["tmp"]/"quiet"/entry["path"]/"frames.jsonl.gz") for entry in quiet["chunks"]] == before


def test_interrupted_run_progress_is_saved(sources):
    with pytest.raises(InterruptedError):
        run_synthetic(sources,stop_after_chunks=1)
    final = read_json(sources["tmp"]/"telemetry/run/progress.json")
    assert final["status"] == "INTERRUPTED" and final["chunks_done"] == 1
    assert final["error_type"] == "InterruptedError"


def test_progress_rejects_image_arrays_and_can_be_disabled(tmp_path):
    import io
    from shadowtrack.detection.progress import Progress
    with Progress(stream=io.StringIO()) as progress:
        with pytest.raises(TypeError,match="scalar metadata"):
            progress.update(image=np.zeros((2,2),dtype=np.uint8))
    stream = io.StringIO()
    with Progress(enabled=False,stream=stream) as quiet:
        quiet.attach(tmp_path/"disabled")
        quiet.update("not_logged",processed_frames=4)
    assert not stream.getvalue() and not (tmp_path/"disabled").exists()


def test_kaggle_download_zip_allows_safe_directory_entries(sources):
    run_synthetic(sources,progress=False)
    archive = export_run(sources["tmp"]/"run",sources["tmp"]/"flat.zip",progress=False)
    download = sources["tmp"]/"kaggle_download.zip"
    with zipfile.ZipFile(archive) as original,zipfile.ZipFile(download,"w") as target:
        parents = {p.as_posix()+"/" for name in original.namelist() for p in Path(name).parents if p != Path(".")}
        for name in sorted(parents):
            target.writestr(name,b"")
        for item in original.infolist():
            target.writestr(item,original.read(item.filename))
    imported = import_run(download,sources["directory"],sources["tmp"]/"kaggle_import",LOCK,
                          allow_synthetic=True,raw_dir=sources["videos"],progress=False)
    validate_run(imported,sources["evidence"])


def test_zip_directory_entries_cannot_escape_destination(sources):
    download = sources["tmp"]/"unsafe_directory.zip"
    with zipfile.ZipFile(download,"w") as target:
        target.writestr("../escaped/",b"")
        target.writestr("export_manifest.json",canonical({"schema_version":1,"package_kind":"shadowtrack_detection_export","files":{}}))
    with pytest.raises(DetectionError,match="escapes"):
        unpack_export(download,sources["tmp"]/"reject_directory")


@pytest.mark.integration
@pytest.mark.parametrize("fps",[20,25])
def test_actual_pyav_sequence_matches_saved_ffprobe_pts(tmp_path,fps):
    from shadowtrack.probe import probe_video
    ffmpeg,ffprobe = ROOT/".tools/ffmpeg.exe",ROOT/".tools/ffprobe.exe"
    if not ffmpeg.exists(): pytest.skip("FFmpeg unavailable")
    video = tmp_path/"synthetic.mkv"
    subprocess.run([str(ffmpeg),"-v","error","-f","lavfi","-i",f"testsrc2=size=64x48:rate={fps}:duration=1",
                    "-vf","setpts=PTS+5/TB","-c:v","libx264","-bf","0","-fps_mode","passthrough",str(video)],
                   check=True,capture_output=True,timeout=30)
    probe_video(video,tmp_path/"probe",str(ffprobe),30)
    frames = read_json(tmp_path/"probe/frames.json")["frames"]
    camera = {"camera_id":"synthetic","source_sha256":sha256(video),"frame_count":len(frames),"width":64,"height":48,
              "frames":[[i,float(f["best_effort_timestamp_time"]),"synthetic:s0"] for i,f in enumerate(frames)]}
    result = list(checked_frames(video,camera,tmp_path/"diagnostic.json"))
    assert len(result) == fps and camera["frames"][0][1] == 5.
    assert not (tmp_path/"diagnostic.json").exists()
