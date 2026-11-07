"""Local export/restore/import. No network calls or automatic cloud upload."""

from __future__ import annotations

import os
import gzip
import hashlib
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from ..probe import sha256
from .common import DetectionError,atomic_json,canonical,read_json,safe_path
from .evidence import load_evidence,verify_local_sources
from .store import validate_run
from .progress import observed


def _file_metadata(path):
    result = {"sha256":sha256(path),"bytes":path.stat().st_size}
    if path.suffix == ".gz":
        payload = gzip.decompress(path.read_bytes())
        result.update(content_sha256=hashlib.sha256(payload).hexdigest(),content_bytes=len(payload))
    return result


@observed("export_zip")
def export_run(run_dir: Path, destination: Path, include_private_quality=False, progress=None):
    """Also exports incomplete checkpoints; import for tracking still requires COMPLETE."""
    run_dir,destination = Path(run_dir).resolve(),Path(destination).resolve()
    if destination.is_relative_to(run_dir):
        raise DetectionError("Export archive must be outside the run directory")
    manifest = read_json(run_dir/"run_manifest.json")
    files = {}
    progress.update("export_inventory",phase="export_inventory",files_done=0)
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or any(part.startswith((".staging-",".pending-")) for part in path.relative_to(run_dir).parts):
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative.startswith("_private_quality/") and not include_private_quality:
            continue
        if path.is_symlink():
            raise DetectionError("Symlinks are not exportable")
        files[relative] = _file_metadata(path)
        progress.update(files_done=len(files),file=path.name)
    export = {"schema_version":1,"package_kind":"shadowtrack_detection_export","run_status":manifest["status"],
              "run_id":manifest["run_id"],"includes_private_quality":include_private_quality,"files":files}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_name(destination.name+".pending")
    try:
        progress.update("export_write",phase="export_zip_write",files_done=0,files_total=len(files))
        with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED) as archive:
            for index,name in enumerate(files,1):
                archive.write(safe_path(run_dir,name),name)
                progress.update(files_done=index,file=Path(name).name)
            archive.writestr("export_manifest.json",canonical(export))
        os.replace(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(destination.with_suffix(destination.suffix+".sha256.json"),{"sha256":sha256(destination),"bytes":destination.stat().st_size,
                "run_status":manifest["status"],"includes_private_quality":include_private_quality})
    progress.update("export_saved",files_done=len(files),files_total=len(files),message=str(destination))
    return destination


def _remove_staging(path: Path, parent: Path):
    if not path.resolve().is_relative_to(parent.resolve()) or not path.name.startswith(".staging-"):
        raise DetectionError("Refusing to remove an unverified temporary directory")
    shutil.rmtree(path)


@observed("export_folder")
def export_run_directory(run_dir: Path, destination: Path, progress=None):
    """Atomically save a data-only folder snapshot for unpacked Kaggle inputs."""
    run_dir,destination = Path(run_dir).resolve(),Path(destination).resolve()
    if destination == run_dir or destination.is_relative_to(run_dir) or run_dir.is_relative_to(destination):
        raise DetectionError("Export directory must be separate from the run directory")
    if destination.exists():
        raise DetectionError("Export destination already exists; use a new snapshot directory")
    progress.attach(run_dir.parent/"telemetry"/run_dir.name/"export")
    manifest = read_json(run_dir/"run_manifest.json")
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-",dir=destination.parent))
    files = {}
    progress.update("export_folder_start",phase="export_copy_and_sha256",files_done=0,message=str(destination))
    try:
        for path in sorted(run_dir.rglob("*")):
            relative = path.relative_to(run_dir)
            if relative.parts[0] == "_private_quality" or any(p.startswith((".staging-",".pending-")) for p in relative.parts):
                continue
            if path.is_symlink():
                raise DetectionError("Symlinks are not exportable")
            if not path.is_file() or relative.as_posix() == "export_manifest.json":
                continue
            target = safe_path(staging,relative.as_posix())
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
            files[relative.as_posix()] = _file_metadata(target)
            progress.update(files_done=len(files),file=path.name)
        atomic_json(staging/"export_manifest.json",{
            "schema_version":1,"package_kind":"shadowtrack_detection_export","run_status":manifest["status"],
            "run_id":manifest["run_id"],"includes_private_quality":False,"files":files})
        os.rename(staging,destination)
    finally:
        if staging.exists():
            _remove_staging(staging,destination.parent)
    progress.update("export_saved",files_done=len(files),files_total=len(files),message=str(destination))
    return destination


def unpack_export(source: Path, destination: Path, progress=None):
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        all_names = [i.filename for i in infos]
        names = [i.filename for i in infos if not i.is_dir()]
        if len(all_names) != len(set(all_names)) or "export_manifest.json" not in names:
            raise DetectionError("Duplicate/missing archive manifest")
        if sum(i.file_size for i in infos) > 20*1024**3:
            raise DetectionError("Archive exceeds 20 GiB extraction limit")
        for entry in infos:
            safe_path(destination,entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise DetectionError("Archive must not contain symlinks")
        export = __import__("json").loads(archive.read("export_manifest.json"))
        if export.get("schema_version") != 1 or export.get("package_kind") != "shadowtrack_detection_export":
            raise DetectionError("Unsupported export schema")
        if set(names)-{"export_manifest.json"} != set(export["files"]):
            raise DetectionError("Interrupted upload: archive file inventory mismatch")
        for index,(name,meta) in enumerate(export["files"].items(),1):
            if progress is not None:
                progress.update(phase="unpack_and_sha256",files_done=index-1,files_total=len(export["files"]),file=Path(name).name)
            target = safe_path(destination,name)
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(name) as src,target.open("wb") as dst:
                shutil.copyfileobj(src,dst)
            if target.stat().st_size != meta["bytes"] or sha256(target) != meta["sha256"]:
                raise DetectionError("Uploaded file size/SHA256 mismatch")
    atomic_json(destination/"export_manifest.json",export)
    return export


@observed("restore_checkpoint")
def restore_checkpoint(source: Path, output: Path, progress=None):
    output = Path(output).resolve()
    if output.exists():
        raise DetectionError("Restore destination must not already exist")
    output.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-",dir=output.parent))
    try:
        source = Path(source)
        progress.update("restore_start",phase="restore_checkpoint",message=str(source))
        if source.is_dir():
            copy_export_directory(source,staging,progress=progress)
        else:
            unpack_export(source,staging,progress=progress)
        if not (staging/"run_manifest.json").exists():
            raise DetectionError("Checkpoint missing run manifest")
        # RunStore will verify descriptor and all chunks against actual source evidence.
        (staging/"export_manifest.json").unlink()
        os.rename(staging,output)
    finally:
        if staging.exists():
            _remove_staging(staging,output.parent)
    progress.update("restore_saved",message=str(output))
    return output


def copy_export_directory(source: Path, destination: Path, progress=None):
    """Validate and copy an unpacked Kaggle checkpoint, without a ZIP input."""
    source,destination = Path(source).resolve(),Path(destination).resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise DetectionError("Checkpoint source and destination must be separate")
    export = read_json(source/"export_manifest.json")
    if export.get("schema_version") != 1 or export.get("package_kind") != "shadowtrack_detection_export":
        raise DetectionError("Unsupported export schema")
    paths = list(source.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise DetectionError("Checkpoint directory contains symlinks")
    actual = {p.relative_to(source).as_posix() for p in paths if p.is_file()}
    resolved = {}
    for index,(name,meta) in enumerate(export["files"].items(),1):
        if progress is not None:
            progress.update(phase="checkpoint_inventory_sha256",files_done=index-1,files_total=len(export["files"]),file=Path(name).name)
        path = safe_path(source,name)
        if path.is_file():
            digest,size = meta["sha256"],meta["bytes"]
        elif path.suffix == ".gz" and path.with_suffix("").is_file() and meta.get("content_sha256"):
            path = path.with_suffix("")
            digest,size = meta["content_sha256"],meta["content_bytes"]
        else:
            raise DetectionError("Interrupted upload: directory file inventory mismatch; missing file or legacy decompressed checkpoint")
        relative = path.relative_to(source).as_posix()
        if relative in resolved:
            raise DetectionError("Ambiguous decompressed checkpoint paths")
        resolved[relative] = path
        if path.stat().st_size != size or sha256(path) != digest:
            raise DetectionError("Interrupted upload/directory SHA256 mismatch")
    if actual != set(resolved)|{"export_manifest.json"}:
        raise DetectionError("Interrupted upload: directory file inventory mismatch")
    for index,(relative,path) in enumerate(resolved.items(),1):
        if progress is not None:
            progress.update(phase="checkpoint_copy",files_done=index-1,files_total=len(resolved),file=path.name)
        target = safe_path(destination,relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    atomic_json(destination/"export_manifest.json",export)
    return export


@observed("import_detections")
def import_run(source: Path, evidence_dir: Path, target_root: Path, model_lock: Path | None = None, allow_synthetic=False, raw_dir=None, progress=None):
    source,target_root = Path(source).resolve(),Path(target_root).resolve()
    with progress.stage("import_evidence",announce=True):
        evidence = load_evidence(Path(evidence_dir))
    verify_local_sources(evidence,Path(raw_dir) if raw_dir else Path.cwd()/"data/raw",progress=progress)
    target_root.mkdir(parents=True,exist_ok=True)
    if source == target_root or source.is_relative_to(target_root) or target_root.is_relative_to(source):
        raise DetectionError("Import source and destination must be separate")
    staging = Path(tempfile.mkdtemp(prefix=".staging-",dir=target_root))
    try:
        if source.is_file():
            export = unpack_export(source,staging,progress=progress)
        else:
            if not (source/"export_manifest.json").exists():
                raise DetectionError("Directory import requires a verified export_manifest.json; export the Kaggle run first")
            export = copy_export_directory(source,staging,progress=progress)
        if export.get("run_status") != "COMPLETE":
            raise DetectionError("Incomplete checkpoint can be resumed in Kaggle, not imported for tracking")
        with progress.stage("import_validate_chunks",announce=True):
            manifest,_ = validate_run(staging,evidence)
        lock = read_json(model_lock if model_lock is not None else Path(__file__).resolve().parents[3]/"configs"/(
            "detection_v2_lock.json" if manifest["schema_version"] == 2 else "detection_model_lock.json"))
        if manifest["descriptor"]["model"] != lock:
            raise DetectionError("Detection model weights/license/config lock differs from local pin")
        if manifest["descriptor"]["backend_kind"] != "ultralytics" and not allow_synthetic:
            raise DetectionError("Synthetic test detections are not real model output")
        # Private quality images never enter the local result store through the import command.
        if export.get("includes_private_quality"):
            raise DetectionError("Use data-only export for import; keep quality media in explicitly private outputs")
        target = target_root/manifest["run_id"]
        if target.exists():
            if sha256(target/"run_manifest.json") != sha256(staging/"run_manifest.json"):
                raise DetectionError("Run ID already exists with different contents")
            validate_run(target,evidence)
            return target
        atomic_json(staging/"IMPORT_RECEIPT.json",{"schema_version":1,"source_archive_sha256":sha256(source) if source.is_file() else None,
                    "manifest_sha256":sha256(staging/"run_manifest.json"),"source_evidence_hash":evidence["manifest_hash"],
                    "validated":True,"synthetic_test_only":manifest["descriptor"]["backend_kind"] != "ultralytics"})
        os.rename(staging,target)
        progress.update("import_saved",message=str(target))
        return target
    finally:
        if staging.exists():
            _remove_staging(staging,target_root)
