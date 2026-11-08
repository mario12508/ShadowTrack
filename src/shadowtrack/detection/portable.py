"""Versioned source-code artifact, deliberately excluding videos and image outputs."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from ..probe import sha256
from .common import DetectionError,atomic_json,canonical,code_fingerprint,read_json,safe_path


def build_code_artifact(root: Path, output: Path, evidence_dir: Path | None = None, existing_evidence=False):
    root,output = root.resolve(),output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    files = sorted((root/"src").glob("**/*.py"))
    files += [root/"pyproject.toml",root/"configs/detection_model_lock.json"]
    files += [root/name for name in ("configs/detection_v2_lock.json","configs/detection_v2.json","configs/scene_filter_v1.json","configs/local_track_v2.toml") if (root/name).exists()]
    files += sorted((root/"licenses").glob("*.txt"))
    files += sorted((root/"schemas").glob("*.json"))
    for name in ("README.md","docs/KAGGLE_SETUP.md","docs/DETECTION_REPORT.md","docs/KAGGLE_DETECTION_V2.md","notebooks/kaggle_detection.ipynb","notebooks/kaggle_detection_v2.ipynb"):
        if (root/name).is_file():
            if name.endswith(".ipynb"):
                notebook = read_json(root/name)
                if any(c.get("outputs") or c.get("execution_count") is not None or c.get("attachments") for c in notebook["cells"]):
                    raise DetectionError("Code artifact notebook must have cleared outputs, execution counts and attachments")
            files.append(root/name)
    sources = {p.relative_to(root).as_posix():p for p in files}
    if evidence_dir is not None:
        from .evidence import load_evidence
        evidence = load_evidence(Path(evidence_dir))
        if not existing_evidence and evidence["manifest"]["producer_code_hash"] != code_fingerprint(root):
            raise DetectionError("Bundled source evidence refers to another code version")
        sources["input_evidence/source_manifest.json"] = Path(evidence_dir)/"source_manifest.json"
        for camera in evidence["manifest"]["cameras"].values():
            sources["input_evidence/"+camera["frames_file"]] = safe_path(Path(evidence_dir),camera["frames_file"])
    manifest = {"schema_version":1,"artifact_kind":"shadowtrack_detection_code","code_hash":code_fingerprint(root),
                "files":{name:sha256(p) for name,p in sources.items()}}
    path = output/f"shadowtrack_detection_code_{manifest['code_hash'][:16]}.zip"
    if path.exists():
        raise DetectionError("Code artifact already exists; use a fresh output directory for a different package snapshot")
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,p in sources.items():
            archive.write(p,name)
        archive.writestr("code_manifest.json",canonical(manifest))
    directory = output/path.stem
    directory.mkdir()
    for name,p in sources.items():
        target = directory/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
    atomic_json(directory/"code_manifest.json",manifest)
    result = {"filename":path.name,"directory":directory.name,"sha256":sha256(path),"code_hash":manifest["code_hash"],
              "includes_plain_source_evidence":evidence_dir is not None,"contains_source_video_or_frames":False}
    atomic_json(output/"code_artifact.json",result)
    return result


def verify_code_tree(root: Path):
    manifest = read_json(root/"code_manifest.json")
    if manifest.get("schema_version") != 1:
        raise DetectionError("Unsupported code artifact")
    for name,digest in manifest["files"].items():
        if sha256(safe_path(root,name)) != digest:
            raise DetectionError("Versioned code artifact changed")
    if manifest["code_hash"] != code_fingerprint(root):
        raise DetectionError("Common detection code fingerprint mismatch")
    return manifest["code_hash"]
