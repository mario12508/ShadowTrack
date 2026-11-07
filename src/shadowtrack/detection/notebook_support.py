"""Small notebook helpers; inference, schema and persistence remain in common modules."""

import os
import tempfile
import urllib.request
from pathlib import Path

from ..probe import sha256
from .common import DetectionError,atomic_json,read_json
from .quality import CHECKS


def pinned_weights(lock_path: Path, path: Path, allow_download=False):
    return _pinned_weights(read_json(lock_path),path,allow_download)


def pinned_detection_weights(lock_path, model_key, path, allow_download=False):
    lock = read_json(lock_path)
    if lock.get("task") != "detect" or model_key not in lock["models"]:
        raise DetectionError("Unknown pinned detection model")
    return _pinned_weights(lock["models"][model_key],Path(path),allow_download)


def _pinned_weights(lock, path, allow_download):
    if path.is_file():
        if path.stat().st_size != lock["weights_bytes"] or sha256(path) != lock["weights_sha256"]:
            raise DetectionError("Weights do not match pinned size/SHA256")
        return path
    if not allow_download:
        raise DetectionError("Attach the pinned weights as a private input or explicitly allow the small public weights download")
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name = tempfile.mkstemp(prefix=".pending-model-",dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with urllib.request.urlopen(lock["weights_url"],timeout=120) as response,temporary.open("wb") as stream:
            received = 0
            while block := response.read(1024*1024):
                received += len(block)
                if received > lock["weights_bytes"]:
                    raise DetectionError("Weights response exceeds pinned model size")
                stream.write(block)
        if received != lock["weights_bytes"] or sha256(temporary) != lock["weights_sha256"]:
            raise DetectionError("Downloaded weights size/SHA256 mismatch")
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def review_template(sample_root: Path, destination: Path):
    root = read_json(sample_root/"run_manifest.json")
    if root.get("status") != "COMPLETE" or root["descriptor"]["plan"]["mode"] != "SAMPLE":
        raise DetectionError("Review template requires a complete SAMPLE")
    template = {"schema_version":1,"sample_manifest_sha256":sha256(sample_root/"run_manifest.json"),
                "benchmark_sha256":sha256(sample_root/"benchmark.json"),"selected_config_hash":None,
                "approved_full":False,"reviewer":"","checks":{k:{"checked":False,"notes":""} for k in CHECKS},
                "instructions":"Review both configurations in PRIVATE quality files, set checks/notes and one selected_config_hash. This template is not approval."}
    atomic_json(destination,template)
    return destination
