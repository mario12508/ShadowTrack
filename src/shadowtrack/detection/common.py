"""Deterministic hashes and durable filesystem transactions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from ..probe import AuditError, sha256


class DetectionError(AuditError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def object_hash(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, data):
    atomic_bytes(path, canonical(data))


def safe_path(root: Path, relative: str) -> Path:
    # Reject Windows paths on Linux as well as traversal and absolute ZIP names.
    if not isinstance(relative, str) or "\\" in relative or ":" in relative or "\x00" in relative:
        raise DetectionError("Invalid relative artifact path")
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(root.resolve()):
        raise DetectionError("Artifact path escapes its root")
    return path


def code_fingerprint(root: Path) -> str:
    paths = sorted((root/"src/shadowtrack/detection").glob("*.py")) + [root/"src/shadowtrack/probe.py"]
    return object_hash({p.relative_to(root).as_posix(): sha256(p) for p in paths})
