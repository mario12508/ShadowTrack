"""Inspection orchestration. Bad inputs produce reports and a nonzero exit code."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from . import __version__
from .previews import make_previews
from .probe import AuditError, binary, probe_video, run, sha256
from .report import write_report
from .review import attach_review


def inventory_metadata(directory: Path) -> dict:
    files = []
    if directory.is_dir():
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            item = {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                try:
                    with Image.open(path) as im:
                        item["width"], item["height"] = im.size
                        im.verify()
                    item["kind"] = "raster_plan_candidate"
                except (OSError, ValueError) as exc:
                    item["error"] = str(exc)
            files.append(item)
    usable = any(f.get("kind") == "raster_plan_candidate" for f in files)
    return {"directory": str(directory), "files": files, "has_raster_plan": usable,
            "calibration": "not_inferred", "camera_mapping": "filename_numbers_are_unverified_labels"}


def consistency_issues(cameras: list[dict], tolerance: float = 0.02) -> list[dict]:
    issues = []
    valid = [c for c in cameras if "timestamps" in c]
    for camera in valid:
        timing = camera["timestamps"]
        measured = timing["observed_fps"]
        rates = [camera.get("nominal_fps"), camera.get("header_average_fps")]
        declared_start = camera.get("video_start_seconds")
        decoded_start = timing.get("first_pts_seconds")
        if declared_start is not None and decoded_start is not None and abs(decoded_start - declared_start) > 0.1:
            issues.append({"code": "decoded_start_mismatch", "severity": "warning", "file": camera["file"],
                           "message": f"Declared video start {declared_start}s; first decoded frame {decoded_start}s."})
        if measured and any(r and abs(r - measured) / measured > tolerance for r in rates):
            issues.append({"code": "fps_header_mismatch", "severity": "warning", "file": camera["file"],
                           "message": f"Header FPS {rates}; observed FPS {measured:.6f} from decoded PTS."})
        if timing["missing_pts"] or timing["duplicate_pts"] or timing["backwards_pts"]:
            issues.append({"code": "invalid_timestamps", "severity": "error", "file": camera["file"],
                           "message": "Missing, duplicate or backwards decoded PTS; see timestamps."})
        if timing["gaps_over_1_5_median"]:
            issues.append({"code": "timestamp_gaps", "severity": "warning", "file": camera["file"],
                           "message": f"{timing['gaps_over_1_5_median']} intervals exceed 1.5 x median; not proof of dropped frames."})
        cadence = timing.get("cadence_fps_excluding_gaps")
        if cadence and any(r and abs(r - cadence) / cadence > tolerance for r in rates):
            issues.append({"code": "fps_cadence_mismatch", "severity": "warning", "file": camera["file"],
                           "message": f"Header FPS {rates}; cadence excluding gaps {cadence:.6f} FPS."})
    for field in ("nominal_fps", "header_average_fps", "observed_fps"):
        values = [c["timestamps"].get(field) if field == "observed_fps" else c.get(field) for c in valid]
        values = [v for v in values if v]
        if values and (max(values) - min(values)) / min(values) > tolerance:
            issues.append({"code": "inconsistent_fps", "severity": "warning", "message": f"Across cameras, {field}: {values}"})
    return issues


def inspect(raw_dir: Path, metadata_dir: Path, output_dir: Path, previews: int = 20,
            expected_cameras: int = 5, timeout_seconds: float = 1800,
            fps_relative_tolerance: float = 0.02, review_file: Path | None = None) -> dict:
    raw_dir, metadata_dir, output = (p.resolve() for p in (raw_dir, metadata_dir, output_dir))
    # Never write audit results inside the source directories.
    if output.is_relative_to(raw_dir) or output.is_relative_to(metadata_dir):
        raise AuditError("output_dir must be outside raw_dir and metadata_dir")
    if not 16 <= previews <= 24:
        raise AuditError("previews must be between 16 and 24")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "shadowtrack_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "options": {"raw_dir": str(raw_dir), "metadata_dir": str(metadata_dir),
                    "output_dir": str(output), "previews": previews, "expected_cameras": expected_cameras,
                    "timeout_seconds": timeout_seconds, "fps_relative_tolerance": fps_relative_tolerance},
        "metadata": inventory_metadata(metadata_dir), "cameras": [], "issues": [],
        "synchronization": {"status": "unverified", "applied_offsets_seconds": None,
                            "reason": "Media PTS and burnt-in clock are distinct. No event-derived correction applied."},
    }
    issues = report["issues"]
    if not report["metadata"]["has_raster_plan"]:
        issues.append({"code": "missing_metadata", "severity": "warning", "message": "No readable raster plan found. Calibration and camera locations remain unknown."})
    paths = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mkv") if raw_dir.is_dir() else []
    if not paths:
        issues.append({"code": "empty_raw", "severity": "error", "message": "No MKV files in raw_dir."})
    if len(paths) != expected_cameras:
        issues.append({"code": "camera_count", "severity": "warning", "message": f"Expected {expected_cameras} MKV files, found {len(paths)}."})
    try:
        ffprobe, ffmpeg = binary("ffprobe"), binary("ffmpeg")
        report["tools"] = {name: run([executable, "-version"], 15).stdout.splitlines()[0]
                           for name, executable in (("ffprobe", ffprobe), ("ffmpeg", ffmpeg))}
    except AuditError as exc:
        if paths:
            issues.append({"code": "missing_tool", "severity": "error", "message": str(exc)})
        ffprobe = ffmpeg = None
    if ffprobe and ffmpeg:
        for i, path in enumerate(paths, start=1):
            # A fixed safe directory name prevents filenames from becoming HTML paths.
            key = f"camera_{i:02d}"
            folder = output / key
            folder.mkdir(exist_ok=True)
            camera = {"file": str(path), "key": key, "status": "error"}
            print(f"[{i}/{len(paths)}] Inspecting {path.name}", flush=True)
            try:
                camera.update(probe_video(path, folder / "timestamps", ffprobe, timeout_seconds))
                if camera["decode_errors"]:
                    issues.append({"code": "decode_errors", "severity": "error", "file": str(path),
                                   "message": "FFprobe reported decoding errors; inspect saved stderr."})
                camera["previews"] = make_previews(path, folder, camera["timestamps"], previews, ffmpeg, timeout_seconds, camera["container_start_seconds"] or 0)
                camera["contact_sheet"] = f"{key}/contact_sheet.jpg"
                camera["status"] = "ok" if not camera["decode_errors"] else "error"
                print(f"  frames={camera['timestamps']['decoded_frames']}, observed_fps={camera['timestamps']['observed_fps']}, previews={len(camera['previews'])}", flush=True)
            except (AuditError, OSError, ValueError) as exc:
                camera["error"] = str(exc)
                issues.append({"code": "video_error", "severity": "error", "file": str(path), "message": str(exc)})
            report["cameras"].append(camera)
    issues.extend(consistency_issues(report["cameras"], fps_relative_tolerance))
    report["options"]["review_file"] = str(review_file.resolve()) if review_file else None
    attach_review(report, review_file)
    report["status"] = "error" if any(i["severity"] == "error" for i in issues) else "warning" if issues else "ok"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(report, output)
    return report
