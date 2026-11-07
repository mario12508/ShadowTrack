"""FFprobe evidence, preserving media time independently of the burnt-in clock."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path


class AuditError(RuntimeError):
    """An expected inspection error, suitable for a user-facing report."""


def binary(name: str) -> str:
    candidate = os.environ.get(f"SHADOWTRACK_{name.upper()}", name)
    resolved = shutil.which(candidate)
    if not resolved:
        raise AuditError(f"{name} not found; add it to PATH or set SHADOWTRACK_{name.upper()}")
    return resolved


def run(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AuditError(str(exc)) from exc


def number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def rate(value) -> float | None:
    try:
        result = float(Fraction(str(value)))
        return result if result > 0 and math.isfinite(result) else None
    except (ValueError, ZeroDivisionError):
        return None


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def timestamp_stats(frames: list[dict]) -> dict:
    pts = [number(f.get("best_effort_timestamp_time")) for f in frames]
    valid = [v for v in pts if v is not None]
    # Do not bridge missing timestamps: those are unknown intervals.
    deltas = [round(b - a, 9) for a, b in zip(pts, pts[1:])
              if a is not None and b is not None]
    positive = [d for d in deltas if d > 0]
    median = statistics.median(positive) if positive else None
    normal = [d for d in positive if d <= 1.5 * median] if median else []
    gaps = [{"previous_frame_index": i, "next_frame_index": i + 1,
             "before_pts_seconds": a, "after_pts_seconds": b,
             "gap_seconds": round(b - a, 9)}
            for i, (a, b) in enumerate(zip(pts, pts[1:]))
            if a is not None and b is not None and median and b - a > 1.5 * median]
    span = valid[-1] - valid[0] if len(valid) > 1 else None
    return {
        "decoded_frames": len(frames), "timestamped_frames": len(valid),
        "missing_pts": len(pts) - len(valid),
        "first_pts_seconds": valid[0] if valid else None,
        "last_pts_seconds": valid[-1] if valid else None,
        "pts_span_seconds": span,
        "observed_fps": (len(valid) - 1) / span if span and span > 0 else None,
        "cadence_fps_excluding_gaps": len(normal) / sum(normal) if normal else None,
        "median_interval_seconds": median,
        "min_interval_seconds": min(deltas) if deltas else None,
        "max_interval_seconds": max(deltas) if deltas else None,
        "duplicate_pts": sum(d == 0 for d in deltas),
        "backwards_pts": sum(d < 0 for d in deltas),
        "gaps_over_1_5_median": sum(d > 1.5 * median for d in positive) if median else 0,
        "gaps": gaps,
        "interval_histogram_seconds": dict(Counter(str(round(d, 6)) for d in deltas)),
    }


def probe_video(path: Path, evidence: Path, ffprobe: str, timeout: float) -> dict:
    evidence.mkdir(parents=True, exist_ok=True)
    commands = [
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        [ffprobe, "-v", "error", "-threads", "2", "-select_streams", "v:0",
         "-show_frames", "-show_entries",
         "frame=best_effort_timestamp,best_effort_timestamp_time,pts,pts_time,pkt_dts_time,duration_time,pkt_duration_time,key_frame,pict_type",
         "-of", "json", str(path)],
    ]
    payloads = []
    errors = []
    for label, command in zip(("probe", "frames"), commands):
        result = run(command, timeout)
        (evidence / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        (evidence / f"{label}.json").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise AuditError(f"ffprobe {label} failed ({result.returncode}): {result.stderr[-1500:]}")
        if result.stderr.strip():
            errors.append(f"ffprobe {label}: {result.stderr[-1500:]}")
        try:
            payloads.append(json.loads(result.stdout))
        except json.JSONDecodeError as exc:
            raise AuditError(f"Invalid ffprobe JSON: {exc}") from exc
        if label == "probe" and not any(s.get("codec_type") == "video" for s in payloads[0].get("streams", [])):
            raise AuditError("No video stream")
    meta, frame_data = payloads
    video = next(s for s in meta["streams"] if s.get("codec_type") == "video")
    frames = frame_data.get("frames", [])
    stats = timestamp_stats(frames)
    if not stats["decoded_frames"] or stats["missing_pts"] == stats["decoded_frames"]:
        raise AuditError("No decoded frames with usable timestamps")
    return {
        "file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        "codec": video.get("codec_name"), "profile": video.get("profile"),
        "width": video.get("width"), "height": video.get("height"),
        "pixel_format": video.get("pix_fmt"),
        "r_frame_rate": video.get("r_frame_rate"), "avg_frame_rate": video.get("avg_frame_rate"),
        "nominal_fps": rate(video.get("r_frame_rate")),
        "header_average_fps": rate(video.get("avg_frame_rate")),
        "declared_frames": video.get("nb_frames"),
        "time_base": video.get("time_base"), "video_start_seconds": number(video.get("start_time")),
        "video_duration_seconds": number(video.get("duration")),
        "video_duration_tag": video.get("tags", {}).get("DURATION"),
        "container_start_seconds": number(meta["format"].get("start_time")),
        "container_duration_seconds": number(meta["format"].get("duration")),
        "format_tags": meta["format"].get("tags", {}),
        "audio_streams": [s for s in meta["streams"] if s.get("codec_type") == "audio"],
        "timestamps": stats, "decode_errors": errors,
        "commands": commands,
    }
