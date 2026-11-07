"""Rebuild presentation artifacts from a completed full probe, checking source hashes.

Useful after changing preview/report layout; does not repeat the full video decode.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from shadowtrack.audit import consistency_issues
from shadowtrack.previews import make_previews
from shadowtrack.probe import binary, sha256, timestamp_stats
from shadowtrack.report import write_report
from shadowtrack.review import attach_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/inspect"))
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args()
    stats_path = args.output / "stats.json"
    report = json.loads(stats_path.read_text(encoding="utf-8"))
    # Verify every source before modifying any previews.
    for camera in report["cameras"]:
        if sha256(Path(camera["file"])) != camera["sha256"]:
            raise RuntimeError(f"Input changed: {camera['file']}; rerun shadowtrack inspect")
    for metadata in report["metadata"]["files"]:
        if sha256(Path(metadata["file"])) != metadata["sha256"]:
            raise RuntimeError(f"Metadata changed: {metadata['file']}; rerun shadowtrack inspect")
    for camera in report["cameras"]:
        folder = args.output / camera["key"]
        frames = json.loads((folder / "timestamps/frames.json").read_text(encoding="utf-8"))["frames"]
        camera["timestamps"] = timestamp_stats(frames)
        camera["previews"] = make_previews(Path(camera["file"]), folder, camera["timestamps"],
                                           report["options"]["previews"], binary("ffmpeg"),
                                           report["options"]["timeout_seconds"], camera["container_start_seconds"] or 0)
        camera["contact_sheet"] = f"{camera['key']}/contact_sheet.jpg"
        print(camera["key"], "refreshed", flush=True)
    timing_codes = {"fps_header_mismatch", "fps_cadence_mismatch", "decoded_start_mismatch", "invalid_timestamps", "timestamp_gaps", "inconsistent_fps"}
    report["issues"] = [i for i in report["issues"] if i["code"] not in timing_codes]
    report["issues"].extend(consistency_issues(report["cameras"], report["options"]["fps_relative_tolerance"]))
    review_path = args.review_file or report["options"].get("review_file")
    report.pop("analyst_review", None)
    attach_review(report, Path(review_path) if review_path else None)
    report["options"]["review_file"] = str(Path(review_path).resolve()) if review_path else None
    report["status"] = "error" if any(i["severity"] == "error" for i in report["issues"]) else "warning" if report["issues"] else "ok"
    report["previews_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(report, args.output)


if __name__ == "__main__":
    main()
