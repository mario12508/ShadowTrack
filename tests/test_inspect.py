import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from shadowtrack.audit import consistency_issues, inspect, inventory_metadata
from shadowtrack.cli import main
from shadowtrack.previews import target_times
from shadowtrack.probe import AuditError, rate, timestamp_stats


def test_empty_directory_writes_actionable_reports(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    report = inspect(raw, tmp_path / "metadata", tmp_path / "out")
    assert report["status"] == "error"
    assert {i["code"] for i in report["issues"]} >= {"empty_raw", "missing_metadata"}
    assert json.loads((tmp_path / "out/stats.json").read_text())["cameras"] == []
    assert "empty_raw" in (tmp_path / "out/report.html").read_text(encoding="utf-8")


def test_missing_metadata_never_invents_calibration(tmp_path):
    meta = inventory_metadata(tmp_path / "missing")
    assert meta["files"] == []
    assert meta["has_raster_plan"] is False
    assert meta["calibration"] == "not_inferred"
    assert not any(k in meta for k in ("homography", "camera_positions", "offsets"))


def test_corrupt_plan_is_not_valid_metadata(tmp_path):
    (tmp_path / "office.png").write_bytes(b"not an image")
    assert not inventory_metadata(tmp_path)["has_raster_plan"]


def camera(fps, pts):
    return {"file": f"camera_{fps}.mkv", "nominal_fps": fps, "header_average_fps": fps,
            "timestamps": timestamp_stats([{"best_effort_timestamp_time": str(t)} for t in pts])}


def test_inconsistent_fps_and_misleading_header():
    issues = consistency_issues([camera(25, [0, 1, 2]), camera(15, [0, 0.1, 0.2])])
    assert sum(i["code"] == "fps_header_mismatch" for i in issues) == 2
    assert any(i["code"] == "inconsistent_fps" for i in issues)


def test_timestamp_anomalies_and_missing_pts_do_not_get_bridged():
    frames = [{"best_effort_timestamp_time": v} for v in ["1", "1", "0.9", None, "2", "3"]]
    result = timestamp_stats(frames)
    assert result["decoded_frames"] == 6
    assert result["missing_pts"] == 1
    assert result["duplicate_pts"] == 1
    assert result["backwards_pts"] == 1
    assert sum(result["interval_histogram_seconds"].values()) == 3


@pytest.mark.parametrize("value", ["0/0", "N/A", None, "nan", "-1/2"])
def test_unknown_fps_is_not_zero(value):
    assert rate(value) is None


def test_preview_targets_cover_whole_video():
    targets = target_times(5.25, 1205.2, 20)
    assert len(targets) == 20
    assert targets[0] == 5.25
    assert targets[-1] == pytest.approx(1204.95)
    assert all(a < b for a, b in zip(targets, targets[1:]))
    with pytest.raises(ValueError):
        target_times(0, 20, 15)


def test_output_cannot_modify_inputs(tmp_path):
    with pytest.raises(AuditError):
        inspect(tmp_path / "raw", tmp_path / "meta", tmp_path / "raw/reports")


def test_cli_config_paths_are_relative_to_config(tmp_path):
    config = tmp_path / "audit.toml"
    config.write_text('[inspect]\nraw_dir="raw"\nmetadata_dir="meta"\noutput_dir="results"\n', encoding="utf-8")
    assert main(["inspect", "--config", str(config)]) == 1
    assert (tmp_path / "results/stats.json").is_file()


@pytest.fixture
def ffmpeg_tools(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    found = []
    for name in ("ffmpeg", "ffprobe"):
        candidate = os.environ.get(f"SHADOWTRACK_{name.upper()}") or shutil.which(name)
        local = root / ".tools" / (name + ".exe")
        if not candidate and local.is_file():
            candidate = str(local)
        if not candidate:
            pytest.skip("Integration test requires ffmpeg and ffprobe")
        monkeypatch.setenv(f"SHADOWTRACK_{name.upper()}", candidate)
        found.append(candidate)
    return found


def make_video(ffmpeg, path, fps=8, offset=0):
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=size=160x120:rate={fps}:duration=3", "-vf", f"setpts=PTS+{offset}/TB",
                    "-c:v", "libx264", "-g", "8", "-bf", "0", "-fps_mode", "passthrough", str(path)],
                   check=True, timeout=30, capture_output=True)


@pytest.mark.integration
def test_corrupt_mkv_is_reported_without_crash(tmp_path, ffmpeg_tools):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "broken.MKV").write_bytes(b"broken matroska stream\x00\xff")
    result = inspect(raw, tmp_path / "meta", tmp_path / "out", expected_cameras=1)
    assert result["status"] == "error"
    assert result["cameras"][0]["status"] == "error"
    assert any(i["code"] == "video_error" for i in result["issues"])
    assert (tmp_path / "out/camera_01/timestamps/probe.stderr.txt").stat().st_size > 0


@pytest.mark.integration
def test_real_different_fps_missing_metadata_and_nonzero_pts(tmp_path, ffmpeg_tools):
    ffmpeg, _ = ffmpeg_tools
    raw = tmp_path / "raw"
    raw.mkdir()
    make_video(ffmpeg, raw / "a.mkv", fps=8, offset=5)
    make_video(ffmpeg, raw / "b.mkv", fps=12)
    result = inspect(raw, tmp_path / "absent_metadata", tmp_path / "out", previews=16, expected_cameras=2)
    assert result["status"] == "warning"
    assert {i["code"] for i in result["issues"]} >= {"inconsistent_fps", "missing_metadata"}
    first, second = result["cameras"]
    assert first["timestamps"]["decoded_frames"] == 24
    assert second["timestamps"]["decoded_frames"] == 36
    assert first["timestamps"]["first_pts_seconds"] == pytest.approx(5)
    for item, expected in zip(result["cameras"], [8, 12]):
        assert item["status"] == "ok"
        assert item["timestamps"]["observed_fps"] == pytest.approx(expected, rel=0.005)
        assert len(item["previews"]) == 16
        for sample in item["previews"]:
            assert sample["actual_pts_seconds"] >= sample["target_pts_seconds"] - 0.001
            assert sample["actual_pts_seconds"] - sample["target_pts_seconds"] <= 1 / expected + 0.002
        with Image.open(tmp_path / "out" / item["contact_sheet"]) as im:
            assert im.width == 1920
    assert result["synchronization"]["applied_offsets_seconds"] is None


def test_missing_tools_are_reported(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.mkv").write_bytes(b"dummy")
    monkeypatch.setenv("SHADOWTRACK_FFPROBE", "definitely-nonexistent-ffprobe")
    result = inspect(raw, tmp_path / "meta", tmp_path / "out")
    assert any(i["code"] == "missing_tool" for i in result["issues"])


def test_invalid_config_returns_2(tmp_path):
    config = tmp_path / "bad.toml"
    config.write_text('[inspect]\npreviews = "twenty"\n', encoding="utf-8")
    assert main(["inspect", "--config", str(config)]) == 2
