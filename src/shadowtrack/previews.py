"""Preview targets are absolute media PTS, never on-screen wall-clock times."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .probe import AuditError, run


def target_times(first: float, last: float, count: int) -> list[float]:
    if not 16 <= count <= 24:
        raise ValueError("previews must be between 16 and 24")
    if last <= first:
        return [first]
    # Include the first frame and approach the end; avoid seeking past the last frame.
    end = max(first, last - 0.25)
    return [first + (end - first) * i / (count - 1) for i in range(count)]


def font(size: int):
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def contact_sheet(samples: list[dict], out: Path, title: str) -> None:
    width, height, columns = 480, 360, 4
    caption, top = 44, 60
    sheet = Image.new("RGB", (columns * width, top + math.ceil(len(samples) / columns) * (height + caption)), "#111820")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 16), title, font=font(23), fill="white")
    for i, sample in enumerate(samples):
        x, y = i % columns * width, top + i // columns * (height + caption)
        with Image.open(out.parent / sample["path"]) as original:
            thumb = ImageOps.contain(original.convert("RGB"), (width, height))
        sheet.paste(thumb, (x + (width - thumb.width) // 2, y + (height - thumb.height) // 2))
        actual = sample["actual_pts_seconds"]
        draw.text((x + 9, y + height + 8), f"{i + 1:02d}  PTS {actual:.3f}s | +{sample['relative_seconds']:.3f}s", font=font(17), fill="white")
    sheet.save(out, quality=92)


def extract_frame(video: Path, path: Path, target: float, container_start: float,
                  ffmpeg: str, timeout: float) -> dict:
    # FFmpeg input -ss is relative to the container start (which can be audio's
    # start). Keep -copyts so showinfo reports original, unshifted video PTS.
    seek = max(0, target - container_start)
    command = [ffmpeg, "-hide_banner", "-loglevel", "info", "-nostdin", "-y",
               "-threads", "2", "-copyts", "-ss", f"{seek:.9f}",
               "-i", str(video), "-map", "0:v:0", "-frames:v", "1",
               "-vf", "showinfo", "-filter_threads", "1", "-threads:v", "1",
               "-pix_fmt", "yuvj420p", "-fps_mode", "passthrough", "-q:v", "3", str(path)]
    result = run(command, timeout)
    path.with_suffix(".log").write_text(result.stderr, encoding="utf-8")
    matches = re.findall(r"\bn:\s*0\s+pts:.*?pts_time:([\d.eE+\-]+)", result.stderr)
    if result.returncode or not path.is_file() or not matches:
        raise AuditError(f"Preview at {target} failed: {result.stderr[-1000:]}")
    actual = float(matches[0])
    if actual < target - 0.005:
        raise AuditError(f"Seek returned PTS {actual} before target {target}")
    return {"target_pts_seconds": target, "actual_pts_seconds": actual, "command": command}


def make_previews(video: Path, output: Path, stats: dict, count: int,
                  ffmpeg: str, timeout: float, container_start: float = 0) -> list[dict]:
    folder = output / "previews"
    folder.mkdir(parents=True, exist_ok=True)
    first = stats["first_pts_seconds"]
    samples = []
    for i, target in enumerate(target_times(first, stats["last_pts_seconds"], count)):
        path = folder / f"preview_{i + 1:02d}.jpg"
        sample = extract_frame(video, path, target, container_start, ffmpeg, timeout)
        sample.update(path=path.relative_to(output).as_posix(), relative_seconds=sample["actual_pts_seconds"] - first)
        samples.append(sample)
    (output / "previews.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    contact_sheet(samples, output / "contact_sheet.jpg", video.stem + " | media PTS / elapsed from first video frame")
    return samples
