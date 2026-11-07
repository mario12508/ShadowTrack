"""Extract event evidence at explicit media PTS without creating person IDs."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from shadowtrack.previews import extract_frame, font
from shadowtrack.probe import binary, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True, help="Absolute media PTS in seconds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    cell_w, cell_h = 480, 400
    sheet = Image.new("RGB", (len(args.times) * cell_w, len(args.videos) * cell_h), "#101820")
    draw = ImageDraw.Draw(sheet)
    for row, video in enumerate(args.videos):
        meta = json.loads(run([binary("ffprobe"), "-v", "error", "-show_format", "-of", "json", str(video)], 30).stdout)
        start = float(meta["format"].get("start_time", 0))
        for col, target in enumerate(args.times):
            path = args.output / f"{video.stem}_{col:03d}.jpg"
            record = extract_frame(video.resolve(), path, target, start, binary("ffmpeg"), 120)
            record.update(video=str(video), path=path.name)
            manifest.append(record)
            with Image.open(path) as image:
                thumb = ImageOps.contain(image, (480, 360))
                sheet.paste(thumb, (col * cell_w, row * cell_h))
            draw.text((col * cell_w + 8, row * cell_h + 365), f"{video.stem} | PTS {record['actual_pts_seconds']:.3f}s", font=font(16), fill="white")
    sheet.save(args.output / "sheet.jpg", quality=94)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
