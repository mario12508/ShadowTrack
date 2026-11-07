"""Make crop evidence for the actually inspected camera 1/3 TV screen.

These pixel rectangles are visual-review crops, not calibration or person labels.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from shadowtrack.previews import font


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    records = json.loads((args.folder / "manifest.json").read_text())
    boxes = {"camera_new_1": (0, 275, 170, 490), "camera_new_3": (1760, 410, 2110, 680)}
    sheet = Image.new("RGB", (1800, ((len(records) + 5) // 6) * 260), "white")
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        stem = Path(record["video"]).stem
        with Image.open(args.folder / record["path"]) as image:
            crop = ImageOps.contain(image.crop(boxes[stem]), (280, 210))
        x, y = i % 6 * 300, i // 6 * 260
        sheet.paste(crop, (x, y))
        draw.text((x, y + 218), f"{stem} | {record['actual_pts_seconds']:.3f}s", font=font(16), fill="black")
    sheet.save(args.folder / "tv_crops.jpg", quality=95)


if __name__ == "__main__":
    main()
