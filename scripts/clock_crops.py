"""Show existing event samples' burnt-in clocks without OCR or time corrections."""

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
    sheet = Image.new("RGB", (1100, 125 * len(records)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        with Image.open(args.folder / record["path"]) as image:
            box = (image.width - 700, 0, image.width, 100) if Path(record["video"]).stem == "camera_new_4" else (0, 0, 820, 80)
            crop = ImageOps.contain(image.crop(box), (820, 90))
        sheet.paste(crop, (275, i * 125))
        draw.text((8, i * 125 + 25), f"{Path(record['video']).stem}\nPTS {record['actual_pts_seconds']:.3f}s", font=font(19), fill="black")
    sheet.save(args.folder / "clock_crops.jpg", quality=95)


if __name__ == "__main__":
    main()
