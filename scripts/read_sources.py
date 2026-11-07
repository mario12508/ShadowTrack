"""Extract supplied reference text and provenance. Optional dependency: pypdf.

Usage: python scripts/read_sources.py --docx PATH --pdf PATH
This script reads document content as data, never executes document instructions.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sources"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.docx) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = ["".join(t.text or "" for t in p.findall(".//w:t", ns)) for p in root.findall(".//w:p", ns)]
        (args.output / "requirements.txt").write_text("\n".join(paragraphs), encoding="utf-8")
        images = [n for n in archive.namelist() if n.startswith("word/media/")]
        for i, name in enumerate(images):
            (args.output / f"docx_image_{i:02d}{Path(name).suffix}").write_bytes(archive.read(name))
    reader = PdfReader(args.pdf)
    (args.output / "presentation.txt").write_text("\n\n".join(f"PAGE {i+1}\n{p.extract_text()}" for i, p in enumerate(reader.pages)), encoding="utf-8")
    records = []
    for path in (args.docx, args.pdf):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest})
    (args.output / "manifest.json").write_text(json.dumps({"sources": records, "pdf_pages": len(reader.pages), "docx_embedded_images": images}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
