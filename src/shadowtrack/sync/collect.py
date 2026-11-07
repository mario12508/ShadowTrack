"""Collect candidate simultaneous events with immutable extraction provenance.

This tool always writes checked=false. A human reviewer selects event bounds,
records the physical event and promotes only checked events into sync evidence.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from ..previews import extract_frame, font
from ..probe import AuditError, binary, sha256


def collect(request: Path, stats_path: Path, evidence_path: Path, output: Path):
    request_data = json.loads(request.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    output = output.resolve()
    if (output/"manifest.json").exists():
        raise AuditError("Refusing to overwrite an event manifest; select a new output directory")
    cameras = {}
    for cid,filename in evidence["camera_files"].items():
        camera = next(c for c in stats["cameras"] if Path(c["file"]).name == filename)
        if output.is_relative_to(Path(camera["file"]).parent) or output.is_relative_to(stats_path.resolve().parent):
            raise AuditError("Event output must be separate from raw data and the completed audit")
        if sha256(Path(camera["file"])) != evidence["input_sha256"][filename]:
            raise AuditError("Candidate source hash changed")
        cameras[cid] = camera
    output.mkdir(parents=True,exist_ok=True)
    manifest = {"schema_version":1,"request":str(request.resolve()),"request_sha256":sha256(request),
                "input_sha256":evidence["input_sha256"],"checked":False,"events":[],
                "policy":"Extraction does not verify simultaneity; never promote by filename, appearance or track outcome."}
    for event_i,event in enumerate(request_data["events"]):
        candidate = {"id":event["id"],"description":event["description"],"checked":False,
                     "reviewer":None,"observations":{},"samples":[],"kind":event.get("kind","candidate_event")}
        for cid,targets in event["targets"].items():
            camera = cameras[cid]
            records = []
            for j,target in enumerate(targets):
                path = output/f"e{event_i:02d}_camera{cid}_{j:03d}.jpg"
                sample = extract_frame(Path(camera["file"]),path,float(target),camera["container_start_seconds"] or 0,binary("ffmpeg"),120)
                sample.update(camera_id=cid,path=path.name,sha256=sha256(path),checked=False)
                records.append(sample)
                candidate["samples"].append(sample)
            candidate["observations"][cid] = None  # Reviewer must choose actual event bounds.
            w,h,cols = 360,310, min(6,len(records))
            if not records:
                raise AuditError("Each candidate camera requires at least one target")
            sheet = Image.new("RGB",(w*cols,((len(records)+cols-1)//cols)*h),"#14212a")
            draw = ImageDraw.Draw(sheet)
            for j,record in enumerate(records):
                with Image.open(output/record["path"]) as im:
                    # Optional scene ROI is for review only, never calibration.
                    box = event.get("crops",{}).get(cid)
                    image = im.crop(tuple(box)) if box else im
                    thumb = ImageOps.contain(image,(w,270))
                    sheet.paste(thumb,(j%cols*w,j//cols*h))
                draw.text((j%cols*w+5,j//cols*h+273),f"cam {cid} PTS {record['actual_pts_seconds']:.3f}s | unchecked",font=font(14),fill="white")
            sheet.save(output/f"e{event_i:02d}_camera{cid}_sheet.jpg",quality=94)
        manifest["events"].append(candidate)
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest
