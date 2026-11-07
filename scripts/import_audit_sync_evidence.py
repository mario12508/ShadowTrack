"""Import only the facts already checked in DATA_AUDIT, bound to original hashes.

This does not create new checked events by examining filenames or people.
The existing audit and manifests are read-only.
"""

import json
from pathlib import Path

from shadowtrack.probe import sha256


def main():
    root = Path(__file__).resolve().parents[1]
    target = root/"configs/sync_evidence.json"
    if target.exists():
        raise SystemExit("Evidence already exists; refusing to overwrite reviewed annotations")
    stats = json.loads((root/"artifacts/inspect/stats.json").read_text(encoding="utf-8"))
    expected = ["744331b88aadaf56a83540d3081f22f11cf207ee93b62bacf03972c679814dc5",
                "6839e486dd14c70b948f5f8e120d3cb339c45a1a26dd1baa496200704fef59c4",
                "f90f08498e306b3e75b07489a253ebdfcbb5f3373b73478b086bd5d7ef9a8873",
                "8ede342c3d0d98ab912e98c1a31ef5458ca04a8329b882634b277d7c4634bef2",
                "6e3dd8bb2bb0c7b640b79b7a2c85f54d8346e930e9d89f55d86cf20f43aa4be5"]
    by_name = {Path(c["file"]).name:c for c in stats["cameras"]}
    camera_files, hashes, frame_hashes = {},{},{}
    for i,digest in enumerate(expected,1):
        cid,name = str(i),f"camera_new_{i}.mkv"
        camera = by_name[name]
        if camera["sha256"] != digest or sha256(Path(camera["file"])) != digest:
            raise SystemExit("Seed importer applies only to the reviewed original files")
        camera_files[cid], hashes[name] = name,digest
        frame_hashes[cid] = sha256(root/"artifacts/inspect"/camera["key"]/"timestamps/frames.json")
    def source(folder):
        path = Path("artifacts/events")/folder/"manifest.json"
        return {"manifest":path.as_posix(),"sha256":sha256(root/path)}
    events = [
        {"id":"audit_tv_A","kind":"shared_tv_cut","checked":True,
         "reviewer":"imported from completed DATA_AUDIT visual review",
         "description":"close shot to wide stage, same physical TV; transition is bracketed",
         "observations":{"1":[1.840,1.918],"3":[.940,1.018]},"source":source("tv_refined")},
        {"id":"audit_tv_B","kind":"shared_tv_cut","checked":True,
         "reviewer":"imported from completed DATA_AUDIT visual review",
         "description":"wide stage to close shot, same physical TV",
         "observations":{"1":[5.172,5.251],"3":[4.275,4.353]},"source":source("tv_refined")},
    ]
    corrections = []
    for label,before,after,clock in [("a",225.902,235.950,"17:42:28"),("b",723.187,733.242,"17:50:46")]:
        corrections.append({"id":f"audit_camera1_jump_{label}","camera_id":"1","checked":True,
                            "kind":"clock_discontinuity","before_pts":before,"after_pts":after,
                            "capture_delta_interval":[.000001,1.0],"estimated_capture_delta":.039,
                            "same_osd_second":clock,"source":source("gaps_camera1"),
                            "assumption":"OSD is running locally; same displayed second bounds elapsed time below 1s. Median step is an estimate, not verified physical cadence."})
    evidence = {"schema_version":1,"camera_files":camera_files,"input_sha256":hashes,
                "frame_pts_sha256":frame_hashes,"events":events,"local_corrections":corrections,
                "regression":{"min_drift_span":60.,"max_drift_ppm":5000.,"systematic_seconds":.02}}
    target.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf-8")
    protected = [root/"DATA_AUDIT.md",root/"artifacts/inspect/stats.json"]
    protected += list((root/"artifacts/events").glob("**/manifest.json"))
    protected += list((root/"artifacts/inspect").glob("*/timestamps/*.json"))
    provenance = {p.relative_to(root).as_posix():sha256(p) for p in protected}
    out = root/"artifacts/sync"
    out.mkdir(exist_ok=True)
    (out/"preserved_audit_hashes.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
