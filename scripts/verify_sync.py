"""Read-only acceptance checks for the reviewed ShadowTrack dataset and outputs."""

import importlib.metadata
import json
import platform
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from shadowtrack.probe import sha256
from shadowtrack.sync import TimeAlignment


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/"artifacts/sync"
    baseline = json.loads((output/"preserved_audit_hashes.json").read_text(encoding="utf-8"))
    unchanged = {name:sha256(root/name)==digest for name,digest in baseline.items()}
    assert all(unchanged.values()), "Completed audit changed"
    api = TimeAlignment.load(output/"time_alignment.json")
    doc = api.document
    assert sha256(Path(doc["source_stats"])) == doc["source_stats_sha256"]
    assert sha256(Path(doc["source_evidence"])) == doc["source_evidence_sha256"]
    raw_hashes = {c["camera_id"]:sha256(Path(c["file"]))==c["sha256"] for c in doc["cameras"]}
    assert all(raw_hashes.values())
    monotonic = {cid:bool(np.all(np.diff(t.estimate)>0)) for cid,t in api.timelines.items()}
    assert all(monotonic.values())
    for event in doc["events"]:
        if not event.get("accepted_for_fit"):
            continue
        for cid,bounds in event["observations"].items():
            for endpoint in bounds:
                assert np.any(np.isclose(api.timelines[cid].raw,endpoint,atol=1e-6,rtol=0)), "Evidence endpoint is not a decoded frame PTS"
    before,after = api.map_local_time("1",225.902),api.map_local_time("1",235.950)
    assert abs(after["relative_capture_time_estimate"]-before["relative_capture_time_estimate"]-.039)<1e-8
    assert api.map_local_time("1",230)["status"] == "unavailable"
    supported = api.map_time("1",3.)
    assert supported[2] == "SUPPORTED_CONDITIONAL"
    inverse = api.inverse_map_time("1",supported[0])
    assert abs(inverse[0]-3)<1e-8
    unavailable = {cid:api.map_time(cid,100)[2] for cid in ("2","4","5")}
    assert set(unavailable.values()) == {"unavailable"}
    assert api.map_time("1",904)[2] == "unavailable"
    assert not doc["full_video_probe_repeated"] and not doc["raw_media_modified"]
    suite = ET.parse(output/"tests.xml").getroot().find("testsuite")
    assert suite is not None and suite.attrib["failures"] == suite.attrib["errors"] == suite.attrib["skipped"] == "0"
    gate_exit = int((output/"qa/gate_exit.txt").read_text(encoding="utf-8-sig").strip())
    assert gate_exit == 1
    sources = [".gitignore","README.md","pyproject.toml","src/shadowtrack/cli.py","TIME_ALIGNMENT.md",
               "tests/test_sync.py","scripts/sync.ps1","scripts/import_audit_sync_evidence.py","scripts/verify_sync.py"]
    sources += [p.relative_to(root).as_posix() for p in (root/"src/shadowtrack/sync").glob("*.py")]
    sources += [p.relative_to(root).as_posix() for p in (root/"configs").glob("sync*.json")]
    verification = {"acceptance_status":doc["acceptance_status"],"tests":suite.attrib,
                    "all_previous_audit_files_unchanged":all(unchanged.values()),"previous_audit_file_count":len(unchanged),
                    "raw_mkv_hashes_match":raw_hashes,"strictly_monotone_local_frame_estimates":monotonic,
                    "accepted_anchor_endpoints_are_real_frame_pts":True,
                    "corrected_local_jump_count":sum(a["corrected"] for c in doc["cameras"] for a in c["anomalies"]),
                    "supported_example":{"raw_pts":3.,"camera_id":"1","global_time":supported[0],"uncertainty":supported[1],"status":supported[2]},
                    "unresolved_camera_checks":unavailable,"late_global_time_unavailable":True,
                    "inverse_mapping_roundtrip_error_seconds":abs(inverse[0]-3),"require_complete_exit_code":gate_exit,
                    "full_video_probe_repeated":False,"audio_window_pairs":sum(len(p["windows"]) for p in doc["audio"].get("pairs",[])),
                    "audio_auto_anchors":doc["audio"]["automatically_applied_anchors"],
                    "python":platform.python_version(),"packages":{p:importlib.metadata.version(p) for p in ["numpy","matplotlib","pytest","Pillow"]},
                    "source_files":{name:sha256(root/name) for name in sorted(set(sources))}}
    (output/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:verification[k] for k in ["acceptance_status","all_previous_audit_files_unchanged","corrected_local_jump_count","require_complete_exit_code"]},indent=2))


if __name__ == "__main__":
    main()
