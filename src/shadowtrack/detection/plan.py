"""Version 2 plans use saved source PTS and always cover every FULL frame."""

from .common import DetectionError,object_hash


def evidence_identity(evidence):
    manifest = evidence["manifest"]
    return object_hash({k:v for k,v in manifest.items() if k != "producer_code_hash"})


def detection_plan(evidence, mode, starts, seconds=6., chunk_frames=128):
    if mode not in ("BENCHMARK","DEMO","FULL") or not 1 <= chunk_frames <= 1024 or seconds <= 0:
        raise DetectionError("Invalid detection v2 mode/window/chunk size")
    windows,chunks,counts = [],[],{}
    for cid,camera in evidence["cameras"].items():
        frames = camera["frames"]
        if mode == "FULL":
            selected = list(range(len(frames)))
        else:
            raw_start = starts.get(cid)
            if not isinstance(raw_start,(int,float)):
                raise DetectionError(f"Missing explicit source PTS window for camera {cid}")
            selected = [i for i,p,sid in frames if raw_start <= p < raw_start+seconds]
            if not selected or selected != list(range(selected[0],selected[-1]+1)):
                raise DetectionError("Demo/benchmark must select a nonempty continuous source frame interval")
            sid = frames[selected[0]][2]
            if any(frames[i][2] != sid for i in selected):
                raise DetectionError("A demo window cannot cross a timestamp segment boundary")
            segment_end = next(s["frame_end"] for s in camera["segments"] if s["id"] == sid)
            if frames[segment_end][1] < raw_start+seconds-.1:
                raise DetectionError("Source segment does not cover the requested complete window")
            windows.append({"camera_id":cid,"segment_id":sid,"raw_interval":[raw_start,raw_start+seconds],
                            "first_frame":selected[0],"stop_frame":selected[-1]+1})
        counts[cid] = len(selected)
        group = []
        for index in selected:
            if group and (index != group[-1]+1 or frames[index][2] != frames[group[0]][2] or len(group) >= chunk_frames):
                chunks.append({"camera_id":cid,"segment_id":frames[group[0]][2],"start":group[0],"stop":group[-1]+1})
                group = []
            group.append(index)
        if group:
            chunks.append({"camera_id":cid,"segment_id":frames[group[0]][2],"start":group[0],"stop":group[-1]+1})
    return {"mode":mode,"raw_starts":starts if mode != "FULL" else {},"window_seconds":seconds,
            "chunk_frames":chunk_frames,"windows":windows,"chunks":chunks,"selected_frame_counts":counts}


def model_for_config(descriptor, config_hash):
    lock = descriptor["model"]
    if lock.get("task") == "detect":
        return lock["models"][descriptor["configurations"][config_hash]["model_key"]]
    return lock
