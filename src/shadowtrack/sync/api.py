"""Public API: unavailable is a result, never a fabricated global timestamp."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fit import predict_interval
from .local import load_local
from ..probe import sha256


def _model_step(model, node, value, low, high):
    if node == model["node_a"]:
        support = model["x_support"]
        if not support[0]-1e-8 <= low <= high <= support[1]+1e-8:
            return None
        bounds = predict_interval(model, low, high)
        return model["node_b"], model["slope"]*value+model["intercept_seconds"], *bounds
    if node == model["node_b"]:
        support = model["y_support"]
        if not support[0]-1e-8 <= low <= high <= support[1]+1e-8:
            return None
        values = [(y-b)/a for a,b in model["parameter_polygon"] for y in (low, high)]
        return model["node_a"], (value-model["intercept_seconds"])/model["slope"], min(values), max(values)
    return None


class TimeAlignment:
    def __init__(self, document: dict, timelines: dict):
        self.document, self.timelines = document, timelines

    @classmethod
    def load(cls, path="artifacts/sync/time_alignment.json"):
        path = Path(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        for c in document["cameras"]:
            if sha256(path.parent/c["local_frame_times_file"]) != c["local_frame_times_sha256"]:
                raise ValueError("Local time mapping checksum mismatch")
        timelines = {c["camera_id"]: load_local(c, path.parent/c["local_frame_times_file"]) for c in document["cameras"]}
        return cls(document, timelines)

    def _routes(self, node, value, low, high, target=None):
        queue = [(node,value,low,high,[],{node})]
        answers = []
        models = [m for m in self.document["pair_models"] if m["status"] == "SUPPORTED"]
        while queue:
            current,v,lo,hi,path,visited = queue.pop(0)
            if (target is None and current == self.document["reference_node"]) or current == target:
                answers.append((v,lo,hi,path,current))
                continue
            for model in models:
                step = _model_step(model,current,v,lo,hi)
                if step and step[0] not in visited:
                    nxt,nv,nlo,nhi = step
                    queue.append((nxt,nv,nlo,nhi,path+[model["id"]],visited|{nxt}))
        return answers

    def map_local_time(self, camera_id, raw_pts):
        timeline = self.timelines.get(str(camera_id))
        result = timeline.at(raw_pts) if timeline else None
        return result or {"relative_capture_time_estimate": None, "uncertainty_seconds": None,
                          "status": "unavailable", "reason": "outside source domain, timestamp break or ambiguous raw PTS"}

    def map_time(self, camera_id, raw_pts):
        if str(camera_id) in self.document.get("invalid_cameras",[]):
            return None,None,"inconsistent"
        if not self.document["reference_support"]:
            return None,None,"unavailable"
        local = self.map_local_time(camera_id, raw_pts)
        if local["relative_capture_time_estimate"] is None or local.get("interval") is None:
            return None, None, "unavailable"
        node, value = local["segment_id"], local["relative_capture_time_estimate"]
        low, high = local["interval"]
        if node == self.document["reference_node"]:
            if not any(a <= low <= high <= b for a,b in self.document["reference_support"]):
                return None, None, "unavailable"
        routes = self._routes(node,value,low,high)
        if not routes:
            return None,None,"unavailable"
        # If independently calibrated paths disagree, fail closed (cycle conflict).
        if max(r[1] for r in routes) > min(r[2] for r in routes)+1e-8:
            return None,None,"inconsistent"
        best = min(routes, key=lambda r:r[2]-r[1])
        lo, hi = min(r[1] for r in routes), max(r[2] for r in routes)
        floor = self.document.get("reference_coordinate_resolution_seconds",0)
        lo,hi = lo-floor,hi+floor
        source_ids = set(m for r in routes for m in r[3])
        if node == self.document["reference_node"]:
            source_ids.update(m["id"] for m in self.document["pair_models"] if m["status"] == "SUPPORTED" and node in (m["node_a"],m["node_b"]))
        return best[0], {"lower_seconds": lo, "upper_seconds": hi,
                         "radius_seconds": max(best[0]-lo,hi-best[0]),
                         "kind": "admissible_model_bounds_not_confidence_interval",
                         "source_model_ids": sorted(source_ids),
                         "basis": self.document["global_time_basis"],
                         "conditional_on": "affine interpolation between checked anchors; local rate not independently verified"}, "SUPPORTED_CONDITIONAL"

    def inverse_map_time(self, camera_id, global_time):
        cid = str(camera_id)
        if cid in self.document.get("invalid_cameras",[]):
            return None,None,"inconsistent"
        root = self.document["reference_node"]
        if cid not in self.timelines or not any(a <= global_time <= b for a,b in self.document["reference_support"]):
            return None, None, "unavailable"
        answers = []
        timeline = self.timelines[cid]
        floor = self.document.get("reference_coordinate_resolution_seconds",0)
        for segment in timeline.segments:
            for v,lo,hi,path,node in self._routes(root,global_time,global_time-floor,global_time+floor,segment["id"]):
                raw = timeline.inverse(v)
                raw_low, raw_high = timeline.inverse(lo), timeline.inverse(hi)
                if raw is not None and raw_low is not None and raw_high is not None and timeline.at(raw) is not None:
                    answers.append((raw, [raw_low,raw_high]))
        if len(answers) != 1:
            return None,None,"unavailable"
        return answers[0][0], {"raw_pts_interval": answers[0][1]}, "SUPPORTED_CONDITIONAL"


def map_time(camera_id, raw_pts, alignment_path="artifacts/sync/time_alignment.json"):
    """Return (global_time_seconds | None, uncertainty | None, status)."""
    if not Path(alignment_path).is_file():
        return None,None,"unavailable"
    return TimeAlignment.load(alignment_path).map_time(camera_id, raw_pts)
