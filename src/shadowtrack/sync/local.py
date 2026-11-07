"""Monotone local estimates with explicit breaks and unknown capture timing.

No cadence inferred from frame_index is promoted to observed elapsed time.
Unchecked gaps and known missing frames retain their full raw duration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalTimeline:
    camera_id: str
    raw: np.ndarray
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    valid: np.ndarray
    segment: np.ndarray
    segments: list[dict]
    anomalies: list[dict]
    nominal_fps: float

    def at(self, raw_pts: float) -> dict | None:
        if not np.isfinite(raw_pts):
            return None
        for anomaly in self.anomalies:
            bounds = anomaly.get("ambiguous_raw_interval")
            if bounds and bounds[0] <= raw_pts <= bounds[1]:
                return None
        matches = np.flatnonzero(np.isfinite(self.raw) & (np.abs(self.raw-raw_pts) < 1e-8))
        if len(matches) > 1:
            return None  # raw PTS cannot disambiguate duplicated frames.
        candidates = []
        for s in self.segments:
            if s["raw_start"]-1e-9 <= raw_pts <= s["raw_end"]+1e-9:
                lo, hi = s["frame_start"], s["frame_end"]+1
                if hi-lo == 1 and not np.isclose(raw_pts, self.raw[lo], atol=1e-8, rtol=0):
                    continue
                t = float(np.interp(raw_pts, self.raw[lo:hi], self.estimate[lo:hi]))
                low = float(np.interp(raw_pts, self.raw[lo:hi], self.lower[lo:hi]))
                high = float(np.interp(raw_pts, self.raw[lo:hi], self.upper[lo:hi]))
                bounded = np.isfinite(low) and np.isfinite(high)
                candidates.append({"relative_capture_time_estimate": t, "interval": [low, high] if bounded else None,
                                   "uncertainty_seconds": max(t-low, high-t) if bounded else None,
                                   "physical_uncertainty_seconds": None, "segment_id": s["id"],
                                   "status": "LOCAL_ESTIMATE", "physical_rate_verified": False})
        return candidates[0] if len(candidates) == 1 else None

    def inverse(self, local_time: float) -> float | None:
        candidates = []
        for s in self.segments:
            lo, hi = s["frame_start"], s["frame_end"]+1
            if self.estimate[lo]-1e-9 <= local_time <= self.estimate[hi-1]+1e-9:
                candidates.append(float(np.interp(local_time, self.estimate[lo:hi], self.raw[lo:hi])))
        return candidates[0] if len(candidates) == 1 else None

    def frame_time(self, frame_index: int) -> dict:
        if frame_index < 0 or frame_index >= len(self.raw):
            raise IndexError(frame_index)
        t = float(self.estimate[frame_index])
        bounded = self.valid[frame_index] and np.isfinite(self.lower[frame_index]) and np.isfinite(self.upper[frame_index])
        return {"frame_index": frame_index,
                "raw_media_pts": float(self.raw[frame_index]) if np.isfinite(self.raw[frame_index]) else None,
                "relative_capture_time_estimate": t,
                "uncertainty_seconds": float(max(t-self.lower[frame_index], self.upper[frame_index]-t)) if bounded else None,
                "physical_uncertainty_seconds": None,
                "synchronized_global_time": None,
                "status": "LOCAL_ESTIMATE" if self.valid[frame_index] else "INDEX_ONLY_UNSUPPORTED"}


def build_local(camera_id: str, raw_pts: list, nominal_fps: float, corrections=()) -> LocalTimeline:
    if not nominal_fps or nominal_fps <= 0:
        raise ValueError("Positive nominal_fps required for ordinal fallback")
    raw = np.array([np.nan if v is None else float(v) for v in raw_pts], dtype=float)
    if not len(raw):
        raise ValueError("Empty frame PTS")
    diffs = np.diff(raw)
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    median = float(np.median(positive)) if len(positive) else 1/nominal_fps
    step = median
    correction_map = {}
    for correction in corrections:
        if correction.get("checked") is not True:
            continue
        if correction.get("kind") != "clock_discontinuity":
            continue  # A real missing interval is never compressed.
        before, after = correction["before_pts"], correction["after_pts"]
        indices = np.flatnonzero(np.isclose(raw[:-1], before, atol=1e-6, rtol=0) &
                                 np.isclose(raw[1:], after, atol=1e-6, rtol=0))
        low, high = correction["capture_delta_interval"]
        if len(indices) != 1 or not 0 < low <= high < after-before or not correction.get("source"):
            raise ValueError(f"Invalid or non-adjacent correction {correction.get('id')}")
        correction_map[int(indices[0])+1] = correction
    estimate, lower, upper = (np.zeros(len(raw)) for _ in range(3))
    valid = np.isfinite(raw)
    breaks = np.zeros(len(raw), dtype=bool)
    breaks[0] = True
    anomalies = []
    for i in range(1, len(raw)):
        delta = raw[i]-raw[i-1]
        correction = correction_map.get(i)
        if correction:
            low, high = correction["capture_delta_interval"]
            advance = min(high, max(low, correction.get("estimated_capture_delta", step)))
            estimate[i] = estimate[i-1]+advance
            lower[i], upper[i] = lower[i-1]+low, upper[i-1]+high
            breaks[i] = True
            anomalies.append({"frame_before": i-1, "frame_after": i, "before_pts": float(raw[i-1]),
                              "after_pts": float(raw[i]), "raw_delta_seconds": float(delta),
                              "classification": "CONFIRMED_CLOCK_DISCONTINUITY", "corrected": True,
                              "estimated_capture_delta": advance, "capture_delta_interval": [low, high],
                              "evidence_id": correction["id"], "source": correction["source"],
                              "assumption":correction.get("assumption","reviewed elapsed interval is valid")})
        elif not np.isfinite(delta) or delta <= 0:
            # Strictly monotone frame ordinal coordinate; elapsed time is unsupported.
            estimate[i] = estimate[i-1]+step
            lower[i], upper[i] = np.nan, np.nan
            valid[i] = False
            breaks[i] = True
            if i+1 < len(raw):
                breaks[i+1] = True
            anomaly = {"frame_after": i, "classification": "MISSING_OR_NONINCREASING_PTS",
                       "corrected": False, "physical_elapsed_time": "UNKNOWN"}
            if np.isfinite(delta) and delta <= 0:
                anomaly["ambiguous_raw_interval"] = [float(raw[i]),float(np.nanmax(raw[:i]))]
            anomalies.append(anomaly)
        else:
            estimate[i], lower[i], upper[i] = estimate[i-1]+delta, lower[i-1]+delta, upper[i-1]+delta
            if delta > 1.5*median or delta < .5*median:
                breaks[i] = True
                anomalies.append({"frame_before": i-1, "frame_after": i,
                                  "before_pts": float(raw[i-1]), "after_pts": float(raw[i]),
                                  "raw_delta_seconds": float(delta), "classification": "UNRESOLVED_GAP_OR_JITTER",
                                  "corrected": False, "physical_elapsed_time": "UNKNOWN"})
    if not valid[0]:
        breaks[0] = True
    # Missing/duplicate PTS contaminates the ordinal-to-capture origin thereafter.
    # Such domains remain usable locally, never silently become capture-calibrated.
    segments, labels = [], np.full(len(raw), -1, dtype=int)
    start = None
    for i in range(len(raw)+1):
        if start is not None and (i == len(raw) or breaks[i] or not valid[i]):
            j = len(segments)
            segments.append({"id": f"{camera_id}:s{j}", "frame_start": start, "frame_end": i-1,
                             "raw_start": float(raw[start]), "raw_end": float(raw[i-1]),
                             "local_start": float(estimate[start]), "local_end": float(estimate[i-1]),
                             "status": "LOCAL_ESTIMATE", "physical_rate_verified": False})
            labels[start:i] = j
            start = None
        if i < len(raw) and valid[i] and start is None:
            start = i
    if not np.all(np.diff(estimate) > 0):
        raise ValueError("Local timeline is not strictly monotone")
    return LocalTimeline(camera_id, raw, estimate, lower, upper, valid, labels, segments, anomalies, nominal_fps)


def save_local(timeline: LocalTimeline, path):
    np.savez_compressed(path, raw_media_pts=timeline.raw,
                        frame_index=np.arange(len(timeline.raw)),
                        relative_capture_time_estimate=timeline.estimate,
                        estimate_lower=timeline.lower, estimate_upper=timeline.upper,
                        raw_mapping_supported=timeline.valid, segment=timeline.segment)


def load_local(info: dict, path) -> LocalTimeline:
    with np.load(path, allow_pickle=False) as a:
        return LocalTimeline(info["camera_id"], a["raw_media_pts"], a["relative_capture_time_estimate"],
                             a["estimate_lower"], a["estimate_upper"], a["raw_mapping_supported"], a["segment"],
                             info["local_segments"], info["anomalies"], info["nominal_fps"])
