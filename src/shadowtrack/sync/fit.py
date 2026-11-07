"""Deterministic robust interval regression, with bounded support and no extrapolation."""

from __future__ import annotations

import itertools
import math
import statistics

import numpy as np


def _clip(poly, a, b, c):
    """Intersect parameter polygon (slope, intercept) with a*x+b*y<=c."""
    result = []
    for p, q in zip(poly, poly[1:]+poly[:1]):
        fp, fq = a*p[0]+b*p[1]-c, a*q[0]+b*q[1]-c
        if fp <= 1e-10:
            result.append(p)
        if (fp < 0) != (fq < 0):
            t = fp/(fp-fq)
            result.append([p[0]+t*(q[0]-p[0]), p[1]+t*(q[1]-p[1])])
    return result


def predict_interval(model: dict, x_low: float, x_high: float | None = None) -> list[float]:
    x_high = x_low if x_high is None else x_high
    values = [a*x+b for a, b in model["parameter_polygon"] for x in (x_low, x_high)]
    return [min(values), max(values)]


def fit_intervals(anchors: list[dict], min_drift_span: float = 60., max_drift_ppm: float = 5000.,
                  systematic_seconds: float = .02) -> dict:
    """Anchors have x/y intervals, unique event id and independent segment domain.

    >=2 agreeing independent events for offset; >=3 events spanning >=60 s for
    drift. RANSAC consensus must contain >=75% of supplied events; outliers are
    recorded. Polygon bounds are admissible parameter bounds, not confidence CIs.
    """
    if len({a["id"] for a in anchors}) != len(anchors):
        return {"status": "INCONSISTENT", "reason": "duplicate event ids"}
    if len(anchors) < 2:
        return {"status": "UNCALIBRATED", "reason": "at least two independent checked events required"}
    for a in anchors:
        if any(not math.isfinite(v) for interval in (a["x"], a["y"]) for v in interval) or any(a[k][0] > a[k][1] for k in ("x", "y")):
            raise ValueError("Invalid anchor interval")
    x = np.array([statistics.mean(a["x"]) for a in anchors])
    y = np.array([statistics.mean(a["y"]) for a in anchors])
    drift = len(anchors) >= 3 and np.ptp(x) >= min_drift_span
    max_drift = max_drift_ppm * 1e-6
    candidates = [(1., float(np.median(y-x)))]
    if drift:
        for i, j in itertools.combinations(range(len(x)), 2):
            if abs(x[j]-x[i]) > 1e-8:
                slope = (y[j]-y[i])/(x[j]-x[i])
                if 1-max_drift <= slope <= 1+max_drift:
                    candidates.append((float(slope), float(y[i]-slope*x[i])))
    candidates += [(1., float(d)) for d in y-x]
    def residuals(a, b):
        errors = []
        for event in anchors:
            low, high = a*event["x"][0]+b, a*event["x"][1]+b
            errors.append(max(event["y"][0]-high, low-event["y"][1], 0))
        return np.array(errors)
    scored = [(int(np.sum(residuals(a,b) <= systematic_seconds+1e-9)),
               -float(np.sum(np.minimum(residuals(a,b), 1))), a, b) for a,b in candidates]
    _, _, slope, intercept = max(scored)
    inliers = np.flatnonzero(residuals(slope, intercept) <= systematic_seconds+1e-9)
    if len(inliers) < 2 or len(inliers) < math.ceil(.75*len(anchors)):
        return {"status": "INCONSISTENT", "reason": "no >=75% interval consensus", "anchor_ids": [a["id"] for a in anchors]}
    drift = drift and len(inliers) >= 3 and np.ptp(x[inliers]) >= min_drift_span
    if drift:
        slopes = [(y[j]-y[i])/(x[j]-x[i]) for i,j in itertools.combinations(inliers,2) if abs(x[j]-x[i]) > 1e-8]
        slope = float(np.clip(np.median(slopes), 1-max_drift, 1+max_drift))
    else:
        slope = 1.
    intercept = float(np.median(y[inliers]-slope*x[inliers]))
    if drift:
        poly = [[1-max_drift,-1e6],[1+max_drift,-1e6],[1+max_drift,1e6],[1-max_drift,1e6]]
        for i in inliers:
            event = anchors[i]
            # a*x_low+b <= y_high+systematic; a*x_high+b >= y_low-systematic
            poly = _clip(poly, event["x"][0], 1, event["y"][1]+systematic_seconds)
            poly = _clip(poly, -event["x"][1], -1, -event["y"][0]+systematic_seconds)
            if not poly:
                break
    else:
        low = max(anchors[i]["y"][0]-anchors[i]["x"][1]-systematic_seconds for i in inliers)
        high = min(anchors[i]["y"][1]-anchors[i]["x"][0]+systematic_seconds for i in inliers)
        poly = [[1.,low],[1.,high]] if low <= high else []
    if not poly:
        return {"status": "INCONSISTENT", "reason": "empty feasible parameter region"}
    # Choose a feasible central estimate; the robust point need not satisfy all bounds.
    centroid = np.mean(np.asarray(poly), axis=0)
    slope, intercept = float(centroid[0]), float(centroid[1])
    support = [float(min(x[inliers])), float(max(x[inliers]))]
    interval_errors = residuals(slope, intercept)
    return {"status": "SUPPORTED", "model": "affine_drift" if drift else "constant_offset",
            "slope": slope, "intercept_seconds": intercept, "drift_ppm": (slope-1)*1e6 if drift else None,
            "drift_status": ("ZERO_NOT_EXCLUDED" if min(p[0] for p in poly) <= 1 <= max(p[0] for p in poly) else "NONZERO_WITHIN_MODEL_BOUNDS") if drift else "NOT_IDENTIFIABLE_ON_THIS_SPAN",
            "drift_interval_ppm": [(min(p[0] for p in poly)-1)*1e6,(max(p[0] for p in poly)-1)*1e6] if drift else None,
            "parameter_polygon": poly, "x_support": support,
            "y_support": [float(min(y[inliers])), float(max(y[inliers]))],
            "systematic_seconds": systematic_seconds,
            "assumption": "affine only between checked anchors within the same uninterrupted segment",
            "anchor_ids": [anchors[i]["id"] for i in inliers],
            "outlier_ids": [a["id"] for i,a in enumerate(anchors) if i not in inliers],
            "residuals": [{"id": a["id"], "center_seconds": float(y[i]-(slope*x[i]+intercept)),
                           "interval_violation_seconds": float(interval_errors[i]), "inlier": bool(i in inliers)} for i,a in enumerate(anchors)],
            "rms_center_residual_seconds": float(np.sqrt(np.mean((y[inliers]-(slope*x[inliers]+intercept))**2)))}
