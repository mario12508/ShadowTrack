"""Audio sample order, packet time and video time are never equated implicitly."""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import numpy as np

from ..probe import AuditError, binary, run, sha256


def informativeness(samples: np.ndarray, sample_rate: int) -> dict:
    x = samples.astype(float)
    ac = float(np.std(x)) if len(x) else 0.0
    blocks = x[:len(x) // 800 * 800].reshape(-1, 800) if len(x) >= 800 else x.reshape(1, -1)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1)) if x.size else np.array([0.])
    db = 20 * np.log10(max(ac, 1e-12) / 32768)
    lag1 = float(np.corrcoef(x[:-1],x[1:])[0,1]) if len(x)>1 and ac>0 else None
    return {"samples": len(x), "sample_rate": sample_rate,
            "decoded_sample_duration_seconds": len(x) / sample_rate,
            "minimum": int(np.min(x)) if len(x) else None,
            "maximum": int(np.max(x)) if len(x) else None,
            "unique_values": int(len(np.unique(samples))),
            "ac_rms_dbfs": float(db), "dc_mean": float(np.mean(x)) if len(x) else None,
            "lag1_autocorrelation":lag1,"near_full_scale_fraction":float(np.mean(np.abs(x)>=32000)) if len(x) else None,
            "rms_block_p10_p90": [float(v) for v in np.quantile(rms, [.1, .9])],
            "informative": bool(db > -60),
            "reason": "AC energy above -60 dBFS" if db > -60 else "below_energy_gate; absence of shared evidence, not contradictory sync evidence"}


def correlate(x: np.ndarray, y: np.ndarray, sample_rate: int, max_lag_seconds: float = 3) -> dict:
    """Overlap-normalized Pearson CCF. Positive lag means x's event occurs later.

    FFT computes dot products; cumulative sums normalize each actual overlap.
    Periodic/flat signals cannot pass peak distinctiveness and energy gates.
    """
    ix, iy = informativeness(x, sample_rate), informativeness(y, sample_rate)
    if not ix["informative"] or not iy["informative"]:
        return {"status": "UNINFORMATIVE", "lag_seconds": None, "x": ix, "y": iy}
    a, b = np.asarray(x, float), np.asarray(y, float)
    # Block means low-pass before reducing the sample rate; avoid a raw stride alias.
    factor = max(1, sample_rate // 2000)
    a = a[:len(a)//factor*factor].reshape(-1, factor).mean(axis=1)
    b = b[:len(b)//factor*factor].reshape(-1, factor).mean(axis=1)
    sr = sample_rate / factor
    size = 1 << (len(a) + len(b) - 1).bit_length()
    conv = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b[::-1], size), size)
    limit = min(int(max_lag_seconds * sr), len(a) - 2, len(b) - 2)
    lags = np.arange(-limit, limit + 1)
    lo_a, lo_b = np.maximum(lags, 0), np.maximum(-lags, 0)
    n = np.minimum(len(a) - lo_a, len(b) - lo_b)
    def moments(z, lo):
        cs, cs2 = np.r_[0, np.cumsum(z)], np.r_[0, np.cumsum(z*z)]
        return cs[lo+n]-cs[lo], cs2[lo+n]-cs2[lo]
    sa, s2a = moments(a, lo_a)
    sb, s2b = moments(b, lo_b)
    denom = np.sqrt(np.maximum(0, (s2a-sa*sa/n)*(s2b-sb*sb/n)))
    cc = np.divide(conv[lags + len(b)-1] - sa*sb/n, denom,
                   out=np.zeros(len(lags)), where=denom > 1e-10)
    peak_i = int(np.argmax(cc))
    lag, peak = int(lags[peak_i]), float(cc[peak_i])
    competitors = cc[np.abs(lags-lag) > max(2, int(.1*sr))]
    second = float(np.max(competitors)) if len(competitors) else 0
    reliable = peak >= .65 and peak-second >= .15 and abs(lag) < limit and n[peak_i] >= 4*sr
    return {"status": "CANDIDATE" if reliable else "AMBIGUOUS", "lag_seconds": lag/sr,
            "lag_interval_seconds": [(lag-1)/sr, (lag+1)/sr],
            "peak_correlation": peak, "secondary_peak": second,
            "peak_margin": peak-second, "overlap_seconds": float(n[peak_i]/sr),
            "checked": False, "av_latency_verified": False,
            "reason": "Candidate requires repeated windows and verified audio/video time association"}


def extract_audio(camera: dict, output: Path, timeout: float = 180) -> tuple[dict, np.ndarray]:
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"camera_{camera['camera_id']}.s16le"
    metadata_path = output / f"camera_{camera['camera_id']}.audio.json"
    frames_path = output / f"camera_{camera['camera_id']}.frames.json"
    cache = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    # Cache is bound to source AND decoded outputs, not just their filenames.
    if (cache.get("input_sha256") == camera["sha256"] and target.exists() and cache.get("pcm_sha256") == sha256(target)
            and frames_path.exists() and cache.get("audio_frames_sha256") == sha256(frames_path)):
        return cache, np.fromfile(target, dtype="<i2")
    ffmpeg, ffprobe = binary("ffmpeg"), binary("ffprobe")
    command = [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", camera["file"],
               "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le", "-f", "s16le", str(target)]
    result = run(command, timeout)
    if result.returncode or result.stderr.strip():
        raise AuditError(f"Audio decode failed: {result.stderr}")
    command_pts = [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_frames", "-show_entries",
                   "frame=pts_time,best_effort_timestamp_time,nb_samples,sample_rate,channels", "-of", "json", camera["file"]]
    probe = run(command_pts, timeout)
    if probe.returncode or probe.stderr.strip():
        raise AuditError(f"Audio timestamp probe failed: {probe.stderr}")
    frames = json.loads(probe.stdout)["frames"]
    frames_path.write_text(probe.stdout, encoding="utf-8")
    samples = np.fromfile(target, dtype="<i2")
    expected = sum(int(f["nb_samples"]) for f in frames)
    if expected != len(samples):
        raise AuditError("Audio frame sample count disagrees with decoded PCM; no time association possible")
    rate = int(camera["audio_streams"][0]["sample_rate"])
    if int(camera["audio_streams"][0]["channels"]) != 1:
        raise AuditError("Audio analysis requires an explicitly selected mono channel")
    segments, gaps = [], []
    cursor, previous_end = 0, None
    for frame in frames:
        pts = float(frame.get("pts_time", frame.get("best_effort_timestamp_time")))
        count = int(frame["nb_samples"])
        if previous_end is None or abs(round(pts-previous_end,6)) > .002:
            if previous_end is not None:
                gaps.append({"before_end_pts": previous_end, "after_pts": pts, "delta_seconds": pts-previous_end})
            segments.append({"sample_start": cursor, "sample_stop": cursor+count, "first_pts": pts, "last_end_pts": pts+count/rate})
        else:
            segments[-1].update(sample_stop=cursor+count, last_end_pts=pts+count/rate)
        previous_end, cursor = pts+count/rate, cursor+count
    record = {"input_sha256": camera["sha256"], "pcm_sha256": sha256(target),
              "audio_frames_sha256":sha256(frames_path),
              "commands": [command, command_pts], "codec": camera["audio_streams"][0]["codec_name"],
              "metrics": informativeness(samples, rate), "segments": segments, "timestamp_gaps": gaps,
              "audio_video_latency": "UNVERIFIED", "full_video_probe_repeated": False}
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record, samples


def audio_window(record: dict, samples: np.ndarray, start: float, duration: float) -> np.ndarray | None:
    rate = record["metrics"]["sample_rate"]
    for segment in record["segments"]:
        if segment["first_pts"] <= start and start+duration <= segment["last_end_pts"]:
            i = segment["sample_start"] + round((start-segment["first_pts"])*rate)
            return samples[i:i+round(duration*rate)]
    return None  # Never concatenate across an audio timestamp break.


def analyze_audio(cameras: list[dict], output: Path, starts=(30, 150, 450, 650, 950, 1150), duration=12.) -> dict:
    records, arrays = {}, {}
    for camera in cameras:
        cid = camera["camera_id"]
        print(f"Audio {cid}: decode + audio-only timestamp probe", flush=True)
        records[cid], arrays[cid] = extract_audio(camera, output)
    pairs = []
    for a, b in itertools.combinations(records, 2):
        windows = []
        for start in starts:
            if records[a]["metrics"]["sample_rate"] != records[b]["metrics"]["sample_rate"]:
                windows.append({"raw_window_start":start,"status":"UNAVAILABLE","reason":"sample rates differ; no implicit resampling"})
                continue
            x, y = audio_window(records[a], arrays[a], start, duration), audio_window(records[b], arrays[b], start, duration)
            result = correlate(x, y, records[a]["metrics"]["sample_rate"]) if x is not None and y is not None else {"status": "UNAVAILABLE", "reason": "window crosses audio PTS break"}
            windows.append({"raw_window_start": start, "duration_seconds": duration, **result})
        pairs.append({"camera_a": a, "camera_b": b, "windows": windows,
                      "calibration_anchor": False,
                      "interpretation": "No anchor without informative distinctive common event and A/V association"})
    return {"status": "NO_RELIABLE_SHARED_EVIDENCE" if not any(w["status"] == "CANDIDATE" for p in pairs for w in p["windows"]) else "CANDIDATES_REQUIRE_VALIDATION",
            "cameras": records, "pairs": pairs, "automatically_applied_anchors": []}
