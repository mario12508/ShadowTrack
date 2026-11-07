"""Sequential decoder guard: verify every actual PTS, then verify EOF frame count."""

from fractions import Fraction
from pathlib import Path

from ..probe import sha256
from .common import DetectionError, atomic_json
from .progress import phase


def pyav_frames(path: Path):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        for frame in container.decode(stream):
            yield frame.pts,frame.time_base,frame


def checked_frames(path: Path, camera: dict, diagnostic: Path, decoder=pyav_frames, progress=None):
    checked = 0
    try:
        digest = progress.hash_file(path,"source_sha256") if progress is not None else sha256(path)
        if digest != camera["source_sha256"]:
            raise DetectionError("Wrong source video SHA256")
        iterator = iter(decoder(path))
        i = 0
        while True:
            with phase(progress,"decode_next_frame"):
                try:
                    pts,time_base,frame = next(iterator)
                except StopIteration:
                    break
            if i >= camera["frame_count"]:
                raise DetectionError(f"Decoder emitted extra frame at index {i}")
            if pts is None or time_base is None:
                raise DetectionError(f"Decoder missing PTS at source frame {i}")
            actual = Fraction(pts)*Fraction(time_base)
            expected = Fraction(str(camera["frames"][i][1]))
            if abs(actual-expected) > Fraction(1,1000000):
                raise DetectionError(f"Decoded frame order/PTS mismatch at index {i}: expected {float(expected)}, actual {float(actual)}")
            if (frame.width,frame.height) != (camera["width"],camera["height"]):
                raise DetectionError(f"Decoded dimensions mismatch at index {i}")
            checked += 1
            if progress is not None:
                progress.update(camera_decoded_frames=checked,source_frame_index=i,raw_media_pts=float(actual),
                                timestamp_segment_id=camera["frames"][i][2])
            yield i,frame
            i += 1
        if checked != camera["frame_count"]:
            raise DetectionError(f"Decoder frame count mismatch: expected {camera['frame_count']}, actual {checked}")
    except Exception as exc:
        atomic_json(diagnostic,{"schema_version":1,"status":"DECODE_MISMATCH","camera_id":camera["camera_id"],
                                "checked_frames":checked,"expected_frames":camera["frame_count"],
                                "error":str(exc),"images_saved":False,"timestamp_policy":"Original audit PTS only; never index/FPS"})
        raise DetectionError(f"Camera {camera['camera_id']} decode validation failed; diagnostic saved") from exc
