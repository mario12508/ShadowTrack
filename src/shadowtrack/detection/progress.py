"""Metadata-only console heartbeat and durable progress; no image logging."""

from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from .common import atomic_json


class Progress:
    def __init__(self, operation="run", interval=10., enabled=True, stream=None):
        if interval <= 0:
            raise ValueError("progress_interval must be positive")
        self.interval,self.enabled = float(interval),enabled
        # Capture the actual notebook stream before YOLO redirects stdout/stderr.
        self.stream = stream if stream is not None else sys.stdout
        self.lock,self.stop_event = threading.RLock(),threading.Event()
        # High-resolution relative clock avoids zero-age heartbeat races on Windows.
        self.started = self.activity = self.phase_started = time.perf_counter()
        self.state = {"operation":operation,"status":"RUNNING","phase":"starting"}
        self.timings,self.directory,self.thread = {},None,None

    def start(self):
        if self.enabled:
            self.emit("start")
            self.thread = threading.Thread(target=self._heartbeat,name="shadowtrack-progress",daemon=True)
            self.thread.start()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, error, traceback):
        status = "COMPLETE" if kind is None else "INTERRUPTED" if isinstance(error,(KeyboardInterrupt,InterruptedError)) else "FAILED"
        self.close(status,kind.__name__ if kind is not None else None)

    def attach(self, directory):
        if self.enabled:
            self.directory = Path(directory)
            self.directory.mkdir(parents=True,exist_ok=True)
            self.emit("telemetry_ready")

    def update(self, event=None, **fields):
        if not self.enabled:
            return
        if any(not isinstance(v,(str,int,float,bool,type(None))) for v in fields.values()):
            raise TypeError("Progress fields must be scalar metadata")
        with self.lock:
            if "phase" in fields and fields["phase"] != self.state["phase"]:
                self.phase_started = time.perf_counter()
            self.state.update(fields)
            self.activity = time.perf_counter()
            if event:
                self.emit(event)

    @contextmanager
    def stage(self, name, announce=False, **fields):
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        with self.lock:
            previous,previous_start = self.state["phase"],self.phase_started
            self.state.update(fields,phase=name)
            self.phase_started = self.activity = started
            if announce:
                self.emit(name+"_start")
        try:
            yield
        except BaseException:
            with self.lock:
                self.state.setdefault("failed_phase",name)
            raise
        finally:
            with self.lock:
                elapsed = time.perf_counter()-started
                self.timings[name] = self.timings.get(name,0.)+elapsed
                if announce:
                    self.emit(name+"_end")
                self.state["phase"],self.phase_started = previous,previous_start
                self.activity = time.perf_counter()

    def emit(self, event):
        if not self.enabled:
            return
        with self.lock:
            now = time.perf_counter()
            record = {**self.state,"event":event,"elapsed_s":round(now-self.started,3),
                      "phase_elapsed_s":round(now-self.phase_started,3),
                      "no_counter_progress_s":now-self.activity,
                      "stage_seconds_inclusive":{k:round(v,3) for k,v in self.timings.items()}}
            parts = [f"[ShadowTrack +{record['elapsed_s']:.1f}s]",event,f"phase={record['phase']}"]
            for key in ("camera_id","config_name","device","file","camera_decoded_frames","camera_total_frames",
                        "decoded_frames_session","total_decode_frames","processed_frames","total_inference_frames",
                        "inferred_frames_session","committed_frames",
                        "chunks_done","chunks_total","source_frame_index","raw_media_pts","detections_session",
                        "hash_bytes_done","hash_bytes_total","files_done","files_total","camera_wall_fps",
                        "camera_eta_s","end_to_end_processed_fps","vram_allocated_bytes","checkpoint_chunks_verified","error_type","failed_phase","message"):
                if key in record and record[key] is not None:
                    parts.append(f"{key}={record[key]}")
            if event == "heartbeat":
                parts += [f"phase_age={record['phase_elapsed_s']:.1f}s",f"idle={record['no_counter_progress_s']:.1f}s"]
            try:
                self.stream.write(" ".join(parts)+"\n")
                self.stream.flush()
            except (OSError,ValueError):
                pass
            if self.directory is not None:
                try:
                    atomic_json(self.directory/"progress.json",record)
                    with (self.directory/"events.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+"\n")
                        stream.flush()
                except OSError as exc:
                    self.state["telemetry_error"] = type(exc).__name__

    def _heartbeat(self):
        while not self.stop_event.wait(self.interval):
            self.emit("heartbeat")

    def close(self, status="COMPLETE", error_type=None):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.update(status=status,error_type=error_type)
        self.emit(status.lower())

    def hash_file(self, path, label="sha256"):
        path = Path(path)
        digest,done = hashlib.sha256(),0
        with self.stage(label,announce=True,file=path.name,hash_bytes_done=0,hash_bytes_total=path.stat().st_size):
            with path.open("rb") as stream:
                while block := stream.read(1024*1024):
                    digest.update(block)
                    done += len(block)
                    self.update(hash_bytes_done=done)
        return digest.hexdigest()


def phase(progress, name, **kwargs):
    return progress.stage(name,**kwargs) if progress is not None else nullcontext()


def observed(operation):
    """Public operations own a reporter unless the caller supplies a shared one."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args,progress=None,progress_interval=10.,**kwargs):
            own = not isinstance(progress,Progress)
            reporter = Progress(operation,progress_interval,enabled=progress is not False).start() if own else progress
            try:
                result = function(*args,progress=reporter,**kwargs)
            except BaseException as exc:
                if own:
                    reporter.close("INTERRUPTED" if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else "FAILED",type(exc).__name__)
                raise
            else:
                if own:
                    reporter.close()
                return result
        return wrapped
    return decorate
