"""Generate a clean notebook containing orchestration only, never copied algorithms."""

from pathlib import Path

import nbformat as nbf


def main():
    cells = []
    md,code = nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells.append(md("""# ShadowTrack private Kaggle person segmentation

This notebook imports the versioned ShadowTrack package. It does not implement a second detector, decoder, schema or writer.

**Default: SAMPLE, execution disabled.** Keep both Dataset and Notebook private; do not publish notebook outputs. Inputs are unpacked folders under `/kaggle/input`, not ZIP files. No image is displayed inline. Quality images/GIFs are sensitive files in private outputs only. Read `docs/KAGGLE_SETUP.md` and the pinned model license before execution.
"""))
    cells.append(code('''from pathlib import Path

EXECUTE = False                 # Set True to run SAMPLE after configuring input folders.
MODE = "SAMPLE"                 # 30 raw-PTS seconds/camera across separate segments; two configs.
DEVICE = "auto"                 # auto / cpu / cuda:0. FULL requires cuda:0.
SAVE_PRIVATE_QUALITY = True
MAX_WALL_SECONDS = 10000        # Configured budget, not a claim about Kaggle's session quota.
PROGRESS_INTERVAL_SECONDS = 10  # Heartbeat even during a long model/decode/I/O call.
SHOW_PROGRESS = True

CODE_DIR = Path("/kaggle/input/datasets/mario12508/shadowtrack-tools")
EVIDENCE_DIR = CODE_DIR          # The current tools package bundles plain JSON input_evidence/.
VIDEO_DIR = Path("/kaggle/input/datasets/mario12508/shadowtrack-videos")
WEIGHTS_INPUT = CODE_DIR        # Folder containing yolo11n-seg.pt; adjust if stored elsewhere.
ALLOW_SMALL_PUBLIC_WEIGHTS_DOWNLOAD = False

RUN_DIR = Path("/kaggle/working/shadowtrack_private") / MODE.lower()
RESUME_DIR = None               # Optional unpacked prior checkpoint folder or its Dataset root.
REVIEWED_SAMPLE_DIR = None      # FULL: unpacked COMPLETE, real SAMPLE export folder.
FULL_REVIEW_JSON = None         # FULL: reviewer-edited JSON after comparing private galleries.
'''))
    cells.append(md("""## Manual input setup

Use private inputs for videos and tools/weights. The current tools package also includes `input_evidence/` with plain JSON files (no inner `.gz`). Replace the old tools Dataset version with this whole package, then restart the Kaggle kernel and Run All. Dataset visibility and Notebook visibility are separate settings. `shadowtrack detection prepare` creates a code folder alongside a ZIP for manual transfer. Kaggle inputs should contain the unpacked folder with `code_manifest.json`, `src/` and `input_evidence/`; no ZIP path or ZIP hash parameter is needed. Paths depend on your Dataset slugs. A unique nested folder is discovered automatically. A separate evidence Dataset is optional.

Select an available GPU accelerator in Notebook settings. Availability, quotas and session limits may change. A CPU fallback can test SAMPLE, but FULL is blocked without CUDA. Nothing below submits a public notebook or dataset.
"""))
    cells.append(code('''if not EXECUTE:
    raise RuntimeError("Execution disabled. Set EXECUTE=True to run the configured SAMPLE.")
if MODE not in ("SAMPLE", "FULL"):
    raise ValueError("MODE must be SAMPLE or FULL")

import importlib, sys

def input_file(folder, filename):
    folder = Path(folder)
    direct = folder / filename
    matches = [direct] if direct.is_file() else sorted(folder.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename} under {folder}, found {len(matches)}. Set the input folder more precisely.")
    return matches[0]

CODE_ROOT = input_file(CODE_DIR, "code_manifest.json").parent
sys.path.insert(0, str(CODE_ROOT / "src"))
# Discard an older ShadowTrack loaded by previous cells in this kernel.
for module_name in tuple(sys.modules):
    if module_name == "shadowtrack" or module_name.startswith("shadowtrack."):
        del sys.modules[module_name]
importlib.invalidate_caches()
from shadowtrack.detection.portable import verify_code_tree
import shadowtrack

print("Checking versioned package files...", flush=True)
CODE_HASH = verify_code_tree(CODE_ROOT)
print("Verified common code:", CODE_HASH)
print("Loaded package:", shadowtrack.__file__)
'''))
    cells.append(code('''# Use a fresh Kaggle kernel. Preserve the image's CUDA-enabled torch build.
import importlib.util, subprocess
from shadowtrack.detection.progress import Progress
if importlib.util.find_spec("torch") is None:
    raise RuntimeError("Use a Kaggle image with preinstalled torch; do not silently install a CPU-only torch build")
with Progress("dependencies", PROGRESS_INTERVAL_SECONDS, enabled=SHOW_PROGRESS) as setup:
    with setup.stage("pip_install", announce=True):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary=:all:",
                        "ultralytics==8.3.161", "av==14.2.0", "opencv-python==4.11.0.86"], check=True)

from shadowtrack.detection.common import atomic_json, read_json
from shadowtrack.detection.runtime import environment
with Progress("environment", PROGRESS_INTERVAL_SECONDS, enabled=SHOW_PROGRESS) as setup:
    with setup.stage("device_and_versions", announce=True):
        RUNTIME = environment(DEVICE)
RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
atomic_json(RUN_DIR.parent / "environment.json", RUNTIME)
print({k: RUNTIME[k] for k in ("selected_device", "torch_cuda_is_available", "packages", "gpu", "nvidia_smi")})
'''))
    cells.append(code('''from shadowtrack.detection.notebook_support import pinned_weights
from shadowtrack.detection.evidence import load_evidence

with Progress("inputs", PROGRESS_INTERVAL_SECONDS, enabled=SHOW_PROGRESS) as setup:
    with setup.stage("evidence_load", announce=True):
        LOCK_PATH = CODE_ROOT / "configs/detection_model_lock.json"
        LOCK = read_json(LOCK_PATH)
        EVIDENCE_DIR = input_file(EVIDENCE_DIR, "source_manifest.json").parent
        EVIDENCE = load_evidence(EVIDENCE_DIR)
        if EVIDENCE["manifest"]["producer_code_hash"] != CODE_HASH:
            raise RuntimeError("Code/evidence version mismatch. Use input_evidence/ from the same current tools package.")
    with setup.stage("weights_input", announce=True):
        if WEIGHTS_INPUT.is_dir():
            WEIGHTS_INPUT = input_file(WEIGHTS_INPUT, LOCK["weights_filename"])
        weights_path = WEIGHTS_INPUT if WEIGHTS_INPUT.is_file() else RUN_DIR.parent / "model_cache" / LOCK["weights_filename"]
        WEIGHTS = pinned_weights(LOCK_PATH, weights_path, allow_download=ALLOW_SMALL_PUBLIC_WEIGHTS_DOWNLOAD)
    with setup.stage("video_inputs", announce=True):
        video_parents = {input_file(VIDEO_DIR, c["source_name"]).parent for c in EVIDENCE["cameras"].values()}
        if len(video_parents) != 1:
            raise RuntimeError("All five source MKV must be in the same input folder")
        VIDEO_DIR = video_parents.pop()
print({"model": LOCK["name"], "model_hash": LOCK["weights_sha256"], "license": LOCK["license"],
       "cameras": len(EVIDENCE["cameras"]), "mode": MODE})
'''))
    cells.append(md("""## Resume, run, and persist

Resume uses the same code, weights, config, source hashes and source evidence. Completed atomic chunks are verified before reuse. Every decode pass still checks every source frame PTS and EOF count, including in SAMPLE; **only the selected 30 seconds per camera receive YOLO inference**. CPU decoding time can therefore exceed inference time.

**SAMPLE is not a 30-second runtime.** There are two complete decode passes per source (one per configuration), including frames outside YOLO windows. The run prints the exact plan before loading the model. Every 10 seconds a heartbeat shows the active phase, camera/config, decoded vs inferred frames, chunks, current source frame/PTS, camera wall FPS and rough camera ETA. A long phase with unchanged counters is reported as such, not as completed work. `source_sha256`, `model_warmup`, `decode_next_frame`, `yolo_predict`, `mask_rle_encode`, `chunk_commit` and export have separate names/timings. Numeric logs live under `RUN_DIR.parent / "telemetry" / RUN_DIR.name`: `progress.json` and `events.jsonl`. No image data are logged. Timing totals are inclusive for nested stages; camera ETA includes inference/I/O and is approximate.

On mismatch the run becomes FAILED_DECODE and cannot be imported. On interruption only complete chunks survive; an unpacked checkpoint folder can be attached privately in a new session. The `finally` block saves a data-only folder snapshot under `checkpoints/`, but it cannot survive an operating-system kill by itself. Save a PRIVATE Kaggle version or download the checkpoint before the session ends.
"""))
    cells.append(code('''from shadowtrack.detection.transport import restore_checkpoint, export_run_directory

if RESUME_DIR is not None and not RUN_DIR.exists():
    restore_checkpoint(input_file(RESUME_DIR, "export_manifest.json").parent, RUN_DIR,
                       progress=SHOW_PROGRESS, progress_interval=PROGRESS_INTERVAL_SECONDS)
SAMPLE_ROOT = None
if MODE == "FULL":
    if REVIEWED_SAMPLE_DIR is None or FULL_REVIEW_JSON is None:
        raise RuntimeError("FULL needs a complete SAMPLE folder and private visual review JSON")
    SAMPLE_ROOT = input_file(REVIEWED_SAMPLE_DIR, "export_manifest.json").parent
'''))
    cells.append(code('''from shadowtrack.detection.pipeline import run

try:
    state = run(EVIDENCE_DIR, VIDEO_DIR, WEIGHTS, LOCK_PATH, RUN_DIR,
                repo_root=CODE_ROOT, mode=MODE, device=DEVICE,
                save_private_quality=SAVE_PRIVATE_QUALITY, chunk_frames=128,
                review_path=Path(FULL_REVIEW_JSON) if FULL_REVIEW_JSON else None,
                sample_root=SAMPLE_ROOT, max_wall_seconds=MAX_WALL_SECONDS,
                progress=SHOW_PROGRESS, progress_interval=PROGRESS_INTERVAL_SECONDS)
finally:
    if (RUN_DIR / "run_manifest.json").is_file():
        import time
        snapshot = RUN_DIR.parent / "checkpoints" / f"{MODE.lower()}_{time.time_ns()}"
        export_run_directory(RUN_DIR, snapshot, progress=SHOW_PROGRESS,
                             progress_interval=PROGRESS_INTERVAL_SECONDS)
        print("Saved data-only checkpoint folder:", snapshot)
        print("Save a PRIVATE Kaggle version or download it before ending the session.")
'''))
    cells.append(code('''# Numeric measurements only. No frames, plots of images, or arrays are emitted to cell outputs.
benchmark = read_json(RUN_DIR / "benchmark.json")
print({"benchmark_status": benchmark["status"], "throughput_definition": benchmark["throughput_definition"]})
for config_hash, measurements in benchmark["configurations"].items():
    print({"config_hash": config_hash, "name": measurements["config"]["name"],
           "processed_frames": measurements["processed_frames"], "detections": measurements["detections"],
           "empty_frames": measurements["empty_frames"], "throughput_fps": measurements["throughput_fps"],
           "peak_vram_allocated_bytes": measurements["peak_vram_allocated_bytes"]})
if MODE == "SAMPLE" and state["status"] == "COMPLETE":
    from shadowtrack.detection.notebook_support import review_template
    review_template(RUN_DIR, RUN_DIR / "_private_quality" / "FULL_REVIEW_TEMPLATE.json")
    print("Review both PRIVATE galleries/previews; the FULL review template remains unchecked.")
'''))
    cells.append(md("""## Private review and export/import

View the gallery JPEGs and short annotated GIFs only through your private output files. Check televisions/reflections, table occlusion, distant people, doors, feet and the 20/25 FPS difference. These are quality examples, not measured AP and not identity annotations. Never publish a notebook version containing frames, masks, galleries or detection outputs.

The checkpoint folder excludes all `_private_quality` media. Download that folder manually, then locally run:

```powershell
shadowtrack detection import --source C:/path/to/sample_checkpoint_folder
shadowtrack local-track --config configs/local_track.toml
```

Set the imported run directory and selected configuration in local_track.toml first. This runs ByteTrack on CPU using saved detections. Keep timestamp segments separate. FULL is a later explicit action: review both SAMPLE configurations, edit the review JSON, select one configuration, set MODE=FULL, attach the completed SAMPLE folder and review, and use a GPU. FULL does not start automatically.
"""))
    notebook = nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                                       "language_info":{"name":"python","version":"3.11"}})
    target = Path(__file__).resolve().parents[1]/"notebooks/kaggle_detection.ipynb"
    target.parent.mkdir(exist_ok=True)
    nbf.validate(notebook)
    nbf.write(notebook,target)
    print(target)


if __name__ == "__main__":
    main()
