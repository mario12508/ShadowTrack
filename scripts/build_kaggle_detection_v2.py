"""Notebook contains setup/orchestration only; detection and outputs use shared code."""

from pathlib import Path
import nbformat as nbf


def main():
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells=[md('''# ShadowTrack YOLO Detection v2

**Machine detections are the primary artifact.** Every processed frame, including an empty frame, has source index, unchanged raw PTS and segment identity. FULL saves every source frame. MP4 files are derived private demonstrations.

Default is **BENCHMARK**, execution disabled. Compare nano/small on the same short continuous windows, inspect private JPEG examples, then explicitly choose a configuration for DEMO. FULL is separate and never starts automatically. No Re-ID, global IDs, BEV or frontend.
'''),code('''from pathlib import Path

EXECUTE = False
MODE = "BENCHMARK"          # BENCHMARK -> DEMO -> explicit FULL
DEVICE = "auto"             # cuda:0 enforces GPU; auto can fall back to CPU
BATCH_SIZE = 8               # 8 or 16 on GPU; reduce if needed
BENCHMARK_CONFIGURATIONS = ["nano_640", "small_640"]  # Optional nano_960/small_960
SELECTED_CONFIGURATION = None  # DEMO/FULL: choose after benchmark + visual review
PROGRESS_INTERVAL_SECONDS = 10
MAX_WALL_SECONDS = 10000

CODE_DIR = Path("/kaggle/input/datasets/mario12508/shadowtrack-tools")
EVIDENCE_DIR = CODE_DIR
VIDEO_DIR = Path("/kaggle/input/datasets/mario12508/shadowtrack-videos")
WEIGHTS_DIR = CODE_DIR
ALLOW_PINNED_WEIGHTS_DOWNLOAD = True  # Known public 5.6/19.3 MB files only, SHA256 checked
WORK_ROOT = Path("/kaggle/working/shadowtrack_detection_v2")
RESUME_DIR = None            # Unpacked checkpoint folder, not a ZIP filename
FULL_DEMO_DIR = None         # FULL: completed matching DEMO folder incl. its five MP4
'''),md('''## Private input folders

Update the tools Dataset with the entire current `prepare-v2` bundle. It includes ordinary JSON `input_evidence/`; a separate evidence Dataset is optional. Kaggle may unpack ZIP and inner gzip files: this notebook reads folders and validates file/content SHA256. Keep Dataset, Notebook and outputs private. No source recordings or images are uploaded by code.

Use a fresh session after updating the Dataset. Input paths vary by account. No archive filename/hash input or cloud-permission flags are required. Public model download needs Notebook internet; alternatively manually attach both pinned `.pt` files and set `ALLOW_PINNED_WEIGHTS_DOWNLOAD=False`.
'''),code('''if not EXECUTE:
    raise RuntimeError("Set input folders and EXECUTE=True to run BENCHMARK")
if MODE not in ("BENCHMARK", "DEMO", "FULL"):
    raise ValueError("Unknown MODE")
import importlib, sys

def input_file(folder, filename):
    folder = Path(folder)
    direct = folder / filename
    paths = [direct] if direct.is_file() else sorted(folder.rglob(filename))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {filename} under {folder}, found {len(paths)}; choose a more precise folder")
    return paths[0]

CODE_ROOT = input_file(CODE_DIR, "code_manifest.json").parent
sys.path.insert(0, str(CODE_ROOT / "src"))
for name in tuple(sys.modules):
    if name == "shadowtrack" or name.startswith("shadowtrack."):
        del sys.modules[name]
importlib.invalidate_caches()
from shadowtrack.detection.portable import verify_code_tree
CODE_HASH = verify_code_tree(CODE_ROOT)
import shadowtrack
print("Code:", CODE_HASH, "package:", shadowtrack.__file__, flush=True)
'''),code('''import importlib.util, subprocess
from shadowtrack.detection.progress import Progress
from shadowtrack.detection.common import read_json, atomic_json, object_hash
if importlib.util.find_spec("torch") is None:
    raise RuntimeError("Select a Kaggle image with preinstalled CUDA torch")
with Progress("dependencies", PROGRESS_INTERVAL_SECONDS) as p:
    with p.stage("pip_install", announce=True):
        subprocess.run([sys.executable,"-m","pip","install","-q","--only-binary=:all:",
                        "ultralytics==8.3.161","av==14.2.0","opencv-python==4.11.0.86"],check=True)
from shadowtrack.detection.runtime import environment
RUNTIME = environment(DEVICE)
WORK_ROOT.mkdir(parents=True, exist_ok=True)
atomic_json(WORK_ROOT / "environment.json", RUNTIME)
print(RUNTIME)
'''),code('''from shadowtrack.detection.evidence import load_evidence
from shadowtrack.detection.notebook_support import pinned_detection_weights

LOCK_PATH = CODE_ROOT / "configs/detection_v2_lock.json"
SCENE_PATH = CODE_ROOT / "configs/scene_filter_v1.json"
LOCK = read_json(LOCK_PATH)
SETTINGS = read_json(CODE_ROOT / "configs/detection_v2.json")
with Progress("inputs", PROGRESS_INTERVAL_SECONDS) as p:
    with p.stage("source_evidence", announce=True):
        EVIDENCE_DIR = input_file(EVIDENCE_DIR, "source_manifest.json").parent
        EVIDENCE = load_evidence(EVIDENCE_DIR)
        parents = {input_file(VIDEO_DIR,c["source_name"]).parent for c in EVIDENCE["cameras"].values()}
        if len(parents) != 1:
            raise RuntimeError("Keep the five MKV together in one input directory")
        VIDEO_DIR = parents.pop()
    MODEL_CACHE = WORK_ROOT / "model_cache"
    MODEL_CACHE.mkdir(exist_ok=True)
    for key, model in LOCK["models"].items():
        with p.stage("pinned_weights_"+key, announce=True):
            matches = sorted(WEIGHTS_DIR.rglob(model["weights_filename"]))
            if len(matches) > 1:
                raise RuntimeError("Multiple weight files; choose WEIGHTS_DIR more precisely")
            target = matches[0] if matches else MODEL_CACHE / model["weights_filename"]
            pinned_detection_weights(LOCK_PATH,key,target,allow_download=ALLOW_PINNED_WEIGHTS_DOWNLOAD)
            if target != MODEL_CACHE / model["weights_filename"]:
                import shutil
                shutil.copyfile(target,MODEL_CACHE / model["weights_filename"])
            print(key, model["weights_sha256"], LOCK["license"])

RUN_KEY = object_hash({"code":CODE_HASH,"mode":MODE,"selected":SELECTED_CONFIGURATION,
                      "benchmark":BENCHMARK_CONFIGURATIONS,"batch":BATCH_SIZE,"settings":SETTINGS})[:16]
RUN_DIR = WORK_ROOT / (MODE.lower()+"_"+RUN_KEY)
print("Run directory:",RUN_DIR)
print("Source evidence producer:",EVIDENCE["manifest"]["producer_code_hash"],"(immutable provenance, not required to equal inference code)")
'''),md('''## Run and checkpoint

One full sequential decode/PTS check per camera serves every selected model. Only the chosen windows receive inference in BENCHMARK/DEMO; EOF validation still reads the entire source, so a 10-second benchmark is not a 10-second job. FULL selects every source frame, including empty frames. No frame skipping or frame-index/FPS timestamps.

Default benchmark uses 10 seconds per camera. DEMO uses one continuous 30-second interval per camera, not concatenated SAMPLE windows. FP16 is used on CUDA; CPU uses FP32. Camera 5 adds a small far-zone crop with source-coordinate translation and explicit duplicate decisions. Scene decisions do not delete raw candidates. Private JPEGs show green accepted, orange uncertain, red rejected; all are retained in data.

Atomic camera/segment/range chunks resume independently. If MP4 fails, complete machine detections are still saved; `demo_videos.json` records the failure. Large GIFs are not generated. Numeric heartbeat and timings live in `telemetry/`.
'''),code('''from shadowtrack.detection.transport import restore_checkpoint, export_run
from shadowtrack.detection.engine import run_detection

if RESUME_DIR is not None and not RUN_DIR.exists():
    restore_checkpoint(input_file(RESUME_DIR,"export_manifest.json").parent,RUN_DIR,
                       progress_interval=PROGRESS_INTERVAL_SECONDS)
demo_source = input_file(FULL_DEMO_DIR,"demo_videos.json").parent if MODE == "FULL" and FULL_DEMO_DIR else None
try:
    state = run_detection(EVIDENCE_DIR,VIDEO_DIR,MODEL_CACHE,LOCK_PATH,SCENE_PATH,RUN_DIR,CODE_ROOT,
        mode=MODE, selected_configuration=SELECTED_CONFIGURATION,
        settings=SETTINGS, batch_size=BATCH_SIZE, device=DEVICE,
        benchmark_configurations=BENCHMARK_CONFIGURATIONS, demo_source=demo_source,
        max_wall_seconds=MAX_WALL_SECONDS, progress_interval=PROGRESS_INTERVAL_SECONDS)
finally:
    if (RUN_DIR / "run_manifest.json").is_file():
        name = "shadowtrack_detection_demo.zip" if MODE == "DEMO" else "shadowtrack_detection_"+MODE.lower()+".zip"
        DOWNLOAD_ZIP = WORK_ROOT / name
        export_run(RUN_DIR,DOWNLOAD_ZIP,progress_interval=PROGRESS_INTERVAL_SECONDS)
        print("Saved checkpoint / download:", DOWNLOAD_ZIP, flush=True)
        print("Save a PRIVATE Version or download outputs before ending the session.", flush=True)
'''),code('''benchmark = read_json(RUN_DIR / "benchmark.json")
print("Status:",benchmark["status"],"GPU:",benchmark["gpu_benchmark_status"])
print("Wall seconds:",benchmark["wall_seconds_all_sessions"],"end-to-end processed FPS:",benchmark["end_to_end_processed_fps"])
for config_hash, row in benchmark["configurations"].items():
    print(row["config"]["name"],"frames:",row["processed_frames"],"raw boxes:",row["raw_detections"],
          "empty:",row["empty_frames"],"predict FPS:",row["predict_fps"],"scene:",row["scene_counts"])
print("Private JPEGs:",RUN_DIR / "_private_quality")
if MODE == "DEMO":
    videos = read_json(RUN_DIR / "demo_videos.json")
    print("Derived video status:",videos["status"])
    for video in videos["videos"]:
        print(RUN_DIR / "demo" / video["path"],"source frames:",video["frames"])
    print("Download one ZIP:",WORK_ROOT / "shadowtrack_detection_demo.zip")
'''),md('''## Choose DEMO and download

After BENCHMARK inspect both models' private JPEGs and statistics. More boxes are not proof of better quality. Set `MODE="DEMO"`, explicitly set `SELECTED_CONFIGURATION="nano_640"` or `"small_640"` and rerun the notebook. The run folder changes automatically. DEMO produces `demo/camera_1_detected.mp4` through `camera_5_detected.mp4` and sidecars containing original frame/PTS. The ZIP includes **all machine records, including empty frames**, five MP4 and a report. JPEG examples remain in private output files, outside the data export.

Download `/kaggle/working/shadowtrack_detection_v2/shadowtrack_detection_demo.zip` from Notebook Outputs. Locally:

```powershell
shadowtrack detection import --source C:/PRIVATE_DOWNLOADS/shadowtrack_detection_demo.zip
# Set detection_run/configuration_name in configs/local_track_v2.toml, then:
shadowtrack local-track --config configs/local_track_v2.toml
```

Local ByteTrack reads stored schema v2 records without Torch/YOLO. Optional `[video]` generates local-ID videos from those track rows; IDs are camera/segment scoped, never global. Keep raw detection records as the primary reusable output.

Only after successful DEMO set `MODE="FULL"`, keep the selected config/batch and set `FULL_DEMO_DIR` to the completed matching DEMO folder (attach its unpacked ZIP if using a new session). FULL saves detections for **every frame of all five sources** and does not automatically render full-length MP4. It never starts as a consequence of BENCHMARK/DEMO completing.
''')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}})
    nbf.validate(nb)
    target=Path(__file__).resolve().parents[1]/"notebooks/kaggle_detection_v2.ipynb"
    nbf.write(nb,target)
    print(target)


if __name__ == "__main__":main()
