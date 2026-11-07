"""Optional GPU extraction uses exactly the same pinned OSNet implementation."""
from pathlib import Path
import nbformat as nbf


def main():
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells=[md('''# ShadowTrack — genuine person Re-ID embeddings

Optional offline acceleration. **No YOLO is imported or executed.** Inputs are unpacked private folders: versioned MVP code and source-bound track crops from `shadowtrack mvp reid-crops`. Keep Notebook, Dataset and outputs private. Images are never displayed inline or uploaded automatically. The CPU command uses the identical encoder.
'''),code('''from pathlib import Path
EXECUTE = False
CODE_DIR = Path("/kaggle/input/shadowtrack-mvp-tools")
CROP_DIR = Path("/kaggle/input/shadowtrack-reid-crops")
WEIGHTS_DIR = CODE_DIR
DEVICE = "cuda:0"
ALLOW_SMALL_PUBLIC_WEIGHTS_DOWNLOAD = True
OUTPUT = Path("/kaggle/working/shadowtrack_reid")
'''),code('''if not EXECUTE:
    raise RuntimeError("Configure private folders and set EXECUTE=True")
import sys, json, hashlib, importlib
def one(folder,name):
    direct=Path(folder)/name
    paths=[direct] if direct.is_file() else sorted(Path(folder).rglob(name))
    if len(paths)!=1:raise RuntimeError(f"Expected one {name}; found {len(paths)}")
    return paths[0]
CODE_ROOT=one(CODE_DIR,"mvp_code_manifest.json").parent
manifest=json.loads((CODE_ROOT/"mvp_code_manifest.json").read_text())
for name,digest in manifest["files"].items():
    path=(CODE_ROOT/name).resolve()
    if not path.is_relative_to(CODE_ROOT.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise RuntimeError("Versioned MVP code changed")
sys.path.insert(0,str(CODE_ROOT/"src"))
for name in tuple(sys.modules):
    if name=="shadowtrack" or name.startswith("shadowtrack."):del sys.modules[name]
importlib.invalidate_caches()
print("Verified MVP code",manifest["code_hash"],flush=True)
'''),code('''import subprocess
subprocess.run([sys.executable,"-m","pip","install","-q","av==14.2.0","Pillow>=10.1,<13"],check=True)
import torch
if DEVICE.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
print("torch",torch.__version__,"device",DEVICE,"CUDA",torch.cuda.is_available(),flush=True)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0),"free/total VRAM",torch.cuda.mem_get_info(0),flush=True)
'''),code('''from shadowtrack.detection.common import read_json
from shadowtrack.probe import sha256
from shadowtrack.mvp.reid import encode_crop_package
import urllib.request, os
LOCK_PATH=CODE_ROOT/"configs/reid_model_lock.json"
LOCK=read_json(LOCK_PATH)
CROP_ROOT=one(CROP_DIR,"crop_manifest.json").parent
OUTPUT.mkdir(parents=True,exist_ok=True)
matches=sorted(Path(WEIGHTS_DIR).rglob(LOCK["weights_filename"]))
if len(matches)>1:raise RuntimeError("Ambiguous weights folder")
weights=matches[0] if matches else OUTPUT/LOCK["weights_filename"]
if not weights.exists():
    if not ALLOW_SMALL_PUBLIC_WEIGHTS_DOWNLOAD:raise RuntimeError("Attach pinned OSNet weights")
    url=f"https://huggingface.co/kaiyangzhou/osnet/resolve/{LOCK['weights_revision']}/{LOCK['weights_filename']}"
    temporary=weights.with_suffix(".pending")
    with urllib.request.urlopen(url,timeout=120) as response,temporary.open("wb") as stream:
        received=0
        while block:=response.read(1024*1024):
            received+=len(block)
            if received>LOCK["weights_bytes"]:raise RuntimeError("Weights exceed pinned size")
            stream.write(block)
    if received!=LOCK["weights_bytes"] or sha256(temporary)!=LOCK["weights_sha256"]:raise RuntimeError("Weights SHA256 mismatch")
    os.replace(temporary,weights)
print("Pinned pretrained MSMT17 OSNet",LOCK["weights_sha256"],LOCK["license"],flush=True)
'''),code('''import zipfile
result=encode_crop_package(CROP_ROOT,weights,LOCK_PATH,OUTPUT/"embeddings",device=DEVICE,batch_size=32)
download=OUTPUT/"shadowtrack_embeddings.zip"
temporary=download.with_suffix(".pending")
with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED) as archive:
    for name in ("embedding_manifest.json","embeddings.npz"):archive.write(result/name,name)
os.replace(temporary,download)
print("Save a PRIVATE Version or download:",download,flush=True)
print("ZIP SHA256",sha256(download),flush=True)
print("Local: shadowtrack mvp embeddings-import --source PATH_TO_ZIP")
print("Then: shadowtrack mvp associate")
'''),md('''Embeddings are normalized track-level averages from at least three distinct source-bound observations. Local import validates model pin, SHA256, source run, frame indices, raw PTS and bounding boxes. External appearance similarity does not prove identity: supported time, uncertainty, spatial adjacency and conflict checks still gate associations. Missing clock evidence remains `GLOBAL_ASSOCIATION_PENDING`.
''')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.validate(nb);target=Path(__file__).resolve().parents[1]/'notebooks/kaggle_reid.ipynb';nbf.write(nb,target);print(target)


if __name__=='__main__':main()
