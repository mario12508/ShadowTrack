"""Lazy GPU dependencies; CPU import/tracking never imports torch or YOLO."""

import contextlib
import importlib.metadata
import io
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..probe import sha256
from .common import DetectionError
from .schema import encode_rle
from .progress import phase


def environment(requested="auto"):
    packages = {}
    for name in ("torch","ultralytics","av","numpy","supervision","opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except ImportError:
        torch,cuda = None,False
    if requested not in ("auto","cpu","cuda:0"):
        raise DetectionError("Supported devices: auto, cpu, cuda:0")
    if requested == "cuda:0" and not cuda:
        raise DetectionError("CUDA requested but torch.cuda.is_available() is false")
    device = "cuda:0" if requested != "cpu" and cuda else "cpu"
    nvidia = shutil.which("nvidia-smi")
    smi = {"status":"UNAVAILABLE"}
    if nvidia:
        result = subprocess.run([nvidia,"--query-gpu=name,memory.total,memory.free,driver_version","--format=csv,noheader,nounits"],
                                capture_output=True,text=True,timeout=15,check=False)
        smi = {"status":"AVAILABLE" if result.returncode == 0 else "ERROR","returncode":result.returncode,
               "gpu_memory_driver_csv":result.stdout.strip(),"error":result.stderr.strip()}
    gpu = None
    if cuda:
        free,total = torch.cuda.mem_get_info(0)
        gpu = {"name":torch.cuda.get_device_name(0),"vram_free_bytes":free,"vram_total_bytes":total,
               "torch_cuda_version":torch.version.cuda}
    return {"selected_device":device,"torch_cuda_is_available":cuda,"packages":packages,"gpu":gpu,"nvidia_smi":smi}


class YoloSegmenter:
    backend_kind = "ultralytics"

    def __init__(self, weights: Path, lock, config, device, progress=None):
        self.progress = progress
        digest = progress.hash_file(weights,"weights_sha256") if progress is not None else sha256(weights)
        if digest != lock["weights_sha256"]:
            raise DetectionError("Weights SHA256 mismatch; no implicit model download allowed")
        if importlib.metadata.version("ultralytics") != lock["ultralytics_version"]:
            raise DetectionError("Ultralytics version does not match model lock")
        # No integrations or image/log uploads. All predictions use already verified local weights.
        os.environ["YOLO_VERBOSE"] = "False"
        os.environ["WANDB_DISABLED"] = "true"
        with phase(progress,"ultralytics_import",announce=True):
            from ultralytics import YOLO, settings
        settings.update({"sync":False,"wandb":False,"mlflow":False,"comet":False,"clearml":False,"neptune":False,
                         "hub":False,"tensorboard":False,"dvc":False,"raytune":False})
        import torch
        self.torch,self.config,self.device = torch,config,device
        with phase(progress,"model_load",announce=True):
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.model = YOLO(str(weights),task="segment")
        if self.model.task != "segment" or self.model.names[0] != "person":
            raise DetectionError("Pinned model must be person segmentation")
        with phase(progress,"model_warmup",announce=True):
            self.predict(np.zeros((config["imgsz"],config["imgsz"],3),dtype=np.uint8))  # excluded synthetic warm-up
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def predict(self, rgb):
        kwargs = {k:self.config[k] for k in ("imgsz","conf","iou","max_det","classes","retina_masks","half")}
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        # Ultralytics numpy sources expect BGR. This is the only channel conversion.
        with phase(self.progress,"yolo_predict"):
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                result = self.model.predict(np.ascontiguousarray(rgb[:,:,::-1]),device=self.device,verbose=False,save=False,show=False,**kwargs)[0]
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        detections = []
        if result.boxes is not None and len(result.boxes):
            if result.masks is None or len(result.masks.data) != len(result.boxes):
                raise DetectionError("Segmentation backend returned boxes without masks")
            with phase(self.progress,"masks_to_cpu"):
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                masks = result.masks.data.cpu().numpy()
            if masks.shape[1:] != rgb.shape[:2]:
                raise DetectionError("retina_masks must retain original frame resolution")
            for box,score,cls,mask in zip(boxes,scores,classes,masks):
                if int(cls) != 0:
                    continue
                with phase(self.progress,"mask_rle_encode"):
                    detections.append({"bbox_xyxy":[float(v) for v in box],"confidence":float(score),"class_id":0,
                                       "mask_rle":encode_rle(mask>.5)})
        return detections,{"inference_seconds":elapsed,
                           "peak_vram_allocated_bytes":int(self.torch.cuda.max_memory_allocated()) if self.device.startswith("cuda") else None,
                           "peak_vram_reserved_bytes":int(self.torch.cuda.max_memory_reserved()) if self.device.startswith("cuda") else None}

    def close(self):
        del self.model
        import gc
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


class YoloDetector:
    """Batched person boxes only. Importing this module does not import torch."""
    backend_kind = "ultralytics"

    def __init__(self, weights, model_lock, config, device, progress=None):
        self.config,self.device,self.progress = config,device,progress
        digest = progress.hash_file(weights,"weights_sha256") if progress else sha256(weights)
        if digest != model_lock["weights_sha256"] or importlib.metadata.version("ultralytics") != model_lock["ultralytics_version"]:
            raise DetectionError("Detection weights/version differ from pinned model")
        os.environ["YOLO_VERBOSE"],os.environ["WANDB_DISABLED"] = "False","true"
        from ultralytics import YOLO,settings
        settings.update({"sync":False,"wandb":False,"mlflow":False,"comet":False,"clearml":False,"neptune":False,
                         "hub":False,"tensorboard":False,"dvc":False,"raytune":False})
        import torch
        self.torch = torch
        self.effective_half = bool(config.get("half") and device.startswith("cuda"))
        with phase(progress,"detection_model_load",announce=True):
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.model = YOLO(str(weights),task="detect")
        if self.model.task != "detect" or self.model.names[0] != "person":
            raise DetectionError("Pinned model must be a person detection model")
        with phase(progress,"detection_warmup",announce=True):
            self.predict_batch([np.zeros((config["imgsz"],config["imgsz"],3),dtype=np.uint8)])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def predict_batch(self, images):
        if not images:
            return [],{"inference_seconds":0.,"effective_half":self.effective_half}
        kwargs = {k:self.config[k] for k in ("imgsz","conf","iou","max_det","classes")}
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with phase(self.progress,"yolo_detection_batch"):
            with self.torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                results = self.model.predict([np.ascontiguousarray(rgb[:,:,::-1]) for rgb in images],
                    device=self.device,half=self.effective_half,batch=len(images),verbose=False,save=False,show=False,**kwargs)
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if len(results) != len(images):
            raise DetectionError("Batch detector changed frame count/order")
        output = []
        with phase(self.progress,"boxes_to_cpu"):
            for result in results:
                boxes = []
                if result.boxes is not None:
                    for raw in result.boxes.data.cpu().numpy():
                        x1,y1,x2,y2,score,cls = raw.tolist()
                        if int(cls) == 0:
                            boxes.append({"bbox_xyxy":[x1,y1,x2,y2],"confidence":score,"class_id":0})
                output.append(boxes)
        return output,{"inference_seconds":elapsed,"effective_half":self.effective_half,
            "peak_vram_allocated_bytes":int(self.torch.cuda.max_memory_allocated()) if self.device.startswith("cuda") else None,
            "peak_vram_reserved_bytes":int(self.torch.cuda.max_memory_reserved()) if self.device.startswith("cuda") else None}

    close = YoloSegmenter.close
