"""Read-only preservation checks plus locally verifiable hybrid-pipeline acceptance."""

import ast
import importlib.metadata
import json
import platform
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import nbformat

from shadowtrack.probe import sha256
from shadowtrack.detection.common import atomic_json,code_fingerprint,read_json
from shadowtrack.detection.evidence import load_evidence,sample_plan
from shadowtrack.detection.portable import verify_code_tree


def main():
    root = Path(__file__).resolve().parents[1]
    output = root/"artifacts/detection"
    protected = read_json(output/"preserved_hashes.json")
    changed = [name for name,digest in protected.items() if sha256(root/name) != digest]
    assert not changed, f"Protected audit/alignment/source files changed: {changed}"
    alignment = read_json(root/"artifacts/sync/time_alignment.json")
    assert all(sha256(root/"artifacts/sync"/c["local_frame_times_file"])==c["local_frame_times_sha256"] for c in alignment["cameras"])
    evidence = load_evidence(output/"input_evidence")
    assert evidence["manifest"]["producer_code_hash"] == code_fingerprint(root)
    plan = sample_plan(evidence)
    artifact = read_json(output/"portable/code_artifact.json")
    code_zip = output/"portable"/artifact["filename"]
    assert artifact["sha256"] == sha256(code_zip) and artifact["code_hash"] == code_fingerprint(root)
    assert verify_code_tree(output/"portable"/artifact["directory"]) == artifact["code_hash"]
    with zipfile.ZipFile(code_zip) as archive:
        manifest = json.loads(archive.read("code_manifest.json"))
        assert not any(name.lower().endswith((".mkv",".jpg",".png",".gif",".pt")) for name in archive.namelist())
        for name,digest in manifest["files"].items():
            source = output/name if name.startswith("input_evidence/") else root/name
            assert sha256(source) == digest, f"Code artifact snapshot differs from repository: {name}"
    notebook = nbformat.read(root/"notebooks/kaggle_detection.ipynb",as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not cell.outputs and cell.execution_count is None
            ast.parse(cell.source)
    lock = read_json(root/"configs/detection_model_lock.json")
    weights = root/".tools/models"/lock["weights_filename"]
    assert weights.stat().st_size == lock["weights_bytes"] and sha256(weights)==lock["weights_sha256"]
    suite = ET.parse(output/"tests.xml").getroot().find("testsuite")
    assert suite is not None and all(suite.attrib[k] == "0" for k in ("errors","failures","skipped"))
    runtime = read_json(output/"runtime.json")
    benchmark = {"status":"NOT_RUN","gpu_available_in_current_environment":runtime["torch_cuda_is_available"],
                 "reason":"No Kaggle execution or real YOLO inference in this task; manual Kaggle execution and visual review pending",
                 "model_hash":lock["weights_sha256"],"configurations":[c["name"] for c in lock["configurations"]],
                 "throughput_fps":None,"peak_vram_bytes":None,"detections":None,"quality_examples":None,"winner":None}
    atomic_json(output/"gpu_benchmark_status.json",benchmark)
    sources = [".gitignore","README.md","pyproject.toml","src/shadowtrack/cli.py","docs/KAGGLE_SETUP.md","docs/DETECTION_REPORT.md",
               "configs/detection_model_lock.json","configs/local_track.toml","notebooks/kaggle_detection.ipynb",
               "scripts/build_kaggle_notebook.py","scripts/verify_detection.py","tests/test_detection.py",
               "schemas/detection_frame.schema.json","licenses/ULTRALYTICS_AGPL-3.0.txt"]
    sources += [p.relative_to(root).as_posix() for p in (root/"src/shadowtrack/detection").glob("*.py")]
    result = {"acceptance_status":"LOCAL_VALIDATED_KAGGLE_GPU_PENDING","tests":suite.attrib,
              "existing_test_count":45,"new_cpu_test_count":int(suite.attrib["tests"])-45,
              "protected_files_unchanged":True,"protected_file_count":len(protected),"alignment_arrays_unchanged":True,
              "source_mkv_unchanged":True,"actual_office_yolo_inference_run":False,"cloud_upload_performed":False,
              "gpu_benchmark_run":False,"full_run":False,"global_association":False,
              "sample_plan_frame_counts_per_configuration":plan["selected_frame_counts"],"sample_plan_total_frames_per_configuration":sum(plan["selected_frame_counts"].values()),
              "planned_windows":len(plan["windows"]),"planned_chunks_per_configuration":len(plan["chunks"]),
              "notebook_valid_and_outputs_empty":True,"portable_artifact":artifact,
              "weights_sha256_verified":True,"license":lock["license"],"python":platform.python_version(),
              "packages":{name:importlib.metadata.version(name) for name in ("numpy","av","supervision","opencv-python","pytest","nbformat")},
              "source_files":{name:sha256(root/name) for name in sorted(set(sources))}}
    atomic_json(output/"verification.json",result)
    print(json.dumps({"acceptance":result["acceptance_status"],"tests":suite.attrib["tests"],
                      "protected_files_unchanged":True,"code_artifact_sha256":artifact["sha256"],
                      "sample_frames_per_config":result["sample_plan_total_frames_per_configuration"]},indent=2))


if __name__ == "__main__":
    main()
