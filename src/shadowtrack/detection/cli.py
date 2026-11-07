"""Local preparation/import/export and CPU tracking command routing."""

from pathlib import Path


def add_parser(sub):
    parser = sub.add_parser("detection",help="Prepare private Kaggle inputs or import validated person detection outputs")
    commands = parser.add_subparsers(dest="detection_command",required=True)
    prepare = commands.add_parser("prepare",help="Create data-free code artifact and portable frame evidence; no upload")
    prepare.add_argument("--output-dir",type=Path,default=Path("artifacts/detection/portable"))
    prepare.add_argument("--evidence-dir",type=Path,default=Path("artifacts/detection/input_evidence"))
    prepare_v2 = commands.add_parser("prepare-v2",help="Bundle detection v2 code with unchanged saved audit evidence")
    prepare_v2.add_argument("--output-dir",type=Path,default=Path("artifacts/detection/v2/portable"))
    prepare_v2.add_argument("--evidence-dir",type=Path,default=Path("artifacts/detection/input_evidence"))
    imp = commands.add_parser("import",help="Validate complete output archive before local storage")
    imp.add_argument("--source",type=Path,required=True)
    imp.add_argument("--evidence-dir",type=Path,default=Path("artifacts/detection/input_evidence"))
    imp.add_argument("--target-dir",type=Path,default=Path("artifacts/detections"))
    imp.add_argument("--model-lock",type=Path,help="Model lock; default chosen from the package schema")
    annotate = commands.add_parser("annotate",help="CPU derived video from saved ByteTrack local IDs, no YOLO")
    annotate.add_argument("--track-dir",type=Path,required=True)
    annotate.add_argument("--evidence-dir",type=Path,default=Path("artifacts/detection/input_evidence"))
    annotate.add_argument("--raw-dir",type=Path,default=Path("data/raw"))
    annotate.add_argument("--output-dir",type=Path,required=True)
    exp = commands.add_parser("export",help="Save complete or partial detection checkpoint to data-only ZIP")
    exp.add_argument("--run-dir",type=Path,required=True)
    exp.add_argument("--destination",type=Path,required=True)
    probe = commands.add_parser("environment",help="Report optional torch/CUDA/nvidia-smi availability")
    probe.add_argument("--device",default="auto")
    probe.add_argument("--output",type=Path,default=Path("artifacts/detection/runtime.json"))
    track = sub.add_parser("local-track",help="Replay imported detections through ByteTrack on CPU")
    track.add_argument("--config",type=Path,required=True)


def execute(args):
    if args.command == "local-track":
        from .tracking import local_track
        summary = local_track(args.config)
        print(f"CPU ByteTrack: {len(summary['files'])} cameras; no YOLO/Re-ID/global association")
        return 0
    if args.detection_command == "prepare-v2":
        from .portable import build_code_artifact
        artifact = build_code_artifact(Path.cwd(),args.output_dir,args.evidence_dir,existing_evidence=True)
        print(f"Detection v2 bundle: {args.output_dir/artifact['filename']} SHA256 {artifact['sha256']}")
        print(f"Unpacked folder: {args.output_dir/artifact['directory']}; source evidence unchanged")
    elif args.detection_command == "prepare":
        from .evidence import prepare_evidence
        from .portable import build_code_artifact
        evidence = prepare_evidence(Path.cwd(),args.evidence_dir)
        artifact = build_code_artifact(Path.cwd(),args.output_dir,args.evidence_dir)
        print(f"Prepared source evidence for {len(evidence['cameras'])} cameras; upload is manual")
        print(f"Code artifact: {artifact['filename']} SHA256 {artifact['sha256']}")
        print(f"Unpacked code folder: {args.output_dir / artifact['directory']}")
    elif args.detection_command == "import":
        from .transport import import_run
        target = import_run(args.source,args.evidence_dir,args.target_dir,args.model_lock)
        print(f"Validated import: {target}")
    elif args.detection_command == "annotate":
        from .evidence import load_evidence
        from .video import render_local_tracks
        result = render_local_tracks(args.track_dir,load_evidence(args.evidence_dir),args.raw_dir,args.output_dir)
        print(f"Derived local-ID videos: {len(result['videos'])}; no inference")
    elif args.detection_command == "export":
        from .transport import export_run
        print(export_run(args.run_dir,args.destination))
    else:
        from .common import atomic_json
        from .runtime import environment
        runtime = environment(args.device)
        atomic_json(args.output,runtime)
        print(f"Device {runtime['selected_device']}; CUDA available: {runtime['torch_cuda_is_available']}")
    return 0
