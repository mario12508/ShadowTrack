"""Nested sync CLI; does not alter inspect behavior or output files."""

import json
from pathlib import Path


def add_parser(sub):
    parser = sub.add_parser("sync",help="Evidence-constrained time alignment")
    modes = parser.add_subparsers(dest="sync_command",required=True)
    analyze = modes.add_parser("analyze",help="Reuse saved frame PTS and investigate audio")
    analyze.add_argument("--stats",type=Path,default=Path("artifacts/inspect/stats.json"))
    analyze.add_argument("--evidence",type=Path,default=Path("configs/sync_evidence.json"))
    analyze.add_argument("--output-dir",type=Path,default=Path("artifacts/sync"))
    analyze.add_argument("--reference-camera",default="3")
    analyze.add_argument("--no-audio",action="store_true",help="Record NOT_REQUESTED, never reuse unspecified audio findings")
    analyze.add_argument("--require-complete",action="store_true",help="Exit 1 when calibration is PARTIAL/UNCALIBRATED")
    report = modes.add_parser("report",help="Render existing time_alignment.json, no media decoding")
    report.add_argument("--alignment",type=Path,default=Path("artifacts/sync/time_alignment.json"))
    report.add_argument("--output-dir",type=Path)
    collect = modes.add_parser("collect",help="Collect unchecked event evidence at explicit per-camera raw PTS")
    collect.add_argument("--request",type=Path,required=True)
    collect.add_argument("--stats",type=Path,default=Path("artifacts/inspect/stats.json"))
    collect.add_argument("--evidence",type=Path,default=Path("configs/sync_evidence.json"))
    collect.add_argument("--output-dir",type=Path,required=True)


def execute(args):
    if args.sync_command == "analyze":
        from .engine import analyze
        result = analyze(args.stats,args.evidence,args.output_dir,args.reference_camera,not args.no_audio)
        print(f"Acceptance: {result['acceptance_status']}; linked cameras: {result['linked_camera_count']}/{len(result['cameras'])}")
        print(args.output_dir.resolve()/"time_alignment.json")
        return 1 if args.require_complete and not result["fully_calibrated"] else 0
    if args.sync_command == "report":
        from .report import render
        render(args.alignment,args.output_dir)
        print((args.output_dir or args.alignment.parent).resolve()/"report.html")
        return 0
    from .collect import collect
    manifest = collect(args.request,args.stats,args.evidence,args.output_dir)
    print(f"Collected {len(manifest['events'])} unchecked candidate events: {args.output_dir/'manifest.json'}")
    return 0
