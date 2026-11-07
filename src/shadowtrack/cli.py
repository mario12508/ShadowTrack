"""Command-line interface with TOML configuration and explicit exit statuses."""

import argparse
import sys
import tomllib
from pathlib import Path

from .audit import inspect
from .probe import AuditError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="shadowtrack")
    sub = parser.add_subparsers(dest="command", required=True)
    from .sync.cli import add_parser
    add_parser(sub)
    from .detection.cli import add_parser as add_detection_parser
    add_detection_parser(sub)
    from .mvp.cli import add_parser as add_mvp_parser
    add_mvp_parser(sub)
    audit = sub.add_parser("inspect", help="Audit MKV metadata, decoded timestamps and previews")
    audit.add_argument("--config", type=Path)
    audit.add_argument("--raw-dir", type=Path)
    audit.add_argument("--metadata-dir", type=Path)
    audit.add_argument("--output-dir", type=Path)
    audit.add_argument("--review-file", type=Path, help="Hash-bound visual review JSON (not source metadata)")
    audit.add_argument("--previews", type=int, choices=range(16, 25))
    audit.add_argument("--expected-cameras", type=int)
    audit.add_argument("--timeout-seconds", type=float)
    audit.add_argument("--strict", action="store_true", help="Return 1 on warnings, too")
    args = parser.parse_args(argv)
    if args.command == 'mvp':
        try:
            from .mvp.cli import execute
            return execute(args)
        except (AuditError,OSError,ValueError,TypeError,KeyError,ImportError) as exc:
            print(f'shadowtrack mvp: {exc}',file=sys.stderr)
            return 2
    if args.command == "sync":
        try:
            from .sync.cli import execute
            return execute(args)
        except (AuditError, OSError, ValueError, TypeError, KeyError, ImportError) as exc:
            print(f"shadowtrack sync: {exc}", file=sys.stderr)
            return 2
    if args.command in ("detection","local-track"):
        try:
            from .detection.cli import execute
            return execute(args)
        except (AuditError,OSError,ValueError,TypeError,KeyError,ImportError) as exc:
            print(f"shadowtrack detection: {exc}",file=sys.stderr)
            return 2
    options = {"raw_dir": Path("data/raw"), "metadata_dir": Path("data/metadata"),
               "output_dir": Path("artifacts/inspect")}
    try:
        if args.config:
            with args.config.open("rb") as stream:
                loaded = tomllib.load(stream).get("inspect", {})
            allowed = {"raw_dir", "metadata_dir", "output_dir", "previews", "expected_cameras", "timeout_seconds", "fps_relative_tolerance", "review_file"}
            if set(loaded) - allowed:
                raise AuditError(f"Unknown config keys: {sorted(set(loaded) - allowed)}")
            options.update(loaded)
            for field in ("raw_dir", "metadata_dir", "output_dir", "review_file"):
                if field in loaded:
                    options[field] = args.config.resolve().parent / loaded[field]
        for field in ("raw_dir", "metadata_dir", "output_dir", "previews", "expected_cameras", "timeout_seconds", "review_file"):
            if getattr(args, field) is not None:
                options[field] = getattr(args, field)
        for field in ("previews", "expected_cameras"):
            value = options.get(field, 20 if field == "previews" else 5)
            if type(value) is not int or value < 1:
                raise AuditError(f"{field} must be a positive integer")
        if options.get("timeout_seconds", 1800) <= 0:
            raise AuditError("timeout_seconds must be positive")
        if not 0 <= options.get("fps_relative_tolerance", 0.02) < 1:
            raise AuditError("fps_relative_tolerance must be in [0, 1)")
        result = inspect(**options)
    except (AuditError, OSError, ValueError, TypeError) as exc:
        print(f"shadowtrack: {exc}", file=sys.stderr)
        return 2
    print(f"Report: {Path(options['output_dir']).resolve() / 'report.html'}")
    print(f"Status: {result['status']}; issues: {len(result['issues'])}")
    return 1 if result["status"] == "error" or (args.strict and result["issues"]) else 0
