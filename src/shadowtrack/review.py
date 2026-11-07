"""Attach an explicitly supplied visual review only to matching source hashes."""

import json
from pathlib import Path


def attach_review(report: dict, path: Path | None) -> None:
    if path is None:
        return
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
        actual = {Path(c["file"]).name: c.get("sha256") for c in report["cameras"]}
        actual.update({Path(m["file"]).name: m["sha256"] for m in report["metadata"]["files"]})
        if actual != review["input_sha256"]:
            raise ValueError("Review input hashes do not match the current inputs")
        if not isinstance(review["sections"], list):
            raise ValueError("Review sections must be a list")
        for section in review["sections"]:
            if not isinstance(section["title"], str) or not isinstance(section["paragraphs"], list) or not all(isinstance(p, str) for p in section["paragraphs"]):
                raise ValueError("Invalid review section")
        report["analyst_review"] = review
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["issues"].append({"code": "review_unavailable", "severity": "warning", "message": str(exc)})
