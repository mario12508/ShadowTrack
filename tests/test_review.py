import json

from shadowtrack.review import attach_review


def test_visual_review_bound_to_source_hashes(tmp_path):
    review = {"input_sha256": {"a.mkv": "abc"}, "sections": [{"title": "Scene", "paragraphs": ["Hypothesis only"]}]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    report = {"cameras": [{"file": "a.mkv", "sha256": "abc"}], "metadata": {"files": []}, "issues": []}
    attach_review(report, path)
    assert report["analyst_review"] == review
    changed = {"cameras": [{"file": "a.mkv", "sha256": "different"}], "metadata": {"files": []}, "issues": []}
    attach_review(changed, path)
    assert "analyst_review" not in changed
    assert changed["issues"][0]["code"] == "review_unavailable"
