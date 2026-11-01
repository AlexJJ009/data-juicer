"""Verify that failed demo preparation preserves evidence and removes owned scratch."""

import json
from pathlib import Path

import pytest

from demos.process_topic2 import run_refine


def test_conversion_failure_cleans_owned_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit conversion error leaves metrics, but no temporary database or Arrow cache."""
    output = tmp_path / "run"

    def fail(*args, **kwargs):
        (output / "tmp/database").write_text("scratch")
        (output / "datasets-cache/arrow").write_text("scratch")
        raise ValueError("synthetic conversion failure")

    monkeypatch.setattr(run_refine, "fasta_to_dj", fail)
    with pytest.raises(ValueError, match="synthetic"):
        run_refine.run("dna", tmp_path, output)
    assert not (output / "tmp").exists()
    assert not (output / "datasets-cache").exists()
    assert json.loads((output / "metrics.json").read_text())["kind"] == "dna"


def test_debug_log_cap_preserves_tail_and_other_levels(tmp_path: Path) -> None:
    """Repeated upstream diagnostics cannot consume the run's disk reserve."""
    debug = tmp_path / "job_DEBUG.txt"
    info = tmp_path / "job.txt"
    debug.write_bytes(b"x" * 4096 + b"last diagnostic")
    info.write_text("all INFO evidence")
    from demos.process_topic2.run_refine import cap_debug_logs

    discarded = cap_debug_logs(tmp_path, limit_bytes=128)
    assert discarded == 4096 + len(b"last diagnostic")
    assert debug.stat().st_size == 0
    assert (tmp_path / "debug_tail.log").read_bytes().endswith(b"last diagnostic")
    assert (tmp_path / "debug_tail.log").stat().st_size <= 128
    assert info.read_text() == "all INFO evidence"
