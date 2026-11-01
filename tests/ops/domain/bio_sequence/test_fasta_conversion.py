"""Synthetic FASTA/annotation merge tests, including unordered and mismatched IDs."""

import json
from pathlib import Path

import pytest

from tools.fmt_conversion.bio_sequence.fasta_to_dj import fasta_to_dj


def test_join_preserves_annotations_and_sample_seed(tmp_path: Path) -> None:
    """Multiline FASTA joins reordered annotations without losing any metadata."""
    fasta, annotations, out = [tmp_path / name for name in ("in.fa", "in.jsonl", "out.jsonl")]
    fasta.write_text(">a description\nAC\nDE\n>b\nGGG\n>c\nAAA\n")
    rows = [
        {"seq_id": i, "length": n, "gtdb.F_name": "family", "lineage": ["x"]} for i, n in [("c", 3), ("b", 3), ("a", 4)]
    ]
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = fasta_to_dj(fasta, annotations, out, "protein", temp_dir=tmp_path)
    actual = [json.loads(line) for line in out.read_text().splitlines()]
    assert report["written"] == 3
    assert actual[0] == dict(rows[2], sequence="ACDE", seq_type="protein")
    fasta_to_dj(fasta, annotations, out, "protein", sample_size=2, seed=42, temp_dir=tmp_path)
    first = out.read_text()
    fasta_to_dj(fasta, annotations, out, "protein", sample_size=2, seed=42, temp_dir=tmp_path)
    assert first == out.read_text()
    assert len(first.splitlines()) == 2


@pytest.mark.parametrize(
    "annotations",
    [
        [{"seq_id": "b"}],
        [{"seq_id": "a"}, {"seq_id": "a"}],
        [{"seq_id": "a", "length": 9}],
        [{"seq_id": "a", "sequence": "overwritten"}],
    ],
)
def test_bad_annotations_leave_no_partial_export(tmp_path: Path, annotations: list[dict]) -> None:
    """Unknown, duplicate, inconsistent or colliding annotations fail explicitly."""
    fasta, ann, out = [tmp_path / name for name in ("a.fa", "a.jsonl", "out.jsonl")]
    fasta.write_text(">a\nACD\n")
    ann.write_text("".join(json.dumps(row) + "\n" for row in annotations))
    with pytest.raises(ValueError):
        fasta_to_dj(fasta, ann, out, "protein", temp_dir=tmp_path)
    assert not out.exists()
    assert not list(tmp_path.glob("dj-fasta-*"))


def test_disk_reserve_stops_before_index_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The converter refuses to write scratch when the configured disk reserve is absent."""
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(100, 99, 1))
    with pytest.raises(RuntimeError, match="disk reserve"):
        fasta_to_dj(
            tmp_path / "absent.fa",
            tmp_path / "absent.jsonl",
            tmp_path / "out.jsonl",
            "protein",
            temp_dir=tmp_path,
            min_free_gib=5,
        )
    assert not list(tmp_path.glob("dj-fasta-*"))


def test_sqlite_connection_closes_before_return(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlinking scratch must also release the SQLite descriptor immediately."""
    import sqlite3

    real_connect = sqlite3.connect
    closed = []

    class TrackingConnection(sqlite3.Connection):
        def close(self):
            closed.append(True)
            return super().close()

    monkeypatch.setattr(sqlite3, "connect", lambda path: real_connect(path, factory=TrackingConnection))
    fasta, ann, out = [tmp_path / name for name in ("in.fa", "in.jsonl", "out.jsonl")]
    fasta.write_text(">a\nACD\n")
    ann.write_text('{"seq_id":"a"}\n')
    fasta_to_dj(fasta, ann, out, "protein", temp_dir=tmp_path)
    assert closed == [True]
